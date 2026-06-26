import json
import os
import pandas as pd
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

NO_SIDE_RELATIVES = {"brother", "sister", "son", "daughter", "wife", "husband", "partner"}

# ------------------------------
# Normalisation helper functions
# ------------------------------

def normalise_condition(condition):
    if condition is None or str(condition).strip() == "":
        return "None"
    if str(condition).lower() in ["nan", "null", "none"]:
        return "None"
    return condition


def normalise_polarity(condition, polarity):
    if condition == "None":
        return "none"
    if polarity is None or str(polarity).strip() == "":
        return "none"
    return polarity


def normalise_side(relative, side):
    if relative == "proband":
        return "self"
    if relative in NO_SIDE_RELATIVES:
        return ""
    if side is None or str(side).strip() == "":
        return ""
    if str(side).lower() in ["nan", "null", "none"]:
        return ""
    return side


def _clean_age(val):
    """
    Strips .0 from float ages returned by the LLM.
    """
    if val is None:
        return None
    age_str = str(val).strip()
    return age_str[:-2] if age_str.endswith(".0") else age_str


def records_to_df(records):
    """
    Converts NLP or LLM records to a DataFrame.
    """
    rows = []

    for record in records:
        # assign proband info
        if record.get("person_id") == "P0":
            record["relative"] = "proband"
            record["side"] = "self"

        # get relative, side, condition, polarity
        relative = record.get("relative")
        side = normalise_side(relative, record.get("side"))
        condition = normalise_condition(record.get("condition"))
        polarity = normalise_polarity(condition, record.get("polarity"))

        # NLP returns age as a nested dict; LLM returns flat fields
        age_dict = record.get("age") or {}
        current_age = _clean_age(age_dict.get("current_age") or record.get("current_age"))
        current_age_unit = age_dict.get("current_age_unit") or record.get("current_age_unit")
        age_at_event = _clean_age(age_dict.get("age_at_event") or record.get("age_at_event"))
        age_at_event_unit = age_dict.get("age_at_event_unit") or record.get("age_at_event_unit")

        rows.append({
            "transcript_id": record.get("transcript_id"),
            "person_id": record.get("person_id"),
            "relative": relative,
            "side": side,
            "sex": record.get("sex"),
            "condition": condition,
            "polarity": polarity,
            "current_age": current_age,
            "current_age_unit": current_age_unit,
            "age_at_event": age_at_event,
            "age_at_event_unit":age_at_event_unit,
            "deceased": record.get("deceased"),
            "source_sentence": record.get("source_sentence"),
            "turn_id": record.get("turn_id"),
            "sent_id": record.get("sent_id"),
            "speaker": record.get("speaker"),
            "priority": record.get("priority"),
            "confidence": record.get("confidence"),
            "source": record.get("source"),
            "condition_score": record.get("condition_score"),
        })

    return pd.DataFrame(rows)

llm_records_to_df = records_to_df


# --------------
# LLM refinement
# --------------

def refine_with_llm(transcript_id, transcript_text, nlp_records):

    # collect low confidence records
    low_confidence_records = [
        row for row in nlp_records
        if row.get("confidence") is not None
        and row["confidence"] < 0.75
    ]

    schema_example = """
{
  "records": [
    {
      "transcript_id": "T1",
      "person_id": "P0",
      "relative": "proband",
      "side": "self",
      "sex": null,
      "condition": null,
      "polarity": "none",
      "current_age": null,
      "current_age_unit": null,
      "age_at_event": null,
      "age_at_event_unit": null,
      "deceased": false,
      "source_sentence": "",
      "turn_id": null,
      "sent_id": null,
      "speaker": "",
      "priority": "",
      "confidence": null,
      "source": "llm_refined",
      "condition_score": null
    }
  ]
}
"""

    prompt = f"""
You are a clinical genetics pedigree extraction system. Refine the NLP draft using the transcript as the primary source of truth.

PROBAND (P0)
- P0 = the person the consultation is about (not necessarily the speaker).
- If a parent discusses their child's symptoms, the child is the proband.
- Always set relative = "proband" and person_id = "P0" for the proband.

SEX
- sex = "male", "female" or null (null only if genuinely unknown).
- Infer from: relative terms, titles (Mr./Mrs.), pronouns (he/she), explicit statements, reproductive history.
- CRITICAL reproductive/sex indicators for the proband:
  - Any mention of miscarriage, pregnancy, periods, giving birth -> proband is female.
  - Any mention of prostate issues, vasectomy -> proband is male.
  - "My daughters/sons" by birth (i.e. the proband bore them) is consistent with female.
- If the patient says "I had a miscarriage" or "one early miscarriage before my first daughter", the proband is FEMALE, do not assign male.
- For the proband: look for doctor pronouns ("he has", "she has"), titles, and any reproductive or gender-specific clinical detail.

STEPS
1. Identify the proband (P0) and the speaker.
   - If the speaker is discussing someone else's condition (e.g. a parent describing their child), P0 is that other person.
   - State explicitly: "Speaker is [role]. P0 is [role]."
2. Translate ALL relative terms into P0's perspective before extracting any records.
   - If speaker is P0's parent, shift every term up one generation (see RELATIVE TERMS rule below).
   - Write out the translation: "speaker's mother -> P0's grandmother", etc.
3. Determine sex for every person.
4. Review low-confidence records: keep, correct, or remove based on the transcript.
5. Extract ALL missing information directly from the transcript (do not rely on the NLP draft alone):
   - relatives (including healthy ones), conditions, ages, deceased status, family side, sex.
6. Resolve pronouns and indirect references (she/he/her/his -> correct person).

RULES
- One record per condition per person.
- polarity: present, absent, uncertain, none.
- side: maternal, paternal, self, null (null for siblings, children, partners unless stated).
- RELATIVE TERMS must always be from P0's perspective, not the speaker's.
  Step 1: identify who is speaking and who is P0.
  Step 2: translate every relative term the speaker uses into P0's equivalent:
    If the speaker IS P0:
      "my mother" -> mother, "my aunt" -> aunt, "my cousin's child" -> cousin once removed
    If the speaker is P0's PARENT (e.g. a mother describing her child's condition):
      "my mother/father" = P0's grandmother/grandfather
      "my brother/sister" = P0's uncle/aunt
      "my aunt/uncle" = P0's great-aunt/great-uncle
      "my grandmother" = P0's great-grandmother
      "my grandfather" = P0's great-grandfather
      "my cousin" = P0's first cousin once removed
      "my cousin's child" = P0's second cousin
      "my husband's/wife's X" = P0's father's/mother's X (use paternal/maternal side)
  Always add the correct side (maternal/paternal) based on which parent is speaking.
- Genetic conditions: extract suspected diagnoses (polarity = "uncertain") and confirmed pathogenic variants (polarity = "present"). 
    - Use the syndrome name when known (e.g. "Marfan syndrome", "Fabry disease"), otherwise use gene and variant type (e.g. "FBN1 pathogenic variant"). 
    - Do NOT extract gene test recommendations (e.g. "we will test for GLA"), these are clinical actions, not findings.
- Approximate ages: use the form from the transcript (e.g. "80s", "mid-70s"), do not convert to a single number.
- source_sentence: must directly support the condition, age, or relationship, not demographics or occupation.
- source = "llm_refined" for all records.
- Prefer recall over precision, missing a finding is worse than including an uncertain one.
- Do not hallucinate. Do not merge distinct individuals.
- Exclude non-clinical details (occupation, lifestyle) unless clinically relevant.

OUTPUT
- Valid JSON only. No markdown, no explanations.
- Schema:
{schema_example}

TRANSCRIPT:
{transcript_text}

NLP DRAFT:
{json.dumps(nlp_records, indent=2)}

LOW-CONFIDENCE RECORDS:
{json.dumps(low_confidence_records, indent=2)}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert clinical genetics pedigree extraction system."
                    "The transcript is the primary source of truth."
                    "The NLP extraction is only a draft that may contain errors."
                    "Return only valid JSON. No explanations, markdown, code fences, or comments."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    result = json.loads(response.choices[0].message.content)
    records = result.get("records", [])

    for record in records:
        record["source"] = "llm_refined"
        record["transcript_id"] = transcript_id

    return records
