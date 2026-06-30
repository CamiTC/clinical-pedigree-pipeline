import re
import spacy
import logging
import medspacy
from loguru import logger
from transformers import pipeline
from medspacy.target_matcher import TargetRule
from spacy.util import filter_spans

# load spaCy
spcy_nlp = spacy.load("en_core_web_sm")

# disables warning logs
logging.getLogger("PyRuSH").setLevel(logging.WARNING)
logger.disable("PyRuSH")


# -------------------
# Global variables
# -------------------

FEMALE_RELATIVES = {"mother", "grandmother", "great-grandmother", "aunt", "great-aunt", "sister", "daughter", "wife", "niece"}
MALE_RELATIVES = {"father", "grandfather", "great-grandfather", "uncle", "great-uncle", "brother", "son", "husband", "nephew"}

SELF_TERMS = {"i", "me", "my", "mine"}

THIRD_PERSON_TERMS = {"he", "she", "his", "her", "they", "their"}

STATUS_TERMS = {"healthy", "born", "alive"}

DEATH_TERMS = {"died", "passed away", "deceased", "dead"}

FAMILY_TERMS = {
    "mother", "father", "sister", "brother",
    "daughter", "son", "aunt", "uncle",
    "cousin", "grandmother", "grandfather",
    "great-aunt", "great-uncle",
    "great-grandmother", "great-grandfather",
    "cousin once removed",
    "wife", "husband", "partner"
}

NO_SIDE_RELATIVES = {"brother", "sister", "son", "daughter", "wife", "husband", "partner"}

PLURAL_RELATIVES = {
    "sons": "son",
    "daughters": "daughter",
    "brothers": "brother",
    "sisters": "sister"
}

COMPOUND_RELATIVES = {
    "father's sister": "paternal aunt",
    "father's brother": "paternal uncle",
    "father's mother": "paternal grandmother",
    "father's father": "paternal grandfather",
    "mother's sister": "maternal aunt",
    "mother's brother": "maternal uncle",
    "mother's mother": "maternal grandmother",
    "mother's father": "maternal grandfather"
}

MATERNAL_TERMS = ["mom's", "mother's", "mom's side", "mother's side", "maternal", "mum's"]
PATERNAL_TERMS = ["dad's", "father's", "dad's side", "father's side", "paternal"]

PROBAND_PATTERNS = {
    "daughter": [
        r"\bmy daughter\b",
        r"\bmy girl\b",
        r"\bshe\b.*\b(has|had|was|is|developed|diagnosed)\b"
    ],
    "son": [
        r"\bmy son\b",
        r"\bmy boy\b",
        r"\bhe\b.*\b(has|had|was|is|developed|diagnosed)\b"
    ],
    "mother": [r"\bmy mother\b", r"\bmy mum\b", r"\bmy mom\b"],
    "father": [r"\bmy father\b", r"\bmy dad\b"],
    "sister": [r"\bmy sister\b"],
    "brother": [r"\bmy brother\b"],
    "aunt": [r"\bmy aunt\b"],
    "uncle": [r"\bmy uncle\b"],
    "grandmother": [r"\bmy grandmother\b"],
    "grandfather": [r"\bmy grandfather\b"],
    "great-aunt": [r"\bmy great[\s-]aunt\b"],
    "great-uncle": [r"\bmy great[\s-]uncle\b"],
    "great-grandmother": [r"\bmy great[\s-]grandmother\b"],
    "great-grandfather": [r"\bmy great[\s-]grandfather\b"],
    "cousin": [r"\bmy cousin\b"],
    "cousin once removed": [r"\bmy cousin once removed\b", r"\bmy (?:first )?cousin(?:'s)? (?:child|son|daughter|baby)\b"],
    "proband": [
        r"\bi have\b",
        r"\bi had\b",
        r"\bi was diagnosed\b",
        r"\bmy symptoms\b",
        r"\bmy gp\b",
        r"\bmy period\b",
        r"\bmy pregnancy\b",
    ]
}


# ------------------------------
# Model initialisation functions
# ------------------------------

def load_family_member_extractor():
    med_nlp = medspacy.load()
    matcher = med_nlp.get_pipe("medspacy_target_matcher")

    # add family member extraction rules
    rules = [TargetRule(term, "FAMILY") for term in FAMILY_TERMS]
    matcher.add(rules)

    return med_nlp

family_member_nlp = load_family_member_extractor()


def load_biobert_ner():
    return pipeline(
        "ner",
        model="alvaroalon2/biobert_diseases_ner",
        tokenizer="alvaroalon2/biobert_diseases_ner",
        aggregation_strategy="simple"
    )

biobert_ner = load_biobert_ner()


# -------------------------------
# Family member entity extraction
# -------------------------------

def extract_family_members(sentence):
    """
    Extracts family member entities from the text.
    """
    doc = family_member_nlp(sentence)
    return [ent.text for ent in doc.ents if ent.label_ == "FAMILY"]


# -----------------------------------
# Medical condition entity extraction
# -----------------------------------

def is_valid_condition(text, sentence):
    """
    Filters BioBERT output to keep only valid medical conditions.
    Rejects sentences, questions, non-nouns and family terms.
    Returns: True if valid, False otherwise.
    """
    text = text.lower().strip().replace("'", "'")
    sentence = sentence.lower().strip()

    clean_text = text.replace("'", "").replace(".", "").strip()

    if len(clean_text) < 3:
        return False

    # reject sentence-like spans
    if len(text.split()) > 5 or "," in text or text.endswith("."):
        return False

    if text == sentence or "?" in text:
        return False

    if not any(char.isalpha() for char in text):
        return False

    # reject spans without a noun or adjective
    doc = spcy_nlp(text)
    if not any(token.pos_ in {"NOUN", "PROPN", "ADJ"} for token in doc):
        return False

    # reject spans starting or ending with function words
    bad_pos = {"PRON", "CCONJ", "SCONJ", "ADP", "DET", "AUX", "VERB"}
    if len(doc) == 0 or doc[0].pos_ in bad_pos or doc[-1].pos_ in bad_pos:
        return False

    # reject spans containing family terms
    if set(text.split()) & FAMILY_TERMS:
        return False

    return True


def extract_conditions(sentence, min_score=0.70):
    """
    Extracts medical conditions using BioBERT NER.
    Returns: a list of valid conditions with scores and character positions.
    """
    conditions = []

    # loop through entities to clean text + reject weak predictions and invalid conditions
    for ent in biobert_ner(sentence):
        text = ent["word"].replace("##", "").replace("'", "'").strip(".,;:!? ")

        if ent["score"] < min_score:
            continue

        if not is_valid_condition(text, sentence):
            continue

        # store valid conditions
        conditions.append({
            "condition": text,
            "condition_score": round(float(ent["score"]), 3),
            "start": ent.get("start"),
            "end": ent.get("end")
        })

    return conditions


# -----------------------------
# medspaCy polarity and context
# -----------------------------

def condition_polarity(sentence, conditions):
    """
    Uses medspaCy context detection to label each condition as present, absent or uncertain.
    Returns: conditions with polarity.
    """

    # create spaCy doc + run medspaCy pipeline
    doc = family_member_nlp(sentence)
    new_entities = list(doc.ents)

    # loop through BioBERT conditions + create spaCy span
    for cond in conditions:
        span = doc.char_span(
            cond["start"], cond["end"],
            label="CONDITION",
            alignment_mode="contract"
        )

        if span is not None:
            new_entities.append(span)

    doc.ents = filter_spans(new_entities)                   # remove duplicate entities from doc

    family_member_nlp.get_pipe("medspacy_context")(doc)     # run medspaCy context detector

    output = []

    # loop through entities and store matched conditions
    for ent in doc.ents:
        if ent.label_ != "CONDITION":
            continue

        # mathces medspaCy entities back to BioBERT condition using character pos.
        matched = next(
            (cond for cond in conditions
             if cond["start"] == ent.start_char and cond["end"] == ent.end_char),
            None
        )

        if matched is None:
            continue

        # polarity labels
        if ent._.is_negated:
            polarity = "absent"
        elif ent._.is_uncertain:
            polarity = "uncertain"
        else:
            polarity = "present"

        output.append({
            "condition": ent.text,
            "condition_score": matched["condition_score"],
            "polarity": polarity,
            "start": matched["start"],
            "end": matched["end"]
        })

    return output


# -----------------
# Proband detection
# -----------------

def detect_proband(preprocessed_output, max_turns=10):
    """
    Detects the likely proband based on regex patterns in the first few turns of conversation.
    Returns: the detected relative label, or "proband" if no match.
    """

    # collecting first sentences into one block
    initial_sentences = " ".join(
        item["sentence"].lower()
        for item in preprocessed_output
        if item["turn_id"] <= max_turns
    )

    scores = {rel: 0 for rel in PROBAND_PATTERNS}

    # loop through relatives and patterns, increase score for matches
    for relative, patterns in PROBAND_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, initial_sentences):
                scores[relative] += 1

    # return best match or "proband" if no score
    best_relative = max(scores, key=scores.get)
    return best_relative if scores[best_relative] > 0 else "proband"


# -------------------------
# Extract person attributes
# -------------------------

def get_side(sentence, relative, proband_role="proband"):
    """
    Determines which side of the family each relative belongs to.
    Returns: self, maternal, paternal, None or unknown.
    """
    sentence = sentence.lower()
    relative = relative.lower()

    if relative in [proband_role, "proband"]:
        return "self"
    if relative in NO_SIDE_RELATIVES:                   # siblings + partners
        return None
    if relative.startswith("maternal"):
        return "maternal"
    if relative.startswith("paternal"):
        return "paternal"
    if any(term in sentence for term in MATERNAL_TERMS):
        return "maternal"
    if relative == "mother":
        return "maternal"
    if any(term in sentence for term in PATERNAL_TERMS):
        return "paternal"
    if relative == "father":
        return "paternal"

    return "unknown"


def extract_age_info(sentence):
    """
    Extracts current age, age at event and units using regex patterns.
    Returns: dict with current_age, current_age_unit, age_at_event, age_at_event_unit.
    """
    sentence = sentence.lower()

    # initialise result dict
    age_info = {
        "current_age": None,
        "current_age_unit": None,
        "age_at_event": None,
        "age_at_event_unit": None,
    }

    def normalise_unit(unit):
        """
        Maps age units to standard labels.
        Defaults to years if no unit.
        """
        if unit is None:
            return "years"
        
        unit = unit.lower()
        
        if unit.startswith("day"): return "days"
        if unit.startswith("week"): return "weeks"
        if unit.startswith(("month", "mo")): return "months"
        
        return "years"

    # search for current age patterns
    # group(1): number, group(2): unit
    match = re.search(
        r"\b(?:i'm|i'm|i am|is|she's|he's|aged|age)\s+(\d{1,3})\s*(days?|weeks?|months?|years?|yrs?|mos?)?\b",
        sentence
    )
    if match:
        age_info["current_age"] = int(match.group(1))
        age_info["current_age_unit"] = normalise_unit(match.group(2))

    match = re.search(
        r"\b(?:at|diagnosed at|died at|developed at|had .* at)\s+(\d{1,3})\s*(days?|weeks?|months?|years?|yrs?|mos?)?\b",
        sentence
    )
    if match:
        age_info["age_at_event"] = int(match.group(1))
        age_info["age_at_event_unit"] = normalise_unit(match.group(2))

    # search for age at even patterns
    match = re.search(r"\b(?:in his|in her|in their)\s+(\d{2,3}s)\b", sentence)
    if match:
        age_info["age_at_event"] = match.group(1)
        age_info["age_at_event_unit"] = "years"

    return age_info


def has_age_info(age_info):
    """
    Checks if age info is extracted.
    Returns: True if info extracted, False otherwise.
    """
    return (
        age_info["current_age"] is not None
        or age_info["age_at_event"] is not None
    )


def is_deceased(sentence):
    """
    Checks if there is a death term in a sentence.
    Returns: True if yes, False otherwise.
    """
    return any(term in sentence.lower() for term in DEATH_TERMS)


# --------------------------
# Relative inference helpers
# --------------------------

def infer_compound_relative(sentence):
    """
    Converts compound relationships (e.g. "father's sister") into a single pedigree label (e.g. "paternal aunt").
    Returns: the inferred relative, or None.
    """
    sentence = sentence.lower().replace("'", "'")

    for phrase, relation in COMPOUND_RELATIVES.items():
        if phrase in sentence:
            return relation
    return None


def resolve_pronoun_relative(sentence, last_relative):
    """
    Link the last seen relative if a sentence has no family member but contains a third-person pronoun.
    Returns: last_relative or None.
    """
    pronouns = {
        token.text.lower()
        for token in spcy_nlp(sentence)
        if token.pos_ == "PRON"
    }

    if pronouns & {"he", "his", "she", "her", "they", "their"}:
        return last_relative
    return None


def extract_plural_relatives(sentence):
    """
    Handles patterns like "two daughters, ages 8 and 5".
    Returns: a list of individual relative records with ages.
    """
    match = re.search(
        r"\b(?:two|2)\s+(sons|daughters|brothers|sisters),?\s+ages?\s+(\d{1,3})\s+and\s+(\d{1,3})",
        sentence.lower()
    )
    if not match:
        return []

    relative = PLURAL_RELATIVES[match.group(1)]

    return [
        {
            "relative": relative,
            "age": {
                "current_age": int(age),
                "current_age_unit": "years",
                "age_at_event": None,
                "age_at_event_unit": None,
            }
        }
        for age in [match.group(2), match.group(3)]
    ]


def normalize_relative(relative):
    """
    Removes side information from the relative label.
    Returns: cleaned relationship label.
    """
    relative = relative.lower().strip()

    if relative == "proband":
        return "proband"
    
    return relative.replace("maternal ", "").replace("paternal ", "")


def is_about_proband(sentence, relatives, proband_relative):
    """
    Determines whether a sentence is about the proband.
    Returns: True or False.
    """
    tokens = {token.text.lower() for token in spcy_nlp(sentence.lower())}

    # if detected relative is the proband, assign to proband
    if proband_relative in relatives:
        return True

    # if another relative is mentioned, don't assume it is about the proband
    if relatives:
        return False

    # if proband is the speaker/patient, self-references indicate proband
    if proband_relative == "proband" and tokens & SELF_TERMS:
        return True

    # if proband is not the speaker, pronoun-only sentences are ambiguous
    if proband_relative != "proband" and tokens & THIRD_PERSON_TERMS:
        return False

    return False


# ----------------
# Record creation
# ----------------

def base_record(item, relative, transcript_id, proband_relative):
    """
    Creates the base record for each relative.
    Returns: a dict.
    """
    sentence = item["sentence"]
    normalized_relative = normalize_relative(relative)

    if normalized_relative == proband_relative:
        pedigree_relative = "proband"
        side = "self"
    else:
        pedigree_relative = normalized_relative
        side = get_side(sentence, relative, proband_relative)

    rel_norm = pedigree_relative.lower()
    if rel_norm in FEMALE_RELATIVES:
        sex = "female"
    elif rel_norm in MALE_RELATIVES:
        sex = "male"
    else:
        sex = None  # proband, cousin, and other ambiguous relatives

    return {
        "transcript_id": transcript_id,
        "person_id": None,
        "speaker_relation_to_proband": None,
        "relative_as_spoken": relative,
        "relative": pedigree_relative,
        "side": side,
        "sex": sex,
        "age": extract_age_info(sentence),
        "deceased": is_deceased(sentence),
        "source_sentence": sentence,
        "turn_id": item["turn_id"],
        "sent_id": item["sent_id"],
        "speaker": item["speaker"],
        "priority": item["priority"],
    }


def build_person_record(item, relative, transcript_id, proband_relative):
    """
    Creates a full record for a relative with no condition.
    Returns: dict.
    """
    record = base_record(item, relative, transcript_id, proband_relative)
    record.update({
        "condition": None,
        "polarity": "none",
        "confidence": 0.75,
        "source": "family_member_only",
    })
    return record


def build_plural_person_record(item, plural_info, transcript_id, proband_relative):
    """
    Creates a full record for a relative extracted from a plural pattern.
    Returns: dict.
    """
    record = base_record(item, plural_info["relative"], transcript_id, proband_relative)
    record["age"] = plural_info["age"]
    record.update({
        "condition": None,
        "polarity": "none",
        "confidence": 0.85,
        "source": "plural_relative_rule",
    })
    return record


# --------------------------
# Relative-condition linking
# --------------------------

def link_relative_to_condition(sentence, conditions):
    """
    Links each condition to the nearest relative by character position.
    Returns: list of (relative, condition) pairs.
    """
    doc = family_member_nlp(sentence)
    family_entities = [ent for ent in doc.ents if ent.label_ == "FAMILY"]
    links = []

    for condition in conditions:
        closest = min(
            family_entities,
            key=lambda ent: abs(ent.start_char - condition["start"]),
            default=None
        )

        if closest:
            links.append({"relative": closest.text.lower(), "condition": condition})

    return links


def link_relations(item, relatives, conditions, transcript_id, proband_relative):
    """
    Links relatives to conditions.
    Returns: list of records with condition info.
    """
    if not relatives:
        return []

    records = []

    # single relative
    if len(relatives) == 1:
        pairs = [{"relative": relatives[0], "condition": cond} for cond in conditions]

    # multiple relatives in sentence
    else:
        pairs = link_relative_to_condition(item["sentence"], conditions)

    # update record to include condition, polarity, etc
    for pair in pairs:
        record = base_record(item, pair["relative"], transcript_id, proband_relative)
        record.update({
            "condition": pair["condition"]["condition"],
            "polarity": pair["condition"]["polarity"],
            "confidence": None,
            "source": "medspacy_biobert",
            "condition_score": pair["condition"]["condition_score"],
        })
        records.append(record)

    return records


# -------------------
# Confidence scoring
# -------------------

def calc_confidence(row):
    """
    Calculate confidence score from 0-1.
    - BioBERT token probability (0.50): reflects condition evidence and is a real model confidence score.
    - Source reliability (0.30): neural extraction model vs rule-based extraction. Neural models score higher.
    - Record completeness (0.20): known polarity, side, and age information.
    """
    # condition evidence (max 0.50)
    condition_score = row.get("condition_score") or 0.0
    has_condition   = row["condition"] not in {None, "None"}
    condition_evidence = condition_score * 0.50 if has_condition else 0.0

    # source reliability (max 0.30)
    # neural pipeline = full score; rule-based only = reduced score
    if row["source"] == "medspacy_biobert":
        source_reliability = 0.30
    else:                         
        source_reliability = 0.15

    # record completeness (max 0.20)
    record_completeness = 0.0
    if row["polarity"] in {"present", "absent"}: record_completeness += 0.10  # clear polarity
    elif row["polarity"] == "uncertain": record_completeness += 0.05  # flagged uncertain
    if row["side"] in {"maternal", "paternal", "self"}: record_completeness += 0.05
    if has_age_info(row["age"]): record_completeness += 0.05

    return round(min(condition_evidence + source_reliability + record_completeness, 1.0), 3)


# -------------------------
# Pipeline helper functions
# -------------------------

def should_add_proband(item, sentence, relatives, raw_conditions, proband_relative):
    """
    Returns: True if the sentence is about the proband and contains condition, age, status or death.
    """
    has_relevant_info = (
        bool(raw_conditions)
        or has_age_info(extract_age_info(sentence))
        or any(term in sentence.lower() for term in STATUS_TERMS)
        or is_deceased(sentence)
    )
    
    return (
        item["speaker"] == "patient"
        and has_relevant_info
        and is_about_proband(sentence, relatives, proband_relative)
        and proband_relative not in relatives
    )


def filter_empty_records(records):
    """
    Removes records with no condition, no age and no deceased status.
    """
    return [
        record for record in records
        if not (
            record["condition"] is None
            and record["source"] == "family_member_only"
            and not has_age_info(record["age"])
            and record["deceased"] is False
        )
    ]


def assign_person_ids(relations, proband_relative):
    """
    Assigns a unique person ID to each individual.
    Proband is always P0. Others are assigned in order.
    """
    person_map = {}
    counter = 1

    for rel in relations:
        # proband always assigned P0 and self.
        if rel["relative"] == "proband":
            rel["person_id"] = "P0"
            rel["side"] = "self"
            continue

        # inlude age in key to distinguish between relatives with same relative label and side info
        key = (
            rel["transcript_id"], rel["relative"], rel["side"], rel["age"]["current_age"]
        ) if rel["source"] == "plural_relative_rule" else (
            rel["transcript_id"], rel["relative"], rel["side"]
        )

        # assign new ID to new individuals
        if key not in person_map:
            person_map[key] = f"P{counter}"
            counter += 1

        # assign same key to same individual
        rel["person_id"] = person_map[key]

    return relations


# ------------------
# Main NLP execution
# ------------------

def extract_from_sentence(item, transcript_id, proband_relative, forced_relatives=None):
    """
    Extracts all pedigree records from a single sentence.
    Returns: a list of records.
    """
    sentence = item["sentence"]
    output = []

    # handle multiple daughters/sons patterns
    for plural_info in extract_plural_relatives(sentence):
        output.append(
            build_plural_person_record(item, plural_info, transcript_id, proband_relative)
        )

    if forced_relatives is not None:
        relatives = forced_relatives
    else:
        compound = infer_compound_relative(sentence)
        relatives = [compound] if compound else extract_family_members(sentence)

    raw_conditions = extract_conditions(sentence)

    if should_add_proband(item, sentence, relatives, raw_conditions, proband_relative):
        relatives.append(proband_relative)

    if not relatives:
        return filter_empty_records(output)

    # person records (no condition)
    for relative in relatives:
        age_info = extract_age_info(sentence)
        if (
            has_age_info(age_info)
            or any(term in sentence.lower() for term in STATUS_TERMS)
            or is_deceased(sentence)
            or bool(raw_conditions)
        ):
            output.append(build_person_record(item, relative, transcript_id, proband_relative))

    # condition records
    if raw_conditions:
        conditions = condition_polarity(sentence, raw_conditions)
        linked = link_relations(item, relatives, conditions, transcript_id, proband_relative)
        for rec in linked:
            rec["confidence"] = calc_confidence(rec)
        output.extend(linked)

    return filter_empty_records(output)


def run_nlp_pipeline(preprocessed_output, transcript_id):
    """
    Runs the full NLP pipeline on a preprocessed transcript.
    Returns: a list of structured pedigree records.
    """
    proband_relative = detect_proband(preprocessed_output)
    results = []
    last_relative = None

    for item in preprocessed_output:
        sentence = item["sentence"]
        relatives = extract_family_members(sentence)        # extract explicitly mentioned family members
        inferred = infer_compound_relative(sentence)        # check for compound relatives

        if inferred:
            relatives = [inferred]

        # if no relatives found, resolve pronouns using last known relative
        if not relatives:
            pronoun_relative = resolve_pronoun_relative(sentence, last_relative)
            if pronoun_relative:
                relatives = [pronoun_relative]

        # update last known relative
        if relatives:
            last_relative = relatives[0]

        # extract all pedigree records
        results.extend(
            extract_from_sentence(
                item=item,
                transcript_id=transcript_id,
                proband_relative=proband_relative,
                forced_relatives=relatives
            )
        )

    #  assign person ids
    return assign_person_ids(results, proband_relative)
