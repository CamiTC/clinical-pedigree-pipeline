# imports
import re                   # regular expression module for pattern matching and cleaning text
import unicodedata          # unicode normalisation
import spacy

# pretrained English model
spacy_nlp = spacy.load("en_core_web_sm")

# dictionary mapping for kinship terms
kinship_map = {
    "mom": "mother", "ma": "mother", "mum": "mother", "mother": "mother",
    "dad": "father", "pop": "father", "father": "father",
    "sis": "sister", "sister": "sister",
    "bro": "brother", "brother": "brother",
    "daughter": "daughter",
    "son": "son",
    "aunt": "aunt", "auntie": "aunt",
    "uncle": "uncle",
    "grandma": "grandmother", "gran": "grandmother",
    "grandpa": "grandfather",
    "cousin": "cousin",
    "wife": "wife",
    "husband": "husband"
}

def preprocess(filepath):
    with open(filepath, encoding="utf-8") as f:
        text = f.read()         # read file into one string

    text = unicodedata.normalize("NFKC", text)          # standardise unicode characters
    text = re.sub(r"\r\n?", "\n", text)                 # consistent line endings

    pattern = r"(?is)\b(patient|clinical geneticist)\s*:\s*(.*?)(?=\n\s*(?:patient|clinical geneticist)\s*:|\Z)"    # split speaker turns   
    turns = re.findall(pattern, text)                   # find and store speaker turns

    preprocessed_output = []

    for turn_id, (speaker, block) in enumerate(turns, start=1):
        speaker = speaker.lower().replace(" ", "_")     # standardise speaker labels
        block = block.lower()
        
        # normalise quotes and apostrophes
        block = (
            block.replace("“", '"')
                .replace("”", '"')
                .replace("’", "'")
                .replace("`", "'")
        )         

        # replace informal terms with formal kinship terms
        for informal, formal in kinship_map.items():
            block = re.sub(rf"\b{re.escape(informal)}\b", formal, block)

        block = re.sub(r"\s+", " ", block).strip()      # clean whitespaces


        # process text with spaCy
        doc = spacy_nlp(block)

        for sent_id, sent in enumerate(doc.sents, start=1):
            sentence = sent.text.strip()                # extract sentence from doc

            # creates a list of dictionary entries containing information about each sentence
            if sentence:
                preprocessed_output.append({
                    "turn_id": turn_id,
                    "sent_id": sent_id,
                    "speaker": speaker,
                    "sentence": sentence,
                    "priority": "primary" if speaker == "patient" else "secondary"
                })

    return preprocessed_output