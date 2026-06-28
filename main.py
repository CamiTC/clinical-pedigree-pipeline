import pandas as pd
from pathlib import Path

from preprocessing import preprocess
from nlp_extraction import run_nlp_pipeline
from llm_refinement import refine_with_llm, records_to_df, llm_records_to_df
from pedigree_generation import draw_pedigree
from evaluation import run_evaluation

# ----------------------
# input and output paths
# ----------------------

PROJECT_ROOT = Path(__file__).parent

TRANSCRIPT_DIR = PROJECT_ROOT / "data" / "transcripts"
GROUND_TRUTH = PROJECT_ROOT / "data" / "ground_truth" / "ground_truth.csv"

PREPROCESS_DIR = PROJECT_ROOT / "outputs" / "preprocessed"
NLP_DIR = PROJECT_ROOT / "outputs" / "nlp"
LLM_DIR = PROJECT_ROOT / "outputs" / "llm"
PEDIGREE_DIR = PROJECT_ROOT / "outputs" / "pedigrees"
EVAL_DIR = PROJECT_ROOT / "outputs" / "evaluation"

for d in [PREPROCESS_DIR, NLP_DIR, LLM_DIR,
          PEDIGREE_DIR / "llm", PEDIGREE_DIR / "ground_truth", EVAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ----------------------
# run pipeline functions
# ----------------------

def run_preprocessing():
    print("Preprocessing:")

    for transcript_file in sorted(TRANSCRIPT_DIR.glob("*.txt")):
        transcript_id = transcript_file.stem
        print(f"  {transcript_id}")

        preprocessed_output = preprocess(transcript_file)
        pd.DataFrame(preprocessed_output).to_csv(
            PREPROCESS_DIR / f"{transcript_id}_preprocessed.csv",
            index=False
        )

    print("Done.\n")


def run_nlp():
    print("NLP Extraction:")

    for preprocess_file in sorted(PREPROCESS_DIR.glob("*_preprocessed.csv")):
        transcript_id = preprocess_file.stem.replace("_preprocessed", "")
        print(f"  {transcript_id}")

        preprocessed_output = pd.read_csv(preprocess_file).to_dict(orient="records")
        nlp_output = run_nlp_pipeline(preprocessed_output, transcript_id)
        records_to_df(nlp_output).to_csv(
            NLP_DIR / f"{transcript_id}_nlp.csv",
            index=False
        )

    print("Done.\n")


def run_llm():
    print("LLM Refinement:")

    for transcript_file in sorted(TRANSCRIPT_DIR.glob("*.txt")):
        transcript_id = transcript_file.stem
        nlp_path = NLP_DIR / f"{transcript_id}_nlp.csv"

        if not nlp_path.exists():
            print(f"  Skip {transcript_id} (missing)")
            continue

        print(f"  {transcript_id}")

        with open(transcript_file, "r", encoding="utf-8") as f:
            transcript_text = f.read()

        nlp_df = pd.read_csv(nlp_path)
        nlp_records = nlp_df.where(pd.notnull(nlp_df), None).to_dict(orient="records")

        llm_records = refine_with_llm(transcript_id, transcript_text, nlp_records)
        llm_records_to_df(llm_records).to_csv(
            LLM_DIR / f"{transcript_id}_llm.csv",
            index=False
        )

    print("Done.\n")


def run_pedigrees():
    print("Pedigree Generation:")

    for llm_file in sorted(LLM_DIR.glob("*_llm.csv")):
        transcript_id = llm_file.stem.replace("_llm", "")
        print(f"  {transcript_id}")

        llm_df = pd.read_csv(llm_file)
        draw_pedigree(llm_df, PEDIGREE_DIR / "llm", f"{transcript_id}_llm_pedigree")

    if GROUND_TRUTH.exists():
        gt_df = pd.read_csv(GROUND_TRUTH)

        for transcript_id in sorted(gt_df["transcript_id"].unique()):
            print(f"  {transcript_id} (ground truth)...")
            gt_t = gt_df[gt_df["transcript_id"] == transcript_id]
            draw_pedigree(gt_t, PEDIGREE_DIR / "ground_truth", f"{transcript_id}_gt_pedigree")

    print("Done.\n")


# -----------
# entry point
# -----------

if __name__ == "__main__":
    run_preprocessing()
    run_nlp()
    run_llm()
    run_pedigrees()
    run_evaluation(
        gold_path=GROUND_TRUTH,
        nlp_dir=NLP_DIR,
        llm_dir=LLM_DIR,
        eval_dir=EVAL_DIR,
    )
