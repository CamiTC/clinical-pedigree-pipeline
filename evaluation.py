import pandas as pd
from pathlib import Path

# define tasks to evaluate
EVAL_TASKS = {
    "family_members":["transcript_id", "relative"],
    "conditions":["transcript_id", "condition"],
    "relative_condition":["transcript_id", "relative", "condition"],
    "side":["transcript_id", "relative", "side"],
    "polarity":["transcript_id", "relative", "condition", "polarity"],
    "deceased":["transcript_id", "relative", "deceased"],
    "sex":["transcript_id", "relative", "sex"],
    "age_at_event":["transcript_id", "relative", "condition", "age_at_event"],
    "full_record":["transcript_id", "relative", "side", "condition", "polarity"],
}


def normalize_df(df):
    df = df.copy()

    for col in df.columns:
        df[col] = (
            df[col]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.strip()
            .str.replace("’", "’", regex=False) # apostrophe
            .str.replace("’", "’", regex=False) # quote
            .str.replace(r"\.0$", "", regex=True) # strip float
            .str.replace(r"^(maternal|paternal)\s+", "", regex=True) # strip side prefix
            .str.replace(r"^first cousin once removed$", "cousin once removed", regex=True)
            .replace({"none": "", "nan": "", "null": "", "false": "false", "true": "true"})
        )

    return df


def make_tuples(df, columns):
    df = normalize_df(df)

    # add missing columns as empty strings
    for col in columns:
        if col not in df.columns:
            df[col] = ""

    # convert each row to a tuple
    return set(tuple(row) for row in df[columns].values)


def evaluate_by_columns(pred_df, gold_df, columns):
    pred_set = make_tuples(pred_df, columns)    # set of predicted tuples
    gold_set = make_tuples(gold_df, columns)    # set of ground truth tuples

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    return {
        "TP": tp, "FP": fp, "FN": fn,
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1": round(f1, 3),
    }


def evaluate_all_tasks(pred_df, gold_df, method_name):
    rows = []

    for task_name, columns in EVAL_TASKS.items():
        metrics = evaluate_by_columns(pred_df, gold_df, columns)
        rows.append({"method": method_name, "task": task_name, **metrics})

    return rows


def run_evaluation(gold_path, nlp_dir, llm_dir, eval_dir):
    gold_df = pd.read_csv(gold_path)    # load full ground truth

    # paths
    nlp_dir = Path(nlp_dir)
    llm_dir = Path(llm_dir)
    eval_dir = Path(eval_dir)

    per_transcript_dir = eval_dir / "per_transcript"
    eval_dir.mkdir(parents=True, exist_ok=True)
    per_transcript_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []       # results for all transcripts

    for transcript_id in sorted(gold_df["transcript_id"].unique()):
        print(f"Evaluating {transcript_id}")

        gold_t = gold_df[gold_df["transcript_id"] == transcript_id]
        nlp_file = nlp_dir / f"{transcript_id}_nlp.csv"
        llm_file = llm_dir / f"{transcript_id}_llm.csv"

        # skip if missing transcript
        if not nlp_file.exists() or not llm_file.exists():
            print(f"Missing {transcript_id} file")
            continue

        nlp_t = pd.read_csv(nlp_file)
        llm_t = pd.read_csv(llm_file)

        transcript_rows = []        # results for individual transcript only

        for row in evaluate_all_tasks(nlp_t, gold_t, "NLP"):
            row["transcript_id"] = transcript_id
            transcript_rows.append(row)
            all_rows.append(row)

        for row in evaluate_all_tasks(llm_t, gold_t, "NLP + LLM"):
            row["transcript_id"] = transcript_id
            transcript_rows.append(row)
            all_rows.append(row)

        # save individual transcript
        transcript_eval_df = pd.DataFrame(transcript_rows)
        transcript_eval_df.to_csv(
            per_transcript_dir / f"{transcript_id}_evaluation.csv",
            index=False
        )
        print(f"Saved {transcript_id}_evaluation.csv")

    # average scores for 10 transcripts
    average_scores_df = (
        pd.DataFrame(all_rows)
        .groupby(["method", "task"])[["Precision", "Recall", "F1"]]
        .mean()
        .reset_index()
        .round(3)
    )
    average_scores_df.to_csv(eval_dir / "average_scores.csv", index=False)
    print("Saved average_scores.csv")

    return average_scores_df
