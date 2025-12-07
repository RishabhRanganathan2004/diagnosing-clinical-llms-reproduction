# src/run_modern_baselines_medlingo.py

import argparse
import csv
from pathlib import Path
from typing import List, Dict

import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from config import Models


def load_medlingo_questions(path: str) -> pd.DataFrame:
    """
    Expects a CSV with at least:
        - question_id
        - prompt (text shown to the model)
        - answer (gold expansion string)
    For convenience you can symlink/copy the upstream questions CSV
    and add a 'prompt' column if needed.
    """
    df = pd.read_csv(path)
    if "prompt" not in df.columns:
        # If using upstream format, you may need to construct the prompt here.
        # For the class project, I precomputed a 'prompt' column offline.
        raise ValueError(
            "CSV must contain a 'prompt' column with the full MedLingo question text."
        )
    required = {"question_id", "prompt", "answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in MedLingo CSV: {missing}")
    return df


def build_generator(model_name: str, device: int = 0):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        device=device,
        max_new_tokens=32,
        do_sample=True,
        temperature=0.2,
        top_p=0.95,
        top_k=40,
    )
    return gen


def evaluate_model_on_medlingo(
    generator,
    questions: pd.DataFrame,
    model_name: str,
) -> Dict[str, float]:
    records: List[Dict] = []
    correct = 0

    for _, row in tqdm(questions.iterrows(), total=len(questions), desc=model_name):
        prompt = row["prompt"]
        gold = str(row["answer"]).lower().strip()
        qid = row["question_id"]

        out = generator(prompt)[0]["generated_text"]
        # Use only the part after the prompt (simple heuristic)
        if out.startswith(prompt):
            answer_text = out[len(prompt) :].strip()
        else:
            answer_text = out.strip()

        norm_pred = answer_text.lower().strip()
        is_correct = gold in norm_pred
        if is_correct:
            correct += 1

        records.append(
            {
                "question_id": qid,
                "prompt": prompt,
                "gold": gold,
                "prediction_raw": answer_text,
                "correct": int(is_correct),
                "model": model_name,
            }
        )

    acc = correct / len(questions)
    return {"accuracy": acc, "records": records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions_csv",
        type=str,
        required=True,
        help="Path to MedLingo questions CSV.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Where to write per-question predictions.",
    )
    args = parser.parse_args()

    questions = load_medlingo_questions(args.questions_csv)
    models = Models()

    results = []
    all_records: List[Dict] = []

    for model_name in [models.gemma2_9b, models.mistral_nemo_12b]:
        gen = build_generator(model_name)
        res = evaluate_model_on_medlingo(gen, questions, model_name)
        print(f"{model_name} MedLingo accuracy: {res['accuracy']:.3f}")
        results.append({"model": model_name, "accuracy": res["accuracy"]})
        all_records.extend(res["records"])

    # Write per-question CSV (for plotting / inspection)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "question_id",
                "prompt",
                "gold",
                "prediction_raw",
                "correct",
            ],
        )
        writer.writeheader()
        for r in all_records:
            writer.writerow(r)

    # Also print a small summary table
    print("\nSummary:")
    for r in results:
        print(f"{r['model']}: {r['accuracy']:.3f}")


if __name__ == "__main__":
    main()
