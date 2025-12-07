# src/llama2_scaling_casi.py

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from config import Models


def load_casi(path: str) -> pd.DataFrame:
    """
    Expects a CSV with at least:
        - example_id
        - acronym
        - expansion
        - prompt: full text shown to model
    This can be an export from the upstream CASI evaluation format.
    """
    df = pd.read_csv(path)
    required = {"example_id", "acronym", "expansion", "prompt"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CASI CSV missing columns: {missing}")
    return df


def load_wimbd_counts(path: str) -> pd.DataFrame:
    """
    wimbd_counts.csv should have:
        - acronym
        - expansion
        - count_final   (the N_hat_final(A, E) in the report)
    """
    df = pd.read_csv(path)
    required = {"acronym", "expansion", "count_final"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"WIMB(D) counts CSV missing columns: {missing}")
    return df


def build_generator(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        max_new_tokens=32,
        do_sample=True,
        temperature=0.2,
        top_p=0.95,
        top_k=40,
    )
    return gen


def eval_casi_model(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    gen = build_generator(model_name)
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=model_name):
        prompt = row["prompt"]
        gold = str(row["expansion"]).lower().strip()
        out = gen(prompt)[0]["generated_text"]
        if out.startswith(prompt):
            ans = out[len(prompt) :].strip()
        else:
            ans = out.strip()
        norm_pred = ans.lower().strip()
        correct = int(norm_pred == gold)
        rows.append(
            {
                "model": model_name,
                "example_id": row["example_id"],
                "acronym": row["acronym"],
                "expansion": row["expansion"],
                "prediction": ans,
                "correct": correct,
            }
        )
    return pd.DataFrame(rows)


def add_frequency_bins(results: pd.DataFrame, wimbd: pd.DataFrame) -> pd.DataFrame:
    merged = results.merge(
        wimbd, on=["acronym", "expansion"], how="left", validate="many_to_one"
    )
    merged["log_count"] = np.log1p(merged["count_final"].fillna(0.0))

    # Tertiles by log_count
    quantiles = merged["log_count"].quantile([1 / 3, 2 / 3]).values
    q1, q2 = quantiles[0], quantiles[1]

    def bin_freq(x):
        if x <= q1:
            return "low"
        if x <= q2:
            return "medium"
        return "high"

    merged["freq_bin"] = merged["log_count"].apply(bin_freq)
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--casi_csv", type=str, required=True)
    parser.add_argument("--wimbd_counts_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    args = parser.parse_args()

    casi = load_casi(args.casi_csv)
    wimbd = load_wimbd_counts(args.wimbd_counts_csv)
    models = Models()

    all_results = []
    for model_name in [
        models.llama2_7b,
        models.llama2_13b,
        models.llama2_34b,
        models.llama2_70b,
    ]:
        df_model = eval_casi_model(casi, model_name)
        all_results.append(df_model)

    results = pd.concat(all_results, ignore_index=True)
    results = add_frequency_bins(results, wimbd)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)

    # Print accuracy per bin for sanity
    summary = (
        results.groupby(["model", "freq_bin"])["correct"]
        .mean()
        .reset_index()
        .rename(columns={"correct": "accuracy"})
    )
    print("\nCASI accuracy by frequency bin and model:")
    print(summary)


if __name__ == "__main__":
    main()
