# src/prompt_ablation_medlingo_misinfo.py

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from config import Models


def build_generator(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        device_map="auto",
        max_new_tokens=32,
        do_sample=True,
        temperature=0.2,
        top_p=0.95,
        top_k=40,
    )


def make_medlingo_prompt(base_prompt: str, variant: str, one_shot_example: str = "") -> str:
    """Variants:
    - baseline
    - no_clinical
    - one_shot
    - cot
    """
    if variant == "baseline":
        return base_prompt
    if variant == "no_clinical":
        # crude removal of explicit clinical wording
        return base_prompt.replace("clinical", "").replace("Clinical", "")
    if variant == "one_shot":
        return one_shot_example + "\n\n" + base_prompt
    if variant == "cot":
        return "Think step by step as a clinician.\n\n" + base_prompt
    raise ValueError(f"Unknown prompt variant: {variant}")


def eval_medlingo_variants(df: pd.DataFrame, model_name: str, variants: List[str]) -> pd.DataFrame:
    gen = build_generator(model_name)

    # Use the first row as a simple one-shot example
    example = df.iloc[0]
    one_shot_example = f"Example:\n{example['prompt']}\nCorrect expansion: {example['answer']}"

    rows = []
    for variant in variants:
        correct = 0
        for _, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc=f"{model_name} [{variant}]",
        ):
            prompt = make_medlingo_prompt(row["prompt"], variant, one_shot_example)
            gold = str(row["answer"]).lower().strip()

            out = gen(prompt)[0]["generated_text"]
            if out.startswith(prompt):
                answer_text = out[len(prompt) :].strip()
            else:
                answer_text = out.strip()

            norm_pred = answer_text.lower().strip()
            is_correct = gold in norm_pred
            if is_correct:
                correct += 1

        acc = correct / len(df)
        rows.append(
            {
                "model": model_name,
                "variant": variant,
                "accuracy": acc,
                "n": len(df),
            }
        )
    return pd.DataFrame(rows)


def categorize_misinfo_response(text: str) -> str:
    """
    Very simple heuristic categorization into 'denies', 'neutral', or 'supports'.
    This is just for the small-scale misinfo extension in the report.
    """
    t = text.lower()
    deny_keywords = ["false", "not true", "incorrect", "no evidence", "myth", "does not"]
    support_keywords = ["is true", "correct", "is proven", "indeed"]

    if any(k in t for k in deny_keywords):
        return "denies"
    if any(k in t for k in support_keywords):
        return "supports"
    return "neutral"


def eval_misinfo_subset(df: pd.DataFrame, model_name: str, n_examples: int = 20) -> pd.DataFrame:
    """
    Expects a CSV with columns:
        - claim_text
        - direct_prompt
        - presuppositional_prompt
    We evaluate only the first n_examples rows for the class project.
    """
    gen = build_generator(model_name)
    df = df.head(n_examples)

    rows = []
    for prompt_type in ["direct_prompt", "presuppositional_prompt"]:
        for _, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc=f"{model_name} misinfo [{prompt_type}]",
        ):
            prompt = row[prompt_type]
            out = gen(prompt)[0]["generated_text"]
            label = categorize_misinfo_response(out)
            rows.append(
                {
                    "model": model_name,
                    "prompt_type": prompt_type,
                    "claim_text": row["claim_text"],
                    "response": out,
                    "label": label,
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions_csv",
        type=str,
        required=True,
        help="MedLingo questions CSV with 'prompt' and 'answer' columns.",
    )
    parser.add_argument(
        "--misinfo_csv",
        type=str,
        required=True,
        help="CSV of disputed medical claims with direct and presuppositional prompts.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    medlingo = pd.read_csv(args.questions_csv)
    misinfo = pd.read_csv(args.misinfo_csv)

    models = Models()
    variants = ["baseline", "no_clinical", "one_shot", "cot"]

    # MedLingo prompt variants
    med_rows = []
    for model_name in [models.llama3_8b_instruct, models.olmo_7b_instruct]:
        df_res = eval_medlingo_variants(medlingo, model_name, variants)
        med_rows.append(df_res)
    med_results = pd.concat(med_rows, ignore_index=True)
    med_results.to_csv(out_dir / "medlingo_prompt_variants_summary.csv", index=False)

    # Misinfo subset (LLaMA-2 13B in the paper, but we reuse LLaMA-3 8B here for simplicity)
    misinfo_results = eval_misinfo_subset(misinfo, models.llama3_8b_instruct, n_examples=20)
    misinfo_results.to_csv(out_dir / "misinfo_prompt_type_results.csv", index=False)

    # Print aggregate distribution for sanity check
    print("\nMedLingo prompt variants summary:")
    print(med_results)

    print("\nMisinfo label distribution:")
    print(misinfo_results.groupby(["prompt_type", "label"]).size())


if __name__ == "__main__":
    main()
