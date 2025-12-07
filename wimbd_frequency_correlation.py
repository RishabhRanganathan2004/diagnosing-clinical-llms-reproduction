# src/wimbd_frequency_correlation.py

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--casi_results_csv",
        type=str,
        required=True,
        help="Output of llama2_scaling_casi.py with columns: model, acronym, expansion, correct, count_final/log_count/freq_bin",
    )
    parser.add_argument(
        "--wimbd_counts_csv",
        type=str,
        required=True,
        help="CSV with acronym, expansion, count_final.",
    )
    args = parser.parse_args()

    results = pd.read_csv(args.casi_results_csv)
    wimbd = pd.read_csv(args.wimbd_counts_csv)

    merged = results.merge(
        wimbd, on=["acronym", "expansion"], how="left", validate="many_to_one"
    )
    merged["log_count"] = np.log1p(merged["count_final"].fillna(0.0))

    # Compute per-(acronym, expansion) accuracy across all models
    grouped = (
        merged.groupby(["acronym", "expansion"])
        .agg(
            log_count=("log_count", "first"),
            acc=("correct", "mean"),
        )
        .reset_index()
    )

    rho, p = spearmanr(grouped["log_count"], grouped["acc"])
    print(f"Spearman correlation between log frequency and CASI accuracy:")
    print(f"rho = {rho:.3f}, p = {p:.3e}")

    # Optionally, save the merged data for plotting
    grouped.to_csv("wimbd_freq_vs_accuracy.csv", index=False)


if __name__ == "__main__":
    main()
