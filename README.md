# diagnosing-clinical-llms-reproduction
CS 598 Deep Learning For Healthcare: Final Project

# Diagnosing Clinical Knowledge in Open-Source LLMs
### Reproduction of “Diagnosing Our Datasets: How Does My Language Model Learn Clinical Information?”

This repository contains the code used for my CS 598: Deep Learning in Healthcare final project.
The goal is to partially reproduce the clinical jargon experiments from Jia et al. (2025) and
implement several extensions and ablations, as described in the accompanying report.

The repo is organized as follows:

- `upstream_diagnosing_our_datasets/` – Unmodified (or lightly modified) copy of the official
  code release from the original paper: https://github.com/Flora-jia-jfr/diagnosing_our_datasets
  This folder contains:
  - Cleaned **CASI** subset in `datasets/casi/`
  - **MedLingo** prompts in `datasets/MedLingo/questions.csv`
  - **Disputed Medical Claims** in `datasets/disputed_medical_claims/`
  - Shell scripts for running CASI / MedLingo / disputed-claims evaluations.

- `src/` – New code written for this course project:
  - `config.py` – Paths and model IDs used across scripts.
  - `run_modern_baselines_medlingo.py` – Evaluates **Gemma-2 9B** and **Mistral-Nemo 12B**
    on MedLingo (Extension 1 in the report, Table 6).
  - `prompt_ablation_medlingo_misinfo.py` – Runs the prompt-sensitivity experiments on
    MedLingo and a small subset of disputed medical claims
    (Extension 2 in the report, Table 7 and Figure 7).
  - `llama2_scaling_casi.py` – Evaluates LLaMA-2 7B/13B/34B/70B on CASI and groups
    accuracy by jargon frequency bin (Extension 3 in the report, Table 8 and Figure 8).
  - `wimbd_frequency_correlation.py` – Computes Spearman correlations between
    estimated jargon counts from WIMB(D) and CASI accuracy (Section 4.3 of the report).

