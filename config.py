# src/config.py
"""
Central configuration for model IDs and default paths.

Update model names here if you used different checkpoints.
"""

from dataclasses import dataclass

@dataclass
class Paths:
    # Default locations – adjust if you use different layout
    medlingo_questions: str = "../data/medlingo_questions.csv"
    casi_csv: str = "../data/casi_dataset.csv"
    wimbd_counts_csv: str = "../data/wimbd_counts.csv"


@dataclass
class Models:
    # Modern open baselines (Extension 1)
    gemma2_9b: str = "google/gemma-2-9b-it"
    mistral_nemo_12b: str = "mistralai/Mistral-Nemo-Instruct-2407"

    # Prompt ablation models (Extension 2)
    llama3_8b_instruct: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    olmo_7b_instruct: str = "allenai/OLMo-7B-Instruct"

    # LLaMA-2 scaling (Extension 3)
    llama2_7b: str = "meta-llama/Llama-2-7b-hf"
    llama2_13b: str = "meta-llama/Llama-2-13b-hf"
    llama2_34b: str = "meta-llama/Llama-2-34b-hf"
    llama2_70b: str = "meta-llama/Llama-2-70b-hf"
