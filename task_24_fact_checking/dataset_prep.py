"""
dataset_prep.py — Task 24: Fact Checking
Prepara il dataset fever/fever per il fine-tuning di Gemma 3 4B.
"""
from __future__ import annotations
import argparse, logging, os
from datasets import load_dataset, DatasetDict

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Verifica il claim usando l'evidenza. Rispondi con: SUPPORTS, REFUTES o NOT_ENOUGH_INFO."""
LABEL_MAP = {"0": "SUPPORTS", "1": "REFUTES", "2": "NOT_ENOUGH_INFO"}

def make_prompt(user_text: str, label_str: str) -> str:
    if SYSTEM_PROMPT:
        return (
            "<start_of_turn>user\n" + SYSTEM_PROMPT + "\n\n"
            "Claim ed Evidenza: " + user_text.strip() + "\n"
            "<end_of_turn>\n<start_of_turn>model\n" + label_str.strip() + "\n<end_of_turn>"
        )
    return (
        "<start_of_turn>user\nClaim ed Evidenza: " + user_text.strip() + "\n"
        "<end_of_turn>\n<start_of_turn>model\n" + label_str.strip() + "\n<end_of_turn>"
    )

def load_main_dataset() -> DatasetDict:
    logger.info("Caricamento dataset: fever/fever...")
    load_kwargs = {}
    kwargs["name"] = "v1.0"
    ds = load_dataset("fever/fever", **load_kwargs)
    def transform(row):
        user_text = str(row.get("claim", ""))
        label_raw = row.get("label", "")
        label_str = {"0": "SUPPORTS", "1": "REFUTES", "2": "NOT_ENOUGH_INFO"}.get(int(label_raw) if str(label_raw).isdigit() else label_raw, str(label_raw))
        return {"text": make_prompt(user_text, label_str)}
    return ds.map(transform, remove_columns=ds["train"].column_names)

def prepare_dataset(save_path: str = "./data/task_24") -> DatasetDict:
    ds = load_main_dataset()
    for split, data in ds.items():
        logger.info(f"  {split}: {len(data)} esempi")
    os.makedirs(save_path, exist_ok=True)
    ds.save_to_disk(save_path)
    logger.info(f"Dataset salvato in: {save_path}")
    return ds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara dataset Task 24 — Fact Checking")
    parser.add_argument("--save_path", default="./data/task_24")
    args = parser.parse_args()
    prepare_dataset(args.save_path)
