"""
dataset_prep.py — Task 02: Intent Classification
Prepara il dataset AmazonScience/massive per il fine-tuning di Gemma 3 1B.
"""
from __future__ import annotations
import argparse, logging, os
from datasets import load_dataset, DatasetDict

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Sei un classificatore di intenti. Dato un utterance, rispondi con il nome esatto dell'intento."""
LABEL_MAP = {}

def make_prompt(user_text: str, label_str: str) -> str:
    if SYSTEM_PROMPT:
        return (
            "<start_of_turn>user\n" + SYSTEM_PROMPT + "\n\n"
            "Utterance: " + user_text.strip() + "\n"
            "<end_of_turn>\n<start_of_turn>model\n" + label_str.strip() + "\n<end_of_turn>"
        )
    return (
        "<start_of_turn>user\nUtterance: " + user_text.strip() + "\n"
        "<end_of_turn>\n<start_of_turn>model\n" + label_str.strip() + "\n<end_of_turn>"
    )

def load_main_dataset() -> DatasetDict:
    logger.info("Caricamento dataset: AmazonScience/massive...")
    load_kwargs = {}
    load_kwargs["name"] = "it-IT"
    ds = load_dataset("AmazonScience/massive", **load_kwargs)
    def transform(row):
        user_text = str(row.get("utt", ""))
        label_raw = row.get("intent", "")
        label_str = str(label_raw)
        return {"text": make_prompt(user_text, label_str)}
    return ds.map(transform, remove_columns=ds["train"].column_names)

def prepare_dataset(save_path: str = "./data/task_02") -> DatasetDict:
    ds = load_main_dataset()
    for split, data in ds.items():
        logger.info(f"  {split}: {len(data)} esempi")
    os.makedirs(save_path, exist_ok=True)
    ds.save_to_disk(save_path)
    logger.info(f"Dataset salvato in: {save_path}")
    return ds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara dataset Task 02 — Intent Classification")
    parser.add_argument("--save_path", default="./data/task_02")
    args = parser.parse_args()
    prepare_dataset(args.save_path)
