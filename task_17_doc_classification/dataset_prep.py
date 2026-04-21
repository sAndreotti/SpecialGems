"""
dataset_prep.py — Task 17: Document Classification
Prepara il dataset fancyzhx/ag_news per il fine-tuning di Gemma 3 1B.
"""
from __future__ import annotations
import argparse, logging, os
from datasets import load_dataset, DatasetDict

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Classifica il documento. Rispondi con un JSON: {"categoria": "World|Sports|Business|Sci-Tech"}."""
LABEL_MAP = {"0": "World", "1": "Sports", "2": "Business", "3": "Sci-Tech"}

def make_prompt(user_text: str, label_str: str) -> str:
    if SYSTEM_PROMPT:
        return (
            "<start_of_turn>user\n" + SYSTEM_PROMPT + "\n\n"
            "Documento: " + user_text.strip() + "\n"
            "<end_of_turn>\n<start_of_turn>model\n" + label_str.strip() + "\n<end_of_turn>"
        )
    return (
        "<start_of_turn>user\nDocumento: " + user_text.strip() + "\n"
        "<end_of_turn>\n<start_of_turn>model\n" + label_str.strip() + "\n<end_of_turn>"
    )

def load_main_dataset() -> DatasetDict:
    logger.info("Caricamento dataset: fancyzhx/ag_news...")
    load_kwargs = {}
    # nessuna config specifica
    ds = load_dataset("fancyzhx/ag_news", **load_kwargs)
    def transform(row):
        user_text = str(row.get("text", ""))
        label_raw = row.get("label", "")
        label_str = {"0": "World", "1": "Sports", "2": "Business", "3": "Sci-Tech"}.get(int(label_raw) if str(label_raw).isdigit() else label_raw, str(label_raw))
        return {"text": make_prompt(user_text, label_str)}
    return ds.map(transform, remove_columns=ds["train"].column_names)

def prepare_dataset(save_path: str = "./data/task_17") -> DatasetDict:
    ds = load_main_dataset()
    for split, data in ds.items():
        logger.info(f"  {split}: {len(data)} esempi")
    os.makedirs(save_path, exist_ok=True)
    ds.save_to_disk(save_path)
    logger.info(f"Dataset salvato in: {save_path}")
    return ds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara dataset Task 17 — Document Classification")
    parser.add_argument("--save_path", default="./data/task_17")
    args = parser.parse_args()
    prepare_dataset(args.save_path)
