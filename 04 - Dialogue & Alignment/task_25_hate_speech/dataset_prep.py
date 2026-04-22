"""
dataset_prep.py — Task 25: Hate Speech Detection
Prepara il dataset evalitahf/hatespeech_detection per il fine-tuning di Gemma 3 1B.
"""
from __future__ import annotations
import argparse, logging, os
from datasets import load_dataset, DatasetDict

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Analizza il testo per rilevare linguaggio d'odio. Rispondi con: odio o non_odio."""
LABEL_MAP = {"0": "non_odio", "1": "odio"}

def make_prompt(user_text: str, label_str: str) -> str:
    if SYSTEM_PROMPT:
        return (
            "<start_of_turn>user\n" + SYSTEM_PROMPT + "\n\n"
            "Testo: " + user_text.strip() + "\n"
            "<end_of_turn>\n<start_of_turn>model\n" + label_str.strip() + "\n<end_of_turn>"
        )
    return (
        "<start_of_turn>user\nTesto: " + user_text.strip() + "\n"
        "<end_of_turn>\n<start_of_turn>model\n" + label_str.strip() + "\n<end_of_turn>"
    )

def load_main_dataset() -> DatasetDict:
    logger.info("Caricamento dataset: evalitahf/hatespeech_detection...")
    load_kwargs = {}
    # nessuna config specifica
    ds = load_dataset("evalitahf/hatespeech_detection", **load_kwargs)
    def transform(row):
        user_text = str(row.get("text", ""))
        label_raw = row.get("label", "")
        label_str = {"0": "non_odio", "1": "odio"}.get(int(label_raw) if str(label_raw).isdigit() else label_raw, str(label_raw))
        return {"text": make_prompt(user_text, label_str)}
    return ds.map(transform, remove_columns=ds["train"].column_names)

def prepare_dataset(save_path: str = "./data/task_25") -> DatasetDict:
    ds = load_main_dataset()
    for split, data in ds.items():
        logger.info(f"  {split}: {len(data)} esempi")
    os.makedirs(save_path, exist_ok=True)
    ds.save_to_disk(save_path)
    logger.info(f"Dataset salvato in: {save_path}")
    return ds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara dataset Task 25 — Hate Speech Detection")
    parser.add_argument("--save_path", default="./data/task_25")
    args = parser.parse_args()
    prepare_dataset(args.save_path)
