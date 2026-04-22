"""
dataset_prep.py — Task 13: Named Entity Recognition
Prepara il dataset Babelscape/wikineural per il fine-tuning di Gemma 3 1B.
"""
from __future__ import annotations
import argparse, logging, os
from datasets import load_dataset, DatasetDict

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Esegui il riconoscimento delle entità nominate. Restituisci un JSON con lista di {testo, tipo, inizio, fine}."""
LABEL_MAP = {}

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
    logger.info("Caricamento dataset: Babelscape/wikineural...")
    load_kwargs = {}
    kwargs["name"] = "it"
    ds = load_dataset("Babelscape/wikineural", **load_kwargs)
    def transform(row):
        user_text = str(row.get("tokens", ""))
        label_raw = row.get("ner_tags", "")
        label_str = str(label_raw)
        return {"text": make_prompt(user_text, label_str)}
    return ds.map(transform, remove_columns=ds["train"].column_names)

def prepare_dataset(save_path: str = "./data/task_13") -> DatasetDict:
    ds = load_main_dataset()
    for split, data in ds.items():
        logger.info(f"  {split}: {len(data)} esempi")
    os.makedirs(save_path, exist_ok=True)
    ds.save_to_disk(save_path)
    logger.info(f"Dataset salvato in: {save_path}")
    return ds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara dataset Task 13 — Named Entity Recognition")
    parser.add_argument("--save_path", default="./data/task_13")
    args = parser.parse_args()
    prepare_dataset(args.save_path)
