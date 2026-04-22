"""
dataset_prep.py — Task 26: Document Segmentation
Prepara il dataset aharley/rvl_cdip per il fine-tuning di Gemma 3 1B.
"""
from __future__ import annotations
import argparse, logging, os
from datasets import load_dataset, DatasetDict

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Analizza la sequenza di pagine e identifica i confini tra documenti. Usa B-DOC per inizio e I-DOC per continuazione."""
LABEL_MAP = {"0": "I-DOC", "1": "B-DOC"}

def make_prompt(user_text: str, label_str: str) -> str:
    if SYSTEM_PROMPT:
        return (
            "<start_of_turn>user\n" + SYSTEM_PROMPT + "\n\n"
            "Testo OCR pagina corrente e pagina precedente: " + user_text.strip() + "\n"
            "<end_of_turn>\n<start_of_turn>model\n" + label_str.strip() + "\n<end_of_turn>"
        )
    return (
        "<start_of_turn>user\nTesto OCR pagina corrente e pagina precedente: " + user_text.strip() + "\n"
        "<end_of_turn>\n<start_of_turn>model\n" + label_str.strip() + "\n<end_of_turn>"
    )

def load_main_dataset() -> DatasetDict:
    logger.info("Caricamento dataset: aharley/rvl_cdip...")
    load_kwargs = {}
    # nessuna config specifica
    ds = load_dataset("aharley/rvl_cdip", **load_kwargs)
    def transform(row):
        user_text = str(row.get("image", ""))
        label_raw = row.get("label", "")
        label_str = {"0": "I-DOC", "1": "B-DOC"}.get(int(label_raw) if str(label_raw).isdigit() else label_raw, str(label_raw))
        return {"text": make_prompt(user_text, label_str)}
    return ds.map(transform, remove_columns=ds["train"].column_names)

def prepare_dataset(save_path: str = "./data/task_26") -> DatasetDict:
    ds = load_main_dataset()
    for split, data in ds.items():
        logger.info(f"  {split}: {len(data)} esempi")
    os.makedirs(save_path, exist_ok=True)
    ds.save_to_disk(save_path)
    logger.info(f"Dataset salvato in: {save_path}")
    return ds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara dataset Task 26 — Document Segmentation")
    parser.add_argument("--save_path", default="./data/task_26")
    args = parser.parse_args()
    prepare_dataset(args.save_path)
