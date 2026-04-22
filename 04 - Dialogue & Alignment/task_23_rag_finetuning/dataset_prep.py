"""
dataset_prep.py — Task 23: RAG Fine-Tuning
Prepara il dataset crux82/squad_it per il fine-tuning di Gemma 3 4B.
"""
from __future__ import annotations
import argparse, logging, os
from datasets import load_dataset, DatasetDict

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Rispondi alla domanda usando solo le informazioni nel contesto. Se la risposta non è presente, rispondi 'Non so'."""
LABEL_MAP = {}

def make_prompt(user_text: str, label_str: str) -> str:
    if SYSTEM_PROMPT:
        return (
            "<start_of_turn>user\n" + SYSTEM_PROMPT + "\n\n"
            "Contesto recuperato e Domanda: " + user_text.strip() + "\n"
            "<end_of_turn>\n<start_of_turn>model\n" + label_str.strip() + "\n<end_of_turn>"
        )
    return (
        "<start_of_turn>user\nContesto recuperato e Domanda: " + user_text.strip() + "\n"
        "<end_of_turn>\n<start_of_turn>model\n" + label_str.strip() + "\n<end_of_turn>"
    )

def load_main_dataset() -> DatasetDict:
    logger.info("Caricamento dataset: crux82/squad_it...")
    load_kwargs = {}
    # nessuna config specifica
    ds = load_dataset("crux82/squad_it", **load_kwargs)
    def transform(row):
        user_text = str(row.get("context", ""))
        label_raw = row.get("answers", "")
        label_str = str(label_raw)
        return {"text": make_prompt(user_text, label_str)}
    return ds.map(transform, remove_columns=ds["train"].column_names)

def prepare_dataset(save_path: str = "./data/task_23") -> DatasetDict:
    ds = load_main_dataset()
    for split, data in ds.items():
        logger.info(f"  {split}: {len(data)} esempi")
    os.makedirs(save_path, exist_ok=True)
    ds.save_to_disk(save_path)
    logger.info(f"Dataset salvato in: {save_path}")
    return ds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara dataset Task 23 — RAG Fine-Tuning")
    parser.add_argument("--save_path", default="./data/task_23")
    args = parser.parse_args()
    prepare_dataset(args.save_path)
