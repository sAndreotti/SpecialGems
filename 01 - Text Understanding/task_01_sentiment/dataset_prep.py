"""
dataset_prep.py — Task 01: Sentiment Analysis
=============================================
Prepara i dataset di sentiment per il fine-tuning di Gemma 3 1B.

Dataset supportati:
    - evalitahf/sentiment_analysis  (SENTIPOLC 2016, IT, 9.4k)
    - stanfordnlp/sst2               (EN, 67k)
    - cardiffnlp/tweet_eval          (EN, config=sentiment, 45k)
    - mteb/amazon_reviews_multi      (IT/EN, 200k, rating 1-5 stelle)

Formato output (prompt Gemma 3 chat):
    <start_of_turn>user
    Analizza il sentiment del seguente testo e rispondi con una sola parola:
    positivo, negativo o neutro.

    Testo: {testo}
    <end_of_turn>
    <start_of_turn>model
    {label}
    <end_of_turn>
"""

from __future__ import annotations

import argparse
import logging
from typing import Literal

from datasets import load_dataset, DatasetDict, concatenate_datasets

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Mappature label → testo italiano
# ──────────────────────────────────────────────────────────────

SENTIPOLC_LABELS = {0: "negativo", 1: "positivo", 2: "neutro", -1: "neutro"}
SST2_LABELS      = {0: "negativo", 1: "positivo"}
STARS_TO_SENTIMENT = {1: "negativo", 2: "negativo", 3: "neutro", 4: "positivo", 5: "positivo"}


# ──────────────────────────────────────────────────────────────
# Prompt template
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Sei un sistema di analisi del sentiment. "
    "Rispondi sempre e solo con una parola: positivo, negativo o neutro."
)

def make_prompt(testo: str, label: str) -> str:
    """Crea la stringa di training nel formato chat Gemma 3."""
    return (
        f"<start_of_turn>user\n"
        f"{SYSTEM_PROMPT}\n\n"
        f"Testo: {testo.strip()}\n"
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
        f"{label}\n"
        f"<end_of_turn>"
    )


# ──────────────────────────────────────────────────────────────
# Loader per ogni dataset
# ──────────────────────────────────────────────────────────────

def load_sentipolc(max_samples: int = 9000) -> DatasetDict:
    """Carica SENTIPOLC 2016 (IT) da evalitahf."""
    logger.info("Caricamento evalitahf/sentiment_analysis (SENTIPOLC 2016)...")
    ds = load_dataset("evalitahf/sentiment_analysis", trust_remote_code=True)

    def transform(row):
        # La label è l'intero 0=neg, 1=pos, 2=neu (campo 'label' o 'polarity')
        label_int = row.get("label", row.get("polarity", 2))
        label_str = SENTIPOLC_LABELS.get(int(label_int), "neutro")
        text = row.get("text", row.get("sentence", ""))
        return {"text": make_prompt(text, label_str)}

    return ds.map(transform, remove_columns=ds["train"].column_names)


def load_sst2(max_samples: int = 20000) -> DatasetDict:
    """Carica SST-2 (EN) da stanfordnlp."""
    logger.info("Caricamento stanfordnlp/sst2...")
    ds = load_dataset("stanfordnlp/sst2", trust_remote_code=True)

    def transform(row):
        label_str = SST2_LABELS.get(int(row["label"]), "neutro")
        return {"text": make_prompt(row["sentence"], label_str)}

    return ds.map(transform, remove_columns=ds["train"].column_names)


def load_amazon_it(max_samples: int = 10000) -> DatasetDict:
    """Carica Amazon Reviews Multi (IT) — rating 1-5 → sentiment."""
    logger.info("Caricamento mteb/amazon_reviews_multi (config=it)...")
    ds = load_dataset("mteb/amazon_reviews_multi", "it", trust_remote_code=True)

    def transform(row):
        stars = int(row.get("label", 3)) + 1  # mteb usa 0-4
        label_str = STARS_TO_SENTIMENT.get(stars, "neutro")
        text = (row.get("text", "") or "").strip()
        if not text:
            text = (row.get("title", "") or "").strip()
        return {"text": make_prompt(text, label_str)}

    return ds.map(transform, remove_columns=ds["train"].column_names)


# ──────────────────────────────────────────────────────────────
# Funzione principale
# ──────────────────────────────────────────────────────────────

def prepare_dataset(
    source: Literal["sentipolc", "sst2", "amazon_it", "all"] = "sentipolc",
    save_path: str = "./data/task_01",
) -> DatasetDict:
    """
    Prepara e salva il dataset per il Task 01 Sentiment Analysis.

    Args:
        source: quale dataset usare ('sentipolc', 'sst2', 'amazon_it', 'all')
        save_path: cartella dove salvare il dataset processato

    Returns:
        DatasetDict con split train/validation/test
    """
    if source == "sentipolc":
        ds = load_sentipolc()
    elif source == "sst2":
        ds = load_sst2()
    elif source == "amazon_it":
        ds = load_amazon_it()
    elif source == "all":
        # Mix IT (SENTIPOLC) + EN (SST2) — bilanciato 70% IT / 30% EN
        ds_it = load_sentipolc()
        ds_en = load_sst2()
        combined_train = concatenate_datasets([
            ds_it["train"],
            ds_en["train"].select(range(min(len(ds_it["train"]) * 3 // 7, len(ds_en["train"]))))
        ])
        ds = DatasetDict({
            "train":      combined_train,
            "validation": ds_it.get("validation", ds_it["train"].select(range(100))),
            "test":       ds_it.get("test",       ds_it["train"].select(range(50))),
        })
    else:
        raise ValueError(f"Source '{source}' non valido. Scegli tra: sentipolc, sst2, amazon_it, all")

    # Stampa statistiche
    for split, data in ds.items():
        logger.info(f"  {split}: {len(data)} esempi")

    # Salva su disco
    import os; os.makedirs(save_path, exist_ok=True)
    ds.save_to_disk(save_path)
    logger.info(f"Dataset salvato in: {save_path}")
    return ds


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara dataset Task 01 Sentiment")
    parser.add_argument("--source",    default="sentipolc",
                        choices=["sentipolc", "sst2", "amazon_it", "all"])
    parser.add_argument("--save_path", default="./data/task_01")
    args = parser.parse_args()
    prepare_dataset(args.source, args.save_path)
