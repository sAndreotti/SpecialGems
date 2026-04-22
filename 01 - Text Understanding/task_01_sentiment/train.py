"""
train.py — Task 01: Sentiment Analysis
=======================================
Fine-tuning di Gemma 3 1B su analisi del sentiment in italiano.

Uso rapido (Colab/Kaggle con GPU T4/L4/A100):
    python train.py

Uso avanzato:
    python train.py \
        --model_size 1b \
        --dataset evalitahf/sentiment_analysis \
        --output_dir ./checkpoints/task_01 \
        --num_epochs 3 \
        --batch_size 8

Dry run (CPU, nessuna GPU richiesta):
    python train.py --dry_run
    oppure:
    python utils/dry_run.py --task 01 \
        --dataset evalitahf/sentiment_analysis \
        --text_col text \
        --required_cols text,label
"""

from __future__ import annotations

import unsloth  # noqa: F401 — must be first to patch CUDA kernels

import argparse
import logging
import os
import sys

# Aggiunge la root del progetto al path per importare utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Costanti del task
# ──────────────────────────────────────────────────────────────

TASK_ID          = "01"
TASK_NAME        = "sentiment_analysis"
DEFAULT_MODEL    = "1b"                              # Gemma 3 1B
DEFAULT_DATASET  = "evalitahf/sentiment_analysis"
DEFAULT_OUT_DIR  = "./checkpoints/task_01_sentiment"
MAX_SEQ_LENGTH   = 512                               # testi di sentiment sono corti
LABEL_CLASSES    = ["positivo", "negativo", "neutro"]


# ──────────────────────────────────────────────────────────────
# Preparazione prompt
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Sei un sistema di analisi del sentiment. "
    "Rispondi sempre e solo con una parola: positivo, negativo o neutro."
)

SENTIPOLC_LABELS = {0: "negativo", 1: "positivo", 2: "neutro"}
SST2_LABELS      = {0: "negativo", 1: "positivo"}

def row_to_prompt(row: dict) -> str:
    """
    Converte una riga del dataset nel prompt chat Gemma 3.
    Supporta SENTIPOLC (campo 'text'+'label') e SST-2 (campo 'sentence'+'label').
    """
    text = (row.get("text") or row.get("sentence") or "").strip()
    label_raw = row.get("label", 2)
    # SENTIPOLC: 0=neg, 1=pos, 2=neu
    label_str = SENTIPOLC_LABELS.get(int(label_raw), "neutro")
    return (
        f"<start_of_turn>user\n"
        f"{SYSTEM_PROMPT}\n\n"
        f"Testo: {text}\n"
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
        f"{label_str}\n"
        f"<end_of_turn>"
    )


# ──────────────────────────────────────────────────────────────
# Caricamento dataset
# ──────────────────────────────────────────────────────────────

def load_and_prepare(dataset_id: str, max_train: int = 8000, max_val: int = 500):
    """Carica il dataset e applica il formatting del prompt."""
    from datasets import load_dataset

    logger.info(f"Caricamento dataset: {dataset_id}")
    ds = load_dataset(dataset_id, trust_remote_code=True)

    def format_fn(batch):
        return {"text": [row_to_prompt(dict(zip(batch.keys(), vals)))
                         for vals in zip(*batch.values())]}

    ds_formatted = ds.map(
        lambda row: {"text": row_to_prompt(row)},
        remove_columns=[c for c in ds["train"].column_names if c != "text"],
    )

    # Limita dimensione per training più veloce
    if len(ds_formatted["train"]) > max_train:
        ds_formatted["train"] = ds_formatted["train"].select(range(max_train))
    if "validation" in ds_formatted and len(ds_formatted["validation"]) > max_val:
        ds_formatted["validation"] = ds_formatted["validation"].select(range(max_val))

    logger.info(f"Train: {len(ds_formatted['train'])} esempi")
    if "validation" in ds_formatted:
        logger.info(f"Validation: {len(ds_formatted['validation'])} esempi")
    return ds_formatted


# ──────────────────────────────────────────────────────────────
# Training principale
# ──────────────────────────────────────────────────────────────

def train(args):
    """Pipeline completa: load → LoRA → train → save."""
    from utils.base_trainer import get_model_and_tokenizer, apply_lora, get_training_args, save_model
    from trl import SFTTrainer, SFTConfig

    # 1. Carica modello + tokenizer via Unsloth
    model, tokenizer = get_model_and_tokenizer(
        size=args.model_size,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    # 2. Applica LoRA
    model = apply_lora(model)

    # 3. Carica e prepara dataset
    ds = load_and_prepare(args.dataset, max_train=args.max_train)

    # 4. Configura training
    train_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=not args.bf16,
        bf16=args.bf16,
        optim="adamw_8bit",
        weight_decay=0.01,
        seed=42,
        report_to="none",
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        packing=True,   # aumenta efficienza su sequenze corte
        dataset_num_proc=1,  # avoids pickle error with Unsloth tokenizer
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        args=train_args,
    )

    # 5. Training
    logger.info("Avvio training...")
    trainer.train()

    # 6. Salvataggio finale
    save_model(model, tokenizer, args.output_dir)
    logger.info(f"✓ Training completato — modello salvato in: {args.output_dir}")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Task 01 — Sentiment Analysis fine-tuning")
    parser.add_argument("--model_size",  default=DEFAULT_MODEL,
                        choices=["1b", "4b", "12b"],
                        help="Dimensione modello Gemma 3")
    parser.add_argument("--dataset",     default=DEFAULT_DATASET,
                        help="HuggingFace dataset ID")
    parser.add_argument("--output_dir",  default=DEFAULT_OUT_DIR)
    parser.add_argument("--num_epochs",  type=int, default=3)
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--max_train",   type=int, default=8000,
                        help="Numero massimo esempi di training")
    parser.add_argument("--bf16",        action="store_true",
                        help="Usa bf16 invece di fp16 (richiede GPU Ampere+)")
    parser.add_argument("--dry_run",     action="store_true",
                        help="Esegui solo la verifica senza training reale")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.dry_run:
        # Rimanda al dry_run utility per la verifica completa
        import subprocess
        script = os.path.join(os.path.dirname(__file__), "..", "utils", "dry_run.py")
        cmd = [
            sys.executable, script,
            "--task",         TASK_ID,
            "--dataset",      DEFAULT_DATASET,
            "--text_col",     "text",
            "--required_cols","text,label",
            "--model",        "google/gemma-3-1b-it",
            "--output_dir",   "./checkpoints",
        ]
        logger.info(f"Modalità dry run — eseguo: {' '.join(cmd)}")
        sys.exit(subprocess.call(cmd))
    else:
        train(args)
