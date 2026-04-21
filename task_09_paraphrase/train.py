"""
train.py — Task 09: Parafrasatura
Uso: python train.py [--dry_run] [--model_size 1b] [--dataset ...] [--output_dir ...] [--num_epochs N] [--batch_size N]
"""
from __future__ import annotations

import unsloth  # noqa: F401 — must be first to patch CUDA kernels
import argparse, logging, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)

TASK_ID         = "09"
TASK_NAME       = "paraphrase"
DEFAULT_MODEL   = "1b"
DEFAULT_DATASET = "google-research-datasets/paws"
DEFAULT_OUT_DIR = "./checkpoints/task_09_paraphrase"
MAX_SEQ_LENGTH  = 256
SYSTEM_PROMPT   = """Riscrivi la frase con parole diverse mantenendo lo stesso significato."""
LABEL_MAP       = {}


def row_to_prompt(row: dict) -> str:
    """Converte riga dataset nel prompt chat Gemma 3."""
    user_text = str(row.get("sentence1", "")).strip()
    label_raw = row.get("label", "")
    label_str = str(label_raw).strip()
    if SYSTEM_PROMPT:
        return (
            "<start_of_turn>user\n" + SYSTEM_PROMPT + "\n\n"
            "Frase originale: " + user_text + "\n"
            "<end_of_turn>\n<start_of_turn>model\n" + label_str + "\n<end_of_turn>"
        )
    return (
        "<start_of_turn>user\nFrase originale: " + user_text + "\n"
        "<end_of_turn>\n<start_of_turn>model\n" + label_str + "\n<end_of_turn>"
    )


def load_and_prepare(dataset_id: str, max_train: int = 10000, max_val: int = 500):
    from datasets import load_dataset
    logger.info(f"Caricamento: {dataset_id}")
    load_kwargs = {"name": "labeled_final"}
    ds = load_dataset(dataset_id, **load_kwargs)
    ds_fmt = ds.map(
        lambda row: {"text": row_to_prompt(row)},
        remove_columns=[c for c in ds["train"].column_names if c != "text"],
    )
    if len(ds_fmt["train"]) > max_train:
        ds_fmt["train"] = ds_fmt["train"].select(range(max_train))
    if "validation" in ds_fmt and len(ds_fmt["validation"]) > max_val:
        ds_fmt["validation"] = ds_fmt["validation"].select(range(max_val))
    logger.info(f"Train: {len(ds_fmt['train'])} | Val: {len(ds_fmt.get('validation', []))}")
    return ds_fmt


def train(args):
    from utils.base_trainer import get_model_and_tokenizer, apply_lora, save_model
    from trl import SFTTrainer, SFTConfig
    # Carica modello Gemma 3 via Unsloth (fp16 patchato, 4-bit)
    model, tokenizer = get_model_and_tokenizer(size=args.model_size, max_seq_length=MAX_SEQ_LENGTH)
    model = apply_lora(model)
    ds    = load_and_prepare(args.dataset, max_train=args.max_train)
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
        seed=42,
        report_to="none",
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        packing=True,
        dataset_num_proc=1,  # avoids pickle error with Unsloth tokenizer
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        args=train_args,
    )
    logger.info("Avvio training Task 09 — Parafrasatura...")
    trainer.train()
    save_model(model, tokenizer, args.output_dir)
    logger.info(f"✓ Modello salvato in: {args.output_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Task 09 — Parafrasatura")
    p.add_argument("--model_size",  default=DEFAULT_MODEL, choices=["1b","4b","12b"])
    p.add_argument("--dataset",     default=DEFAULT_DATASET)
    p.add_argument("--output_dir",  default=DEFAULT_OUT_DIR)
    p.add_argument("--num_epochs",  type=int, default=3)
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--max_train",   type=int, default=10000)
    p.add_argument("--bf16",        action="store_true")
    p.add_argument("--dry_run",     action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.dry_run:
        import subprocess
        script = os.path.join(os.path.dirname(__file__), "..", "utils", "dry_run.py")
        sys.exit(subprocess.call([
            sys.executable, script,
            "--task", TASK_ID,
            "--dataset", DEFAULT_DATASET,
            "--text_col", "sentence1",
            "--model", f"google/gemma-3-{DEFAULT_MODEL}-it",
            "--output_dir", "./checkpoints",
        ]))
    else:
        train(parse_args())
