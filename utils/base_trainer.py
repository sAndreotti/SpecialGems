"""
base_trainer.py — Utilità condivise per il fine-tuning con Unsloth.

Tutte le funzioni sono riutilizzate dai 26 task. Non modificare
direttamente: creare sottoclassi o wrapper per logica specifica del task.
"""

from __future__ import annotations

import unsloth  # noqa: F401 — must be first import to apply all optimizations

import os
import json
import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import TrainingArguments

# Configurazione logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Configurazione LoRA di base (condivisa da tutti i task)
# ──────────────────────────────────────────────────────────────

LORA_DEFAULT = dict(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# ──────────────────────────────────────────────────────────────
# Dimensioni modello → ID HuggingFace
# ──────────────────────────────────────────────────────────────

MODEL_MAP = {
    "1b":  "unsloth/gemma-3-1b-it-bnb-4bit",
    "4b":  "unsloth/gemma-3-4b-it-bnb-4bit",
    "12b": "unsloth/gemma-3-12b-it-bnb-4bit",
    # Gemma 4 (Apache 2.0)
    "4b-g4":  "unsloth/gemma-4-4b-it-bnb-4bit",
    "12b-g4": "unsloth/gemma-4-12b-it-bnb-4bit",
}


def get_model_and_tokenizer(
    size: str = "1b",
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
    dtype: Optional[torch.dtype] = None,
):
    """
    Carica il modello Gemma via Unsloth e imposta il pad token.

    Args:
        size: chiave in MODEL_MAP ("1b", "4b", "12b", ...)
        max_seq_length: lunghezza massima sequenza (contesto 128K disponibile)
        load_in_4bit: usa QLoRA 4-bit (consigliato su T4/L4)
        dtype: None = auto-detect (bf16 su Ampere+, fp16 altrimenti)

    Returns:
        (model, tokenizer) pronti per get_peft_model
    """
    try:
        from unsloth import FastLanguageModel  # noqa: F811
    except ImportError:
        raise ImportError(
            "Unsloth not found. Install with:\n"
            "  pip install unsloth[colab-new]"
        )

    model_name = MODEL_MAP.get(size)
    if model_name is None:
        raise ValueError(f"Dimensione modello '{size}' non valida. Scegli tra: {list(MODEL_MAP)}")

    logger.info(f"Caricamento modello: {model_name} (seq_len={max_seq_length}, 4bit={load_in_4bit})")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # Gotcha Gemma 3: pad_token deve essere impostato esplicitamente
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info("Modello e tokenizer caricati con successo.")
    return model, tokenizer


def apply_lora(model, lora_config: dict | None = None):
    """
    Applica i layer LoRA al modello via Unsloth.

    Args:
        model: modello caricato con get_model_and_tokenizer
        lora_config: dizionario di parametri LoRA (default: LORA_DEFAULT)

    Returns:
        model con adapter LoRA attivi
    """
    from unsloth import FastLanguageModel

    cfg = {**LORA_DEFAULT, **(lora_config or {})}
    logger.info(f"Applicazione LoRA: r={cfg['r']}, alpha={cfg['lora_alpha']}, "
                f"dropout={cfg['lora_dropout']}")

    model = FastLanguageModel.get_peft_model(model, **cfg)
    return model


def get_training_args(
    output_dir: str,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    max_steps: int = -1,
    warmup_ratio: float = 0.03,
    logging_steps: int = 10,
    save_strategy: str = "epoch",
    fp16: bool = False,
    bf16: bool = False,
    **kwargs,
) -> TrainingArguments:
    """
    Restituisce TrainingArguments pre-configurati per Unsloth.

    Se né fp16 né bf16 sono specificati, il framework sceglie automaticamente
    in base all'hardware disponibile (bf16 su Ampere+, fp16 altrimenti).
    """
    # Auto-detect precisione se non specificata
    if not fp16 and not bf16:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            bf16 = True
            logger.info("Precisione: bf16 (rilevato GPU Ampere+)")
        else:
            fp16 = True
            logger.info("Precisione: fp16 (GPU non Ampere)")

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        save_total_limit=2,
        load_best_model_at_end=False,
        fp16=fp16,
        bf16=bf16,
        optim="adamw_8bit",
        weight_decay=0.01,
        seed=42,
        report_to="none",      # disabilita wandb/tensorboard di default
        **kwargs,
    )


def save_model(model, tokenizer, output_dir: str, push_to_hub: bool = False, repo_id: str = ""):
    """
    Salva il modello fine-tuned (adapter LoRA + tokenizer).

    Salva sia il checkpoint merged (per inferenza) sia il solo adapter
    (per distribuzione leggera). Struttura output:
        output_dir/
            adapter/        ← solo pesi LoRA (~10-50 MB)
            merged/         ← modello completo merged (opzionale)
    """
    output_path = Path(output_dir)
    adapter_path = output_path / "adapter"
    merged_path  = output_path / "merged"

    # Salva adapter LoRA
    logger.info(f"Salvataggio adapter LoRA → {adapter_path}")
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    # Salva metadati del training
    meta = {
        "task": output_path.parent.name,
        "adapter_path": str(adapter_path),
        "merged_path": str(merged_path),
        "push_to_hub": push_to_hub,
    }
    with open(output_path / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info("Salvataggio completato.")

    # Push opzionale su HuggingFace Hub
    if push_to_hub and repo_id:
        logger.info(f"Upload su HuggingFace Hub → {repo_id}")
        model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)

    return str(adapter_path)


# ──────────────────────────────────────────────────────────────
# Chat template Gemma 3 (nessun ruolo system)
# Gemma 4 aggiunge system, da usare con apply_chat_template
# ──────────────────────────────────────────────────────────────

GEMMA3_CHAT_TEMPLATE = (
    "<start_of_turn>user\n{user}<end_of_turn>\n"
    "<start_of_turn>model\n{model}<end_of_turn>"
)


def format_chat(user_msg: str, model_msg: str) -> str:
    """
    Formatta una coppia (user, model) secondo il chat template di Gemma 3.
    Per Gemma 4 usare tokenizer.apply_chat_template con messages list.
    """
    return GEMMA3_CHAT_TEMPLATE.format(user=user_msg, model=model_msg)


def format_chat_list(messages: list[dict]) -> str:
    """
    Formatta una lista di messaggi multi-turn per Gemma 3.
    messages = [{"role": "user", "content": "..."}, {"role": "model", "content": "..."}, ...]
    """
    result = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        result.append(f"<start_of_turn>{role}\n{content}<end_of_turn>")
    result.append("<start_of_turn>model\n")  # prompt di apertura per generazione
    return "\n".join(result)
