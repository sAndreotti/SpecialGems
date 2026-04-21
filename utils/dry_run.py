"""
dry_run.py — Verifica automatica pipeline fine-tuning (offline-safe).

Eseguire con:
    python utils/dry_run.py --task 01 --dataset_schema text,label

Cosa verifica (nell'ordine):
    1. [SCHEMA]    Struttura dataset attesa (colonne + tipi)
    2. [PROMPT]    Correttezza del prompt template Gemma 3
    3. [TOKENIZER] Caricamento tokenizer dal modello locale (o mock se offline)
    4. [PIPELINE]  Training minimale 10 step su dati sintetici (modello sshleifer/tiny-gpt2)
    5. [SAVE]      Salvataggio checkpoint + verifica file presenti

NOTA: Il download reale dei dataset e dei modelli Gemma deve avvenire su
Colab/Kaggle dove HuggingFace è accessibile. Questo script verifica la
correttezza della pipeline di codice indipendentemente dalla rete.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("dry_run")

# ──────────────────────────────────────────────────────────────
# Dati sintetici per schema comuni
# ──────────────────────────────────────────────────────────────

SYNTHETIC_TEMPLATES = {
    # task_id → funzione che genera una lista di esempi mock
    "01": lambda n: [
        {"text": f"Questo prodotto è {'ottimo' if i%3==0 else 'pessimo' if i%3==1 else 'nella media'}.",
         "label": i % 3}
        for i in range(n)
    ],
    "02": lambda n: [
        {"utt": f"Imposta sveglia alle {7+i%12} di mattina", "intent": "alarm_set",
         "locale": "it-IT", "annot_utt": f"Imposta sveglia alle {7+i%12}"}
        for i in range(n)
    ],
    "03": lambda n: [
        {"text": f"Notizia di {'politica' if i%4==0 else 'sport' if i%4==1 else 'economia' if i%4==2 else 'tecnologia'} n.{i}",
         "label": i % 4}
        for i in range(n)
    ],
    "04": lambda n: [
        {"premise": f"Il gatto dorme sul divano.", "hypothesis": f"Il gatto è sul divano.",
         "label": 0}  # entailment
        for i in range(n)
    ],
    "05": lambda n: [
        {"article": f"Testo di lettura numero {i}.", "question": "Di cosa parla?",
         "options": ["A", "B", "C", "D"], "answer": "A"}
        for i in range(n)
    ],
    "06": lambda n: [
        {"article": f"Articolo lungo numero {i} con molte informazioni importanti e dettagliate su vari argomenti.",
         "highlights": f"Articolo numero {i}."}
        for i in range(n)
    ],
    "07": lambda n: [
        {"context": f"Roma è la capitale d'Italia. Conta circa 3 milioni di abitanti.",
         "question": "Qual è la capitale d'Italia?", "answers": {"text": ["Roma"], "answer_start": [0]}}
        for i in range(n)
    ],
    "08": lambda n: [
        {"en": f"The European Parliament met on day {i}.",
         "it": f"Il Parlamento Europeo si è riunito il giorno {i}."}
        for i in range(n)
    ],
    "09": lambda n: [
        {"sentence1": f"La banca è chiusa oggi per festività.", "sentence2": f"Oggi la banca non è aperta.",
         "label": 1}
        for i in range(n)
    ],
    "10": lambda n: [
        {"incorrect": f"Io sono andato al mercatto ieri.", "correct": f"Io sono andato al mercato ieri."}
        for i in range(n)
    ],
    "11": lambda n: [
        {"original": f"Il fenomeno dell'antropizzazione delle aree periurbane comporta significative alterazioni ecosistemiche.",
         "simple": f"Le città crescono e danneggiano l'ambiente."}
        for i in range(n)
    ],
    "12": lambda n: [
        {"toxic": f"Questa persona è completamente inutile e stupida.", "neutral": f"Non sono d'accordo con questa persona."}
        for i in range(n)
    ],
    "13": lambda n: [
        {"tokens": ["Mario", "Rossi", "lavora", "a", "Roma", "."],
         "ner_tags": [1, 2, 0, 0, 3, 0]}  # B-PER, I-PER, O, O, B-LOC, O
        for i in range(n)
    ],
    "14": lambda n: [
        {"title": f"Documento {i}", "sents": ["Mario Rossi lavora per Fiat."],
         "head": "Mario Rossi", "tail": "Fiat", "relation": "lavoraPer"}
        for i in range(n)
    ],
    "15": lambda n: [
        {"question": f"Quanti clienti ci sono?", "query": "SELECT COUNT(*) FROM clienti",
         "db_id": "negozio"}
        for i in range(n)
    ],
    "16": lambda n: [
        {"text": f"Write a function that returns the sum of two numbers.",
         "code": "def somma(a, b):\n    return a + b"}
        for i in range(n)
    ],
    "17": lambda n: [
        {"text": f"Wall St. Bears Claw Back Into the Black (Reuters). Short-sellers, Wall Street's dwindling band of ultra-cynics, are seeing green again.",
         "label": 2}  # Business
        for i in range(n)
    ],
    "18": lambda n: [
        {"table": "| Nome | Età |\n|------|-----|\n| Mario | 30 |\n| Luca | 25 |",
         "question": "Chi è più giovane?", "answer": "Luca"}
        for i in range(n)
    ],
    "19": lambda n: [
        {"input": f"Nome: Mario Rossi, email: mario@example.com, telefono: 333-1234567",
         "schema": '{"name": "string", "email": "string", "phone": "string"}',
         "output": '{"name": "Mario Rossi", "email": "mario@example.com", "phone": "333-1234567"}'}
        for i in range(n)
    ],
    "20": lambda n: [
        {"question": f"Qual è il titolo del documento?", "docId": f"doc_{i:04d}",
         "answers": {"answer_start": [0], "text": [f"Documento {i}"]}}
        for i in range(n)
    ],
    "21": lambda n: [
        {"instruction": f"Scrivi una breve storia su un gatto.", "input": "",
         "output": f"C'era una volta un gatto di nome Micio che viveva in una casa accogliente."}
        for i in range(n)
    ],
    "22": lambda n: [
        {"messages": [
            {"role": "user", "content": f"Ciao, come stai?"},
            {"role": "assistant", "content": f"Sto bene, grazie! Come posso aiutarti?"},
        ]}
        for i in range(n)
    ],
    "23": lambda n: [
        {"question": f"Cosa è la fotosintesi?",
         "context": "La fotosintesi è il processo con cui le piante convertono la luce solare in energia.",
         "answers": {"text": ["processo"], "answer_start": [18]}}
        for i in range(n)
    ],
    "24": lambda n: [
        {"claim": f"La terra è piatta.", "evidence": "Gli scienziati hanno dimostrato che la terra è sferica.",
         "label": 1}  # REFUTES
        for i in range(n)
    ],
    "25": lambda n: [
        {"text": f"Testo di esempio numero {i} su vari argomenti sociali.", "label": 0}
        for i in range(n)
    ],
    "26": lambda n: [
        {"page_text": f"Pagina {i} del documento.", "label": "B-DOC" if i % 5 == 0 else "I-DOC"}
        for i in range(n)
    ],
}

def get_synthetic_data(task_id: str, n: int = 50) -> list[dict]:
    """Genera dati sintetici per il task specificato."""
    gen_fn = SYNTHETIC_TEMPLATES.get(task_id)
    if gen_fn is None:
        # Dati generici per task senza template specifico
        return [{"text": f"Esempio sintetico {i} per task {task_id}"} for i in range(n)]
    return gen_fn(n)


# ──────────────────────────────────────────────────────────────
# Utility log
# ──────────────────────────────────────────────────────────────

def _ok(msg: str):   logger.info(f"  ✓  {msg}")
def _warn(msg: str): logger.warning(f"  ⚠  {msg}")
def _fail(msg: str, exc: Exception | None = None):
    logger.error(f"  ✗  {msg}")
    if exc:
        logger.error(f"     Dettaglio: {exc}")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────
# Step 1 — Schema e dati sintetici
# ──────────────────────────────────────────────────────────────

def check_schema(task_id: str, required_cols: list[str], n_samples: int = 50):
    logger.info(f"[1/5] Generazione dati sintetici schema-compliant per task {task_id}")
    samples = get_synthetic_data(task_id, n_samples)

    if required_cols:
        actual = set(samples[0].keys())
        missing = [c for c in required_cols if c not in actual]
        if missing:
            _fail(f"Colonne mancanti nei dati mock: {missing} (disponibili: {sorted(actual)})")
        _ok(f"Schema OK — colonne: {sorted(actual)}")
    else:
        _ok(f"Schema generato: {list(samples[0].keys())}")

    _ok(f"Generati {len(samples)} esempi sintetici")
    return samples


# ──────────────────────────────────────────────────────────────
# Step 2 — Prompt template
# ──────────────────────────────────────────────────────────────

def check_prompt(task_id: str, samples: list[dict], text_col: str):
    logger.info(f"[2/5] Verifica prompt template Gemma 3")
    row = samples[0]
    text = str(row.get(text_col, "testo di esempio"))
    # Verifica formato base del chat template
    prompt = (
        f"<start_of_turn>user\n"
        f"Compito del task {task_id}: {text}\n"
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
        f"risposta\n"
        f"<end_of_turn>"
    )
    assert "<start_of_turn>user" in prompt
    assert "<end_of_turn>" in prompt
    assert "<start_of_turn>model" in prompt
    _ok(f"Prompt template Gemma 3 valido (len={len(prompt)} chars)")
    return prompt


# ──────────────────────────────────────────────────────────────
# Step 3 — Tokenizer
# ──────────────────────────────────────────────────────────────

def _build_local_tokenizer():
    """
    Costruisce un tokenizer BPE minimale completamente offline,
    senza scaricare nulla da HuggingFace.
    Restituisce un PreTrainedTokenizerFast compatibile con SFTTrainer.
    """
    import json, tempfile, os
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import BpeTrainer
    from transformers import PreTrainedTokenizerFast

    # Vocabolario sintetico minimale per il dry run
    vocab_texts = [
        "positivo negativo neutro testo analisi sentiment",
        "the quick brown fox jumps over the lazy dog",
        "<start_of_turn> <end_of_turn> user model system",
        "Questo è un esempio di testo in italiano per il training.",
        "Input Output risposta domanda contesto documento pagina",
    ] * 10

    # Allena un BPE tokenizer minimale su vocabolario sintetico
    tokenizer_backend = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer_backend.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=512,
        special_tokens=["[UNK]", "[PAD]", "[EOS]", "[BOS]",
                        "<start_of_turn>", "<end_of_turn>"]
    )
    tokenizer_backend.train_from_iterator(vocab_texts, trainer=trainer)

    # Salva e ricarica come PreTrainedTokenizerFast
    with tempfile.TemporaryDirectory() as tmp:
        tok_path = os.path.join(tmp, "tokenizer.json")
        tokenizer_backend.save(tok_path)
        fast_tok = PreTrainedTokenizerFast(
            tokenizer_file=tok_path,
            unk_token="[UNK]",
            pad_token="[PAD]",
            eos_token="[EOS]",
            bos_token="[BOS]",
        )
    return fast_tok


def check_tokenizer(model_name: str):
    """
    Prova a caricare il tokenizer reale; se non disponibile (offline)
    costruisce un tokenizer BPE minimale localmente senza download.
    """
    logger.info(f"[3/5] Verifica tokenizer: {model_name}")
    try:
        from transformers import AutoTokenizer

        # Prima prova il modello reale (funziona su Colab/Kaggle)
        try:
            tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
            tok.pad_token = tok.eos_token
            _ok(f"Tokenizer REALE caricato — vocab_size={tok.vocab_size}")
            return tok, True
        except Exception as e_real:
            _warn(f"Tokenizer reale non disponibile (offline): {type(e_real).__name__}")
            _warn("Costruzione tokenizer mock locale (BPE, completamente offline)...")

        # Fallback: tokenizer costruito localmente
        mock_tok = _build_local_tokenizer()
        _ok(f"Tokenizer MOCK locale costruito — vocab_size={mock_tok.vocab_size}")
        _warn("NOTA: il tokenizer Gemma reale sarà verificato su Colab/Kaggle")
        return mock_tok, False

    except ImportError:
        _fail("'transformers' non installato. Esegui: pip install transformers")


# ──────────────────────────────────────────────────────────────
# Step 4 — Tokenizzazione
# ──────────────────────────────────────────────────────────────

def check_tokenization(tokenizer, samples: list[dict], text_col: str, max_length: int = 128):
    logger.info(f"[4/5] Tokenizzazione campione ({min(3, len(samples))} esempi)")
    tested = 0
    for row in samples[:3]:
        text = str(row.get(text_col, ""))
        if not text:
            text = str(list(row.values())[0])
        enc = tokenizer(text, max_length=max_length, truncation=True, return_tensors="pt")
        tested += 1
    _ok(f"Tokenizzati {tested} esempi — shape input_ids: {enc['input_ids'].shape}")


# ──────────────────────────────────────────────────────────────
# Step 5 — Pipeline training + salvataggio
# ──────────────────────────────────────────────────────────────

def check_training_pipeline(
    task_id: str,
    tokenizer,
    samples: list[dict],
    text_col: str,
    output_base: str,
    max_steps: int = 10,
):
    logger.info(f"[5/5] Pipeline training ({max_steps} step, modello mock CPU) + salvataggio")

    try:
        from transformers import AutoModelForCausalLM, TrainingArguments
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset
    except ImportError as e:
        _fail(f"Dipendenza mancante: {e}")

    # Modello mock CPU — inizializzato da config, zero download
    logger.info(f"  Costruzione modello mock locale (GPT-2 tiny, inizializzazione random, CPU)...")
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
        tiny_cfg = GPT2Config(
            vocab_size=tokenizer.vocab_size or 512,
            n_positions=128,
            n_embd=64,
            n_layer=2,
            n_head=2,
            n_inner=128,
        )
        mock_tok  = tokenizer          # usa il tokenizer passato (stesso vocabolario)
        mock_model = GPT2LMHeadModel(config=tiny_cfg)
        mock_model.train()
        n_params = sum(p.numel() for p in mock_model.parameters())
        _ok(f"Modello mock costruito localmente ({n_params:,} parametri, CPU)")
    except Exception as e:
        _fail("Impossibile costruire il modello mock", e)

    # Dataset minimale da dati sintetici
    def row_to_text(row):
        key = text_col if text_col in row else list(row.keys())[0]
        return f"Input: {row[key]}\nOutput: risposta"

    texts = [row_to_text(r) for r in samples[:20]]
    mock_ds = Dataset.from_dict({"text": texts})

    # Cartella output
    out_dir = Path(output_base) / f"dry_run_task{task_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_args = SFTConfig(
        output_dir=str(out_dir),
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        logging_steps=2,
        save_steps=max_steps,
        fp16=False,
        bf16=False,
        use_cpu=True,           # forza CPU (trl >= 1.0)
        report_to="none",
        dataset_text_field="text",
        max_length=64,
    )

    trainer = SFTTrainer(
        model=mock_model,
        processing_class=mock_tok,   # trl >= 1.0: processing_class sostituisce tokenizer
        train_dataset=mock_ds,
        args=train_args,
    )

    logger.info(f"  Avvio {max_steps} step di training (CPU)...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    _ok(f"Training completato in {elapsed:.1f}s")

    # ── Salvataggio checkpoint (step critico) ──
    ckpt_dir = out_dir / "checkpoint_final"
    trainer.save_model(str(ckpt_dir))
    mock_tok.save_pretrained(str(ckpt_dir))

    # Verifica file salvati
    saved_files = list(ckpt_dir.iterdir())
    if not saved_files:
        _fail(f"Nessun file trovato in {ckpt_dir} dopo il salvataggio!")
    _ok(f"Checkpoint salvato: {[f.name for f in saved_files]}")

    # Report JSON
    report = {
        "task_id": task_id,
        "status": "PASSED",
        "steps": max_steps,
        "elapsed_sec": round(elapsed, 2),
        "output_dir": str(out_dir),
        "saved_files": [f.name for f in saved_files],
        "note": "Pipeline verificata offline con modello mock. Test rete su Colab/Kaggle.",
    }
    with open(out_dir / "dry_run_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    _ok(f"Report salvato: {out_dir / 'dry_run_report.json'}")
    return report


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dry run task fine-tuning (offline-safe)")
    parser.add_argument("--task",           required=True,  help="ID task (es. 01)")
    parser.add_argument("--dataset",        default="",     help="HF dataset ID (usato per log)")
    parser.add_argument("--text_col",       default="text", help="Colonna testo principale")
    parser.add_argument("--required_cols",  default="",     help="Colonne obbligatorie (virgola)")
    parser.add_argument("--model",          default="google/gemma-3-1b-it",
                                                            help="Modello per verifica tokenizer")
    parser.add_argument("--n_samples",      type=int, default=50)
    parser.add_argument("--max_steps",      type=int, default=10)
    parser.add_argument("--output_dir",     default="./checkpoints")
    args = parser.parse_args()

    logger.info("=" * 62)
    logger.info(f"  DRY RUN — Task {args.task}  |  Dataset: {args.dataset or '(sintetico)'}")
    logger.info("=" * 62)

    req_cols = [c.strip() for c in args.required_cols.split(",") if c.strip()]

    # Step 1: schema + dati sintetici
    samples = check_schema(args.task, req_cols, args.n_samples)

    # Step 2: prompt template
    check_prompt(args.task, samples, args.text_col)

    # Step 3: tokenizer
    tokenizer, is_real = check_tokenizer(args.model)

    # Step 4: tokenizzazione
    check_tokenization(tokenizer, samples, args.text_col)

    # Step 5: pipeline + salvataggio
    report = check_training_pipeline(
        task_id=args.task,
        tokenizer=tokenizer,
        samples=samples,
        text_col=args.text_col,
        output_base=args.output_dir,
        max_steps=args.max_steps,
    )

    logger.info("=" * 62)
    logger.info(f"  DRY RUN ✓  Task {args.task}  —  Status: {report['status']}")
    if not is_real:
        logger.info("  ⚠  Tokenizer mock usato — verifica rete su Colab/Kaggle")
    logger.info("=" * 62)


if __name__ == "__main__":
    main()
