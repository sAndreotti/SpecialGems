# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fine-tuning framework for Google Gemma 3/4 models across 26 NLP tasks, with systematic comparison of specialized vs. base models. Heavy focus on Italian-language NLP. Each task lives in its own folder (`task_NN_name/`) containing `train.py`, `dataset_prep.py`, and `evaluate.ipynb`.

## Common Commands

```bash
# Install primary framework (Colab/Kaggle)
pip install unsloth

# Train a task
python task_01_sentiment/train.py \
    --model google/gemma-3-1b-it \
    --dataset evalitahf/sentiment_analysis \
    --output_dir ./checkpoints/task_01

# Evaluate
jupyter notebook task_01_sentiment/evaluate.ipynb
```

## Hardware & Framework Matrix

| Platform | Framework | Notes |
|---|---|---|
| Colab / Kaggle (T4, L4, A100) | **Unsloth** | Only framework with fp16 patch for Gemma 3; 50–60% VRAM savings |
| Mac (Apple Silicon) | **MLX-LM** | `mlx_lm.lora` + `mlx_lm.fuse` for LoRA merge |
| 2× GTX 1080 Ti (Pascal, sm_61) | **LLaMA Factory** + DeepSpeed ZeRO-2 | fp16, `attn_implementation=eager`; pin PyTorch ≤ 2.4 cu118 |

## Base LoRA Configuration

```python
r=16, alpha=16, dropout=0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
lr=2e-4, scheduler="cosine", warmup=0.03
```

## Critical Gemma 3 Gotchas

- **fp16 NaN:** activations can exceed 65,504 — use Unsloth or bf16 on Ampere+ GPUs
- **Chat template:** `<start_of_turn>user…<end_of_turn><start_of_turn>model…<end_of_turn>` — Gemma 3 has no `system` role; Gemma 4 introduces it
- **`attn_implementation="eager"`** required for Gemma 3 — SDPA/FA2 cause numerical degradation
- **Pad token:** always set `tokenizer.pad_token = tokenizer.eos_token` to avoid silent loss masking

## Task → Model Size Mapping

- **1B models:** Tasks 1 (Sentiment), 2 (Intent), 3 (Topic), 4 (NLI), 13 (NER), 17 (Doc Classification), 25 (Hate Speech), 26 (Doc Segmentation pairwise)
- **4B models:** Tasks 5, 6, 7, 8, 9, 10, 11, 12, 14, 18, 19, 20, 21, 22, 23, 24, 25, 26 (stream)
- **12B+ models:** Tasks 8 (Translation), 10 (GEC), 15 (Text-to-SQL), 16 (Code Gen), 20 (DocVQA)

## Key Italian Datasets

| Task | Dataset |
|---|---|
| Sentiment | `evalitahf/sentiment_analysis` (SENTIPOLC 2016) |
| Intent | `AmazonScience/massive` it-IT |
| NLI | `MoritzLaurer/multilingual-NLI-26lang-2mil7` IT split |
| Summarization | `ARTeLab/fanpage`, `ARTeLab/ilpost` |
| QA | `crux82/squad_it` |
| Translation | `Helsinki-NLP/europarl` en-it |
| NER | `Babelscape/wikineural` it, `dhfbk/KIND` |
| SFT | `mchl-labs/stambecco_data_it`, `DeepMount00/italian_conversations` |
| Hate Speech | `evalitahf/hatespeech_detection` (HaSpeeDe 2) |

**SFT note:** use a 70% IT / 30% EN data mix to avoid catastrophic forgetting in Italian.

## Task Categories

1. **Text Understanding (1–5):** Sentiment, Intent, Topic, NLI, Reading Comprehension
2. **Generation (6–12):** Summarization, QA, Translation, Paraphrasing, GEC, Simplification, Style Transfer
3. **Structured Extraction (13–20):** NER, Relation Extraction, Text-to-SQL, Code Gen, Doc Classification, Table Extraction, Structured Data Extraction, DocVQA
4. **Dialogue & Alignment (21–25):** Instruction Following/SFT, Multi-Turn Chat, RAG Fine-Tuning, Fact Checking, Hate Speech Detection
5. **Advanced Extraction (26):** Document Segmentation / Page-Stream Segmentation
