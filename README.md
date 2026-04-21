# Special Application Models — Fine-Tuning Gemma 3/4 on 26 NLP Tasks

> Specialized versions of Google Gemma 3 (and Gemma 4) trained on individual NLP tasks, with systematic comparison against the base model.

**Model License:** Gemma 1/2/3 → Gemma Terms of Use · Gemma 4 → Apache 2.0  
**Code License:** Apache 2.0

---

## Objective

Each folder in this repository contains:
1. A **fine-tuning script** (LoRA/QLoRA) for a specific NLP task.
2. The **dataset** used (HuggingFace link or download/preparation script).
3. An **evaluation notebook** comparing the fine-tuned model against the corresponding Gemma base model.

---

## Supported Hardware

| Platform | Recommended Framework | Notes |
|---|---|---|
| Colab / Kaggle (T4, L4, A100) | **Unsloth** | Only framework with fp16 patch for Gemma 3; 50–60% VRAM savings |
| Mac mini 24 GB (Apple Silicon) | **MLX-LM** | `mlx_lm.lora` + `mlx_lm.fuse` for LoRA merge |
| 2× GTX 1080 Ti (Pascal, sm_61) | **LLaMA Factory** + DeepSpeed ZeRO-2 | fp16, `attn_implementation=eager`; pin PyTorch ≤ 2.4 cu118 |

---

## Base LoRA Configuration

```python
r             = 16
alpha         = 16
dropout       = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
lr            = 2e-4
scheduler     = "cosine"
warmup        = 0.03   # 3% of total steps
```

---

## Critical Gemma 3 Gotchas

- **fp16 NaN:** activations can exceed 65,504 — use Unsloth or bf16 on Ampere+ GPUs.
- **Chat template:** `<start_of_turn>user…<end_of_turn><start_of_turn>model…<end_of_turn>`. Gemma 3 has no `system` role; Gemma 4 introduces it.
- **`attn_implementation="eager"`** required for Gemma 3 — SDPA/FA2 cause numerical degradation.
- **Pad token:** always set `tokenizer.pad_token = tokenizer.eos_token` to avoid silent loss masking.

---

## Implemented Tasks

### Text Understanding (Tasks 1–5)

| # | Task | Model | Metrics | Colab |
|---|---|---|---|---|
| 1 | **Sentiment Analysis** — pos/neg/neu polarity or 1–5 star rating | Gemma 3 1B | Accuracy, macro-F1 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sAndreotti/SpecialGems/blob/main/task_01_sentiment/colab_train.ipynb) |
| 2 | **Intent Classification** — 60–150 intent classes + OOD | Gemma 3 1B | Top-1 accuracy, macro-F1, OOS recall | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sAndreotti/SpecialGems/blob/main/task_02_intent/colab_train.ipynb) |
| 3 | **Topic Classification** — single- or multi-label on news/social | Gemma 3 1B | Accuracy, macro-F1, Hamming loss | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sAndreotti/SpecialGems/blob/main/task_03_topic/colab_train.ipynb) |
| 4 | **Natural Language Inference (NLI)** — entailment / neutral / contradiction | Gemma 3 1B–4B | Accuracy matched/mismatched, ANLI per-round | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sAndreotti/SpecialGems/blob/main/task_04_nli/colab_train.ipynb) |
| 5 | **Reading Comprehension (Multi-Choice)** — passage + 4 choices | Gemma 3 4B | Accuracy | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sAndreotti/SpecialGems/blob/main/task_05_reading_comprehension/colab_train.ipynb) |

### Generation (Tasks 6–12)

| # | Task | Model | Metrics | Colab |
|---|---|---|---|---|
| 6 | **Text Summarization** — abstractive/extractive from articles and dialogues | Gemma 3 4B (1B for SAMSum) | ROUGE-1/2/L, BERTScore | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sAndreotti/SpecialGems/blob/main/task_06_summarization/colab_train.ipynb) |
| 7 | **Question Answering** — extractive (span) and generative | Gemma 3 1B–4B | Exact Match, token F1, ROUGE-L | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sAndreotti/SpecialGems/blob/main/task_07_question_answering/colab_train.ipynb) |
| 8 | **Translation EN↔IT** | Gemma 3 4B–12B | SacreBLEU, chrF++, COMET-22, TER | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sAndreotti/SpecialGems/blob/main/task_08_translation/colab_train.ipynb) |
| 9 | **Paraphrasing** — semantically equivalent reformulation | Gemma 3 1B–4B | BLEU, self-BLEU, BERTScore, iBLEU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sAndreotti/SpecialGems/blob/main/task_09_paraphrase/colab_train.ipynb) |
| 10 | **Grammatical Error Correction (GEC)** | Gemma 3 4B–12B | M2/ERRANT F0.5, GLEU+ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sAndreotti/SpecialGems/blob/main/task_10_gec/colab_train.ipynb) |
| 11 | **Text Simplification** — readable rewriting with readability constraints | Gemma 3 4B | SARI, BLEU, Gulpease, BERTScore | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sAndreotti/SpecialGems/blob/main/task_11_simplification/colab_train.ipynb) |
| 12 | **Text Style Transfer** — formality, tone, toxicity | Gemma 3 4B | Transfer-accuracy, BERTScore, perplexity | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sAndreotti/SpecialGems/blob/main/task_12_style_transfer/colab_train.ipynb) |

### Structured Extraction (Tasks 13–20)

| # | Task | Model | Metrics | Colab |
|---|---|---|---|---|
| 13 | **Named Entity Recognition (Generative)** — inline or JSON output | Gemma 3 1B / 4B | Span micro/macro entity-F1 (seqeval) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sAndreotti/SpecialGems/blob/main/task_13_ner/colab_train.ipynb) |
| 14 | **Relation Extraction** — JSON triples `{head, relation, tail}` | Gemma 3 4B / 12B | Micro-F1 on triples | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sAndreotti/SpecialGems/blob/main/task_14_relation_extraction/colab_train.ipynb) |
| 15 | **Text-to-SQL** — NL + DB schema → executable SQL query | Gemma 3 4B / 12B+ | Exact Set Match, Execution Accuracy | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sAndreotti/SpecialGems/blob/main/task_15_text_to_sql/colab_train.ipynb) |
| 16 | **Code Generation / Completion** — docstring → implementation | Gemma 3 4B / 12B+ | pass@1 / pass@10 (EvalPlus) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sAndreotti/SpecialGems/blob/main/task_16_code_generation/colab_train.ipynb) |
| 17 | **Document Classification** — text → label or JSON | Gemma 3 1B | Accuracy, macro-F1, Hamming loss | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sAndreotti/SpecialGems/blob/main/task_17_doc_classification/colab_train.ipynb) |
| 18 | **Table Extraction / Table QA** — table + question → answer | Gemma 3 4B / 12B | Exact-match, BLEU/ROUGE/BERTScore | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sAndreotti/SpecialGems/blob/main/task_18_table_qa/colab_train.ipynb) |
| 19 | **Structured Data Extraction** — text → JSON with schema | Gemma 3 4B | Schema-validity rate, field-level P/R/F1 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sAndreotti/SpecialGems/blob/main/task_19_structured_extraction/colab_train.ipynb) |
| 20 | **DocVQA Multimodal** — page image + question → answer | Gemma 3 4B / 12B+ | ANLS, VQA accuracy, entity-F1 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sAndreotti/SpecialGems/blob/main/task_20_docvqa/colab_train.ipynb) |

### Dialogue & Alignment (Tasks 21–25)

| # | Task | Model | Metrics | Colab |
|---|---|---|---|---|
| 21 | **Instruction Following / General SFT** — baseline for DPO/ORPO | Gemma 3 4B / 12B | MT-Bench-IT, AlpacaEval-IT, IFEval | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sAndreotti/SpecialGems/blob/main/task_21_sft/colab_train.ipynb) |
| 22 | **Multi-Turn Chat** — history management and long context | Gemma 3 4B–12B | MT-Bench(-IT), AlpacaEval 2 LC win-rate | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sAndreotti/SpecialGems/blob/main/task_22_multiturn_chat/colab_train.ipynb) |
| 23 | **RAG Fine-Tuning** — generator grounded on retrieved passages | Gemma 3 4B / 12B | RAGAS faithfulness, AIS, EM/F1 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sAndreotti/SpecialGems/blob/main/task_23_rag_finetuning/colab_train.ipynb) |
| 24 | **Fact Checking / Claim Verification** — SUPPORTS / REFUTES / NEI | Gemma 3 4B / 12B | Label accuracy, macro-F1, FEVER score | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sAndreotti/SpecialGems/blob/main/task_24_fact_checking/colab_train.ipynb) |
| 25 | **Hate Speech / Toxic Content Detection** | Gemma 3 1B / 4B | Macro-F1 (EVALITA), HateCheck | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sAndreotti/SpecialGems/blob/main/task_25_hate_speech/colab_train.ipynb) |

### Advanced Extraction (Task 26)

| # | Task | Model | Metrics | Colab |
|---|---|---|---|---|
| 26 | **Document Segmentation / Page-Stream Segmentation** — BIO boundary detection on composite PDFs | Gemma 3 1B (pairwise) / 4B (stream) | Boundary F1, Window-Diff, Pk, Panoptic Quality | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sAndreotti/SpecialGems/blob/main/task_26_doc_segmentation/colab_train.ipynb) |

---

## Key Italian Datasets

| Task | Recommended IT Dataset |
|---|---|
| Sentiment | `evalitahf/sentiment_analysis` (SENTIPOLC 2016, 9.4k) |
| Intent | `AmazonScience/massive` it-IT (16.5k) |
| NLI | `MoritzLaurer/multilingual-NLI-26lang-2mil7` IT split |
| Summarization | `ARTeLab/fanpage` (84k), `ARTeLab/ilpost` (44k) |
| QA | `crux82/squad_it` (54k train / 7.6k test) |
| Translation | `Helsinki-NLP/europarl` en-it (1.9M) |
| GEC | `lang-uk/omnigec` IT split |
| NER | `Babelscape/wikineural` it, `dhfbk/KIND` |
| SFT | `mchl-labs/stambecco_data_it` (52k), `DeepMount00/italian_conversations` |
| Chat | OASST1 filtered `lang==it`, `mii-community/ultrachat-translated-ita` |
| RAG | `crux82/squad_it`, `unicamp-dl/mmarco` Italian config |
| Fact Checking | FEVER-IT (`crux82` on GitHub) |
| Hate Speech | `evalitahf/hatespeech_detection` (HaSpeeDe 2), `RiTA-nlp/ami_2020` |

> **SFT Note:** use a 70% IT / 30% EN data mix to avoid catastrophic forgetting in Italian.

---

## Repository Structure

```
Special Application Models/
├── README.md
├── LICENSE
├── requirements.txt
├── utils/
│   ├── base_trainer.py     ← shared LoRA/Unsloth utilities
│   └── dry_run.py          ← offline pipeline verifier
├── .github/
│   └── workflows/
│       ├── ci.yml          ← automatic dry run on every push
│       ├── release.yml     ← tag completed tasks
│       └── notify.yml      ← training completion notifications
├── task_01_sentiment/
│   ├── train.py
│   ├── dataset_prep.py
│   └── evaluate.ipynb
├── task_02_intent/
│   └── ...
└── task_26_doc_segmentation/
    └── ...
```

---

## How to Use

### 1 — Repository Setup on GitHub

```bash
# From the project folder on your PC
git init
git add .
git commit -m "feat: Gemma 3/4 fine-tuning project across 26 NLP tasks"

# Create a repo on github.com, then:
git remote add origin https://github.com/<your-username>/special-application-models.git
git push -u origin main
```

### 2 — Training on Google Colab

Open a new notebook at [colab.research.google.com](https://colab.research.google.com), select a GPU (Runtime → Change runtime type → T4/L4/A100) and paste:

```python
# Clone the project
!git clone https://github.com/<your-username>/special-application-models.git
%cd special-application-models

# Install Unsloth (includes fp16 patch for Gemma 3)
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install trl datasets evaluate rouge_score bert_score sacrebleu

# Run training for the desired task
!python task_01_sentiment/train.py --model_size 1b --num_epochs 3 --batch_size 8

# Evaluate
!jupyter nbconvert --to notebook --execute task_01_sentiment/evaluate.ipynb
```

To pull the latest code after local changes, just run `!git pull` before launching.

### 3 — Training on Kaggle

1. Go to **kaggle.com → Notebooks → New Notebook**
2. In the right panel, enable **GPU T4 x2** or **P100**
3. In a cell at the top:

```python
# Kaggle already has PyTorch and CUDA; install only extra dependencies
!pip install "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git" trl datasets

!git clone https://github.com/<your-username>/special-application-models.git
%cd special-application-models

!python task_01_sentiment/train.py --model_size 1b --bf16
```

Checkpoints are saved to `/kaggle/working/checkpoints/` and downloadable from the Kaggle UI.

---

## GitHub Actions — CI and Automation

The repository includes three workflows in `.github/workflows/`:

### `ci.yml` — Automatic check on every push
Runs syntax checking of all Python scripts and a dry run of every task on CPU, without GPU. Triggers on every `git push` and every Pull Request.

```yaml
# .github/workflows/ci.yml
name: CI — Dry Run all tasks

on: [push, pull_request]

jobs:
  dry-run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install lightweight dependencies (no GPU)
        run: pip install datasets transformers trl accelerate tokenizers

      - name: Python syntax check
        run: |
          python -m py_compile utils/base_trainer.py utils/dry_run.py
          for f in task_*/train.py task_*/dataset_prep.py; do
            python -m py_compile "$f" && echo "✓ $f"
          done

      - name: Dry run all tasks (CPU, synthetic data)
        run: |
          for tid in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26; do
            python utils/dry_run.py --task "$tid" --max_steps 5 --output_dir /tmp/checkpoints
          done
```

### `release.yml` — Automatic tag on task completion
When you push a tag `task-NN-done` (e.g., `git tag task-01-done && git push --tags`), it automatically creates a GitHub Release with attached log files.

```yaml
# .github/workflows/release.yml
name: Release — Task completed

on:
  push:
    tags:
      - "task-*-done"

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Create Release
        uses: softprops/action-gh-release@v2
        with:
          name: "✅ ${{ github.ref_name }}"
          body: "Fine-tuning completed for task ${{ github.ref_name }}."
          generate_release_notes: true
```

### `notify.yml` — Email/Slack notification when training ends
You can add a webhook (e.g., Slack or email via SendGrid) to receive a notification when a Colab training run finishes. At the end of the training script in Colab, add:

```python
# At the end of train.py, notify via GitHub Actions dispatch
import requests, os
requests.post(
    "https://api.github.com/repos/<your-username>/special-application-models/dispatches",
    headers={"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
             "Accept": "application/vnd.github+json"},
    json={"event_type": "training-done", "client_payload": {"task": "01"}}
)
```

### How to create the workflow files

```bash
mkdir -p .github/workflows
# Copy the YAML contents shown above into their respective files, then:
git add .github/
git commit -m "ci: add GitHub Actions workflows"
git push
```

After pushing, go to **github.com → your repo → Actions** to see live run status.

---

## References

- [Gemma 3 Technical Report](https://ai.google.dev/gemma)
- [Unsloth](https://github.com/unslothai/unsloth)
- [MLX-LM](https://github.com/ml-explore/mlx-lm)
- [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory)
- [EvalPlus](https://github.com/evalplus/evalplus) (code generation)
- [RAGAS](https://github.com/explodinggradients/ragas) (RAG evaluation)
