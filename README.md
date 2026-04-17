# Unifying retrieval and re-ranking: A multimodal approach to detecting fact-checked information

## MMBT - Multimodal BiTransformers for automatic fact-checking

The system combines BERT (text) + ResNet-152 (images) in a multimodal model (MMBT) for ranking supporting documents in fact-checking. Document indexing and retrieval are handled through FAISS.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#1-environment-setup)
3. [Local BERT Models](#local-bert-models)
4. [Datasets](#2-datasets)
5. [Training](#3-training)
6. [Evaluation](#4-evaluation)
7. [Experiments](#5-experiments)
8. [Output Structure](#6-output-structure)
9. [Full Project Structure](#7-full-project-structure)
10. [Complete Step-by-Step Guide](#complete-step-by-step-guide)
11. [FAISS GPU](#faiss-gpu)
12. [Quick Reference](#quick-reference)
13. [Citation](#citation)
14. [Troubleshooting](#troubleshooting)
15. [License](#license)

---

## Prerequisites

| Component | Requirement |
|-----------|-------------|
| OS | Linux (tested on Ubuntu 22.04) |
| GPU | NVIDIA with >= 4 GB VRAM (RTX 2070 8 GB used) |
| CUDA | 11.x or 12.x |
| Python | 3.8+ (3.11 used) |
| RAM | >= 8 GB |

---

## 1. Environment Setup

### 1.1 Quick Installation

```bash
# Clone the repository
git clone <repo-url> && cd MMBT_FactChecking_Thesis

# Create and activate the virtual environment
./bin/setup_env.sh

# Or manually:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1.2 Verify Installation

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"

python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import faiss; print('FAISS GPU:', faiss.get_num_gpus())"
python -c "from mmbt.models.mmbt import MultimodalBertForClassification; print('MMBT: OK')"
```

> **Note**: every time you open a new terminal, activate the environment:
> ```bash
> source .venv/bin/activate
> export PYTHONPATH="$PWD:$PYTHONPATH"
> ```

### 1.3 Makefile Automation

The Makefile handles all major operations. To list the available commands:

```bash
make help
```

Most commonly used commands:

| Command | Description |
|---------|-------------|
| `make setup` | Full environment setup |
| `make train DATASET=snopes` | Generic training |
| `make train-snopes` | Snopes training |
| `make train-m3check-multilingual` | M3-Check training |
| `make eval DATASET=snopes` | Evaluation |
| `make exp-noise DATASET=Snopes` | White noise experiment |
| `make info` | System info (GPU, Python, CUDA) |

---

## Local BERT Models

The BERT models must be downloaded **only once** and stored locally. Two different models are needed depending on the dataset:

| Model | Used for | Parameters | `hidden_sz` |
|-------|----------|------------|-------------|
| `bert-tiny` | Snopes, Politifact | 17M | 128 |
| `bert-base-multilingual-cased` | MuMiN, M3-Check | 177M | 768 |

```bash
# For Snopes/Politifact (bert-tiny, 17M parameters)
python -c "
from transformers import BertModel, BertTokenizer
BertModel.from_pretrained('prajjwal1/bert-tiny').save_pretrained('/root/bert_tiny_local')
BertTokenizer.from_pretrained('prajjwal1/bert-tiny').save_pretrained('/root/bert_tiny_local')
"

# For multilingual MuMiN / M3-Check (bert-base-multilingual-cased, 177M parameters, 104 languages)
python -c "
from transformers import BertModel, BertTokenizer
BertModel.from_pretrained('bert-base-multilingual-cased').save_pretrained('/root/bert_multilingual_local')
BertTokenizer.from_pretrained('bert-base-multilingual-cased').save_pretrained('/root/bert_multilingual_local')
"
```

Or with the Makefile:

```bash
make download-models
```

---

## 2. Datasets

The project supports **4 datasets** (+ 1 variant) for multimodal fact-checking:

| Dataset | Type | Languages | Train | Val | Test | BERT |
|---------|------|-----------|-------|-----|------|------|
| **Snopes** | English | 1 (en) | 399k pairs | 50k | 58k | bert-tiny |
| **Politifact** | English | 1 (en) | (included in Snopes) | (included in Snopes) | (included in Snopes) | bert-tiny |
| **MuMiN** | Multilingual (no EN) | ~40 | 165k pairs | 21k | 21k | mBERT |
| **M3-Check** | Multilingual | 92 | 1.97M pairs | 424k | 428k | mBERT |
| **M3-Check No-Eng** | Multilingual (no EN) | 91 | 1.68M pairs | 360k | 368k | mBERT |

All datasets use the same TSV format with 7 columns:

```
QueryID  QueryText  QueryImages  DocID  DocText  DocImages  Label
```

- **Query** = claim/post to be verified (text + optional image)
- **Document** = fact-check article (text + optional image)
- **Label** = 1 (relevant) or 0 (non-relevant, generated with negative sampling)

### 2.1 Snopes & Politifact (ready to use)

For the original Snopes and Politifact resources, refer to the EMNLP2020 repository:
https://github.com/nguyenvo09/EMNLP2020

Use the dataset links and structure documented there, then place the prepared files in this project as follows:

Place them in this structure:

```
datasets/
└── fakenewsdet/
    └── annotations/
        ├── train.tsv
        ├── val.tsv
        ├── test.tsv
        └── Snopes_DocumentsDataset.tsv
```

> Snopes and Politifact share the same `fakenewsdet/` directory. The TSV files are already ready to use, so no further preparation is required.

### 2.2 MuMiN (requires preparation)

The original MuMiN dataset must be converted to the MMBT format with negative sampling.

**Step 1 - Place the raw data:**

```
datasets/
└── mumin/
    └── (original MuMiN CSV files)
```

**Step 2 - Prepare the dataset:**

```bash
# Full MuMiN dataset (non-EN claims only, 30 negatives per positive)
python scripts/prepare_mumin_dataset_v2.py \
    --neg-ratio 30 \
    --exclude-langs en \
    --output-dir datasets/mumin_mmbt

# Or with the Makefile:
make prepare-mumin-multilingual
```

**Step 3 - Verify:**

```bash
wc -l datasets/mumin_mmbt/annotations/*.tsv
# Expected: ~165k train, ~21k val, ~21k test
```

Generated output:

```
datasets/
└── mumin_mmbt/
    └── annotations/
        ├── train.tsv
        ├── val.tsv
        └── test.tsv
```

### 2.3 M3-Check (requires preparation)

M3-Check is a multilingual and multimodal dataset for fact-check retrieval given a social media post. It covers 92 languages (main ones: spa, eng, por, fra, msa, hin, deu, tha, ara, zho).

**Step 1 - Place the raw data:**

Download the M3-Check dataset and place it in:

```
datasets/
└── M3-check/
    ├── training_set.csv
    ├── validation_set.csv
    ├── testing_set.csv
    ├── fact_checks.csv
    ├── pairs.csv
    └── user_posts.csv
```

**Step 2 - Prepare the dataset (full version, all languages):**

```bash
python scripts/prepare_m3check_dataset.py \
    --input-dir datasets/M3-check \
    --output-dir datasets/mmbt_m3check_multilingual \
    --neg-ratio 30 \
    --seed 2024
```

This generates ~1.97M training pairs with 30 negatives for each positive.

**Step 3 (optional) - Prepare the no-English variant:**

```bash
python scripts/prepare_m3check_dataset.py \
    --input-dir datasets/M3-check \
    --output-dir datasets/mmbt_m3check_no_eng \
    --neg-ratio 30 \
    --exclude-langs eng \
    --seed 2024
```

**Step 4 - Verify:**

```bash
# Full version
wc -l datasets/mmbt_m3check_multilingual/annotations/*.tsv
# Expected: ~1.97M train, ~424k val, ~428k test

# No-English version
wc -l datasets/mmbt_m3check_no_eng/annotations/*.tsv
# Expected: ~1.68M train, ~360k val, ~368k test

# Check the first lines
head -2 datasets/mmbt_m3check_multilingual/annotations/train.tsv
```

Generated output:

```
datasets/
├── mmbt_m3check_multilingual/
│   └── annotations/
│       ├── train.tsv
│       ├── val.tsv
│       └── test.tsv
└── mmbt_m3check_no_eng/
    └── annotations/
        ├── train.tsv
        ├── val.tsv
        └── test.tsv
```

### Complete dataset structure summary

```
datasets/
├── fakenewsdet/                    # Snopes + Politifact (ready to use)
│   └── annotations/
│       ├── train.tsv
│       ├── val.tsv
│       ├── test.tsv
│       └── Snopes_DocumentsDataset.tsv
├── M3-check/                       # Raw M3-Check (original CSVs)
│   ├── training_set.csv
│   ├── validation_set.csv
│   ├── testing_set.csv
│   ├── fact_checks.csv
│   ├── pairs.csv
│   └── user_posts.csv
├── mumin/                          # Raw MuMiN (original CSVs)
├── mumin_mmbt/                     # MuMiN prepared for MMBT
│   └── annotations/
│       ├── train.tsv, val.tsv, test.tsv
├── mmbt_m3check_multilingual/      # Prepared M3-Check (all languages)
│   └── annotations/
│       ├── train.tsv, val.tsv, test.tsv
└── mmbt_m3check_no_eng/            # Prepared M3-Check (without English)
    └── annotations/
        ├── train.tsv, val.tsv, test.tsv
```

---

## 3. Training

### 3.1 Snopes

```bash
python main.py -c configs/snopes_config.conf

# Or:
make train-snopes
```

| Parameter | Value |
|-----------|-------|
| BERT | `bert-tiny` (hidden_sz=128) |
| Batch | 8 x 24 acc. steps = 192 effective |
| Epochs | 3 |
| Learning rate | 1e-5 |
| Checkpoint | `models/checkpoints/mmbt_snopes/` |

### 3.2 Politifact

```bash
python main.py -c configs/politifact_config.conf

# Or:
make train-politifact
```

Same architecture as Snopes. Checkpoint in `models/checkpoints/mmbt_politifact/`.

### 3.3 MuMiN (Multilingual)

```bash
python main.py -c configs/mumin_multilingual_config.conf

# Or:
make train-mumin-multilingual
```

| Parameter | Value |
|-----------|-------|
| BERT | `bert-base-multilingual-cased` (hidden_sz=768, 104 languages) |
| Dataset | Non-EN claims from MuMiN |
| Epochs | 3 |
| Learning rate | 2e-5 |
| Checkpoint | `models/checkpoints/mmbt_mumin_multilingual/` |

### 3.4 M3-Check (Multilingual)

```bash
python main.py -c configs/m3check_multilingual_config.conf

# Or:
make train-m3check-multilingual
```

| Parameter | Value |
|-----------|-------|
| BERT | `bert-base-multilingual-cased` (hidden_sz=768) |
| Dataset | 92 languages (spa, eng, por, fra, msa, ...) |
| Epochs | 6 |
| Learning rate | 1e-5 |
| Checkpoint | `models/checkpoints/mmbt_m3check/` |

### 3.5 M3-Check (Without English)

```bash
python main.py -c configs/m3check_no_eng_config.conf

# Or:
make train-m3check-no-eng
```

| Parameter | Value |
|-----------|-------|
| BERT | `bert-base-multilingual-cased` (hidden_sz=768) |
| Dataset | 91 languages (without eng) - main languages: spa, por, fra, msa, hin |
| Epochs | 6 |
| Learning rate | 1e-5 |
| Checkpoint | `models/checkpoints/mmbt_m3check_no_eng/` |

### Resume Interrupted Training

If training is interrupted, it can be resumed from the checkpoint:

```bash
make train-m3check-multilingual RESUME=1
```

### Verify Checkpoints

```bash
ls -lh models/checkpoints/mmbt_snopes/model_best.pt
ls -lh models/checkpoints/mmbt_politifact/model_best.pt
ls -lh models/checkpoints/mmbt_mumin_multilingual/model_best.pt
ls -lh models/checkpoints/mmbt_m3check/model_best.pt
ls -lh models/checkpoints/mmbt_m3check_no_eng/model_best.pt
```

---

## 4. Evaluation

To evaluate an already trained model on the test set:

```bash
# Snopes
python main.py -c configs/snopes_config.conf --is_evaluation Yes

# Politifact
python main.py -c configs/politifact_config.conf --is_evaluation Yes

# MuMiN
python main.py -c configs/mumin_multilingual_config.conf --is_evaluation Yes

# M3-Check (all languages)
python main.py -c configs/m3check_multilingual_config.conf --is_evaluation Yes

# M3-Check (without English)
python main.py -c configs/m3check_no_eng_config.conf --is_evaluation Yes
```

Or with the Makefile:

```bash
make eval DATASET=snopes
make eval DATASET=politifact
make eval DATASET=mumin
make eval DATASET=m3check_multilingual
make eval DATASET=m3check_no_eng
```

The results are saved in `models/checkpoints/{name}/test_results_{dataset}.json`.

The computed metrics are:
- **NDCG@k** (k=1,3,5,10) - Normalized Discounted Cumulative Gain
- **Hit@k** (k=1,3,5,10) - percentage of queries with at least one relevant result in the top k
- **MAP** - Mean Average Precision

---

## 5. Experiments

All outputs are organized by dataset in `outputs/{Dataset}/{experiment}/`.

### 5.1 Experiment 1 - Noise Robustness (White Noise)

Adds white Gaussian noise to images and measures Hit@k and NDCG@k.

```bash
# With the default thresholds (0.0, 0.05, 0.10, 0.20, 0.40)
python scripts/run_experiments.py \
    --experiment white_noise \
    --dataset Snopes \
    --output-dir outputs

# With custom thresholds to explore the robustness limit
python scripts/run_experiments.py \
    --experiment white_noise \
    --dataset Snopes \
    --noise-levels "0.0,0.1,0.2,0.4,0.6,0.8,1.0,1.5,2.0" \
    --output-dir outputs

# Or with the Makefile:
make exp-noise DATASET=Snopes
```

Available for all datasets: `Snopes`, `Politifact`, `MuMiN`, `M3Check`, `M3CheckNoEng`.

**`--noise-levels` parameter**: list of eta values (Gaussian noise standard deviation). Higher values = more noise. To find the system breaking point, increase progressively until the metrics drop significantly.

**PSNR reference** (for 224x224 images):
| eta | PSNR (dB) | Visual effect |
|-----|-----------|---------------|
| 0.05 | ~26 dB | Almost imperceptible |
| 0.20 | ~14 dB | Visible noise |
| 0.40 | ~8 dB | Heavily degraded |
| 1.00 | ~0 dB | Almost unrecognizable |
| 2.00 | ~-6 dB | Pure noise |

Output:
```
outputs/Snopes/white_noise/
├── results.json   # Metrics for each eta
├── results.csv    # Same data in CSV format
└── REPORT.md      # Summary table
```

### 5.2 Experiment 2 - Multilingual Support

Evaluation on multilingual (non-EN) claims from the MuMiN and M3-Check datasets.

```bash
# MuMiN
python main.py -c configs/mumin_multilingual_config.conf
python main.py -c configs/mumin_multilingual_config.conf --is_evaluation Yes

# M3-Check (92 languages)
python main.py -c configs/m3check_multilingual_config.conf
python main.py -c configs/m3check_multilingual_config.conf --is_evaluation Yes

# M3-Check without English (comparison to measure the impact of English)
python main.py -c configs/m3check_no_eng_config.conf
python main.py -c configs/m3check_no_eng_config.conf --is_evaluation Yes
```

### 5.3 Experiment 3 - FAISS Scalability

Benchmark of indexing and retrieval time with FAISS as the document collection grows.

```bash
python scripts/run_experiments.py \
    --experiment complexity \
    --dataset Snopes \
    --output-dir outputs

# Or with the Makefile:
make exp-complexity DATASET=Snopes
```

Procedure: it takes the 1,703 Snopes documents, generates variants by removing 25% of the words with scaling factors x10 and x100, creates FlatL2 (exact) and IVFFlat (approximate) indexes, and measures runtime for k=50 and k=200.

Output:
```
outputs/Snopes/complexity/
├── scalability.json   # Detailed timings
└── REPORT.md          # Summary table
```

> **Note on FAISS GPU**: `faiss-cpu` is currently installed. For very large corpora (>1M documents), switching to `faiss-gpu` is recommended:
> ```bash
> pip uninstall faiss-cpu && pip install faiss-gpu
> ```
> The current code uses `faiss.IndexFlatL2` and `faiss.IndexIVFFlat`, which are CPU-only. To use the GPU, wrap the indexes with `faiss.index_cpu_to_gpu()`. See the [FAISS GPU](#faiss-gpu) section for details.

### Run All Experiments

```bash
python scripts/run_experiments.py \
    --experiment all \
    --dataset Snopes \
    --output-dir outputs

# Or:
make exp-all DATASET=Snopes
```

### 5.4 Baseline MAN (Vo & Lee, EMNLP 2020)

Comparison with the **MAN (Multimodal Attention Network)** model from the paper
[Where Are the Facts?](https://aclanthology.org/2020.emnlp-main.621/) (EMNLP 2020),
adapted to the multilingual setting by using mBERT instead of GloVe + ELMo.

| Component | Original (EN-only) | Multilingual adaptation |
|-----------|--------------------|-------------------------|
| Word Embeddings | GloVe (300d) | mBERT token embeddings (768d) |
| Contextual | ELMo (1024d) | mBERT hidden states (768d) |
| Similarity | Similarity matrix + CNN | Similarity matrix + CNN (unchanged) |
| PACRR Head | Top-k max-pooling | Top-k max-pooling (unchanged) |
| Loss | Hinge (margin-based) | Hinge (margin-based, unchanged) |
| Visual | ResNet50 features | Not used (text only) |

```bash
# MuMiN (multilingual, without English)
python -m scripts.benchmark_baselines_man --dataset mumin

# M3-Check multilingual (92 languages)
python -m scripts.benchmark_baselines_man --dataset m3check_multilingual

# M3-Check without English (91 languages)
python -m scripts.benchmark_baselines_man --dataset m3check_no_eng

# Or with the Makefile:
make baseline-man-mumin
make baseline-man-m3check
make baseline-man-m3check-no-eng

# Evaluation only (already trained model)
python -m scripts.benchmark_baselines_man --dataset mumin --eval-only
```

Customizable parameters via Makefile:

```bash
# More epochs and a larger batch size
make baseline-man-mumin MAN_EPOCHS=5 MAN_BATCH=32

# Quick test with a subsample
make baseline-man DATASET=mumin MAN_MAX_TRAIN=5000
```

Output:

```
outputs/{dataset}/baselines/
└── man_results_{timestamp}.json   # Metrics: NDCG@1,3,5,10, HIT@1,3,5,10

models/checkpoints/man_{dataset}/
└── man_best.pt                    # Best model checkpoint
```

---

## 6. Output Structure

```
outputs/
├── Snopes/
│   ├── white_noise/
│   │   ├── results.json
│   │   ├── results.csv
│   │   └── REPORT.md
│   ├── complexity/
│   │   ├── scalability.json
│   │   └── REPORT.md
│   └── logs/
├── Politifact/
│   └── ...
├── mumin/
│   └── ...
└── M3Check/
    ├── white_noise/
    └── logs/
```

---

## 7. Full Project Structure

```
MMBT_FactChecking_Thesis/
├── bin/                        # Shell scripts
│   ├── setup_env.sh            # Full environment setup
│   ├── activate_env.sh         # Activate .venv + PYTHONPATH
│   ├── train.sh                # Launch training (dataset switch)
│   ├── run_experiment.sh       # Experiment wrapper
│   └── test.sh                 # Test suite
├── configs/                    # Configuration files
│   ├── snopes_config.conf
│   ├── politifact_config.conf
│   ├── mumin_multilingual_config.conf
│   ├── m3check_multilingual_config.conf
│   ├── m3check_no_eng_config.conf
│   └── experiments.yaml
├── datasets/                   # Datasets (raw + prepared)
│   ├── fakenewsdet/            # Snopes + Politifact (ready TSV files)
│   ├── M3-check/               # Raw M3-Check (original CSVs)
│   ├── mumin/                  # Raw MuMiN
│   ├── mumin_mmbt/             # Prepared MuMiN
│   ├── mmbt_m3check_multilingual/  # Prepared M3-Check
│   └── mmbt_m3check_no_eng/    # M3-Check without English
├── mmbt/                       # Main source code
│   ├── models/                 # MMBT architecture (BERT + ResNet)
│   ├── data/                   # DataLoader, dataset.py, helpers.py
│   ├── losses/                 # CrossSimilarity loss
│   ├── metrics/                # Ranking metrics (NDCG, Hit, MAP)
│   ├── train.py                # Training loop
│   └── evaluate.py             # Evaluation loop
├── scripts/                    # Preparation and experiment scripts
│   ├── prepare_mumin_dataset_v2.py    # MuMiN preparation
│   ├── prepare_m3check_dataset.py     # M3-Check preparation
│   ├── benchmark_baselines_man.py     # Multilingual MAN baseline (EMNLP 2020)
│   ├── run_experiments.py             # Experiment entry point
│   └── run_all_experiments.py         # Batch runner
├── models/checkpoints/         # Trained model checkpoints
│   ├── mmbt_snopes/
│   ├── mmbt_politifact/
│   ├── mmbt_mumin_multilingual/
│   ├── mmbt_m3check/
│   └── mmbt_m3check_no_eng/
├── outputs/                    # Experiment results
├── tests/                      # Test suite
├── main.py                     # Training/evaluation entry point
├── Makefile                    # Full automation
└── requirements.txt            # Python dependencies
```

---

## Complete Step-by-Step Guide

Here are all the steps from cloning the repository to obtaining the final results, in the correct order.

### Step 1 - Environment Setup

```bash
git clone <repo-url> && cd MMBT_FactChecking_Thesis
make setup
# or: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

### Step 2 - Download BERT Models

```bash
make download-models
# Saves bert-tiny to /root/bert_tiny_local
# Saves mBERT to /root/bert_multilingual_local
```

### Step 3 - Prepare the Datasets

```bash
# Snopes/Politifact: place the TSV files in datasets/fakenewsdet/annotations/
# (already ready to use, no preparation required)

# MuMiN:
python scripts/prepare_mumin_dataset_v2.py --neg-ratio 30 --exclude-langs en --output-dir datasets/mumin_mmbt

# M3-Check (all languages):
python scripts/prepare_m3check_dataset.py --input-dir datasets/M3-check --output-dir datasets/mmbt_m3check_multilingual --neg-ratio 30 --seed 2024

# M3-Check (without English, optional):
python scripts/prepare_m3check_dataset.py --input-dir datasets/M3-check --output-dir datasets/mmbt_m3check_no_eng --neg-ratio 30 --exclude-langs eng --seed 2024
```

### Step 4 - Training

```bash
# Choose the dataset to train:
make train-snopes
make train-politifact
make train-mumin-multilingual
make train-m3check-multilingual
make train-m3check-no-eng
```

Training automatically saves the best checkpoint in `models/checkpoints/{name}/model_best.pt`.

### Step 5 - Evaluation

```bash
# After training, evaluate on the test set:
make eval DATASET=snopes
make eval DATASET=m3check_multilingual
make eval DATASET=m3check_no_eng
```

### Step 6 - Experiments

```bash
# White noise (robustness to noise in images):
make exp-noise DATASET=Snopes

# FAISS scalability:
make exp-complexity DATASET=Snopes
```

---

## FAISS GPU

To switch from FAISS CPU to GPU (useful for corpora >500k documents):

```bash
# 1. Install faiss-gpu
pip uninstall faiss-cpu
pip install faiss-gpu

# 2. Verify
python -c "import faiss; print('Available GPUs:', faiss.get_num_gpus())"
```

In the code, the only required change is in `run_experiments.py` inside `ComplexityExperiment.measure_faiss_scaling()`:

```python
# Before (CPU)
index_flat = faiss.IndexFlatL2(dim)

# After (GPU)
res = faiss.StandardGpuResources()
index_flat_cpu = faiss.IndexFlatL2(dim)
index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat_cpu)
```

With the RTX 2070 (8 GB), datasets up to ~500k documents (dim=128) fit comfortably in VRAM. For larger corpora, use `faiss.index_cpu_to_gpu_multiple()` or stay on CPU with IVFFlat, which is already very fast (~0.04 ms/query on 170k documents).

---

## Quick Reference

```bash
# Activate environment
source .venv/bin/activate && export PYTHONPATH="$PWD:$PYTHONPATH"

# === TRAINING ===
python main.py -c configs/snopes_config.conf                   # Snopes
python main.py -c configs/politifact_config.conf                # Politifact
python main.py -c configs/mumin_multilingual_config.conf        # MuMiN
python main.py -c configs/m3check_multilingual_config.conf      # M3-Check
python main.py -c configs/m3check_no_eng_config.conf            # M3-Check without EN

# === EVALUATION (add --is_evaluation Yes) ===
python main.py -c configs/snopes_config.conf --is_evaluation Yes
python main.py -c configs/m3check_multilingual_config.conf --is_evaluation Yes

# === EXPERIMENTS ===
python scripts/run_experiments.py --experiment white_noise --dataset Snopes --output-dir outputs
python scripts/run_experiments.py --experiment white_noise --dataset M3Check --output-dir outputs
python scripts/run_experiments.py --experiment complexity --dataset Snopes --output-dir outputs

# === BASELINE MAN (EMNLP 2020) ===
python -m scripts.benchmark_baselines_man --dataset mumin                  # MuMiN
python -m scripts.benchmark_baselines_man --dataset m3check_multilingual   # M3-Check
python -m scripts.benchmark_baselines_man --dataset m3check_no_eng         # M3-Check no EN
python -m scripts.benchmark_baselines_man --dataset mumin --eval-only      # Evaluation only

# === DATASET PREPARATION ===
python scripts/prepare_mumin_dataset_v2.py --neg-ratio 30 --exclude-langs en --output-dir datasets/mumin_mmbt
python scripts/prepare_m3check_dataset.py --input-dir datasets/M3-check --output-dir datasets/mmbt_m3check_multilingual --neg-ratio 30
python scripts/prepare_m3check_dataset.py --input-dir datasets/M3-check --output-dir datasets/mmbt_m3check_no_eng --neg-ratio 30 --exclude-langs eng
```

---

## Citation

If you use this repository in your work, please cite the following paper:

```bibtex
@article{FORMISANO2026133703,
title = {Unifying retrieval and re-ranking: A multimodal approach to detecting fact-checked information},
journal = {Neurocomputing},
pages = {133703},
year = {2026},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2026.133703},
url = {https://www.sciencedirect.com/science/article/pii/S0925231226011008},
author = {Raffaele Formisano and Valerio {La Gatta} and Vincenzo Moscato and Giancarlo Sperl\`i},
keywords = {Fact-checking, Verified claim retrieval, Multimodal disinformation mining},
abstract = {Although recently several fact-checking organizations have emerged to verify disinformation, fake news has continued to proliferate, especially exploiting multimodal data on social media. As a result, the fact-checking verification process cannot keep up with this overwhelming and uncertain content, thus raising the need to adopt improved strategies for disinformation detection. The fact-checking process could be optimised considering the tendency of viral claims to be reshared over time and in different contexts. In other words, verifying whether a (multimodal) claim has been previously fact-checked can ease fact-checkers' manual effort and would provide reliable evidence for the input claim. In this paper, considering the task's ranking formulation, we propose a novel multimodal information retrieval approach aiming at retrieving and re-ranking a list of verified documents according to their relevance with the input claim. Specifically, we exploit text and image's modalities and leverage the modern visual-language models to extract powerful representations that capture their complex relationships. Our experiments on three benchmark datasets prove the superiority of the proposed system: in re-ranking settings, it exceeds competitors up to 15 NDCG points; in retrieval settings, it is the only one which overcomes the standard BM25 baseline.}
}
```

---

## Troubleshooting

### `invalid choice: 'M3Check'` for `--type_dataset`

Check that `mmbt/utils/parseargs.py` includes `M3Check` and `M3CheckNoEng` in the `choices` list for the `--type_dataset` argument.

### `FileNotFoundError` during eval

Eval looks for the `test.tsv` file in the directory configured in `mmbt/data/helpers.py` (`get_dataset_annotations_path()`). Verify that:
1. The dataset has been prepared and the TSV files exist.
2. The mapping in `helpers.py` points to the correct directory.

### CUDA out of memory

Reduce `batch_sz` in the `.conf` file (for example from 8 to 4), or increase `gradient_accumulation_steps` to compensate.

### Placeholder Images (Gray)

If the dataset images are not extracted (ZIP archives not unpacked), the system uses gray placeholders. The model still works thanks to the textual component. To use the real images, extract the archives into the correct directory under `datasets/`.

---

## License

MIT License - See [LICENSE](LICENSE)
