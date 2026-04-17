# Complete Documentation: MultiModal BiTransformers (MMBT) for Fact-Checking

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Code Structure](#3-code-structure)
4. [Training Pipeline](#4-training-pipeline)
5. [Ranking and Retrieval System](#5-ranking-and-retrieval-system)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [Datasets Used](#7-datasets-used)
8. [Configuration and Parameters](#8-configuration-and-parameters)
9. [Execution Flow](#9-execution-flow)

---

## 1. Project Overview

### 1.1 Objective
This project implements an **automatic fact-checking** system based on multimodal techniques that combine:
- **Text**: analyzed through BERT (Bidirectional Encoder Representations from Transformers)
- **Images**: processed through ResNet (Residual Neural Network)

The system addresses the **fact-checking document retrieval** problem given a claim composed of text and image.

### 1.2 Approach
The MMBT (MultiModal BiTransformers) model merges text and image representations into a single vector space, making it possible to:
1. Encode multimodal queries (claims to verify)
2. Encode fact-checking documents (verification articles)
3. Compute similarity between queries and documents
4. Rank documents by relevance

### 1.3 Main Contributions
- MMBT implementation for fact-checking
- Cross-Similarity system for multimodal ranking
- Integration with approxNDCG loss for Learning-to-Rank
- Feature extraction pipeline for efficient FAISS-based retrieval

---

## 2. System Architecture

### 2.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           MULTIMODAL INPUT                             │
├──────────────────────────┬──────────────────────────────────────────────┤
│         QUERY            │                DOCUMENT                      │
│  ┌──────────┬─────────┐  │  ┌──────────────┬──────────────────┐         │
│  │   Text   │  Image  │  │  │     Text     │      Image       │         │
│  │  Claim   │  Claim  │  │  │  Fact-Check  │   Fact-Check     │         │
│  └────┬─────┴────┬────┘  │  └──────┬───────┴────────┬─────────┘         │
│       │          │       │         │                │                    │
│       ▼          ▼       │         ▼                ▼                    │
│  ┌─────────┐ ┌────────┐  │  ┌───────────┐    ┌───────────┐              │
│  │  BERT   │ │ ResNet │  │  │   BERT    │    │  ResNet   │              │
│  │Tokenizer│ │Encoder │  │  │ Tokenizer │    │ Encoder   │              │
│  └────┬────┘ └───┬────┘  │  └─────┬─────┘    └─────┬─────┘              │
│       │          │       │        │                │                     │
│       ▼          ▼       │        ▼                ▼                     │
│  ┌───────────────────┐   │  ┌────────────────────────────┐              │
│  │ Image BERT Embed  │   │  │    Image BERT Embed        │              │
│  │  (Modal Fusion)   │   │  │      (Modal Fusion)        │              │
│  └─────────┬─────────┘   │  └──────────────┬─────────────┘              │
│            │             │                 │                             │
│            ▼             │                 ▼                             │
│  ┌───────────────────┐   │  ┌────────────────────────────┐              │
│  │   BERT Encoder    │   │  │      BERT Encoder          │              │
│  │    (12 layers)    │   │  │       (12 layers)          │              │
│  └─────────┬─────────┘   │  └──────────────┬─────────────┘              │
│            │             │                 │                             │
│            ▼             │                 ▼                             │
│  ┌───────────────────┐   │  ┌────────────────────────────┐              │
│  │    BERT Pooler    │   │  │       BERT Pooler          │              │
│  │    (CLS Token)    │   │  │        (CLS Token)         │              │
│  └─────────┬─────────┘   │  └──────────────┬─────────────┘              │
│            │             │                 │                             │
│            ▼             │                 ▼                             │
│      Query Vector        │          Document Vector                      │
│      [hidden_sz]         │           [hidden_sz]                         │
└──────────────────────────┴──────────────────────────────────────────────┘
                    │                        │
                    └──────────┬─────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Cross Similarity   │
                    │   (Cosine Distance)  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    approxNDCG Loss   │
                    │   (Learning to Rank) │
                    └──────────────────────┘
```

### 2.2 Main Components

#### 2.2.1 Image Encoder (ResNet152/50)
- **Input**: 224x224 RGB image
- **Output**: 2048-dimensional features
- **Process**:
  1. Pre-processing (resize, crop, normalize)
  2. Feature extraction with ResNet (removing the last 2 layers)
  3. Adaptive pooling to handle different resolutions

#### 2.2.2 Text Encoder (BERT)
- **Input**: token sequence (max 128)
- **Output**: contextual embeddings
- **Supported models**: bert-base-uncased, bert-tiny, and others

#### 2.2.3 Image-BERT Embeddings
Converts image features into BERT-compatible embeddings:
1. Linear projection from 2048 to hidden_sz
2. Addition of position embeddings
3. Addition of token type embeddings
4. Layer normalization and dropout

#### 2.2.4 Multimodal Encoder
Fuses text and image:
1. Concatenates [CLS] + Image_Embed + [SEP] + Text_Embed
2. Applies the BERT encoder (12 transformer layers)
3. Pools with the CLS token for the final representation

---

## 3. Code Structure

The repository is organized around a main training entry point, configuration files, dataset preparation scripts, experiments, and the core MMBT package.

```text
MMBT_FactChecking_Thesis/
├── README.md
├── main.py                          # Main entry point for training/evaluation
├── configs/                         # Configuration files
│   ├── configurazione.conf
│   ├── snopes_config.conf
│   ├── politifact_config.conf
│   ├── mumin_multilingual_config.conf
│   ├── m3check_multilingual_config.conf
│   └── experiments.yaml
├── mmbt/                            # Core package (models, data, losses, metrics, utils)
├── scripts/                         # Dataset prep, baselines, experiments, reporting
├── docs/                            # Extended project documentation
├── tests/                           # Unit and integration tests
├── datasets/                        # Raw and processed datasets
├── models/                          # Saved checkpoints
└── outputs/                         # Experiment outputs and reports
```

At a high level, the core package is divided into:
- **models**: multimodal encoders and image/text submodules
- **data**: datasets, transforms, loading helpers, vocabulary utilities
- **losses**: cross-similarity and ranking losses
- **metrics**: NDCG, Hit, MAP, and related ranking metrics
- **utils**: argument parsing, logging, optimizer, scheduler, and helpers

---

## 4. Training Pipeline

### 4.1 Training Flow

```python
# Simplified pseudocode
def train():
    # 1. Data loading
    train_loader, val_loader, test_loader = get_data_loaders()

    # 2. Model initialization
    model = MMBT(args)
    optimizer = AdamW(model.parameters())
    scheduler = ReduceLROnPlateau(optimizer)

    # 3. Training loop
    for epoch in range(max_epochs):
        for batch in train_loader:
            # Forward pass with a triplet (query, positive_doc, negative_doc)
            out_query, out_doc, out_neg = model(batch)

            # Similarity and loss computation (approxNDCG)
            similarity = cross_similarity(out_query, out_doc)
            loss = approxNDCG_loss(similarity, labels)

            # Backward pass and parameter update
            loss.backward()
            optimizer.step()

        # Validation
        metrics = evaluate(val_loader, model)
        scheduler.step(metrics['ndcg_5'])

        # Early stopping
        if no_improvement > patience:
            break

    # 4. Final test
    test_metrics = evaluate(test_loader, model)
```

### 4.2 Training Strategy

1. **Triplet Learning**: each batch contains:
   - Query (claim to verify)
   - Positive document (correct fact-check article)
   - Negative document (random, unrelated article)

2. **Gradient Accumulation**: to simulate larger batch sizes on limited GPUs
   ```python
   gradient_accumulation_steps = 24
   loss = loss / gradient_accumulation_steps
   ```

3. **Mixed Precision**: use of `torch.cuda.amp.autocast()` for efficiency

4. **Learning Rate Schedule**: ReduceLROnPlateau based on NDCG@5

---

## 5. Ranking and Retrieval System

### 5.1 Cross-Similarity Loss

The loss implemented in `CrossSimilarity.py`:

```python
class CrossSimilarity:
    def similarities(self, query, doc):
        # Euclidean distance (used as dissimilarity)
        return (query - doc).pow(2).sum(axis=-1).sqrt()

    def forward(self, query, doc, labels, k):
        similarity_scores = self.similarities(query, doc)

        # Top-k retrieval with FAISS (via sentence_transformers)
        hits = semantic_search(query, doc, top_k=k)

        # approxNDCG loss for learning-to-rank
        loss = approxNDCG_loss(similarity_scores, labels)

        return hits, similarity_scores, loss
```

### 5.2 Feature Extraction for Retrieval

For efficient inference on large datasets:

1. **Offline extraction**: compute document embeddings once
2. **Disk persistence**: save features as `.pt` files
3. **FAISS retrieval**: efficient nearest-neighbor search

```python
# Document feature extraction
for doc_batch in doc_loader:
    features = model.encode(doc_batch)
    save_to_disk(features, f"doc_{batch_id}.pt")

# Retrieval
doc_features = load_all_features()  # Tensor [N_docs, hidden_sz]
for query in query_loader:
    query_feat = model.encode(query)
    results = semantic_search(query_feat, doc_features, top_k=50)
```

---

## 6. Evaluation Metrics

### 6.1 Implemented Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| **NDCG@k** | Normalized Discounted Cumulative Gain | $\frac{DCG@k}{IDCG@k}$ |
| **Hit@k** | Hit Rate (at least 1 relevant document in top-k) | $\frac{\text{queries with a hit}}{N_{queries}}$ |
| **MAP@k** | Mean Average Precision | $\frac{1}{Q}\sum_{q=1}^{Q} AP_q$ |
| **MRR** | Mean Reciprocal Rank | $\frac{1}{Q}\sum_{q=1}^{Q} \frac{1}{rank_q}$ |

### 6.2 NDCG Computation

```python
def ndcg(y_pred, y_true, k):
    # DCG = sum (2^rel - 1) / log2(rank + 1)
    dcg = sum([(2**rel - 1) / log2(rank + 2)
               for rank, rel in enumerate(sorted_by_pred[:k])])

    # IDCG = ideal DCG (documents sorted by relevance)
    idcg = sum([(2**rel - 1) / log2(rank + 2)
                for rank, rel in enumerate(sorted_by_true[:k])])

    return dcg / idcg if idcg > 0 else 0
```

---

## 7. Datasets Used

### 7.1 FakeNewsDet (Snopes + Politifact)

Main fact-checking dataset with:
- **Query**: claim to verify (text + image)
- **Documents**: fact-check articles (text + image)
- **Label**: 0 (non-relevant) / 1 (relevant)

TSV structure:
```
QueryID | QueryText | DocID | DocText | Label
```

### 7.2 MuMiN Dataset

Multilingual misinformation dataset with:
- 21.5M tweets
- 2M users
- 12,914 fact-checked claims
- 41 languages

---

## 8. Configuration and Parameters

### 8.1 Main Parameters

```properties
# Model
bert_model='prajjwal1/bert-tiny'     # BERT model
image_model='resnet152'              # Image encoder
hidden_sz=128                        # Hidden size
dropout=0.4                          # Dropout rate

# Training
batch_sz=8                           # Batch size
max_epochs=100                       # Maximum epochs
lr=1e-5                              # Learning rate
gradient_accumulation_steps=24       # Accumulation steps
patience=10                          # Early stopping patience

# Data
max_seq_len=128                      # Text sequence length
num_image_embeds=3                   # Number of image embeddings

# Feature extraction
doc_extraction='No'                  # Extraction mode
top_k_extract=50                     # Top-k for retrieval
```

### 8.2 Execution Modes

| Mode | Description |
|------|-------------|
| `is_evaluation='No'` | Training + evaluation |
| `is_evaluation='Yes'` | Evaluation only (loads a trained model) |
| `doc_extraction='Yes'` | Document feature extraction |

---

## 9. Execution Flow

### 9.1 Standard Training

```bash
python main.py -c configs/configurazione.conf
```

Flow:
1. Configuration parsing -> `parseargs.py`
2. Logger setup -> `logger.py`
3. Data loading -> `helpers.py` -> `dataset.py`
4. Model initialization -> `mmbt.py`
5. Training loop -> `train.py`
6. Validation -> `evaluate.py`
7. Test -> `evaluate.py`
8. Result saving

### 9.2 Feature Extraction

```bash
# Modify configs/configurazione.conf
doc_extraction='Yes'
save_documents='Yes'
doc_training='training'
```

Flow:
1. Train the model for extraction
2. Save document features to disk
3. Run inference against the document collection
4. Compute ranking metrics

### 9.3 Generated Outputs

- `model_best.pt`: best model checkpoint
- `times_test_{dataset}.csv`: execution timings
- `means_{dataset}_top{k}.csv`: average metrics
- `embedding_final_{dataset}_top{k}.txt`: similarity scores
