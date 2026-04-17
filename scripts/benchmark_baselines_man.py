#!/usr/bin/env python3
"""
Benchmark MAN (Multimodal Attention Network) Baseline
=====================================================

Implementazione della baseline MAN (Vo & Lee, EMNLP 2020) adattata per il
contesto multilingue. Sostituisce GloVe + ELMo con mBERT per supportare
dataset non-inglesi (MuMiN, M3-Check).

Architettura MAN:
- Encoding testo con mBERT (token embeddings)
- Matrice di similarità query-document
- CNN n-gram + PACRR-style max-pooling
- Hinge loss per learning-to-rank

Il modello viene addestrato sul train set e valutato sul test set con
NDCG@k e HIT@k.

Usage:
    # MuMiN (multilingua, senza inglese)
    python -m scripts.benchmark_baselines_man --dataset mumin

    # M3-Check multilingua
    python -m scripts.benchmark_baselines_man --dataset m3check_multilingual

    # M3-Check senza inglese
    python -m scripts.benchmark_baselines_man --dataset m3check_no_eng

    # Solo valutazione (modello già addestrato)
    python -m scripts.benchmark_baselines_man --dataset mumin --eval-only

Riferimento:
    Vo, N. & Lee, K. (2020). "Where Are the Facts? Searching for Fact-checked
    Information to Alleviate the Spread of Fake News." EMNLP 2020.
    https://aclanthology.org/2020.emnlp-main.621/
"""

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ---------------------------------------------------------------------------
# Metriche (stesse usate nelle altre baseline)
# ---------------------------------------------------------------------------

def ndcg_at_k(relevance_scores: List[int], k: int) -> float:
    scores = relevance_scores[:k]
    if not scores:
        return 0.0
    dcg = scores[0]
    for i, rel in enumerate(scores[1:], start=2):
        dcg += rel / np.log2(i + 1)
    ideal = sorted(scores, reverse=True)
    idcg = ideal[0] if ideal else 0
    for i, rel in enumerate(ideal[1:], start=2):
        idcg += rel / np.log2(i + 1)
    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k(relevance_scores: List[int], k: int) -> float:
    return 1.0 if any(relevance_scores[:k]) else 0.0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

DATASET_PATHS = {
    "mumin": "mumin_mmbt",
    "m3check_multilingual": "mmbt_m3check_multilingual",
    "m3check_no_eng": "mmbt_m3check_no_eng",
}


class PairwiseRankDataset(Dataset):
    """
    Dataset per training pairwise: genera triple (query, doc+, doc-).
    Legge i file TSV nel formato MMBT standard.
    """

    def __init__(
        self,
        tsv_path: str,
        tokenizer,
        max_query_len: int = 64,
        max_doc_len: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len

        logger.info(f"Caricamento dati da {tsv_path}...")
        df = pd.read_csv(tsv_path, sep="\t")
        logger.info(f"  Righe caricate: {len(df)}")

        # Raggruppa per query
        self.query_groups: Dict[str, Dict] = {}
        for _, row in df.iterrows():
            qid = str(row["QueryID"])
            if qid not in self.query_groups:
                self.query_groups[qid] = {
                    "text": str(row["QueryText"]) if pd.notna(row["QueryText"]) else "",
                    "pos_docs": [],
                    "neg_docs": [],
                }
            doc_text = str(row["DocText"]) if pd.notna(row["DocText"]) else ""
            label = int(float(row["Label"])) if pd.notna(row["Label"]) else 0
            if label == 1:
                self.query_groups[qid]["pos_docs"].append(doc_text)
            else:
                self.query_groups[qid]["neg_docs"].append(doc_text)

        # Filtra query con almeno 1 pos e 1 neg
        self.valid_qids = [
            qid
            for qid, g in self.query_groups.items()
            if g["pos_docs"] and g["neg_docs"]
        ]
        logger.info(f"  Query valide (con pos+neg): {len(self.valid_qids)}")

        # Genera coppie (query, pos, neg) — campiona 1 negativo per positivo
        self.triples = []
        for qid in self.valid_qids:
            g = self.query_groups[qid]
            for pos_doc in g["pos_docs"]:
                self.triples.append((g["text"], pos_doc, qid))

        logger.info(f"  Triple di training: {len(self.triples)}")

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        query_text, pos_text, qid = self.triples[idx]
        # Campiona un negativo random
        neg_text = np.random.choice(self.query_groups[qid]["neg_docs"])

        q_enc = self.tokenizer(
            query_text,
            max_length=self.max_query_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        pos_enc = self.tokenizer(
            pos_text,
            max_length=self.max_doc_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        neg_enc = self.tokenizer(
            neg_text,
            max_length=self.max_doc_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "query_ids": q_enc["input_ids"].squeeze(0),
            "query_mask": q_enc["attention_mask"].squeeze(0),
            "pos_ids": pos_enc["input_ids"].squeeze(0),
            "pos_mask": pos_enc["attention_mask"].squeeze(0),
            "neg_ids": neg_enc["input_ids"].squeeze(0),
            "neg_mask": neg_enc["attention_mask"].squeeze(0),
        }


class EvalDataset(Dataset):
    """Dataset per valutazione: coppie query-doc con label."""

    def __init__(
        self,
        tsv_path: str,
        tokenizer,
        max_query_len: int = 64,
        max_doc_len: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len

        df = pd.read_csv(tsv_path, sep="\t")

        self.items = []
        for _, row in df.iterrows():
            qid = str(row["QueryID"])
            did = str(row["DocID"])
            qt = str(row["QueryText"]) if pd.notna(row["QueryText"]) else ""
            dt = str(row["DocText"]) if pd.notna(row["DocText"]) else ""
            label = int(float(row["Label"])) if pd.notna(row["Label"]) else 0
            self.items.append((qid, did, qt, dt, label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        qid, did, qt, dt, label = self.items[idx]
        q_enc = self.tokenizer(
            qt,
            max_length=self.max_query_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        d_enc = self.tokenizer(
            dt,
            max_length=self.max_doc_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "query_ids": q_enc["input_ids"].squeeze(0),
            "query_mask": q_enc["attention_mask"].squeeze(0),
            "doc_ids": d_enc["input_ids"].squeeze(0),
            "doc_mask": d_enc["attention_mask"].squeeze(0),
            "label": label,
            "qid": qid,
            "did": did,
        }


# ---------------------------------------------------------------------------
# MAN Model (adattato per mBERT)
# ---------------------------------------------------------------------------


class PACRRMaxPool(nn.Module):
    """
    PACRR-style max-pooling per estrarre segnali dalla matrice
    di similarità multi-canale: [sim, ctx, sim-ctx, sim*ctx].
    """

    def __init__(
        self,
        input_channels: int = 4,
        n_filters: int = 16,
        n_conv_layers: int = 2,
        n_s: int = 32,
    ):
        super().__init__()
        self.n_s = n_s
        layers = []
        in_ch = input_channels
        for i in range(n_conv_layers):
            out_ch = n_filters if i < n_conv_layers - 1 else n_filters
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.out_channels = n_filters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, L, R)  — matrice multi-canale
        Returns: (B, n_filters * n_s)
        """
        out = self.conv(x)  # (B, F, L, R)
        # Max-pool sui top-n_s segnali per ogni filtro
        B, F, L, R = out.shape
        out = out.view(B, F, L * R)
        # top n_s valori per filtro
        k = min(self.n_s, L * R)
        top_vals, _ = torch.topk(out, k=k, dim=-1)
        return top_vals.view(B, -1)


class MANMultilingual(nn.Module):
    """
    Multimodal Attention Network adattato per mBERT (text-only).

    Sostituisce GloVe+ELMo con mBERT:
    - mBERT produce sia word embeddings sia rappresentazioni contestuali
    - La matrice di similarità usa gli embedding dei token
    - CNNs estraggono pattern n-gram dalla matrice di similarità
    - PACRR max-pool seleziona i segnali più forti
    - Linear head produce lo score di rilevanza
    """

    def __init__(
        self,
        bert_model_path: str,
        max_ngram: int = 2,
        n_filters: int = 256,
        pacrr_filters: int = 16,
        n_conv_layers: int = 2,
        n_s: int = 32,
        dropout: float = 0.1,
        freeze_bert: bool = True,
    ):
        super().__init__()
        from transformers import BertModel

        self.bert = BertModel.from_pretrained(bert_model_path)
        self.hidden_size = self.bert.config.hidden_size

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.max_ngram = max_ngram
        self.input_channels = 4  # [sim, ctx, sim-ctx, sim*ctx]

        # Proiezione per ridurre la dimensionalità dei token embeddings
        self.proj = nn.Linear(self.hidden_size, n_filters)

        # CNN n-gram per gli embedding (word-level)
        self.q_convs = nn.ModuleList()
        self.d_convs = nn.ModuleList()
        # CNN n-gram per le rappresentazioni contestuali
        self.q_ctx_convs = nn.ModuleList()
        self.d_ctx_convs = nn.ModuleList()

        for i in range(max_ngram):
            ks = i + 1
            conv = nn.Sequential(
                nn.Conv1d(n_filters, n_filters, kernel_size=ks, padding=0),
                nn.Tanh(),
            )
            conv_ctx = nn.Sequential(
                nn.Conv1d(n_filters, n_filters, kernel_size=ks, padding=0),
                nn.Tanh(),
            )
            self.q_convs.append(conv)
            self.d_convs.append(conv)
            self.q_ctx_convs.append(conv_ctx)
            self.d_ctx_convs.append(conv_ctx)

        # PACRR head per ogni livello n-gram
        self.pacrr_heads = nn.ModuleList()
        for _ in range(max_ngram):
            self.pacrr_heads.append(
                PACRRMaxPool(
                    input_channels=self.input_channels,
                    n_filters=pacrr_filters,
                    n_conv_layers=n_conv_layers,
                    n_s=n_s,
                )
            )

        # Calcolo dimensione output
        pacrr_out_dim = max_ngram * pacrr_filters * n_s

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(pacrr_out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _encode(self, input_ids, attention_mask):
        """Encoding con BERT, restituisce embedding + hidden states."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # word embeddings (primo layer)
        word_emb = self.bert.embeddings.word_embeddings(input_ids)
        # hidden states contestuali (ultimo layer)
        ctx_emb = outputs.last_hidden_state
        return word_emb, ctx_emb, attention_mask

    def _compute_sim_matrices(
        self, q_word, q_ctx, q_mask, d_word, d_ctx, d_mask, ngram_idx
    ):
        """
        Calcola le 4 matrici di similarità per un livello n-gram:
        [sim, context_sim, sim - context_sim, sim * context_sim]
        """
        # Proietta e passa attraverso CNN n-gram
        q_proj = self.proj(q_word).transpose(1, 2)  # (B, F, L)
        d_proj = self.proj(d_word).transpose(1, 2)  # (B, F, R)
        q_ctx_proj = self.proj(q_ctx).transpose(1, 2)
        d_ctx_proj = self.proj(d_ctx).transpose(1, 2)

        q_conv = self.q_convs[ngram_idx](q_proj).transpose(1, 2)  # (B, L', F)
        d_conv = self.d_convs[ngram_idx](d_proj).transpose(1, 2)  # (B, R', F)
        q_ctx_conv = self.q_ctx_convs[ngram_idx](q_ctx_proj).transpose(1, 2)
        d_ctx_conv = self.d_ctx_convs[ngram_idx](d_ctx_proj).transpose(1, 2)

        # Normalizza
        q_conv = F.normalize(q_conv, p=2, dim=-1)
        d_conv = F.normalize(d_conv, p=2, dim=-1)
        q_ctx_conv = F.normalize(q_ctx_conv, p=2, dim=-1)
        d_ctx_conv = F.normalize(d_ctx_conv, p=2, dim=-1)

        # Matrice di similarità testuale (dot product)
        sim = torch.bmm(q_conv, d_conv.transpose(1, 2))  # (B, L', R')

        # Matrice di similarità contestuale
        ctx_sim = torch.bmm(q_ctx_conv, d_ctx_conv.transpose(1, 2))

        # Stack 4 canali: [sim, ctx, sim-ctx, sim*ctx]
        tensors = torch.stack(
            [sim, ctx_sim, sim - ctx_sim, sim * ctx_sim], dim=1
        )  # (B, 4, L', R')

        return tensors

    def _score(self, q_ids, q_mask, d_ids, d_mask):
        """Calcola lo score di rilevanza per una coppia query-doc."""
        q_word, q_ctx, _ = self._encode(q_ids, q_mask)
        d_word, d_ctx, _ = self._encode(d_ids, d_mask)

        phis = []
        for i in range(self.max_ngram):
            sim_tensor = self._compute_sim_matrices(
                q_word, q_ctx, q_mask, d_word, d_ctx, d_mask, i
            )
            phi = self.pacrr_heads[i](sim_tensor)
            phis.append(phi)

        features = torch.cat(phis, dim=-1)
        score = self.classifier(features).squeeze(-1)
        return score

    def forward(self, batch, mode="train"):
        """
        Training: batch con query, pos_doc, neg_doc -> hinge loss
        Eval: batch con query, doc -> score
        """
        if mode == "train":
            pos_score = self._score(
                batch["query_ids"],
                batch["query_mask"],
                batch["pos_ids"],
                batch["pos_mask"],
            )
            neg_score = self._score(
                batch["query_ids"],
                batch["query_mask"],
                batch["neg_ids"],
                batch["neg_mask"],
            )
            return pos_score, neg_score
        else:
            score = self._score(
                batch["query_ids"],
                batch["query_mask"],
                batch["doc_ids"],
                batch["doc_mask"],
            )
            return score


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------


def train_man(
    model: MANMultilingual,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_epochs: int = 3,
    lr: float = 1e-4,
    margin: float = 1.0,
    checkpoint_dir: Optional[str] = None,
):
    """
    Addestra il modello MAN con hinge loss (margin-based ranking loss).
    """
    # Solo parametri non-frozen
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    hinge_loss = nn.MarginRankingLoss(margin=margin)

    best_val_ndcg = -1
    best_model_state = None

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            pos_score, neg_score = model(batch, mode="train")
            target = torch.ones_like(pos_score)
            loss = hinge_loss(pos_score, neg_score, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if batch_idx % 100 == 0:
                logger.info(
                    f"  Epoch {epoch + 1}/{n_epochs} | "
                    f"Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / max(n_batches, 1)
        logger.info(f"Epoch {epoch + 1}/{n_epochs} | Avg Loss: {avg_loss:.4f}")

        # Validazione
        val_metrics = evaluate_man(model, val_loader, device)
        val_ndcg = val_metrics.get("ndcg@3", 0.0)
        logger.info(
            f"  Val NDCG@3: {val_ndcg:.4f} | "
            f"Val HIT@3: {val_metrics.get('hit@3', 0):.4f}"
        )

        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info(f"  -> Nuovo best model (NDCG@3={val_ndcg:.4f})")

            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
                ckpt_path = os.path.join(checkpoint_dir, "man_best.pt")
                torch.save(best_model_state, ckpt_path)
                logger.info(f"  -> Checkpoint salvato: {ckpt_path}")

    # Ripristina best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def evaluate_man(
    model: MANMultilingual,
    eval_loader: DataLoader,
    device: torch.device,
    k_values: List[int] = None,
) -> Dict[str, float]:
    """
    Valuta il modello MAN e calcola NDCG@k e HIT@k.
    """
    if k_values is None:
        k_values = [1, 3, 5]

    model.eval()
    all_scores: Dict[str, List[Tuple[str, int, float]]] = defaultdict(list)

    with torch.no_grad():
        for batch in eval_loader:
            device_batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            scores = model(device_batch, mode="eval")
            scores_np = scores.cpu().numpy()

            qids = batch["qid"]
            dids = batch["did"]
            labels = batch["label"].numpy()

            for i in range(len(qids)):
                all_scores[qids[i]].append(
                    (dids[i], int(labels[i]), float(scores_np[i]))
                )

    # Calcola metriche per query
    results = {f"hit@{k}": [] for k in k_values}
    results.update({f"ndcg@{k}": [] for k in k_values})

    for qid, items in all_scores.items():
        # Ordina per score decrescente
        ranked = sorted(items, key=lambda x: x[2], reverse=True)
        rel_scores = [label for _, label, _ in ranked]

        if not any(rel_scores):
            continue

        for k in k_values:
            results[f"hit@{k}"].append(hit_at_k(rel_scores, k))
            results[f"ndcg@{k}"].append(ndcg_at_k(rel_scores, k))

    metrics = {}
    for key, values in results.items():
        metrics[key] = float(np.mean(values)) if values else 0.0
    metrics["n_queries_evaluated"] = len(
        [q for q in all_scores if any(l == 1 for _, l, _ in all_scores[q])]
    )

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_dataset_dir(dataset: str) -> Path:
    """Restituisce il path della directory annotations per il dataset."""
    subdir = DATASET_PATHS.get(dataset)
    if subdir is None:
        raise ValueError(
            f"Dataset '{dataset}' non supportato. "
            f"Scelte: {list(DATASET_PATHS.keys())}"
        )
    return PROJECT_ROOT / "datasets" / subdir / "annotations"


def run_man_baseline(
    dataset: str,
    bert_path: str = "/root/bert_multilingual_local",
    n_epochs: int = 3,
    batch_size: int = 16,
    lr: float = 1e-4,
    max_query_len: int = 64,
    max_doc_len: int = 128,
    max_ngram: int = 2,
    n_filters: int = 128,
    pacrr_filters: int = 16,
    n_conv_layers: int = 2,
    n_s: int = 32,
    margin: float = 1.0,
    freeze_bert: bool = True,
    eval_only: bool = False,
    max_train_samples: int = 0,
    seed: int = 42,
):
    """Esegue training e valutazione del modello MAN multilingue."""

    # Riproducibilità
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    ann_dir = get_dataset_dir(dataset)
    checkpoint_dir = str(PROJECT_ROOT / "models" / "checkpoints" / f"man_{dataset}")

    logger.info("=" * 60)
    logger.info(f"MAN Baseline — Dataset: {dataset}")
    logger.info(f"BERT model: {bert_path}")
    logger.info(f"Annotations: {ann_dir}")
    logger.info("=" * 60)

    # Tokenizer
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_path)

    # Modello
    model = MANMultilingual(
        bert_model_path=bert_path,
        max_ngram=max_ngram,
        n_filters=n_filters,
        pacrr_filters=pacrr_filters,
        n_conv_layers=n_conv_layers,
        n_s=n_s,
        freeze_bert=freeze_bert,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parametri totali: {n_params:,}")
    logger.info(f"Parametri addestrabili: {n_trainable:,}")

    # Carica checkpoint se eval_only
    ckpt_path = os.path.join(checkpoint_dir, "man_best.pt")
    if eval_only:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint non trovato: {ckpt_path}. "
                "Esegui prima il training senza --eval-only."
            )
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        logger.info(f"Checkpoint caricato: {ckpt_path}")
    else:
        # --- Training ---
        train_path = ann_dir / "train.tsv"
        val_path = ann_dir / "val.tsv"

        if not train_path.exists():
            raise FileNotFoundError(f"Train set non trovato: {train_path}")
        if not val_path.exists():
            raise FileNotFoundError(f"Validation set non trovato: {val_path}")

        train_ds = PairwiseRankDataset(
            str(train_path), tokenizer, max_query_len, max_doc_len
        )
        if max_train_samples > 0 and max_train_samples < len(train_ds):
            indices = np.random.choice(len(train_ds), max_train_samples, replace=False)
            train_ds = torch.utils.data.Subset(train_ds, indices)
            logger.info(f"Sottocampione training: {max_train_samples} campioni")

        val_ds = EvalDataset(str(val_path), tokenizer, max_query_len, max_doc_len)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        logger.info(f"Training samples: {len(train_ds)}")
        logger.info(f"Validation samples: {len(val_ds)}")

        t0 = time.time()
        model = train_man(
            model,
            train_loader,
            val_loader,
            device,
            n_epochs=n_epochs,
            lr=lr,
            margin=margin,
            checkpoint_dir=checkpoint_dir,
        )
        train_time = time.time() - t0
        logger.info(f"Training completato in {train_time:.1f}s")

    # --- Evaluation su test set ---
    test_path = ann_dir / "test.tsv"
    if not test_path.exists():
        raise FileNotFoundError(f"Test set non trovato: {test_path}")

    test_ds = EvalDataset(str(test_path), tokenizer, max_query_len, max_doc_len)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    logger.info(f"Test samples: {len(test_ds)}")

    t0 = time.time()
    test_metrics = evaluate_man(model, test_loader, device, k_values=[1, 3, 5, 10])
    eval_time = time.time() - t0

    test_metrics["eval_time_s"] = eval_time
    test_metrics["dataset"] = dataset
    test_metrics["model"] = "MAN-mBERT"
    test_metrics["bert_model"] = bert_path
    test_metrics["freeze_bert"] = freeze_bert
    test_metrics["n_epochs"] = n_epochs

    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS — MAN Baseline")
    logger.info("=" * 60)
    for k, v in test_metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")

    # Salva risultati
    output_dir = PROJECT_ROOT / "outputs" / dataset / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"man_results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"Risultati salvati in: {json_path}")

    # Tabella riepilogativa
    print("\n" + "=" * 80)
    print(f"MAN BASELINE — {dataset.upper()}")
    print("=" * 80)
    print(
        f"{'Metric':<15} {'@1':<10} {'@3':<10} {'@5':<10} {'@10':<10}"
    )
    print("-" * 55)
    for metric_name in ["hit", "ndcg"]:
        vals = []
        for k in [1, 3, 5, 10]:
            key = f"{metric_name}@{k}"
            vals.append(test_metrics.get(key, 0.0))
        print(
            f"{metric_name.upper():<15} "
            + "".join(f"{v:.4f}    " for v in vals)
        )
    print("=" * 80)

    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MAN Baseline (EMNLP 2020) adattato per multilingue"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["mumin", "m3check_multilingual", "m3check_no_eng"],
        help="Dataset da usare",
    )
    parser.add_argument(
        "--bert-path",
        type=str,
        default="/root/bert_multilingual_local",
        help="Path al modello BERT locale",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Numero di epoche di training"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--max-query-len", type=int, default=64, help="Lunghezza max query"
    )
    parser.add_argument(
        "--max-doc-len", type=int, default=128, help="Lunghezza max documento"
    )
    parser.add_argument(
        "--max-ngram", type=int, default=2, help="Livelli n-gram (1-3)"
    )
    parser.add_argument(
        "--n-filters", type=int, default=128, help="Filtri CNN"
    )
    parser.add_argument(
        "--margin", type=float, default=1.0, help="Margine per hinge loss"
    )
    parser.add_argument(
        "--freeze-bert",
        action="store_true",
        default=True,
        help="Congela i pesi BERT (default: True)",
    )
    parser.add_argument(
        "--finetune-bert",
        action="store_true",
        default=False,
        help="Fine-tuna i pesi BERT (sovrascrive --freeze-bert)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Solo valutazione (richiede checkpoint esistente)",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="Max campioni di training (0=tutti). Utile per test rapidi.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    freeze = args.freeze_bert and not args.finetune_bert

    run_man_baseline(
        dataset=args.dataset,
        bert_path=args.bert_path,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_query_len=args.max_query_len,
        max_doc_len=args.max_doc_len,
        max_ngram=args.max_ngram,
        n_filters=args.n_filters,
        margin=args.margin,
        freeze_bert=freeze,
        eval_only=args.eval_only,
        max_train_samples=args.max_train_samples,
        seed=args.seed,
    )
