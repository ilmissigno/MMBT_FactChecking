#!/usr/bin/env python3
"""
Benchmark Baseline Methods on M3-Check Multilingual Dataset
=============================================================

Confronta le performance di retrieval di MMBT con baseline classiche:
- BM25 (Best Match 25 - retrieval lessicale)
- sentence-BERT (embedding semantici)

Metriche: HIT@3, HIT@5, NDCG@1, NDCG@3, NDCG@5

Usage:
    python -m scripts.benchmark_baselines_m3check
    python -m scripts.benchmark_baselines_m3check --mode rerank
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Riusa le stesse classi BM25/sentence-BERT dal benchmark MuMiN
from scripts.benchmark_baselines_mumin import (
    BM25Baseline,
    SentenceBERTBaseline,
    evaluate_full_corpus,
    evaluate_reranker,
    hit_at_k,
    ndcg_at_k,
)

PROJECT_ROOT = Path(__file__).parent.parent


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


logger = setup_logging()


def load_m3check_data(
    use_full_corpus: bool = True,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, List[Tuple[str, int]]]]:
    """
    Carica i dati M3-Check multilingual per il benchmark.

    Il file test.tsv è grande (>50MB), lo leggiamo in modo efficiente.

    Returns:
        queries: QueryID -> QueryText
        doc_corpus: DocID -> DocText
        query_candidates: QueryID -> [(DocID, Label), ...]
    """
    ann_dir = PROJECT_ROOT / "datasets" / "mmbt_m3check_multilingual" / "annotations"
    test_path = ann_dir / "test.tsv"

    if not test_path.exists():
        raise FileNotFoundError(f"Test set non trovato: {test_path}")

    logger.info(f"Caricamento test set M3-Check: {test_path}")
    test_df = pd.read_csv(test_path, sep="\t")
    logger.info(f"Caricati {len(test_df)} campioni di test M3-Check")

    # Corpus documenti unici
    doc_corpus = {}
    for _, row in test_df.drop_duplicates(subset="DocID").iterrows():
        doc_id = str(row["DocID"])
        doc_text = str(row["DocText"]) if pd.notna(row["DocText"]) else ""
        if doc_text:
            doc_corpus[doc_id] = doc_text

    # Query uniche
    queries = {}
    for _, row in test_df.drop_duplicates(subset="QueryID").iterrows():
        q_id = str(row["QueryID"])
        q_text = str(row["QueryText"]) if pd.notna(row["QueryText"]) else ""
        if q_text:
            queries[q_id] = q_text

    logger.info(f"Query uniche: {len(queries)}")
    logger.info(f"Documenti unici: {len(doc_corpus)}")

    # Candidati per ogni query
    query_candidates = defaultdict(list)
    for _, row in test_df.iterrows():
        query_id = str(row["QueryID"])
        doc_id = str(row["DocID"])
        label = int(float(row["Label"])) if pd.notna(row["Label"]) else 0
        query_candidates[query_id].append((doc_id, label))

    # Statistiche
    n_cands = [len(v) for v in query_candidates.values()]
    n_with_pos = sum(
        1 for v in query_candidates.values() if any(l == 1 for _, l in v)
    )
    logger.info(
        f"Candidati per query: media {np.mean(n_cands):.1f}, "
        f"min {min(n_cands)}, max {max(n_cands)}"
    )
    logger.info(f"Query con almeno 1 documento rilevante: {n_with_pos}")

    return queries, doc_corpus, dict(query_candidates)


def run_benchmark(mode: str = "full"):
    """
    Esegue il benchmark BM25 e sentence-BERT su M3-Check multilingual.
    """
    logger.info("=" * 60)
    logger.info("Benchmark Baseline Methods on M3-Check Multilingual")
    logger.info(f"Mode: {mode}")
    logger.info("=" * 60)

    queries, doc_corpus, query_candidates = load_m3check_data()

    doc_ids = list(doc_corpus.keys())
    doc_texts = [doc_corpus[d] for d in doc_ids]

    all_results = {}
    k_values = [1, 3, 5]
    eval_func = evaluate_full_corpus if mode == "full" else evaluate_reranker

    # === BM25 ===
    logger.info("\n" + "=" * 40)
    logger.info("BM25 Baseline")
    logger.info("=" * 40)

    bm25 = BM25Baseline()
    bm25.index(doc_texts, doc_ids)

    bm25_metrics = eval_func(bm25, queries, query_candidates, k_values)
    bm25_metrics["index_time_s"] = bm25.index_time
    all_results["BM25"] = bm25_metrics

    logger.info("BM25 Results:")
    for k, v in bm25_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # === sentence-BERT ===
    logger.info("\n" + "=" * 40)
    logger.info("sentence-BERT Baseline")
    logger.info("=" * 40)

    try:
        sbert = SentenceBERTBaseline()
        sbert.index(doc_texts, doc_ids)

        sbert_metrics = eval_func(sbert, queries, query_candidates, k_values)
        sbert_metrics["index_time_s"] = sbert.index_time
        all_results["sentence-BERT"] = sbert_metrics

        logger.info("sentence-BERT Results:")
        for k, v in sbert_metrics.items():
            logger.info(f"  {k}: {v:.4f}")
    except Exception as e:
        logger.warning(f"sentence-BERT non disponibile: {e}")

    # === Salva risultati ===
    output_dir = PROJECT_ROOT / "outputs" / "m3check_multilingual" / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"baseline_results_{mode}_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nRisultati salvati in: {json_path}")

    # Tabella riepilogativa
    mode_desc = "Full Corpus" if mode == "full" else "Re-ranking"
    print("\n" + "=" * 80)
    print(f"RIEPILOGO RISULTATI - M3-Check Multilingual ({mode_desc})")
    print("=" * 80)
    print(
        f"{'Method':<20} {'HIT@1':<10} {'HIT@3':<10} {'HIT@5':<10} "
        f"{'NDCG@1':<10} {'NDCG@3':<10} {'NDCG@5':<10}"
    )
    print("-" * 80)
    for method, metrics in all_results.items():
        print(
            f"{method:<20} "
            f"{metrics.get('hit@1', 0):.3f}     "
            f"{metrics.get('hit@3', 0):.3f}     "
            f"{metrics.get('hit@5', 0):.3f}     "
            f"{metrics.get('ndcg@1', 0):.3f}     "
            f"{metrics.get('ndcg@3', 0):.3f}     "
            f"{metrics.get('ndcg@5', 0):.3f}"
        )
    print("=" * 80)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark baseline methods on M3-Check multilingual"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "rerank"],
        help="full=corpus completo, rerank=candidati pre-selezionati",
    )
    args = parser.parse_args()
    run_benchmark(args.mode)
