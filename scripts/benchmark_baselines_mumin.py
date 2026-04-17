#!/usr/bin/env python3
"""
Benchmark Baseline Methods on MuMiN/Snopes/Politifact Datasets
===============================================================

Confronta le performance di retrieval di MMBT con baseline classiche:
- BM25 (Best Match 25 - retrieval lessicale)
- sentence-BERT (embedding semantici)

Metriche: HIT@3, HIT@5, NDCG@1, NDCG@3, NDCG@5

Usage:
    python scripts/benchmark_baselines_mumin.py --dataset mumin
    python scripts/benchmark_baselines_mumin.py --dataset snopes
    python scripts/benchmark_baselines_mumin.py --dataset politifact
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

import re
import string
import numpy as np
import pandas as pd

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Stopwords inglesi comuni (per evitare dipendenza NLTK installazione)
ENGLISH_STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were',
    'will', 'with', 'this', 'have', 'had', 'been', 'but', 'or', 'not', 'do',
    'also', 'which', 'who', 'what', 'when', 'where', 'how', 'all', 'each',
    'if', 'then', 'than', 'so', 'no', 'can', 'would', 'could', 'should',
    'their', 'them', 'they', 'these', 'there', 'we', 'our', 'you', 'your',
    'i', 'my', 'me', 'him', 'her', 'she', 'his', 'about', 'after', 'before',
    'up', 'down', 'out', 'into', 'over', 'under', 'again', 'just', 'now',
    'some', 'such', 'own', 'same', 'other', 'very', 'more', 'most', 'any',
    'only', 'said', 'says', 'did', 'does', 'doing', 'being'
}


class SimplePorterStemmer:
    """Stemmer Porter semplificato per ridurre dipendenze esterne."""
    
    def __init__(self):
        self.vowels = 'aeiou'
    
    def _is_consonant(self, word, i):
        if word[i] in self.vowels:
            return False
        if word[i] == 'y':
            if i == 0:
                return True
            return not self._is_consonant(word, i - 1)
        return True
    
    def _measure(self, stem):
        """Calcola il measure (numero di sequenze VC)."""
        cv_sequence = ''
        for i, char in enumerate(stem):
            if self._is_consonant(stem, i):
                cv_sequence += 'c'
            else:
                cv_sequence += 'v'
        return cv_sequence.count('vc')
    
    def stem(self, word):
        """Applica regole di stemming semplificato."""
        word = word.lower()
        
        # Step 1: Plurali e participi passati
        if word.endswith('sses'):
            word = word[:-2]
        elif word.endswith('ies'):
            word = word[:-2]
        elif word.endswith('ss'):
            pass
        elif word.endswith('s'):
            word = word[:-1]
        
        # Step 2: Finali -ed, -ing
        if word.endswith('eed'):
            if self._measure(word[:-3]) > 0:
                word = word[:-1]
        elif word.endswith('ed'):
            if any(c in self.vowels for c in word[:-2]):
                word = word[:-2]
                if word.endswith('at') or word.endswith('bl') or word.endswith('iz'):
                    word += 'e'
        elif word.endswith('ing'):
            if any(c in self.vowels for c in word[:-3]):
                word = word[:-3]
                if word.endswith('at') or word.endswith('bl') or word.endswith('iz'):
                    word += 'e'
        
        # Step 3: -ional, -tion, -ness, -ment, -ful, -ly
        if word.endswith('ational'):
            word = word[:-7] + 'ate'
        elif word.endswith('tional'):
            word = word[:-2]
        elif word.endswith('ness') or word.endswith('ment'):
            word = word[:-4]
        elif word.endswith('ful') or word.endswith('ous'):
            word = word[:-3]
        elif word.endswith('ive') or word.endswith('ize'):
            word = word[:-3]
        elif word.endswith('ly'):
            word = word[:-2]
        
        return word


def preprocess_text_for_bm25(text: str, stemmer=None) -> List[str]:
    """
    Preprocessa il testo per BM25: tokenizzazione, lowercase, pulizia, 
    rimozione stopwords, stemming.
    """
    # Lowercase
    text = text.lower()
    
    # Rimuovi punteggiatura e caratteri speciali
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Tokenizza
    tokens = text.split()
    
    # Rimuovi stopwords e token corti
    tokens = [t for t in tokens if t not in ENGLISH_STOPWORDS and len(t) > 2]
    
    # Stemming
    if stemmer:
        tokens = [stemmer.stem(t) for t in tokens]
    
    return tokens

# Setup logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logging()


def ndcg_at_k(relevance_scores: List[int], k: int) -> float:
    """
    Calcola NDCG@k (Normalized Discounted Cumulative Gain).
    
    Args:
        relevance_scores: Lista di score di rilevanza (1=rilevante, 0=non rilevante)
        k: Numero di risultati da considerare
    """
    relevance_scores = relevance_scores[:k]
    if not relevance_scores:
        return 0.0
    
    # DCG
    dcg = relevance_scores[0]
    for i, rel in enumerate(relevance_scores[1:], start=2):
        dcg += rel / np.log2(i + 1)
    
    # Ideal DCG (tutti documenti rilevanti prima)
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = ideal_scores[0] if ideal_scores else 0
    for i, rel in enumerate(ideal_scores[1:], start=2):
        idcg += rel / np.log2(i + 1)
    
    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k(relevance_scores: List[int], k: int) -> float:
    """
    HIT@k = 1 se c'è almeno un documento rilevante nei primi k risultati.
    """
    return 1.0 if any(relevance_scores[:k]) else 0.0


class BM25Baseline:
    """BM25 retrieval baseline usando rank_bm25 con preprocessing avanzato"""
    
    def __init__(self, use_preprocessing: bool = True):
        from rank_bm25 import BM25Okapi
        self.bm25 = None
        self.corpus = None
        self.doc_ids = None
        self.doc_id_to_idx = None
        self.use_preprocessing = use_preprocessing
        self.stemmer = SimplePorterStemmer() if use_preprocessing else None
        
    def _tokenize(self, text: str) -> List[str]:
        """Tokenizza applicando preprocessing se abilitato."""
        if self.use_preprocessing:
            return preprocess_text_for_bm25(text, self.stemmer)
        else:
            return text.lower().split()
        
    def index(self, documents: List[str], doc_ids: List[str]):
        """Indicizza i documenti"""
        from rank_bm25 import BM25Okapi
        
        logger.info(f"BM25: Indicizzazione di {len(documents)} documenti...")
        logger.info(f"BM25: Preprocessing {'attivo' if self.use_preprocessing else 'disattivo'}")
        self.corpus = documents
        self.doc_ids = doc_ids
        self.doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}
        
        # Tokenizza con preprocessing
        tokenized_corpus = [self._tokenize(doc) for doc in documents]
        
        t0 = time.time()
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.index_time = time.time() - t0
        logger.info(f"BM25: Indicizzazione completata in {self.index_time:.3f}s")
    
    def score_candidates(self, query: str, candidate_doc_ids: List[str]) -> List[float]:
        """
        Calcola gli score BM25 per un set specifico di candidati.
        """
        tokenized_query = self._tokenize(query)
        all_scores = self.bm25.get_scores(tokenized_query)
        
        # Estrai solo gli score per i candidati
        scores = []
        for doc_id in candidate_doc_ids:
            idx = self.doc_id_to_idx.get(doc_id)
            if idx is not None:
                scores.append(all_scores[idx])
            else:
                scores.append(0.0)
        return scores
        
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Cerca i documenti più rilevanti per una query"""
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Top-k
        top_indices = np.argsort(scores)[::-1][:k]
        results = [(self.doc_ids[i], scores[i]) for i in top_indices]
        return results


class SentenceBERTBaseline:
    """sentence-BERT retrieval baseline"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer
        
        logger.info(f"sentence-BERT: Caricamento modello {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.doc_embeddings = None
        self.doc_ids = None
        self.doc_id_to_idx = None
        
    def index(self, documents: List[str], doc_ids: List[str]):
        """Codifica e indicizza i documenti"""
        logger.info(f"sentence-BERT: Encoding di {len(documents)} documenti...")
        self.doc_ids = doc_ids
        self.doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}
        
        t0 = time.time()
        self.doc_embeddings = self.model.encode(
            documents, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        self.index_time = time.time() - t0
        logger.info(f"sentence-BERT: Encoding completato in {self.index_time:.3f}s")
    
    def score_candidates(self, query: str, candidate_doc_ids: List[str]) -> List[float]:
        """
        Calcola similarità coseno per un set specifico di candidati.
        """
        query_emb = self.model.encode([query], convert_to_numpy=True)[0]
        
        scores = []
        for doc_id in candidate_doc_ids:
            idx = self.doc_id_to_idx.get(doc_id)
            if idx is not None:
                doc_emb = self.doc_embeddings[idx]
                sim = np.dot(query_emb, doc_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(doc_emb) + 1e-8
                )
                scores.append(float(sim))
            else:
                scores.append(0.0)
        return scores
        
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Cerca per similarità coseno"""
        query_emb = self.model.encode([query], convert_to_numpy=True)[0]
        
        # Similarità coseno
        similarities = np.dot(self.doc_embeddings, query_emb) / (
            np.linalg.norm(self.doc_embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        
        top_indices = np.argsort(similarities)[::-1][:k]
        results = [(self.doc_ids[i], similarities[i]) for i in top_indices]
        return results


def load_mumin_data() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, List[Tuple[str, int]]]]:
    """
    Carica i dati MuMiN per il benchmark in formato re-ranking.
    
    Il task è: per ogni query, rankare i suoi candidati specifici (non tutto il corpus).
    
    Returns:
        queries: Dizionario QueryID -> QueryText
        doc_corpus: Dizionario DocID -> DocText
        query_candidates: Per ogni QueryID, lista di (DocID, Label) candidati
    """
    ann_dir = PROJECT_ROOT / "datasets" / "mumin_mmbt" / "annotations"
    
    # Carica test set
    test_path = ann_dir / "test.tsv"
    if not test_path.exists():
        raise FileNotFoundError(f"Test set non trovato: {test_path}")
    
    test_df = pd.read_csv(test_path, sep="\t")
    logger.info(f"Caricati {len(test_df)} campioni di test MuMiN")
    
    # Costruisci corpus documenti unici
    doc_corpus = {}
    for _, row in test_df.iterrows():
        doc_id = str(row['DocID'])
        doc_text = str(row['DocText']) if pd.notna(row['DocText']) else ""
        if doc_id not in doc_corpus and doc_text:
            doc_corpus[doc_id] = doc_text
    
    # Costruisci query uniche
    queries = {}
    for _, row in test_df.iterrows():
        q_id = str(row['QueryID'])
        q_text = str(row['QueryText']) if pd.notna(row['QueryText']) else ""
        if q_id not in queries and q_text:
            queries[q_id] = q_text
    
    logger.info(f"Query uniche: {len(queries)}")
    logger.info(f"Documenti unici: {len(doc_corpus)}")
    
    # Costruisci candidati per ogni query (solo i documenti associati a quella query)
    query_candidates = defaultdict(list)
    for _, row in test_df.iterrows():
        query_id = str(row['QueryID'])
        doc_id = str(row['DocID'])
        label = int(row['Label']) if pd.notna(row['Label']) else 0
        query_candidates[query_id].append((doc_id, label))
    
    # Statistiche
    n_cands = [len(v) for v in query_candidates.values()]
    n_with_pos = sum(1 for v in query_candidates.values() if any(l == 1 for _, l in v))
    logger.info(f"Candidati per query: media {np.mean(n_cands):.1f}, min {min(n_cands)}, max {max(n_cands)}")
    logger.info(f"Query con almeno 1 documento rilevante: {n_with_pos}")
    
    return queries, doc_corpus, dict(query_candidates)


def load_dataset(dataset: str, use_full_corpus: bool = True) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, List[Tuple[str, int]]]]:
    """
    Carica i dati per il benchmark in formato re-ranking.
    
    Args:
        dataset: 'mumin', 'snopes', o 'politifact'
        use_full_corpus: Se True, carica il corpus completo per retrieval full
    
    Returns:
        queries: Dizionario QueryID -> QueryText
        doc_corpus: Dizionario DocID -> DocText
        query_candidates: Per ogni QueryID, lista di (DocID, Label) candidati
    """
    if dataset == 'mumin':
        ann_dir = PROJECT_ROOT / "datasets" / "mumin_mmbt" / "annotations"
        docs_file = None  # MuMiN non ha un corpus separato
    elif dataset == 'snopes':
        ann_dir = PROJECT_ROOT / "datasets" / "fakenewsdet" / "annotations" / "snopes"
        docs_file = ann_dir / "Snopes_DocumentsDataset.tsv"
    elif dataset == 'politifact':
        ann_dir = PROJECT_ROOT / "datasets" / "fakenewsdet" / "annotations" / "politifact"
        docs_file = ann_dir / "Politifact_DocumentsDataset.tsv"
    else:
        raise ValueError(f"Dataset sconosciuto: {dataset}")
    
    # Carica test set
    test_path = ann_dir / "test.tsv"
    if not test_path.exists():
        raise FileNotFoundError(f"Test set non trovato: {test_path}")
    
    test_df = pd.read_csv(test_path, sep="\t")
    logger.info(f"Caricati {len(test_df)} campioni di test {dataset.upper()}")
    
    # Costruisci corpus documenti
    doc_corpus = {}
    
    if use_full_corpus and docs_file and docs_file.exists():
        # Carica corpus completo per Snopes/Politifact
        docs_df = pd.read_csv(docs_file, sep="\t")
        for _, row in docs_df.iterrows():
            doc_id = str(row['DocID'])
            doc_text = str(row['DocText']) if pd.notna(row['DocText']) else ""
            if doc_text:
                doc_corpus[doc_id] = doc_text
        logger.info(f"Corpus completo caricato: {len(doc_corpus)} documenti")
    else:
        # Usa solo documenti dal test set
        for _, row in test_df.iterrows():
            doc_id = str(row['DocID'])
            doc_text = str(row['DocText']) if pd.notna(row['DocText']) else ""
            if doc_id not in doc_corpus and doc_text:
                doc_corpus[doc_id] = doc_text
    
    # Costruisci query uniche
    queries = {}
    for _, row in test_df.iterrows():
        q_id = str(row['QueryID'])
        q_text = str(row['QueryText']) if pd.notna(row['QueryText']) else ""
        if q_id not in queries and q_text:
            queries[q_id] = q_text
    
    logger.info(f"Query uniche: {len(queries)}")
    logger.info(f"Documenti nel corpus: {len(doc_corpus)}")
    
    # Costruisci candidati per ogni query (ground truth)
    query_candidates = defaultdict(list)
    for _, row in test_df.iterrows():
        query_id = str(row['QueryID'])
        doc_id = str(row['DocID'])
        label = int(row['Label']) if pd.notna(row['Label']) else 0
        query_candidates[query_id].append((doc_id, label))
    
    # Statistiche
    n_cands = [len(v) for v in query_candidates.values()]
    n_with_pos = sum(1 for v in query_candidates.values() if any(l == 1 for _, l in v))
    logger.info(f"Candidati annotati per query: media {np.mean(n_cands):.1f}, min {min(n_cands)}, max {max(n_cands)}")
    logger.info(f"Query con almeno 1 documento rilevante: {n_with_pos}")
    
    return queries, doc_corpus, dict(query_candidates)


def evaluate_reranker(
    retriever,
    queries: Dict[str, str],
    query_candidates: Dict[str, List[Tuple[str, int]]],
    k_values: List[int] = [1, 3, 5]
) -> Dict[str, float]:
    """
    Valuta un retriever in modalità re-ranking.
    
    Per ogni query, rankare i suoi candidati specifici e calcolare le metriche.
    
    Returns:
        Dizionario con metriche aggregate (HIT@k, NDCG@k)
    """
    results = {f'hit@{k}': [] for k in k_values}
    results.update({f'ndcg@{k}': [] for k in k_values})
    
    search_times = []
    n_evaluated = 0
    
    for query_id, query_text in queries.items():
        if query_id not in query_candidates:
            continue
        
        candidates = query_candidates[query_id]
        if len(candidates) < 2:
            # Skip query con un solo candidato (niente da rankare)
            continue
        
        # Estrai doc_ids e labels
        doc_ids = [doc_id for doc_id, _ in candidates]
        labels = [label for _, label in candidates]
        
        # Skip se non ci sono documenti positivi
        if not any(labels):
            continue
            
        t0 = time.time()
        scores = retriever.score_candidates(query_text, doc_ids)
        search_times.append(time.time() - t0)
        
        # Ordina candidati per score decrescente
        ranked = sorted(zip(doc_ids, labels, scores), key=lambda x: x[2], reverse=True)
        rel_scores = [label for _, label, _ in ranked]
        
        # Calcola metriche
        for k in k_values:
            results[f'hit@{k}'].append(hit_at_k(rel_scores, k))
            results[f'ndcg@{k}'].append(ndcg_at_k(rel_scores, k))
        
        n_evaluated += 1
    
    # Media delle metriche
    metrics = {}
    for key, values in results.items():
        metrics[key] = np.mean(values) if values else 0.0
    
    metrics['avg_search_time_ms'] = np.mean(search_times) * 1000 if search_times else 0.0
    metrics['n_queries_evaluated'] = n_evaluated
    
    return metrics


def evaluate_full_corpus(
    retriever,
    queries: Dict[str, str],
    query_candidates: Dict[str, List[Tuple[str, int]]],
    k_values: List[int] = [1, 3, 5]
) -> Dict[str, float]:
    """
    Valuta un retriever in modalità full-corpus retrieval.
    
    Per ogni query, cerca nell'intero corpus e calcola le metriche.
    Più simile al setup originale della tabella.
    
    Returns:
        Dizionario con metriche aggregate (HIT@k, NDCG@k)
    """
    results = {f'hit@{k}': [] for k in k_values}
    results.update({f'ndcg@{k}': [] for k in k_values})
    
    max_k = max(k_values)
    search_times = []
    n_evaluated = 0
    
    for query_id, query_text in queries.items():
        if query_id not in query_candidates:
            continue
        
        candidates = query_candidates[query_id]
        
        # Trova i doc_ids positivi per questa query
        positive_doc_ids = set(doc_id for doc_id, label in candidates if label == 1)
        
        if not positive_doc_ids:
            continue
            
        t0 = time.time()
        # Cerca nell'intero corpus
        retrieved = retriever.search(query_text, k=max_k)
        search_times.append(time.time() - t0)
        
        # Costruisci lista di rilevanza per i risultati
        rel_scores = []
        for doc_id, _ in retrieved:
            rel = 1 if doc_id in positive_doc_ids else 0
            rel_scores.append(rel)
        
        # Calcola metriche
        for k in k_values:
            results[f'hit@{k}'].append(hit_at_k(rel_scores, k))
            results[f'ndcg@{k}'].append(ndcg_at_k(rel_scores, k))
        
        n_evaluated += 1
    
    # Media delle metriche
    metrics = {}
    for key, values in results.items():
        metrics[key] = np.mean(values) if values else 0.0
    
    metrics['avg_search_time_ms'] = np.mean(search_times) * 1000 if search_times else 0.0
    metrics['n_queries_evaluated'] = n_evaluated
    
    return metrics


def run_benchmark(dataset: str = 'mumin', mode: str = 'full'):
    """
    Esegue il benchmark completo.
    
    Args:
        dataset: 'mumin', 'snopes', o 'politifact'
        mode: 'full' = retrieval su tutto il corpus (come nella tabella)
              'rerank' = re-ranking sui candidati pre-selezionati
    """
    logger.info("=" * 60)
    logger.info(f"Benchmark Baseline Methods on {dataset.upper()} Dataset")
    logger.info(f"Mode: {mode} ({'full corpus retrieval' if mode == 'full' else 're-ranking candidati'})")
    logger.info("=" * 60)
    
    # Carica dati
    queries, doc_corpus, query_candidates = load_dataset(dataset)
    
    # Prepara documenti per l'indice
    doc_ids = list(doc_corpus.keys())
    doc_texts = [doc_corpus[d] for d in doc_ids]
    
    all_results = {}
    k_values = [1, 3, 5]
    
    # Scegli funzione di valutazione
    eval_func = evaluate_full_corpus if mode == 'full' else evaluate_reranker
    
    # === BM25 Baseline ===
    logger.info("\n" + "=" * 40)
    logger.info("BM25 Baseline")
    logger.info("=" * 40)
    
    bm25 = BM25Baseline()
    bm25.index(doc_texts, doc_ids)
    
    bm25_metrics = eval_func(bm25, queries, query_candidates, k_values)
    bm25_metrics['index_time_s'] = bm25.index_time
    all_results['BM25'] = bm25_metrics
    
    logger.info(f"BM25 Results:")
    for k, v in bm25_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    
    # === sentence-BERT Baseline ===
    logger.info("\n" + "=" * 40)
    logger.info("sentence-BERT Baseline")
    logger.info("=" * 40)
    
    try:
        sbert = SentenceBERTBaseline()
        sbert.index(doc_texts, doc_ids)
        
        sbert_metrics = eval_func(sbert, queries, query_candidates, k_values)
        sbert_metrics['index_time_s'] = sbert.index_time
        all_results['sentence-BERT'] = sbert_metrics
        
        logger.info(f"sentence-BERT Results:")
        for k, v in sbert_metrics.items():
            logger.info(f"  {k}: {v:.4f}")
    except Exception as e:
        logger.warning(f"sentence-BERT non disponibile: {e}")
    
    # === Salva risultati ===
    output_dir = PROJECT_ROOT / "outputs" / dataset / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON
    json_path = output_dir / f"baseline_results_{mode}_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nRisultati salvati in: {json_path}")
    
    # Tabella riepilogativa
    mode_desc = "Full Corpus" if mode == 'full' else "Re-ranking"
    print("\n" + "=" * 80)
    print(f"RIEPILOGO RISULTATI - {dataset.upper()} Dataset ({mode_desc})")
    print("=" * 80)
    print(f"{'Method':<20} {'HIT@1':<10} {'HIT@3':<10} {'HIT@5':<10} {'NDCG@1':<10} {'NDCG@3':<10} {'NDCG@5':<10}")
    print("-" * 80)
    
    for method, metrics in all_results.items():
        print(f"{method:<20} "
              f"{metrics.get('hit@1', 0):.3f}     "
              f"{metrics.get('hit@3', 0):.3f}     "
              f"{metrics.get('hit@5', 0):.3f}     "
              f"{metrics.get('ndcg@1', 0):.3f}     "
              f"{metrics.get('ndcg@3', 0):.3f}     "
              f"{metrics.get('ndcg@5', 0):.3f}")
    
    print("=" * 80)
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark baseline methods')
    parser.add_argument('--dataset', type=str, default='mumin',
                        choices=['mumin', 'snopes', 'politifact'],
                        help='Dataset da valutare')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'rerank'],
                        help='full=corpus completo, rerank=candidati pre-selezionati')
    args = parser.parse_args()
    
    results = run_benchmark(args.dataset, args.mode)
