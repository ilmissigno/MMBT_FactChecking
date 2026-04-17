#!/usr/bin/env python3
"""
Compute Dataset Statistics for MuMiN and M3-Check
===================================================

Calcola statistiche dei dataset per il paper:
- Numero di claim (query) unici
- Numero di documenti unici
- Lingue claim e documenti
- Lunghezza media (parole) claim e documenti
- Vocabolario claim e documenti (parole uniche)

Usage:
    python -m scripts.compute_dataset_statistics --dataset mumin
    python -m scripts.compute_dataset_statistics --dataset m3check
    python -m scripts.compute_dataset_statistics --dataset all
"""

import argparse
import logging
import pickle
import lzma
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def word_count(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(text.split())


def unique_words(texts) -> set:
    vocab = set()
    for t in texts:
        if isinstance(t, str) and t.strip():
            vocab.update(t.lower().split())
    return vocab


def compute_mumin_statistics():
    """Calcola statistiche per MuMiN (solo claim non inglesi)."""
    logger.info("=" * 60)
    logger.info("Computing MuMiN Statistics")
    logger.info("=" * 60)

    ann_dir = PROJECT_ROOT / "datasets" / "mumin_mmbt" / "annotations"

    # Carica tutti gli split
    dfs = []
    for split in ["train", "val", "test"]:
        path = ann_dir / f"{split}.tsv"
        if path.exists():
            df = pd.read_csv(path, sep="\t")
            df["split"] = split
            dfs.append(df)
            logger.info(f"  {split}: {len(df)} righe")
    all_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Totale righe: {len(all_df)}")

    # Claim unici (QueryID + QueryText)
    claims = all_df.drop_duplicates(subset="QueryID")[["QueryID", "QueryText"]].copy()
    claims["QueryText"] = claims["QueryText"].fillna("").astype(str)
    n_claims = len(claims)

    # Documenti unici (DocID + DocText)
    docs = all_df.drop_duplicates(subset="DocID")[["DocID", "DocText"]].copy()
    docs["DocText"] = docs["DocText"].fillna("").astype(str)
    n_docs = len(docs)

    # Lunghezza media
    claim_lengths = claims["QueryText"].apply(word_count)
    doc_lengths = docs["DocText"].apply(word_count)
    avg_claim_len = claim_lengths.mean()
    avg_doc_len = doc_lengths.mean()

    # Vocabolario
    claim_vocab = unique_words(claims["QueryText"])
    doc_vocab = unique_words(docs["DocText"])

    # Lingue — dal log di preparazione (datasets/mumin_mmbt/logs/)
    # Il pickle raw non è compatibile con la versione corrente di pandas.
    # Dati estratti dal log: mumin_preparation_v2_20260306_154202.log
    # Filtro lingua (escluse: ['en']): 12924 → 7379 claims (57.1% mantenuti)
    claim_langs = ["pt", "es", "hi", "ar", "fr", "de", "id", "it", "bn", "tr"]
    doc_langs = claim_langs.copy()  # In MuMiN, docs sono anch'essi claim (cluster-based)

    stats = {
        "dataset": "MuMiN",
        "n_claims": n_claims,
        "n_docs": n_docs,
        "claim_langs": claim_langs,
        "doc_langs": doc_langs,
        "avg_claim_words": avg_claim_len,
        "avg_doc_words": avg_doc_len,
        "claim_vocab_size": len(claim_vocab),
        "doc_vocab_size": len(doc_vocab),
    }

    _print_stats(stats)
    return stats


def _get_mumin_languages():
    """Estrai le lingue dal file raw MuMiN claim pickle."""
    claim_path = PROJECT_ROOT / "datasets" / "mumin" / "mumin" / "claim"
    if not claim_path.exists():
        logger.warning(f"File claim non trovato: {claim_path}")
        return []

    try:
        # Prova prima lzma (formato xz)
        with lzma.open(claim_path, "rb") as f:
            claims_df = pickle.load(f)
    except lzma.LZMAError:
        try:
            with open(claim_path, "rb") as f:
                claims_df = pickle.load(f)
        except Exception as e:
            logger.warning(f"Impossibile caricare claim pickle: {e}")
            return []

    if isinstance(claims_df, pd.DataFrame) and "language" in claims_df.columns:
        lang_counts = claims_df["language"].value_counts()
        # Escludi inglese (coerente con il setup multilingual)
        lang_counts = lang_counts[lang_counts.index != "en"]
        langs = list(lang_counts.index)
        logger.info(f"Lingue MuMiN (non-en): {lang_counts.to_dict()}")
        return langs
    elif isinstance(claims_df, dict):
        # Potrebbe essere un dict con le colonne
        logger.info(f"Claims tipo dict con chiavi: {list(claims_df.keys())[:10]}")
        return []
    else:
        logger.info(f"Claims tipo: {type(claims_df)}")
        return []


def compute_m3check_statistics():
    """Calcola statistiche per M3-Check multilingual."""
    logger.info("=" * 60)
    logger.info("Computing M3-Check Multilingual Statistics")
    logger.info("=" * 60)

    ann_dir = PROJECT_ROOT / "datasets" / "mmbt_m3check_multilingual" / "annotations"

    # Carica tutti gli split
    dfs = []
    for split in ["train", "val", "test"]:
        path = ann_dir / f"{split}.tsv"
        if path.exists():
            logger.info(f"  Caricamento {split}...")
            df = pd.read_csv(path, sep="\t")
            df["split"] = split
            dfs.append(df)
            logger.info(f"  {split}: {len(df)} righe")
    all_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Totale righe: {len(all_df)}")

    # Claim unici
    claims = all_df.drop_duplicates(subset="QueryID")[["QueryID", "QueryText"]].copy()
    claims["QueryText"] = claims["QueryText"].fillna("").astype(str)
    n_claims = len(claims)

    # Documenti unici
    docs = all_df.drop_duplicates(subset="DocID")[["DocID", "DocText"]].copy()
    docs["DocText"] = docs["DocText"].fillna("").astype(str)
    n_docs = len(docs)

    # Lunghezza media
    claim_lengths = claims["QueryText"].apply(word_count)
    doc_lengths = docs["DocText"].apply(word_count)
    avg_claim_len = claim_lengths.mean()
    avg_doc_len = doc_lengths.mean()

    # Vocabolario
    claim_vocab = unique_words(claims["QueryText"])
    doc_vocab = unique_words(docs["DocText"])

    # Lingue — dal raw CSV
    claim_langs, doc_langs = _get_m3check_languages()

    stats = {
        "dataset": "M3-Check",
        "n_claims": n_claims,
        "n_docs": n_docs,
        "claim_langs": claim_langs,
        "doc_langs": doc_langs,
        "avg_claim_words": avg_claim_len,
        "avg_doc_words": avg_doc_len,
        "claim_vocab_size": len(claim_vocab),
        "doc_vocab_size": len(doc_vocab),
    }

    _print_stats(stats)
    return stats


def _get_m3check_languages():
    """Estrai le lingue dai CSV raw M3-Check."""
    raw_dir = PROJECT_ROOT / "datasets" / "M3-check"

    claim_langs = []
    doc_langs = []

    # Lingue claim (post) — unisci train+val+test
    all_post_langs = Counter()
    for fname in ["m3_check_training.csv", "m3_check_validation.csv", "m3_check_testing.csv"]:
        path = raw_dir / fname
        if path.exists():
            df = pd.read_csv(path, usecols=["post_lang"])
            counts = df["post_lang"].dropna().value_counts()
            for lang, cnt in counts.items():
                all_post_langs[lang] += cnt

    claim_langs = [lang for lang, _ in all_post_langs.most_common()]
    logger.info(f"Lingue claim (post): {dict(all_post_langs.most_common(15))}")

    # Lingue documenti (fact-checks)
    fc_path = raw_dir / "m3_check_fact_checks.csv"
    if fc_path.exists():
        fc_df = pd.read_csv(fc_path, usecols=["fact_check_title_lang"])
        doc_lang_counts = fc_df["fact_check_title_lang"].dropna().value_counts()
        doc_langs = list(doc_lang_counts.index)
        logger.info(f"Lingue documenti (fact-check): {doc_lang_counts.to_dict()}")

    return claim_langs, doc_langs


def _print_stats(stats: dict):
    """Stampa le statistiche in formato tabella."""
    print("\n" + "=" * 70)
    print(f"  Dataset Statistics: {stats['dataset']}")
    print("=" * 70)

    # Mappa codici lingua → nomi
    LANG_NAMES = {
        "eng": "English", "en": "English", "spa": "Spanish", "es": "Spanish",
        "por": "Portuguese", "pt": "Portuguese", "fra": "French", "fr": "French",
        "msa": "Malay", "ms": "Malay", "hin": "Hindi", "hi": "Hindi",
        "deu": "German", "de": "German", "tha": "Thai", "th": "Thai",
        "ara": "Arabic", "ar": "Arabic", "zho": "Chinese", "zh": "Chinese",
        "sin": "Sinhala", "si": "Sinhala", "ind": "Indonesian", "id": "Indonesian",
        "ita": "Italian", "it": "Italian", "ben": "Bengali", "bn": "Bengali",
        "tur": "Turkish", "tr": "Turkish", "jpn": "Japanese", "ja": "Japanese",
        "kor": "Korean", "ko": "Korean", "rus": "Russian", "ru": "Russian",
        "tam": "Tamil", "ta": "Tamil", "mya": "Burmese", "my": "Burmese",
        "nld": "Dutch", "nl": "Dutch", "pol": "Polish", "pl": "Polish",
        "urd": "Urdu", "ur": "Urdu", "cat": "Catalan", "mar": "Marathi",
        "tgl": "Tagalog", "fil": "Filipino",
    }

    def lang_list(codes):
        if not codes:
            return "N/A"
        names = []
        for c in codes:
            name = LANG_NAMES.get(c, c)
            if name not in names:
                names.append(name)
        return ", ".join(names)

    rows = [
        ("Numero di claim utilizzati", f"{stats['n_claims']:,}"),
        ("Numero di documenti utilizzati", f"{stats['n_docs']:,}"),
        ("Lingue claim", lang_list(stats["claim_langs"])),
        ("Lingue documenti", lang_list(stats["doc_langs"])),
        ("Lunghezza media claim (parole)", f"{stats['avg_claim_words']:.1f}"),
        ("Lunghezza media documenti (parole)", f"{stats['avg_doc_words']:.1f}"),
        ("Vocabolario claim (parole uniche)", f"{stats['claim_vocab_size']:,}"),
        ("Vocabolario documenti (parole uniche)", f"{stats['doc_vocab_size']:,}"),
    ]

    # Markdown table
    print("\n### Markdown Table\n")
    print("| Statistica | Valore |")
    print("|---|---|")
    for label, value in rows:
        print(f"| {label} | {value} |")

    # LaTeX table
    print("\n### LaTeX Table\n")
    print("\\begin{table}[h]")
    print("\\centering")
    print(f"\\caption{{Dataset statistics: {stats['dataset']}}}")
    print("\\begin{tabular}{l r}")
    print("\\toprule")
    print("Statistic & Value \\\\")
    print("\\midrule")
    for label, value in rows:
        # Escape special chars in LaTeX
        label_tex = label.replace("(", "\\textrm{(}").replace(")", "\\textrm{)}")
        value_tex = value.replace(",", "{,}")
        print(f"{label_tex} & {value_tex} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute dataset statistics")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["mumin", "m3check", "all"],
        help="Dataset da analizzare",
    )
    args = parser.parse_args()

    if args.dataset in ("mumin", "all"):
        compute_mumin_statistics()
    if args.dataset in ("m3check", "all"):
        compute_m3check_statistics()
