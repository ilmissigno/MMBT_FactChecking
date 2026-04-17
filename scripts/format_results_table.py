#!/usr/bin/env python3
"""
Format Results Tables for Paper (Table 4 style)
=================================================

Produce tabelle per MuMiN e M3-Check nel formato di Table 4:
Method | MM | HIT@3 | HIT@5 | NDCG@1 | NDCG@3 | NDCG@5

Con bold per il best e underline per il runner-up.

Usage:
    python -m scripts.format_results_table
"""

import json
import logging
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


METRIC_COLS = ["HIT@3", "HIT@5", "NDCG@1", "NDCG@3", "NDCG@5"]


def load_mumin_results() -> dict:
    """Carica risultati MuMiN da JSON esistenti."""
    results = {}

    # BM25 + sentence-BERT baseline
    baselines_dir = PROJECT_ROOT / "outputs" / "mumin" / "baselines"
    baseline_files = sorted(baselines_dir.glob("baseline_results_full_*.json"))
    if baseline_files:
        with open(baseline_files[-1]) as f:
            baselines = json.load(f)

        if "BM25" in baselines:
            b = baselines["BM25"]
            results["BM25"] = {
                "mm": False,
                "HIT@3": b.get("hit@3", 0),
                "HIT@5": b.get("hit@5", 0),
                "NDCG@1": b.get("ndcg@1", 0),
                "NDCG@3": b.get("ndcg@3", 0),
                "NDCG@5": b.get("ndcg@5", 0),
            }

        if "sentence-BERT" in baselines:
            b = baselines["sentence-BERT"]
            results["sentence-BERT"] = {
                "mm": False,
                "HIT@3": b.get("hit@3", 0),
                "HIT@5": b.get("hit@5", 0),
                "NDCG@1": b.get("ndcg@1", 0),
                "NDCG@3": b.get("ndcg@3", 0),
                "NDCG@5": b.get("ndcg@5", 0),
            }

    # Ours (MMBT multilingual)
    ours_path = (
        PROJECT_ROOT
        / "models"
        / "checkpoints"
        / "mmbt_mumin_multilingual"
        / "test_results_MuMiN.json"
    )
    if ours_path.exists():
        with open(ours_path) as f:
            ours = json.load(f)
        # Mapping: acc_K -> HIT@K
        results["Ours"] = {
            "mm": True,
            "HIT@3": ours.get("acc_3", 0),
            "HIT@5": ours.get("acc_5", 0),
            "NDCG@1": ours.get("ndcg_1", 0),
            "NDCG@3": ours.get("ndcg_3", 0),
            "NDCG@5": ours.get("ndcg_5", 0),
        }

    return results


def load_m3check_results() -> dict:
    """Carica risultati M3-Check da JSON esistenti."""
    results = {}

    # BM25 + sentence-BERT baseline
    baselines_dir = PROJECT_ROOT / "outputs" / "m3check_multilingual" / "baselines"
    baseline_files = sorted(baselines_dir.glob("baseline_results_full_*.json"))
    if baseline_files:
        with open(baseline_files[-1]) as f:
            baselines = json.load(f)

        if "BM25" in baselines:
            b = baselines["BM25"]
            results["BM25"] = {
                "mm": False,
                "HIT@3": b.get("hit@3", 0),
                "HIT@5": b.get("hit@5", 0),
                "NDCG@1": b.get("ndcg@1", 0),
                "NDCG@3": b.get("ndcg@3", 0),
                "NDCG@5": b.get("ndcg@5", 0),
            }

        if "sentence-BERT" in baselines:
            b = baselines["sentence-BERT"]
            results["sentence-BERT"] = {
                "mm": False,
                "HIT@3": b.get("hit@3", 0),
                "HIT@5": b.get("hit@5", 0),
                "NDCG@1": b.get("ndcg@1", 0),
                "NDCG@3": b.get("ndcg@3", 0),
                "NDCG@5": b.get("ndcg@5", 0),
            }

    # Ours (MMBT multilingual)
    ours_path = (
        PROJECT_ROOT
        / "models"
        / "checkpoints"
        / "mmbt_m3check_multilingual"
        / "test_results_M3Check.json"
    )
    if ours_path.exists():
        with open(ours_path) as f:
            ours = json.load(f)
        results["Ours"] = {
            "mm": True,
            "HIT@3": ours.get("acc_3", 0),
            "HIT@5": ours.get("acc_5", 0),
            "NDCG@1": ours.get("ndcg_1", 0),
            "NDCG@3": ours.get("ndcg_3", 0),
            "NDCG@5": ours.get("ndcg_5", 0),
        }

    return results


def _format_value(val: float) -> str:
    """Formatta un valore a 3 decimali senza lo zero iniziale."""
    return f".{int(round(val * 1000)):03d}"


def _find_best_runner(results: dict, metric: str):
    """Trova il miglior valore e il runner-up per una metrica."""
    vals = [(name, r[metric]) for name, r in results.items()]
    vals.sort(key=lambda x: x[1], reverse=True)
    best = vals[0][0] if vals else None
    runner = vals[1][0] if len(vals) > 1 else None
    return best, runner


def format_markdown(dataset_name: str, results: dict):
    """Formatta la tabella in Markdown."""
    print(f"\n### {dataset_name} — Re-ranking Performance\n")
    header = f"| Method | MM | {' | '.join(METRIC_COLS)} |"
    sep = "|---|---|" + "|".join(["---"] * len(METRIC_COLS)) + "|"
    print(header)
    print(sep)

    # Trova best/runner-up per ogni metrica
    best_runner = {m: _find_best_runner(results, m) for m in METRIC_COLS}

    for method, r in results.items():
        mm = "✓" if r["mm"] else ""
        vals = []
        for m in METRIC_COLS:
            v = _format_value(r[m])
            best, runner = best_runner[m]
            if method == best:
                v = f"**{v}**"
            elif method == runner:
                v = f"_{v}_"
            vals.append(v)
        print(f"| {method} | {mm} | {' | '.join(vals)} |")


def format_latex(dataset_name: str, results: dict):
    """Formatta la tabella in LaTeX (stile Table 4 del paper)."""
    print(f"\n% === {dataset_name} ===")
    print("\\begin{table}[h]")
    print("\\centering")
    print(f"\\caption{{Re-ranking performance: {dataset_name}}}")
    print("\\begin{tabular}{l c " + "c " * len(METRIC_COLS) + "}")
    print("\\toprule")
    print("Method & MM & " + " & ".join(METRIC_COLS) + " \\\\")
    print("\\midrule")

    best_runner = {m: _find_best_runner(results, m) for m in METRIC_COLS}

    for method, r in results.items():
        mm = "$\\checkmark$" if r["mm"] else ""
        vals = []
        for m in METRIC_COLS:
            v = _format_value(r[m])
            best, runner = best_runner[m]
            if method == best:
                v = f"\\textbf{{{v}}}"
            elif method == runner:
                v = f"\\underline{{{v}}}"
            vals.append(v)
        print(f"{method} & {mm} & {' & '.join(vals)} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def main():
    print("=" * 80)
    print("RESULTS TABLES FOR PAPER")
    print("=" * 80)

    # MuMiN
    mumin = load_mumin_results()
    if mumin:
        format_markdown("MuMiN", mumin)
        format_latex("MuMiN", mumin)
    else:
        print("\n⚠ Nessun risultato MuMiN trovato")

    # M3-Check
    m3check = load_m3check_results()
    if m3check:
        format_markdown("M3-Check", m3check)
        format_latex("M3-Check", m3check)
    else:
        print("\n⚠ Nessun risultato M3-Check trovato")


if __name__ == "__main__":
    main()
