#!/usr/bin/env python3
"""
MMBT - MuMiN Dataset Preparation Script v2
===========================================

Prepara il dataset MuMiN per MMBT generando coppie query-document
con negative sampling, simile al formato Snopes/Politifact.

Formato output: QueryID, QueryText, QueryImages, DocID, DocText, DocImages, Label
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging"""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"mumin_preparation_v2_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger("mumin_prep_v2")


class MuMiNPreparer:
    """Prepara MuMiN nel formato MMBT con negative sampling"""
    
    def __init__(
        self,
        raw_dir: Path,
        output_dir: Path,
        neg_ratio: int = 30,  # Numero di negative per ogni positive
        exclude_langs: List[str] = None,  # es. ['en'] per tenere solo non-EN
        logger: logging.Logger = None
    ):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.neg_ratio = neg_ratio
        self.exclude_langs = [l.lower() for l in exclude_langs] if exclude_langs else []
        self.logger = logger or logging.getLogger(__name__)
        
        self.claims_df = None
        
    def load_data(self) -> bool:
        """Carica i claims dal dataset MuMiN raw"""
        claims_path = self.raw_dir / "claim"
        
        if not claims_path.exists():
            self.logger.error(f"Claims file not found: {claims_path}")
            return False
        
        self.logger.info("Caricamento claims...")
        self.claims_df = pd.read_pickle(claims_path, compression='xz')
        self.logger.info(f"Caricati {len(self.claims_df)} claims totali")
        
        # Filtro per lingua
        if self.exclude_langs and 'language' in self.claims_df.columns:
            before = len(self.claims_df)
            mask = ~self.claims_df['language'].str.lower().isin(self.exclude_langs)
            self.claims_df = self.claims_df[mask].reset_index(drop=True)
            kept_langs = self.claims_df['language'].value_counts()
            self.logger.info(
                f"Filtro lingua (escluse: {self.exclude_langs}): "
                f"{before} → {len(self.claims_df)} claims "
                f"({len(self.claims_df)/before*100:.1f}% mantenuti)"
            )
            self.logger.info(f"Lingue presenti: {dict(kept_langs.head(10))}")
        
        # Log distribuzione label
        if 'label' in self.claims_df.columns:
            label_dist = self.claims_df['label'].value_counts()
            self.logger.info(f"Distribuzione label: {dict(label_dist)}")
        
        return True
    
    def generate_pairs(self) -> pd.DataFrame:
        """
        Genera coppie query-document con cluster-based sampling.
        
        Strategia:
        - Positive: claim X keywords → keywords di altro claim nello STESSO cluster (label=1)
        - Negative: claim X keywords → keywords di claim da CLUSTER DIVERSO (label=0)
        
        Questo ha senso semantico: claims nello stesso cluster parlano della stessa storia/topic.
        """
        self.logger.info(f"Generazione coppie cluster-based con neg_ratio={self.neg_ratio}...")
        
        claims = self.claims_df.reset_index(drop=True)
        
        # Raggruppa per cluster
        cluster_groups = {}
        for idx, row in claims.iterrows():
            c = row['cluster']
            cluster_groups.setdefault(c, []).append(idx)
        
        all_clusters = list(cluster_groups.keys())
        self.logger.info(f"Cluster trovati: {len(all_clusters)} | Claims totali: {len(claims)}")
        
        def get_text(row):
            t = row.get('keywords', None)
            if pd.isna(t) or t is None or str(t).strip() in ('', 'nan'):
                return None
            return str(t).strip()
        
        pairs = []
        for idx, claim in tqdm(claims.iterrows(), total=len(claims), desc="Generating pairs"):
            query_id = str(claim['id'])
            query_text = get_text(claim)
            
            if not query_text:
                continue
            
            my_cluster = claim['cluster']
            same_cluster_idxs = [i for i in cluster_groups[my_cluster] if i != idx]
            
            # Serve almeno 1 positivo
            if not same_cluster_idxs:
                continue
            
            # Positive pairs: altri claims dello stesso cluster
            n_pos = max(1, min(3, len(same_cluster_idxs)))
            pos_idxs = random.sample(same_cluster_idxs, n_pos)
            
            for pos_idx in pos_idxs:
                doc_text = get_text(claims.iloc[pos_idx])
                if not doc_text:
                    continue
                pairs.append({
                    'QueryID': query_id,
                    'QueryText': query_text,
                    'QueryImages': '',
                    'DocID': str(claims.iloc[pos_idx]['id']),
                    'DocText': doc_text,
                    'DocImages': '',
                    'Label': 1.0
                })
            
            # Negative pairs: claims da cluster DIVERSI
            other_cluster_keys = [c for c in all_clusters if c != my_cluster]
            neg_per_cluster = max(1, self.neg_ratio // len(other_cluster_keys))
            
            neg_count = 0
            for other_cluster in random.sample(other_cluster_keys, len(other_cluster_keys)):
                if neg_count >= self.neg_ratio:
                    break
                other_idxs = cluster_groups[other_cluster]
                sample_size = min(neg_per_cluster, len(other_idxs))
                for neg_idx in random.sample(other_idxs, sample_size):
                    if neg_count >= self.neg_ratio:
                        break
                    doc_text = get_text(claims.iloc[neg_idx])
                    if not doc_text:
                        continue
                    pairs.append({
                        'QueryID': query_id,
                        'QueryText': query_text,
                        'QueryImages': '',
                        'DocID': str(claims.iloc[neg_idx]['id']),
                        'DocText': doc_text,
                        'DocImages': '',
                        'Label': 0.0
                    })
                    neg_count += 1
        
        df = pd.DataFrame(pairs)
        self.logger.info(f"Generate {len(df)} coppie totali")
        label_dist = df['Label'].value_counts()
        self.logger.info(f"Distribuzione: {dict(label_dist)}")
        return df
    
    def split_data(self, df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split in train/val/test"""
        self.logger.info(f"Split: {train_ratio:.0%} train, {val_ratio:.0%} val, {1-train_ratio-val_ratio:.0%} test")
        
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df[:train_end]
        val_df = df[train_end:val_end]
        test_df = df[val_end:]
        
        self.logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def _clean_text_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rimuovi righe con testi null/nan/vuoti"""
        original_len = len(df)
        
        # Rimuovi righe con QueryText o DocText null/nan
        df = df.dropna(subset=['QueryText', 'DocText'])
        
        # Rimuovi righe con testo 'nan' come stringa
        df = df[df['QueryText'].astype(str) != 'nan']
        df = df[df['DocText'].astype(str) != 'nan']
        
        # Rimuovi righe con testo vuoto
        df = df[df['QueryText'].astype(str).str.strip() != '']
        df = df[df['DocText'].astype(str).str.strip() != '']
        
        removed = original_len - len(df)
        if removed > 0:
            self.logger.info(f"Rimosse {removed} righe con testi invalidi")
        
        return df.reset_index(drop=True)
    
    def save_datasets(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Salva i dataset in formato TSV"""
        annotations_dir = self.output_dir / "annotations"
        annotations_dir.mkdir(parents=True, exist_ok=True)
        
        # Crea cartelle immagini vuote
        images_dir = self.output_dir / "images"
        (images_dir / "query").mkdir(parents=True, exist_ok=True)
        (images_dir / "doc").mkdir(parents=True, exist_ok=True)
        
        for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            # Pulisci testi prima di salvare
            df = self._clean_text_data(df)
            path = annotations_dir / f"{name}.tsv"
            df.to_csv(path, sep='\t', index=False)
            self.logger.info(f"Salvato {path} ({len(df)} righe)")
        
        # Log statistiche finali
        self._log_stats(train_df, val_df, test_df)
    
    def _log_stats(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Log statistiche finali"""
        self.logger.info("\n" + "="*60)
        self.logger.info("STATISTICHE FINALI")
        self.logger.info("="*60)
        
        for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            pos = (df['Label'] == 1.0).sum()
            neg = (df['Label'] == 0.0).sum()
            self.logger.info(f"{name}: {len(df)} totali | {pos} positivi ({pos/len(df)*100:.1f}%) | {neg} negativi ({neg/len(df)*100:.1f}%)")
        
        self.logger.info("="*60)
    
    def run(self):
        """Esegue la pipeline completa"""
        self.logger.info("="*60)
        self.logger.info("MuMiN Dataset Preparation v2")
        self.logger.info("="*60)
        
        if not self.load_data():
            return False
        
        pairs_df = self.generate_pairs()
        train_df, val_df, test_df = self.split_data(pairs_df)
        self.save_datasets(train_df, val_df, test_df)
        
        self.logger.info("\n✅ Preparazione completata!")
        return True


def main():
    parser = argparse.ArgumentParser(description="Prepare MuMiN dataset for MMBT")
    parser.add_argument("--raw-dir", type=Path, 
                       default=PROJECT_ROOT / "datasets" / "mumin" / "mumin",
                       help="Directory with raw MuMiN files")
    parser.add_argument("--output-dir", type=Path,
                       default=PROJECT_ROOT / "datasets" / "mumin_mmbt",
                       help="Output directory")
    parser.add_argument("--neg-ratio", type=int, default=30,
                       help="Negative samples per positive (default: 30)")
    parser.add_argument("--exclude-langs", type=str, nargs="+", default=[],
                       metavar="LANG",
                       help="Codici lingua ISO da escludere (es. --exclude-langs en)")
    
    args = parser.parse_args()
    
    logger = setup_logging(args.output_dir)
    
    preparer = MuMiNPreparer(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        neg_ratio=args.neg_ratio,
        exclude_langs=args.exclude_langs,
        logger=logger
    )
    
    success = preparer.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
