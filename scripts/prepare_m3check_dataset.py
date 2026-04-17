#!/usr/bin/env python3
"""
MMBT - M3-Check Dataset Preparation Script
============================================

Prepara il dataset M3-Check per MMBT generando coppie query-document
con negative sampling, nel formato standard MMBT.

M3-Check è un dataset multilingue e multimodale per il retrieval
di fact-check già pubblicati dato un post social media.

Formato output: QueryID, QueryText, QueryImages, DocID, DocText, DocImages, Label

Mapping:
  - Query = Post social media (post_translated_text + post_image)
  - Document = Fact-check article (fact_check_translated_title + fact_check_image)
  - Positive pair: post associato al suo fact-check (ground truth)
  - Negative pair: post associato a un fact-check random non correlato
"""

import argparse
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging"""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"m3check_preparation_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

    return logging.getLogger("m3check_prep")


class M3CheckPreparer:
    """Prepara M3-Check nel formato MMBT con negative sampling"""

    def __init__(
        self,
        raw_dir: Path,
        output_dir: Path,
        neg_ratio: int = 30,
        use_translated: bool = True,
        extract_images: bool = False,
        exclude_langs: list = None,
        logger: logging.Logger = None,
    ):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.neg_ratio = neg_ratio
        self.use_translated = use_translated
        self.extract_images = extract_images
        self.exclude_langs = [l.lower() for l in exclude_langs] if exclude_langs else []
        self.logger = logger or logging.getLogger(__name__)

        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.fc_df = None
        self.pairs_df = None

    def load_data(self) -> bool:
        """Carica i dati grezzi M3-Check"""
        self.logger.info("Caricamento dati M3-Check...")

        files = {
            "training": self.raw_dir / "m3_check_training.csv",
            "validation": self.raw_dir / "m3_check_validation.csv",
            "testing": self.raw_dir / "m3_check_testing.csv",
            "fact_checks": self.raw_dir / "m3_check_fact_checks.csv",
            "pairs": self.raw_dir / "m3_check_post_article_pairs.csv",
        }

        for name, path in files.items():
            if not path.exists():
                self.logger.error(f"File non trovato: {path}")
                return False

        self.train_df = pd.read_csv(files["training"])
        self.val_df = pd.read_csv(files["validation"])
        self.test_df = pd.read_csv(files["testing"])
        self.fc_df = pd.read_csv(files["fact_checks"])
        self.pairs_df = pd.read_csv(files["pairs"])

        self.logger.info(f"Training:    {len(self.train_df)} righe")
        self.logger.info(f"Validation:  {len(self.val_df)} righe")
        self.logger.info(f"Testing:     {len(self.test_df)} righe")
        self.logger.info(f"Fact-checks: {len(self.fc_df)} nel corpus")
        self.logger.info(f"Pairs:       {len(self.pairs_df)} associazioni post-fc")

        # Log distribuzione lingue
        for name, df in [("Train", self.train_df), ("Val", self.val_df), ("Test", self.test_df)]:
            if "post_lang" in df.columns:
                top_langs = df["post_lang"].dropna().value_counts().head(5)
                self.logger.info(f"{name} - Top lingue: {dict(top_langs)}")

        # NaN analysis
        nan_text = self.train_df["post_translated_text"].isna().sum()
        nan_orig = self.train_df["post_original_text"].isna().sum()
        self.logger.info(f"Train NaN: post_translated_text={nan_text}, post_original_text={nan_orig}")

        # Filtro lingue da escludere
        if self.exclude_langs:
            self.logger.info(f"Filtraggio lingue da escludere: {self.exclude_langs}")
            for name, attr in [("train_df", "train_df"), ("val_df", "val_df"), ("test_df", "test_df")]:
                df = getattr(self, attr)
                before = len(df)
                # Rimuovi post con lingua nelle exclude_langs e post senza lingua (NaN)
                mask = df["post_lang"].notna() & ~df["post_lang"].str.lower().isin(self.exclude_langs)
                df_filtered = df[mask].reset_index(drop=True)
                setattr(self, attr, df_filtered)
                self.logger.info(
                    f"  {name}: {before} → {len(df_filtered)} "
                    f"(-{before - len(df_filtered)} rimossi, {len(df_filtered)/before*100:.1f}% mantenuti)"
                )
            # Log lingue rimaste
            top = self.train_df["post_lang"].value_counts().head(10)
            self.logger.info(f"Lingue rimaste (train top 10): {dict(top)}")

        return True

    def _get_post_text(self, row) -> str:
        """Ottieni il testo del post (tradotto o originale)"""
        if self.use_translated:
            text = row.get("post_translated_text", None)
            if pd.isna(text) or str(text).strip() in ("", "nan"):
                text = row.get("post_original_text", None)
        else:
            text = row.get("post_original_text", None)
            if pd.isna(text) or str(text).strip() in ("", "nan"):
                text = row.get("post_translated_text", None)

        if pd.isna(text) or str(text).strip() in ("", "nan"):
            return ""
        return str(text).strip()

    def _get_fc_text(self, row) -> str:
        """Ottieni il testo del fact-check (claim tradotta + titolo tradotto)"""
        # Usa claim tradotta come testo principale, con fallback su titolo
        claim = row.get("fact_check_translated_claim", None)
        title = row.get("fact_check_translated_title", None)

        parts = []
        if not pd.isna(title) and str(title).strip() not in ("", "nan"):
            parts.append(str(title).strip())
        if not pd.isna(claim) and str(claim).strip() not in ("", "nan"):
            # Evita duplicazione se claim == title
            claim_str = str(claim).strip()
            if not parts or claim_str != parts[0]:
                parts.append(claim_str)

        return " | ".join(parts) if parts else ""

    def _build_fc_lookup(self) -> dict:
        """Costruisci lookup per fact-check dal corpus"""
        fc_lookup = {}
        for _, row in self.fc_df.iterrows():
            fc_id = row["fact_check_id"]
            text = self._get_fc_text(row)
            fc_lookup[fc_id] = {
                "text": text,
                "image": row.get("fact_check_image", ""),
            }
        return fc_lookup

    def _generate_pairs_for_split(
        self, split_df: pd.DataFrame, split_name: str, fc_lookup: dict
    ) -> pd.DataFrame:
        """
        Genera coppie query-document per uno split.
        
        Per ogni post:
        - 1 coppia positiva con il fact-check ground truth
        - N coppie negative con fact-check casuali
        """
        self.logger.info(f"Generazione coppie per {split_name} (neg_ratio={self.neg_ratio})...")

        all_fc_ids = list(fc_lookup.keys())
        pairs = []
        skipped = 0

        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"{split_name}"):
            post_id = row["post_id"]
            query_text = self._get_post_text(row)
            post_image = row.get("post_image", "")

            # Skip post senza testo
            if not query_text:
                skipped += 1
                continue

            fc_id = row["fact_check_id"]

            # Positive pair
            fc_info = fc_lookup.get(fc_id, None)
            if fc_info is None or not fc_info["text"]:
                skipped += 1
                continue

            pairs.append({
                "QueryID": str(int(post_id)),
                "QueryText": query_text,
                "QueryImages": str(post_image) if not pd.isna(post_image) else "",
                "DocID": str(int(fc_id)),
                "DocText": fc_info["text"],
                "DocImages": str(fc_info["image"]) if not pd.isna(fc_info["image"]) else "",
                "Label": 1.0,
            })

            # Negative pairs: fact-check non correlati
            # Ottieni tutti i FC associati a questo post (dalle pairs)
            positive_fc_ids = set(
                self.pairs_df[self.pairs_df["post_id"] == post_id]["fact_check_id"].tolist()
            )
            # Aggiungi anche il fc_id corrente
            positive_fc_ids.add(fc_id)

            # Campiona negative evitando i positivi
            candidate_ids = [fid for fid in all_fc_ids if fid not in positive_fc_ids]
            n_neg = min(self.neg_ratio, len(candidate_ids))
            neg_fc_ids = random.sample(candidate_ids, n_neg)

            for neg_fc_id in neg_fc_ids:
                neg_fc_info = fc_lookup.get(neg_fc_id, None)
                if neg_fc_info is None or not neg_fc_info["text"]:
                    continue
                pairs.append({
                    "QueryID": str(int(post_id)),
                    "QueryText": query_text,
                    "QueryImages": str(post_image) if not pd.isna(post_image) else "",
                    "DocID": str(int(neg_fc_id)),
                    "DocText": neg_fc_info["text"],
                    "DocImages": str(neg_fc_info["image"]) if not pd.isna(neg_fc_info["image"]) else "",
                    "Label": 0.0,
                })

        df = pd.DataFrame(pairs)
        self.logger.info(
            f"{split_name}: {len(df)} coppie generate "
            f"(skip={skipped}, pos={(df['Label']==1.0).sum()}, neg={(df['Label']==0.0).sum()})"
        )
        return df

    def _clean_text_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rimuovi righe con testi null/nan/vuoti"""
        original_len = len(df)

        df = df.dropna(subset=["QueryText", "DocText"])
        df = df[df["QueryText"].astype(str) != "nan"]
        df = df[df["DocText"].astype(str) != "nan"]
        df = df[df["QueryText"].astype(str).str.strip() != ""]
        df = df[df["DocText"].astype(str).str.strip() != ""]

        removed = original_len - len(df)
        if removed > 0:
            self.logger.info(f"Rimosse {removed} righe con testi invalidi")

        return df.reset_index(drop=True)

    def _organize_images(self):
        """
        Organizza le immagini M3-Check nel formato atteso da MMBT.

        MMBT cerca:
          images/query/{post_id}.png
          images/doc/{fact_check_id}.png

        M3-Check fornisce:
          Post:  drive/fb_images/{hash}.jpg  o  drive/ig_images/{hash}.jpg
                 (possono essere multipli, separati da virgola → si usa il primo)
          FC:    drive_images/image_{id}/  (directory con JPEG multipli → si usa il primo)

        Usa symlink per evitare duplicazione dei ~4GB di immagini.
        Le immagini devono già essere estratte dagli ZIP in raw_dir.
        """
        images_dir = self.output_dir / "images"
        query_dir = images_dir / "query"
        doc_dir = images_dir / "doc"
        query_dir.mkdir(parents=True, exist_ok=True)
        doc_dir.mkdir(parents=True, exist_ok=True)

        # --- Post images (query) ---
        all_posts = pd.concat(
            [self.train_df, self.val_df, self.test_df], ignore_index=True
        )
        post_imgs = all_posts.drop_duplicates("post_id")[["post_id", "post_image"]]

        linked_q, missing_q = 0, 0
        for _, row in tqdm(post_imgs.iterrows(), total=len(post_imgs), desc="Query images"):
            post_id = int(row["post_id"])
            link_path = query_dir / f"{post_id}.png"
            if link_path.exists() or link_path.is_symlink():
                linked_q += 1
                continue

            img_ref = row.get("post_image", "")
            if pd.isna(img_ref) or str(img_ref).strip() == "":
                missing_q += 1
                continue

            # Se ci sono più immagini (comma-separated), prendi la prima
            first_img = str(img_ref).split(",")[0].strip()
            src = (self.raw_dir / first_img).resolve()
            if src.exists():
                os.symlink(src, link_path)
                linked_q += 1
            else:
                missing_q += 1

        self.logger.info(
            f"Query images: {linked_q} linkati, {missing_q} mancanti "
            f"(su {len(post_imgs)} post unici)"
        )

        # --- Fact-check images (doc) ---
        fc_imgs = all_posts.drop_duplicates("fact_check_id")[
            ["fact_check_id", "fact_check_image"]
        ]

        linked_d, missing_d = 0, 0
        for _, row in tqdm(fc_imgs.iterrows(), total=len(fc_imgs), desc="Doc images"):
            fc_id = int(row["fact_check_id"])
            link_path = doc_dir / f"{fc_id}.png"
            if link_path.exists() or link_path.is_symlink():
                linked_d += 1
                continue

            img_ref = row.get("fact_check_image", "")
            if pd.isna(img_ref) or str(img_ref).strip() == "":
                missing_d += 1
                continue

            fc_dir = (self.raw_dir / str(img_ref).strip()).resolve()
            if fc_dir.is_dir():
                # La directory contiene uno o più JPEG; prendi il primo
                jpegs = sorted(
                    f for f in os.listdir(fc_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
                )
                if jpegs:
                    os.symlink(fc_dir / jpegs[0], link_path)
                    linked_d += 1
                else:
                    missing_d += 1
            elif fc_dir.is_file():
                os.symlink(fc_dir, link_path)
                linked_d += 1
            else:
                missing_d += 1

        self.logger.info(
            f"Doc images: {linked_d} linkati, {missing_d} mancanti "
            f"(su {len(fc_imgs)} fc unici)"
        )

    def save_datasets(
        self, train_pairs: pd.DataFrame, val_pairs: pd.DataFrame, test_pairs: pd.DataFrame
    ):
        """Salva i dataset in formato TSV"""
        annotations_dir = self.output_dir / "annotations"
        annotations_dir.mkdir(parents=True, exist_ok=True)

        # Crea cartelle immagini
        images_dir = self.output_dir / "images"
        (images_dir / "query").mkdir(parents=True, exist_ok=True)
        (images_dir / "doc").mkdir(parents=True, exist_ok=True)

        for name, df in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
            df = self._clean_text_data(df)
            path = annotations_dir / f"{name}.tsv"
            df.to_csv(path, sep="\t", index=False)
            self.logger.info(f"Salvato {path} ({len(df)} righe)")

        self._log_stats(train_pairs, val_pairs, test_pairs)

    def _log_stats(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
    ):
        """Log statistiche finali"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STATISTICHE FINALI M3-Check")
        self.logger.info("=" * 60)

        for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            pos = (df["Label"] == 1.0).sum()
            neg = (df["Label"] == 0.0).sum()
            n_queries = df["QueryID"].nunique()
            n_docs = df["DocID"].nunique()
            self.logger.info(
                f"{name}: {len(df)} totali | {pos} positivi ({pos / len(df) * 100:.1f}%) | "
                f"{neg} negativi ({neg / len(df) * 100:.1f}%) | "
                f"{n_queries} query uniche | {n_docs} doc unici"
            )

        self.logger.info("=" * 60)

    def run(self):
        """Esegue la pipeline completa"""
        self.logger.info("=" * 60)
        self.logger.info("M3-Check Dataset Preparation per MMBT")
        self.logger.info(f"  Raw dir: {self.raw_dir}")
        self.logger.info(f"  Output dir: {self.output_dir}")
        self.logger.info(f"  Neg ratio: {self.neg_ratio}")
        self.logger.info(f"  Use translated: {self.use_translated}")
        self.logger.info("=" * 60)

        if not self.load_data():
            return False

        fc_lookup = self._build_fc_lookup()
        self.logger.info(f"FC lookup: {len(fc_lookup)} fact-check indicizzati")

        train_pairs = self._generate_pairs_for_split(self.train_df, "Train", fc_lookup)
        val_pairs = self._generate_pairs_for_split(self.val_df, "Val", fc_lookup)
        test_pairs = self._generate_pairs_for_split(self.test_df, "Test", fc_lookup)

        self.save_datasets(train_pairs, val_pairs, test_pairs)

        if self.extract_images:
            self._organize_images()

        self.logger.info("\n✅ Preparazione M3-Check completata!")
        return True


def main():
    parser = argparse.ArgumentParser(description="Prepare M3-Check dataset for MMBT")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "M3-check",
        help="Directory con i file CSV M3-Check originali",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "m3check_mmbt",
        help="Directory di output per il formato MMBT",
    )
    parser.add_argument(
        "--neg-ratio",
        type=int,
        default=30,
        help="Numero di negative samples per coppia positiva (default: 30)",
    )
    parser.add_argument(
        "--use-original",
        action="store_true",
        help="Usa testo originale invece del tradotto (default: usa tradotto)",
    )
    parser.add_argument(
        "--exclude-langs",
        type=str,
        nargs="+",
        default=[],
        metavar="LANG",
        help="Codici lingua ISO da escludere dai post (es. --exclude-langs eng)",
    )
    parser.add_argument(
        "--extract-images",
        action="store_true",
        help="Organizza immagini nel formato MMBT (crea symlink da raw_dir)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed per riproducibilità",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    logger = setup_logging(args.output_dir)

    preparer = M3CheckPreparer(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        neg_ratio=args.neg_ratio,
        use_translated=not args.use_original,
        extract_images=args.extract_images,
        exclude_langs=args.exclude_langs,
        logger=logger,
    )

    success = preparer.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
