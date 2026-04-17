#!/usr/bin/env python3
"""
MMBT - MuMiN Dataset Preparation Script
========================================

Prepares the MuMiN dataset for use with MMBT:
1. Downloads/compiles the dataset using mumin package
2. Converts to MMBT TSV format
3. Downloads/processes images
4. Creates train/val/test splits

Requirements:
    pip install mumin[all]
    
    Twitter Bearer Token (get from https://developer.twitter.com/):
    export TWITTER_BEARER_TOKEN=your_token_here

Usage:
    python prepare_mumin_dataset.py --size small
    python prepare_mumin_dataset.py --size medium --include-images
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil

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
    log_file = log_dir / f"mumin_preparation_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger("mumin_prep")


class MuMiNConverter:
    """Convert MuMiN dataset to MMBT format"""
    
    def __init__(
        self,
        size: str = 'small',
        bearer_token: Optional[str] = None,
        output_dir: Path = Path("datasets/mumin_mmbt"),
        include_images: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        self.size = size
        self.bearer_token = bearer_token or os.environ.get("TWITTER_BEARER_TOKEN")
        self.output_dir = output_dir
        self.include_images = include_images
        self.logger = logger or logging.getLogger(__name__)
        
        self.dataset = None
        
    def download_and_compile(self) -> bool:
        """Download and compile MuMiN dataset"""
        try:
            from mumin import MuminDataset
        except ImportError:
            self.logger.error(
                "MuMiN package not installed. Run: pip install mumin[all]"
            )
            return False
        
        if not self.bearer_token:
            self.logger.error(
                "Twitter Bearer Token not provided. "
                "Set TWITTER_BEARER_TOKEN environment variable."
            )
            return False
        
        self.logger.info(f"Initializing MuMiN dataset (size={self.size})...")
        
        try:
            self.dataset = MuminDataset(
                self.bearer_token,
                size=self.size,
                include_extra_images=self.include_images
            )
            
            self.logger.info("Compiling dataset (this may take a while)...")
            self.dataset.compile()
            
            self.logger.info("Dataset compiled successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to compile dataset: {e}")
            return False
    
    def load_from_raw(self) -> bool:
        """Load from raw MuMiN files if already downloaded"""
        raw_dir = PROJECT_ROOT / "datasets" / "mumin" / "mumin"
        
        if not raw_dir.exists():
            self.logger.warning(f"Raw MuMiN data not found at {raw_dir}")
            return False
        
        self.logger.info("Loading from raw MuMiN files...")
        
        try:
            # Load claims
            claims_path = raw_dir / "claim"
            if claims_path.exists():
                self.claims_df = pd.read_pickle(claims_path, compression='xz')
                self.logger.info(f"Loaded {len(self.claims_df)} claims")
            
            # Load tweets
            tweets_path = raw_dir / "tweet"
            if tweets_path.exists():
                self.tweets_df = pd.read_pickle(tweets_path, compression='xz')
                self.logger.info(f"Loaded {len(self.tweets_df)} tweets")
            
            # Load tweet-claim relations
            rel_path = raw_dir / "tweet_discusses_claim"
            if rel_path.exists():
                self.tweet_claim_rel = pd.read_pickle(rel_path, compression='xz')
                self.logger.info(f"Loaded {len(self.tweet_claim_rel)} tweet-claim relations")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load raw data: {e}")
            return False
    
    def convert_to_mmbt_format(self) -> Dict[str, pd.DataFrame]:
        """Convert MuMiN data to MMBT TSV format"""
        self.logger.info("Converting to MMBT format...")
        
        # Create output directories
        annotations_dir = self.output_dir / "annotations"
        images_dir = self.output_dir / "images"
        annotations_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        (images_dir / "query").mkdir(exist_ok=True)
        (images_dir / "doc").mkdir(exist_ok=True)
        
        # Prepare data structures
        train_data = []
        val_data = []
        test_data = []
        
        # Get split masks
        if hasattr(self, 'claims_df'):
            claims = self.claims_df
        elif self.dataset is not None:
            claims = self.dataset.nodes['claim']
        else:
            self.logger.error("No data loaded")
            return {}
        
        # Determine split column prefix based on size
        split_prefix = f"{self.size}_"
        
        self.logger.info("Processing claims and creating pairs...")
        
        for idx, claim in tqdm(claims.iterrows(), total=len(claims), desc="Processing claims"):
            # Create document entry from claim
            doc_entry = {
                'DocID': claim['id'] if 'id' in claim else str(idx),
                'DocText': self._get_claim_text(claim),
                'Label': 1 if claim.get('label', 'factual') == 'factual' else 0
            }
            
            # Create query entry (simplified - using claim keywords as query)
            query_entry = {
                'QueryID': f"q_{claim['id'] if 'id' in claim else idx}",
                'QueryText': claim.get('keywords', doc_entry['DocText'][:200]),
            }
            
            # Combine for TSV format
            entry = {
                **query_entry,
                **doc_entry
            }
            
            # Assign to split
            is_train = claim.get(f'{split_prefix}train_mask', False)
            is_val = claim.get(f'{split_prefix}val_mask', False)
            is_test = claim.get(f'{split_prefix}test_mask', False)
            
            if is_train:
                train_data.append(entry)
            elif is_val:
                val_data.append(entry)
            elif is_test:
                test_data.append(entry)
            else:
                # Default to train if no mask
                train_data.append(entry)
        
        # Create DataFrames
        result = {
            'train': pd.DataFrame(train_data),
            'val': pd.DataFrame(val_data),
            'test': pd.DataFrame(test_data)
        }
        
        # Log statistics
        self.logger.info(f"Train samples: {len(result['train'])}")
        self.logger.info(f"Val samples: {len(result['val'])}")
        self.logger.info(f"Test samples: {len(result['test'])}")
        
        return result
    
    def _get_claim_text(self, claim) -> str:
        """Extract text from claim entry"""
        # Try different possible text fields
        text_fields = ['text', 'claim_text', 'keywords', 'cluster_keywords']
        
        for field in text_fields:
            if field in claim and pd.notna(claim[field]):
                return str(claim[field])
        
        return ""
    
    def save_datasets(self, datasets: Dict[str, pd.DataFrame]) -> None:
        """Save datasets to TSV files"""
        annotations_dir = self.output_dir / "annotations"
        
        for split_name, df in datasets.items():
            if len(df) > 0:
                output_path = annotations_dir / f"{split_name}.tsv"
                df.to_csv(output_path, sep='\t', index=False)
                self.logger.info(f"Saved {split_name} to {output_path}")
    
    def create_interaction_file(self, datasets: Dict[str, pd.DataFrame]) -> None:
        """Create query-document interaction file"""
        interactions = []
        
        for df in datasets.values():
            if len(df) > 0:
                for _, row in df.iterrows():
                    interactions.append({
                        'QueryID': row['QueryID'],
                        'DocID': row['DocID'],
                        'Label': row['Label']
                    })
        
        interactions_df = pd.DataFrame(interactions)
        output_path = self.output_dir / "annotations" / "query_doc_interactions.csv"
        interactions_df.to_csv(output_path, index=False)
        
        self.logger.info(f"Created interaction file with {len(interactions)} entries")
    
    def run(self) -> bool:
        """Run the full conversion pipeline"""
        self.logger.info("="*60)
        self.logger.info("MuMiN Dataset Preparation")
        self.logger.info(f"Size: {self.size}")
        self.logger.info(f"Output: {self.output_dir}")
        self.logger.info("="*60)
        
        # Try loading from raw first
        if self.load_from_raw():
            self.logger.info("Loaded from raw files")
        elif not self.download_and_compile():
            self.logger.error("Failed to load/download dataset")
            return False
        
        # Convert to MMBT format
        datasets = self.convert_to_mmbt_format()
        
        if not datasets:
            self.logger.error("No data to save")
            return False
        
        # Save datasets
        self.save_datasets(datasets)
        
        # Create interaction file
        self.create_interaction_file(datasets)
        
        self.logger.info("\n" + "="*60)
        self.logger.info("MuMiN dataset preparation complete!")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info("="*60)
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MuMiN dataset for MMBT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Prepare small dataset
    python prepare_mumin_dataset.py --size small

    # Prepare medium dataset with images
    python prepare_mumin_dataset.py --size medium --include-images

    # Use specific output directory
    python prepare_mumin_dataset.py --size small --output-dir ./data/mumin
        """
    )
    
    parser.add_argument(
        "--size",
        type=str,
        default="small",
        choices=["small", "medium", "large"],
        help="MuMiN dataset size"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/mumin_mmbt",
        help="Output directory for prepared dataset"
    )
    
    parser.add_argument(
        "--include-images",
        action="store_true",
        help="Include image data (requires more time/space)"
    )
    
    parser.add_argument(
        "--bearer-token",
        type=str,
        default=None,
        help="Twitter Bearer Token (or set TWITTER_BEARER_TOKEN env)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir)
    
    converter = MuMiNConverter(
        size=args.size,
        bearer_token=args.bearer_token,
        output_dir=output_dir,
        include_images=args.include_images,
        logger=logger
    )
    
    success = converter.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
