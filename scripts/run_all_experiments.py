#!/usr/bin/env python3
"""
MMBT - Run All Experiments
==========================

Master script to orchestrate all experiments with full reproducibility.

Usage:
    python run_all_experiments.py --config ./configs/experiments.yaml
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging"""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"all_experiments_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger("all_experiments")


def run_command(cmd: List[str], logger: logging.Logger) -> int:
    """Run command and log output"""
    logger.info(f"Running: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    for line in process.stdout:
        logger.info(line.strip())
    
    process.wait()
    return process.returncode


def main():
    parser = argparse.ArgumentParser(description="Run all MMBT experiments")
    parser.add_argument("--config", type=str, default="configs/experiments.yaml")
    parser.add_argument("--output-dir", type=str, default="outputs/all_experiments")
    parser.add_argument("--skip-training", action="store_true", help="Skip training, only run evaluation")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    
    logger.info("="*60)
    logger.info("MMBT Fact-Checking - Full Experiment Suite")
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info("="*60)
    
    results = {}
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {
            "experiments": ["white_noise", "complexity"],
            "datasets": ["Snopes", "Politifact"],
            "noise_levels": [0.1, 0.2, 0.3, 0.5],
            "dataset_sizes": [100, 500, 1000, 2000, 5000]
        }
    
    datasets = config.get("datasets", ["Snopes", "Politifact"])
    experiments = config.get("experiments", ["white_noise", "complexity", "mumin"])
    
    # 1. Baseline Training (if not skipping)
    if not args.skip_training:
        for dataset in datasets:
            logger.info(f"\n{'='*40}")
            logger.info(f"Training baseline on {dataset}")
            logger.info("="*40)
            
            cmd = [
                sys.executable, "main.py",
                "-c", f"./configs/{dataset.lower()}_config.conf"
            ]
            
            returncode = run_command(cmd, logger)
            results[f"baseline_{dataset}"] = {
                "status": "success" if returncode == 0 else "failed",
                "returncode": returncode
            }
    
    # 2. White Noise Experiments
    if "white_noise" in experiments:
        noise_levels = ",".join(map(str, config.get("noise_levels", [0.1, 0.2, 0.3, 0.5])))
        
        for dataset in datasets:
            logger.info(f"\n{'='*40}")
            logger.info(f"White Noise Experiment on {dataset}")
            logger.info("="*40)
            
            cmd = [
                sys.executable, "scripts/run_experiments.py",
                "--experiment", "white_noise",
                "--dataset", dataset,
                "--noise-levels", noise_levels,
                "--output-dir", str(output_dir / "white_noise")
            ]
            
            returncode = run_command(cmd, logger)
            results[f"white_noise_{dataset}"] = {
                "status": "success" if returncode == 0 else "failed",
                "returncode": returncode
            }
    
    # 3. Complexity Analysis
    if "complexity" in experiments:
        dataset_sizes = ",".join(map(str, config.get("dataset_sizes", [100, 500, 1000, 2000, 5000])))
        
        logger.info(f"\n{'='*40}")
        logger.info("Computational Complexity Analysis")
        logger.info("="*40)
        
        cmd = [
            sys.executable, "scripts/run_experiments.py",
            "--experiment", "complexity",
            "--dataset-sizes", dataset_sizes,
            "--output-dir", str(output_dir / "complexity")
        ]
        
        returncode = run_command(cmd, logger)
        results["complexity"] = {
            "status": "success" if returncode == 0 else "failed",
            "returncode": returncode
        }
    
    # 4. MuMiN Integration (optional)
    if "mumin" in experiments and os.environ.get("TWITTER_BEARER_TOKEN"):
        logger.info(f"\n{'='*40}")
        logger.info("MuMiN Dataset Integration")
        logger.info("="*40)
        
        cmd = [
            sys.executable, "scripts/run_experiments.py",
            "--experiment", "mumin",
            "--mumin-size", config.get("mumin_size", "small"),
            "--output-dir", str(output_dir / "mumin")
        ]
        
        returncode = run_command(cmd, logger)
        results["mumin"] = {
            "status": "success" if returncode == 0 else "failed",
            "returncode": returncode
        }
    
    # Save experiment summary
    summary_path = output_dir / "experiment_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "results": results
        }, f, indent=2)
    
    # Generate final report
    generate_final_report(output_dir, results, logger)
    
    logger.info("\n" + "="*60)
    logger.info("All experiments completed!")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info("="*60)


def generate_final_report(output_dir: Path, results: Dict, logger: logging.Logger):
    """Generate final summary report"""
    report = [
        "# MMBT Fact-Checking - Complete Experiment Report\n\n",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
        "## Experiment Status Summary\n\n",
        "| Experiment | Status |\n",
        "|------------|--------|\n"
    ]
    
    for exp_name, exp_results in results.items():
        status = "✅ Success" if exp_results.get("status") == "success" else "❌ Failed"
        report.append(f"| {exp_name} | {status} |\n")
    
    report.append("\n## How to Reproduce\n\n")
    report.append("```bash\n")
    report.append("# Install dependencies\n")
    report.append("pip install -r requirements.txt\n\n")
    report.append("# Run all experiments\n")
    report.append("python scripts/run_all_experiments.py --config configs/experiments.yaml\n")
    report.append("```\n")
    
    report_path = output_dir / "FINAL_REPORT.md"
    with open(report_path, 'w') as f:
        f.writelines(report)
    
    logger.info(f"Final report: {report_path}")


if __name__ == "__main__":
    main()
