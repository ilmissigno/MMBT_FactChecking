#!/usr/bin/env python3
"""Entry point for MMBT training/evaluation workflows."""

import warnings

import torch

from mmbt.train import Trainer


def _configure_runtime() -> None:
    """Set PyTorch backend flags used across training runs."""
    # Keep runs reproducible unless overridden in downstream code.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def main() -> None:
    warnings.filterwarnings("ignore")
    _configure_runtime()
    Trainer()


if __name__ == "__main__":
    main()
