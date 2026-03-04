"""
src/utils.py
Shared utility helpers used across the chess analysis project.
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd

from src.logger    import logger
from src.exception import ChessAnalysisException


MODELS_DIR = "models"


def tc_label(perf_type: str) -> str:
    """Map perf_type string → file-naming label ('blitz' or 'standard')."""
    return "blitz" if perf_type == "blitz" else "standard"


def model_base(username: str, target: str, perf_type: str) -> str:
    """Return the base path prefix for all pickle files for one model."""
    return os.path.join(MODELS_DIR, f"{username}_{target}_{tc_label(perf_type)}")


def save_pickle(obj, path: str) -> None:
    """Pickle-dump *obj* to *path*, creating parent dirs as needed."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Saved: {path}")
    except Exception as e:
        raise ChessAnalysisException(e, sys) from e


def load_pickle(path: str):
    """Load and return a pickled object from *path*."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise ChessAnalysisException(e, sys) from e


def models_available(username: str, perf_type: str) -> bool:
    """
    Return True if all required model artifact files exist for the given
    username / perf_type combination.
    """
    tc = tc_label(perf_type)
    required = [
        f"{MODELS_DIR}/{username}_{target}_{tc}_{suffix}.pkl"
        for target in ("blunder", "inaccuracy")
        for suffix in (
            "model", "features", "medians", "base_rate",
            "scaler", "log_caps", "winsor_bounds", "scale_cols", "thresholds",
        )
    ]
    return all(os.path.exists(p) for p in required)
