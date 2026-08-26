"""
advanced_pipeline
=================

Minimal utilities for the simplified churn prediction model.
Provides data loading and logging utilities.
"""

from .kaggle_data import load_kaggle_full
from .utils import get_logger

__version__ = "2.0.0"

__all__ = ["load_kaggle_full", "get_logger"]