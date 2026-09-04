"""
utils.py
========

Shared helpers for the simplified churn prediction model:
* :func:`get_logger` -- a configured module-level logger.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_logger_cache: dict[str, logging.Logger] = {}


def get_logger(name: str = "advanced_pipeline", level: int = logging.INFO) -> logging.Logger:
    """Return a cached logger configured with a consistent format.

    Parameters
    ----------
    name:
        Logger name (usually ``__name__`` of the calling module).
    level:
        Logging level (defaults to ``logging.INFO``).

    Returns
    -------
    logging.Logger
        The configured logger instance.
    """
    if name in _logger_cache:
        return _logger_cache[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Avoid duplicate handlers when re-imported / re-configured.
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
    logger.propagate = False
    _logger_cache[name] = logger
    return logger