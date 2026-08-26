"""
kaggle_data.py
==============

Exclusive data source for the Churn Prediction Model.

Every training and testing dataset is sourced from the Kaggle dataset
``muhammadshahidazeem/customer-churn-dataset``.  The Kaggle release ships two
CSV files:

    * ``customer_churn_dataset-training-master.csv``
    * ``customer_churn_dataset-testing-master.csv``

Both files use a schema that differs from the one expected by the project
(``tenure_months``, ``monthly_charges``, ``contract_type``,
``payment_method`` ... plus a ``churn`` binary target).  This
module downloads the dataset on demand (via ``kagglehub``), pools the two files,
and remaps the raw Kaggle columns onto the project schema.

Remap (Kaggle -> project)
-------------------------
    CustomerID        -> customer_id
    Tenure            -> tenure_months
    Total Spend       -> total_charges
    Usage Frequency   -> login_frequency_weekly
    Support Calls     -> support_tickets_last_3mo
    Subscription Type -> contract_type
    Contract Length   -> payment_method
    Age               -> nps_score (age bucket 0-10, higher = younger/healthier)
    Last Interaction  -> avg_session_duration_min (days since interaction)
    Payment Delay     -> monthly_charges (proxy)
    Churn             -> churn

The datasets are pooled and a fresh, stratified train/test split is created
internally (the Kaggle ``training``/``testing`` split is not used directly).
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd

from .utils import get_logger

logger = get_logger(__name__)

# The only permitted data source for this project.
KAGGLE_SLUG = "muhammadshahidazeem/customer-churn-dataset"

# Expected schema after remapping (churn-only).
PROJECT_ID = ["customer_id"]
PROJECT_NUMERIC = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_tickets_last_3mo",
    "avg_session_duration_min",
    "login_frequency_weekly",
    "nps_score",
]
PROJECT_CATEGORICAL = ["contract_type", "payment_method"]
PROJECT_TARGETS = ["churn"]


def _download_kaggle_dir() -> str:
    """Download (or locate cached) the Kaggle dataset directory."""
    kagglehub = __import__("kagglehub")
    return kagglehub.dataset_download(KAGGLE_SLUG)


def _load_raw_kaggle(directory: str) -> pd.DataFrame:
    """Load and concatenate the two raw Kaggle CSVs."""
    files = [
        os.path.join(directory, "customer_churn_dataset-training-master.csv"),
        os.path.join(directory, "customer_churn_dataset-testing-master.csv"),
    ]
    frames = []
    for f in files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            logger.info(
                "Loaded %s -> %d rows x %d cols", os.path.basename(f), *df.shape
            )
            frames.append(df)
        else:
            logger.warning("Expected Kaggle file not found: %s", f)
    if not frames:
        raise FileNotFoundError("No Kaggle dataset files found after download.")
    return pd.concat(frames, ignore_index=True)


def _remap_to_project_schema(raw: pd.DataFrame) -> pd.DataFrame:
    """Map raw Kaggle columns onto the project's churn-only schema."""
    raw = raw.copy()
    numeric_raw = [
        "Age",
        "Tenure",
        "Usage Frequency",
        "Support Calls",
        "Payment Delay",
        "Total Spend",
        "Last Interaction",
        "Churn",
    ]
    for c in numeric_raw:
        if c in raw.columns:
            raw[c] = raw[c].fillna(raw[c].median())
    for c in ("Contract Length", "Subscription Type"):
        if c in raw.columns:
            raw[c] = raw[c].fillna(
                raw[c].mode().iloc[0] if raw[c].notna().any() else "unknown"
            )

    df = pd.DataFrame()

    # Identifiers.
    if "CustomerID" in raw.columns:
        cid = raw["CustomerID"].astype("float").round()
        cid = cid.where(cid.notna(), pd.Series(range(len(raw)), index=raw.index))
        df["customer_id"] = cid.astype("Int64").astype(str)
    else:
        df["customer_id"] = [f"KAGGLE{i:07d}" for i in range(1, len(raw) + 1)]

    # Numeric features.
    df["tenure_months"] = raw["Tenure"].astype(float)
    df["total_charges"] = raw["Total Spend"].astype(float)
    df["monthly_charges"] = (raw["Payment Delay"].astype(float) * 5.0).clip(15, 150)
    df["support_tickets_last_3mo"] = raw["Support Calls"].astype(float)
    df["avg_session_duration_min"] = raw["Last Interaction"].astype(float)
    df["login_frequency_weekly"] = raw["Usage Frequency"].astype(float)

    # NPS proxy: bucket age into 0-10, higher = younger cohort (lower churn risk).
    if "Age" in raw.columns:
        age = raw["Age"].astype(float)
        df["nps_score"] = (
            (10 - (age - age.min()) / max(age.max() - age.min(), 1e-9) * 10)
            .round()
            .clip(0, 10)
            .astype(int)
        )
    else:
        df["nps_score"] = 5

    # Categorical features.
    df["contract_type"] = raw["Contract Length"].astype(str).str.strip().str.lower()
    df["payment_method"] = raw["Subscription Type"].astype(str).str.strip().str.lower()

    # Target.
    df["churn"] = raw["Churn"].astype(float).round().astype(int)

    return df[PROJECT_ID + PROJECT_NUMERIC + PROJECT_CATEGORICAL + PROJECT_TARGETS]


def load_kaggle_full(random_state: int = 42) -> pd.DataFrame:
    """Return the full pooled Kaggle dataset in the project schema (no split)."""
    directory = _download_kaggle_dir()
    raw = _load_raw_kaggle(directory)
    return _remap_to_project_schema(raw)


def load_kaggle_split(
    random_state: int = 42,
    test_size: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test) Kaggle-sourced DataFrames in the project schema.

    Both splits are pooled from the Kaggle training + testing CSVs and then
    re-split with stratification on the churn target.
    """
    directory = _download_kaggle_dir()
    raw = _load_raw_kaggle(directory)
    df = _remap_to_project_schema(raw)

    from sklearn.model_selection import train_test_split

    X = df.drop(columns=["churn"])
    y_churn = df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_churn, test_size=test_size, random_state=random_state, stratify=y_churn
    )
    train = X_train.join(df.loc[X_train.index, ["churn"]]).reset_index(drop=True)
    test = X_test.join(df.loc[X_test.index, ["churn"]]).reset_index(drop=True)

    logger.info(
        "Kaggle dataset ready -> train=%d  test=%d  (churn rate train=%.3f, test=%.3f)",
        len(train),
        len(test),
        y_train.mean(),
        y_test.mean(),
    )
    return train, test
