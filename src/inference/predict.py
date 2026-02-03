# src/inference/predict.py
"""
Inference utilities for claim-risk model.

Responsibilities:
- Load the trained model artifact (joblib) produced by src.train.train
- Align inference feature DataFrame to the exact training feature columns
- Return p_claim (probability of claim)

This module assumes that feature engineering has already been applied
(e.g., via src.features.build or an equivalent runtime feature builder).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.config import get_paths


@dataclass(frozen=True)
class ModelArtifact:
    model: Any
    model_name: str
    feature_columns: list[str]
    target: str


def load_model(model_path: Optional[str] = None) -> ModelArtifact:
    """
    Load model artifact saved by joblib.dump in src.train.train.

    If model_path is not provided:
      - Use MODEL_LOCAL_PATH (default: /tmp/claim_risk_model.joblib)
      - If file doesn't exist, download from MODEL_S3_URI
    """
    try:
        import joblib  # type: ignore
    except ImportError as e:
        raise ImportError("joblib is required for inference. Install with: pip install joblib") from e

    # --- NEW: S3-backed local path resolution ---
    from src.utils.model_store import (
        ensure_model_downloaded,
        get_aws_region,
        get_model_local_path,
        get_model_s3_uri,
    )

    if model_path:
        path = Path(model_path)
    else:
        local_path = get_model_local_path(default_path="/tmp/claim_risk_model.joblib")
        s3_uri = get_model_s3_uri()
        aws_region = get_aws_region()
        ensured = ensure_model_downloaded(model_s3_uri=s3_uri, local_path=local_path, aws_region=aws_region)
        path = Path(ensured)
    # ------------------------------------------

    obj: Dict[str, Any] = joblib.load(path)

    missing = [k for k in ["model", "model_name", "feature_columns", "target"] if k not in obj]
    if missing:
        raise KeyError(f"Model artifact missing keys: {missing}. Found keys: {list(obj.keys())}")

    return ModelArtifact(
        model=obj["model"],
        model_name=str(obj["model_name"]),
        feature_columns=list(obj["feature_columns"]),
        target=str(obj["target"]),
    )


def align_features(X: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """
    Align inference features to training columns:
    - add missing columns with 0
    - drop extra columns
    - reorder columns
    """
    X_aligned = X.copy()

    # Add missing with 0
    missing = [c for c in feature_columns if c not in X_aligned.columns]
    for c in missing:
        X_aligned[c] = 0

    # Drop extras
    extra = [c for c in X_aligned.columns if c not in feature_columns]
    if extra:
        X_aligned = X_aligned.drop(columns=extra)

    # Reorder
    X_aligned = X_aligned[feature_columns]

    # Ensure numeric
    for c in X_aligned.columns:
        if not pd.api.types.is_numeric_dtype(X_aligned[c].dtype):
            X_aligned[c] = pd.to_numeric(X_aligned[c], errors="coerce")

    # NaNs -> 0 (should be rare if pipeline is consistent)
    X_aligned = X_aligned.fillna(0)

    return X_aligned


def predict_proba_claim(
    X_features: pd.DataFrame,
    model_path: Optional[str] = None,
) -> Tuple[np.ndarray, ModelArtifact]:
    """
    Predict probability of claim for each row in X_features.

    Returns:
      (p_claim, artifact)
    """
    artifact = load_model(model_path=model_path)
    X_aligned = align_features(X_features, artifact.feature_columns)

    # predict_proba -> column 1 = P(y=1)
    p_claim = artifact.model.predict_proba(X_aligned)[:, 1]
    return p_claim.astype(float), artifact


def predict_single(
    features: Dict[str, Any],
    model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function for a single record (dict -> one-row DataFrame).
    """
    X = pd.DataFrame([features])
    p_claim, artifact = predict_proba_claim(X, model_path=model_path)
    return {
        "model_name": artifact.model_name,
        "p_claim": float(p_claim[0]),
    }
