# src/utils/model_store.py
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

import boto3


def ensure_model_downloaded(*, model_s3_uri: str, local_path: str, aws_region: str | None = None) -> str:
    """
    Ensure model exists at local_path. If not, download from S3.
    Returns local_path.
    """
    lp = Path(local_path)
    if lp.exists() and lp.stat().st_size > 0:
        return str(lp)

    p = urlparse(model_s3_uri)
    if p.scheme != "s3":
        raise ValueError(f"MODEL_S3_URI must be s3://..., got: {model_s3_uri}")

    bucket = p.netloc
    key = p.path.lstrip("/")

    lp.parent.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3", region_name=aws_region) if aws_region else boto3.client("s3")
    s3.download_file(bucket, key, str(lp))
    return str(lp)


def get_model_local_path(default_path: str = "/tmp/claim_risk_model.joblib") -> str:
    """
    Read MODEL_LOCAL_PATH from env (recommended for Lambda).
    """
    return os.getenv("MODEL_LOCAL_PATH", default_path)


def get_model_s3_uri() -> str:
    """
    Read MODEL_S3_URI from env (required for S3-backed model).
    """
    uri = os.getenv("MODEL_S3_URI", "").strip()
    if not uri:
        raise ValueError("MODEL_S3_URI is not set. Example: s3://bucket/path/model.joblib")
    return uri


def get_aws_region() -> str | None:
    """
    Prefer AWS_REGION env var (works in Lambda).
    """
    return os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
