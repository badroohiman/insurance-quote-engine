# src/utils/config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(key)
    return v if v is not None and v != "" else default


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    reports_dir: Path
    artifacts_dir: Path
    models_dir: Path


def get_project_root() -> Path:
    """
    Resolve repo root robustly.
    Assumes this file lives at: <root>/src/utils/config.py
    """
    return Path(__file__).resolve().parents[2]


def get_paths() -> ProjectPaths:
    root = get_project_root()
    data_dir = root / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    reports_dir = root / "reports"
    artifacts_dir = root / "artifacts"
    models_dir = artifacts_dir / "models"
    return ProjectPaths(
        root=root,
        data_dir=data_dir,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        artifacts_dir=artifacts_dir,
        models_dir=models_dir,
    )


@dataclass(frozen=True)
class AwsConfig:
    region: str
    s3_bucket: Optional[str]
    s3_prefix: str

    @property
    def enabled(self) -> bool:
        return self.s3_bucket is not None


def get_aws_config() -> AwsConfig:
    """
    Configure S3 usage via environment variables.
    Keep it optional so local runs are frictionless.

    Env:
      AWS_REGION (default: eu-west-2)
      S3_BUCKET  (optional)
      S3_PREFIX  (default: insurance-quote-engine)
    """
    return AwsConfig(
        region=_env("AWS_REGION", "eu-west-2") or "eu-west-2",
        s3_bucket=_env("S3_BUCKET", None),
        s3_prefix=_env("S3_PREFIX", "insurance-quote-engine")
        or "insurance-quote-engine",
    )
