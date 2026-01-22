from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)

    if is_dataclass(obj):
        payload = asdict(obj)
    else:
        payload = obj

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def read_df(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataframe format: {suf}")


def write_df(df: pd.DataFrame, path: Union[str, Path]) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    suf = path.suffix.lower()
    if suf == ".csv":
        df.to_csv(path, index=False)
        return
    if suf == ".parquet":
        df.to_parquet(path, index=False)
        return
    raise ValueError(f"Unsupported dataframe format: {suf}")


# ---------------------------
# Optional S3 support
# ---------------------------
def _boto3_client(service: str, region: Optional[str] = None):
    try:
        import boto3  # type: ignore
    except ImportError as e:
        raise ImportError(
            "boto3 is required for S3 operations. Install with: pip install boto3"
        ) from e
    return boto3.client(service, region_name=region)


def s3_upload_file(
    local_path: Path, bucket: str, key: str, region: Optional[str] = None
) -> None:
    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")
    s3 = _boto3_client("s3", region=region)
    s3.upload_file(str(local_path), bucket, key)


def s3_download_file(
    bucket: str, key: str, local_path: Path, region: Optional[str] = None
) -> None:
    local_path = Path(local_path)
    ensure_dir(local_path.parent)
    s3 = _boto3_client("s3", region=region)
    s3.download_file(bucket, key, str(local_path))
