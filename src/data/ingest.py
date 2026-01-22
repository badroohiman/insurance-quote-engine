# src/data/ingest.py
"""
Ingest Auto Insurance Claims raw data into a canonical, reproducible "raw" artifact.

What it does:
- Reads a source file (CSV or Parquet)
- Writes a canonical raw Parquet to data/raw/
- Writes an ingest manifest JSON to reports/ (row/col counts, dtypes, hash, sizes)
- Optionally uploads the canonical raw artifact + manifest to S3 (if S3_BUCKET is set)

Usage:
  python -m src.data.ingest --in_path data/raw/insurance_claims.csv

Optional:
  python -m src.data.ingest --in_path data/raw/insurance_claims.csv \
    --out_path data/raw/insurance_claims.parquet \
    --manifest_path reports/ingest_manifest.json \
    --upload_s3

Env (optional for S3):
  AWS_REGION=eu-west-2
  S3_BUCKET=your-bucket
  S3_PREFIX=insurance-quote-engine
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.utils.config import get_aws_config, get_paths
from src.utils.io import read_df, sha256_file, s3_upload_file, write_df, write_json


@dataclass
class IngestManifest:
    dataset: str
    source_path: str
    canonical_path: str
    created_utc: str
    rows: int
    cols: int
    columns: list[str]
    dtypes: Dict[str, str]
    file_size_bytes: int
    sha256: str
    notes: list[str]


def _utc_now_iso() -> str:
    # Avoid external deps; ISO-like UTC timestamp
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _default_out_path(in_path: Path) -> Path:
    """
    Canonical raw artifact: always parquet, placed in data/raw/.
    """
    paths = get_paths()
    stem = in_path.stem
    return paths.raw_dir / f"{stem}.parquet"


def _default_manifest_path() -> Path:
    paths = get_paths()
    return paths.reports_dir / "ingest_manifest.json"


def build_manifest(
    df: pd.DataFrame,
    source_path: Path,
    canonical_path: Path,
    sha: str,
    file_size: int,
    notes: Optional[list[str]] = None,
) -> IngestManifest:
    if notes is None:
        notes = []

    dtypes = {c: str(df[c].dtype) for c in df.columns}

    return IngestManifest(
        dataset="auto_insurance_claims",
        source_path=str(source_path),
        canonical_path=str(canonical_path),
        created_utc=_utc_now_iso(),
        rows=int(df.shape[0]),
        cols=int(df.shape[1]),
        columns=[str(c) for c in df.columns],
        dtypes=dtypes,
        file_size_bytes=int(file_size),
        sha256=sha,
        notes=notes,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest raw insurance dataset into canonical parquet + manifest.")
    p.add_argument("--in_path", type=str, required=True, help="Input CSV/Parquet path (e.g., data/raw/insurance_claims.csv)")
    p.add_argument("--out_path", type=str, default=None, help="Canonical output path (.parquet). Default: data/raw/<stem>.parquet")
    p.add_argument("--manifest_path", type=str, default=None, help="Manifest JSON path. Default: reports/ingest_manifest.json")
    p.add_argument("--upload_s3", action="store_true", help="Upload canonical artifact + manifest to S3 (requires env S3_BUCKET)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = get_paths()
    aws = get_aws_config()

    in_path = Path(args.in_path)

    if args.out_path is None:
        out_path = _default_out_path(in_path)
    else:
        out_path = Path(args.out_path)

    if out_path.suffix.lower() != ".parquet":
        raise ValueError("out_path must end with .parquet (canonical raw artifact should be parquet).")

    manifest_path = Path(args.manifest_path) if args.manifest_path else _default_manifest_path()

    # Ensure dirs exist
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Read source data
    df = read_df(in_path)

    notes: list[str] = []
    if df.empty:
        notes.append("WARNING: Input dataframe is empty.")

    # Write canonical raw parquet
    write_df(df, out_path)

    # Compute artifact metadata
    file_size = out_path.stat().st_size
    sha = sha256_file(out_path)

    manifest = build_manifest(
        df=df,
        source_path=in_path,
        canonical_path=out_path,
        sha=sha,
        file_size=file_size,
        notes=notes,
    )

    write_json(manifest, manifest_path)

    print(f"[OK] Ingested source       : {in_path}")
    print(f"[OK] Canonical raw saved  : {out_path}")
    print(f"[OK] Manifest saved       : {manifest_path}")
    print(f"Rows: {manifest.rows} | Cols: {manifest.cols} | SHA256: {manifest.sha256[:12]}...")

    # Optional: upload to S3
    if args.upload_s3:
        if not aws.enabled:
            raise RuntimeError("S3 upload requested but S3_BUCKET is not set in environment.")
        bucket = aws.s3_bucket  # type: ignore[assignment]
        prefix = aws.s3_prefix.rstrip("/")

        # Key layout:
        # s3://<bucket>/<prefix>/raw/<filename>
        # s3://<bucket>/<prefix>/manifests/ingest_manifest.json
        raw_key = f"{prefix}/raw/{out_path.name}"
        manifest_key = f"{prefix}/manifests/{manifest_path.name}"

        s3_upload_file(out_path, bucket=bucket, key=raw_key, region=aws.region)
        s3_upload_file(manifest_path, bucket=bucket, key=manifest_key, region=aws.region)

        print(f"[OK] Uploaded raw to S3   : s3://{bucket}/{raw_key}")
        print(f"[OK] Uploaded manifest    : s3://{bucket}/{manifest_key}")


if __name__ == "__main__":
    main()
