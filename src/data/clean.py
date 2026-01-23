# src/data/clean.py
"""
Clean and standardise the Auto Insurance dataset based on EDA decisions.

EDA-derived decisions implemented:
- Target column: claim_status (binary 0/1)
- Identifier: policy_id (unique) -> excluded from modeling dataset
- Yes/No columns -> mapped to 1/0 (kept as int)
- No missingness observed -> no imputation
- Torque/Power parsing deferred to feature engineering (kept as strings here)
- Drop duplicate rows (safety)

Outputs:
1) Clean dataset (keeps policy_id): data/processed/insurance_clean.parquet
2) Model input dataset (drops policy_id, binary mapped): data/processed/insurance_model_input.parquet
3) Data quality report: reports/data_quality.json

Usage:
  python -m src.data.clean --in_path data/raw/Insurance_claims.parquet
  python -m src.data.clean --in_path data/raw/Insurance_claims.csv

If --in_path is omitted, the script will try to find a single .parquet/.csv in data/raw.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.utils.config import get_paths
from src.utils.io import ensure_dir, read_df, write_df, write_json


YES_NO = {"yes", "no"}
TARGET_COL = "claim_status"
ID_COL = "policy_id"


@dataclass
class DataQualityReport:
    dataset: str
    rows_in: int
    cols_in: int
    rows_out_clean: int
    cols_out_clean: int
    rows_out_model: int
    cols_out_model: int
    duplicate_rows_removed: int
    target_positive_rate: float
    target_counts: Dict[str, int]
    yes_no_columns: List[str]
    dominant_binary_columns: List[Dict[str, Any]]
    notes: List[str]


def _auto_find_input(raw_dir: Path) -> Path:
    """Find a single dataset file in data/raw. Prefer parquet over csv."""
    parquets = sorted(raw_dir.glob("*.parquet"))
    csvs = sorted(raw_dir.glob("*.csv"))

    if len(parquets) == 1:
        return parquets[0]
    if len(parquets) > 1:
        # Prefer a file that looks like insurance claims
        for p in parquets:
            if "insurance" in p.name.lower() or "claims" in p.name.lower():
                return p
        return parquets[0]

    if len(csvs) == 1:
        return csvs[0]
    if len(csvs) > 1:
        for p in csvs:
            if "insurance" in p.name.lower() or "claims" in p.name.lower():
                return p
        return csvs[0]

    raise FileNotFoundError(f"No .parquet or .csv found in {raw_dir}")


def _strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and normalise string casing."""
    out = df.copy()
    text_cols = out.select_dtypes(include=["string", "object"]).columns

    for c in text_cols:
        s = out[c].astype("string")
        out[c] = s.str.strip().str.lower()

    return out
def _infer_yes_no_cols(df: pd.DataFrame) -> List[str]:
    """Identify columns whose non-null unique values are subset of {'Yes','No'}."""
    yes_no_cols: List[str] = []
    for c in df.columns:
        if c in (TARGET_COL, ID_COL):
            continue
        if not pd.api.types.is_string_dtype(df[c]):
            continue
        uniq = set(df[c].dropna().unique().tolist())
        if len(uniq) > 0 and uniq.issubset(YES_NO):
            yes_no_cols.append(c)
    return yes_no_cols


def _map_yes_no_to_int(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    mapping = {"yes": 1, "no": 0}
    for c in cols:
        out[c] = out[c].map(mapping).astype("int64")
    return out


def _validate_schema(df: pd.DataFrame) -> None:
    missing_required = [c for c in [TARGET_COL, ID_COL] if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    # Validate target domain (allow int or bool-like)
    bad = df[~df[TARGET_COL].isin([0, 1])][TARGET_COL]
    if len(bad) > 0:
        examples = bad.head(10).tolist()
        raise ValueError(f"{TARGET_COL} must be binary 0/1. Found invalid examples: {examples}")


def _dominance_report_for_binary(df: pd.DataFrame, yes_no_cols: List[str], top_k: int = 15) -> List[Dict[str, Any]]:
    """
    Report highly dominant binary features (useful for later feature review).
    Returns list of dicts sorted by dominance.
    """
    rows = len(df)
    items: List[Dict[str, Any]] = []
    for c in yes_no_cols:
        vc = df[c].value_counts(dropna=False)
        if vc.empty:
            continue
        top_val = int(vc.index[0])
        top_count = int(vc.iloc[0])
        top_pct = float(top_count / rows) if rows else 0.0
        items.append({"column": c, "top_value": top_val, "top_pct": top_pct})
    items.sort(key=lambda d: d["top_pct"], reverse=True)
    return items[:top_k]


def clean(in_path: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame, DataQualityReport]:
    paths = get_paths()
    notes: List[str] = []

    if in_path is None:
        in_path = _auto_find_input(paths.raw_dir)
        notes.append(f"Auto-selected input: {in_path}")

    df = read_df(in_path)
    rows_in, cols_in = df.shape

    # Basic string cleanup
    df = _strip_strings(df)

    # Drop duplicates
    before = len(df)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    dup_removed = before - len(df_clean)
    if dup_removed:
        notes.append(f"Removed {dup_removed} duplicate rows.")

    # Validate required columns and target
    _validate_schema(df_clean)

    # Ensure target is int 0/1
    df_clean[TARGET_COL] = df_clean[TARGET_COL].astype("int64")

    # Identify and map Yes/No columns
    yes_no_cols = _infer_yes_no_cols(df_clean)
    df_clean_mapped = _map_yes_no_to_int(df_clean, yes_no_cols)

    # Model input: drop identifier
    df_model = df_clean_mapped.drop(columns=[ID_COL]).copy()

    # Target stats
    target_counts = df_clean[TARGET_COL].value_counts().to_dict()
    pos_rate = float(df_clean[TARGET_COL].mean())

    # Dominance report for binary cols (post mapping)
    dom = _dominance_report_for_binary(df_clean_mapped, yes_no_cols, top_k=15)

    report = DataQualityReport(
        dataset="auto_insurance_claims",
        rows_in=int(rows_in),
        cols_in=int(cols_in),
        rows_out_clean=int(df_clean.shape[0]),
        cols_out_clean=int(df_clean.shape[1]),
        rows_out_model=int(df_model.shape[0]),
        cols_out_model=int(df_model.shape[1]),
        duplicate_rows_removed=int(dup_removed),
        target_positive_rate=pos_rate,
        target_counts={str(k): int(v) for k, v in target_counts.items()},
        yes_no_columns=yes_no_cols,
        dominant_binary_columns=dom,
        notes=notes,
    )

    return df_clean, df_model, report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean and standardise the auto insurance dataset (EDA-aligned).")
    p.add_argument(
        "--in_path",
        type=str,
        default=None,
        help="Input dataset path (.csv or .parquet). If omitted, auto-detect in data/raw/.",
    )
    p.add_argument(
        "--out_clean",
        type=str,
        default="data/processed/insurance_clean.parquet",
        help="Output path for cleaned dataset (keeps policy_id).",
    )
    p.add_argument(
        "--out_model",
        type=str,
        default="data/processed/insurance_model_input.parquet",
        help="Output path for model input dataset (drops policy_id, Yes/No->0/1).",
    )
    p.add_argument(
        "--report_path",
        type=str,
        default="reports/data_quality.json",
        help="Output path for data quality report JSON.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = get_paths()

    in_path = Path(args.in_path) if args.in_path else None
    out_clean = paths.root / args.out_clean
    out_model = paths.root / args.out_model
    report_path = paths.root / args.report_path

    ensure_dir(out_clean.parent)
    ensure_dir(out_model.parent)
    ensure_dir(report_path.parent)

    df_clean, df_model, report = clean(in_path=in_path)

    write_df(df_clean, out_clean)
    write_df(df_model, out_model)
    write_json(asdict(report), report_path)

    print(f"[OK] Clean saved      : {out_clean}")
    print(f"[OK] Model input saved: {out_model}")
    print(f"[OK] Report saved     : {report_path}")
    print(f"Rows: {report.rows_in} -> {report.rows_out_clean} | PosRate={report.target_positive_rate:.4f}")
    if report.duplicate_rows_removed:
        print(f"Duplicates removed: {report.duplicate_rows_removed}")
    print(f"Yes/No cols mapped: {len(report.yes_no_columns)}")


if __name__ == "__main__":
    main()
