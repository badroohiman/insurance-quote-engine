# src/features/build.py
"""
Feature engineering for the Auto Insurance Claims dataset (EDA-aligned).

Inputs:
- data/processed/insurance_model_input.parquet  (from src.data.clean)
  - policy_id already removed
  - Yes/No columns already mapped to 0/1
  - target: claim_status (0/1)

What this step does:
- Parses max_torque / max_power strings into numeric components (value + rpm)
- One-hot encodes categorical columns (safe baseline)
- Creates stratified train/valid split (important due to ~6.4% positives)
- Writes feature datasets + a feature report

Outputs:
- data/processed/features_train.parquet
- data/processed/features_valid.parquet
- reports/feature_report.json

Usage:
  python -m src.features.build
  python -m src.features.build --in_path data/processed/insurance_model_input.parquet --test_size 0.2 --random_state 42
"""

from __future__ import annotations

import argparse
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.config import get_paths
from src.utils.io import ensure_dir, read_df, write_df, write_json

TARGET_COL = "claim_status"

# Common columns in your dataset (if present)
TORQUE_COL = "max_torque"
POWER_COL = "max_power"


@dataclass
class FeatureReport:
    dataset: str
    rows_in: int
    cols_in: int
    rows_train: int
    rows_valid: int
    target_positive_rate_all: float
    target_positive_rate_train: float
    target_positive_rate_valid: float
    categorical_cols_encoded: List[str]
    numeric_cols_used: List[str]
    parsed_columns_created: List[str]
    final_feature_count: int
    notes: List[str]


def _extract_value_and_rpm(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Extracts two numbers from strings like:
      "250Nm@2750rpm" -> value=250, rpm=2750
      "75PS@5000rpm"  -> value=75,  rpm=5000

    Returns:
      (value_series, rpm_series) as float
    """
    s = series.astype(str)

    # First number = value
    val = pd.to_numeric(s.str.extract(r"([0-9]+(?:\.[0-9]+)?)", expand=False), errors="coerce")

    # RPM number typically after '@'
    rpm = pd.to_numeric(s.str.extract(r"@([0-9]+(?:\.[0-9]+)?)", expand=False), errors="coerce")

    return val.astype("float64"), rpm.astype("float64")


def _parse_torque_power(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Adds parsed numeric columns if torque/power columns exist.
    Keeps original string columns for traceability (you can drop later if you want).
    """
    out = df.copy()
    created: List[str] = []
    notes: List[str] = []

    if TORQUE_COL in out.columns:
        v, r = _extract_value_and_rpm(out[TORQUE_COL])
        out["torque_value"] = v
        out["torque_rpm"] = r
        created += ["torque_value", "torque_rpm"]
        notes.append(f"Parsed {TORQUE_COL} -> torque_value, torque_rpm")

    if POWER_COL in out.columns:
        v, r = _extract_value_and_rpm(out[POWER_COL])
        out["power_value"] = v
        out["power_rpm"] = r
        created += ["power_value", "power_rpm"]
        notes.append(f"Parsed {POWER_COL} -> power_value, power_rpm")

    return out, created, notes


def _infer_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    feature_cols = [c for c in df.columns if c != TARGET_COL]

    cat_cols: List[str] = []
    num_cols: List[str] = []

    for c in feature_cols:
        dt = df[c].dtype

        # Treat pandas string/object/category as categorical
        if pd.api.types.is_object_dtype(dt) or pd.api.types.is_string_dtype(dt) or pd.api.types.is_categorical_dtype(dt):
            cat_cols.append(c)
        else:
            num_cols.append(c)

    return cat_cols, num_cols


def _stratified_split(
    df: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified split with sklearn if available; otherwise a simple stratified fallback.
    """
    try:
        from sklearn.model_selection import train_test_split  # type: ignore

        idx = np.arange(len(df))
        train_idx, valid_idx = train_test_split(
            idx,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        return df.iloc[train_idx].reset_index(drop=True), df.iloc[valid_idx].reset_index(drop=True)

    except ImportError:
        # Fallback: manual stratified split (deterministic)
        rng = np.random.default_rng(random_state)
        df0 = df[y == 0].copy()
        df1 = df[y == 1].copy()

        n0_valid = int(round(len(df0) * test_size))
        n1_valid = int(round(len(df1) * test_size))

        idx0 = rng.permutation(len(df0))
        idx1 = rng.permutation(len(df1))

        valid0 = df0.iloc[idx0[:n0_valid]]
        train0 = df0.iloc[idx0[n0_valid:]]
        valid1 = df1.iloc[idx1[:n1_valid]]
        train1 = df1.iloc[idx1[n1_valid:]]

        train = pd.concat([train0, train1], axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        valid = pd.concat([valid0, valid1], axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        return train, valid


def build_features(
    in_path: Path,
    test_size: float,
    random_state: int,
    drop_raw_torque_power: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, FeatureReport]:
    notes: List[str] = []

    df = read_df(in_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    rows_in, cols_in = df.shape

    # Ensure target type
    df[TARGET_COL] = df[TARGET_COL].astype("int64")

    # Parse torque/power strings into numeric features
    df_parsed, created_cols, parse_notes = _parse_torque_power(df)
    notes.extend(parse_notes)

    if drop_raw_torque_power:
        drop_cols = [c for c in [TORQUE_COL, POWER_COL] if c in df_parsed.columns]
        df_parsed = df_parsed.drop(columns=drop_cols)
        notes.append(f"Dropped raw columns: {drop_cols}")

    # Identify feature types
    cat_cols, num_cols = _infer_feature_types(df_parsed)

    for c in list(num_cols):
        if df_parsed[c].dtype == "object":
            num_cols.remove(c)
            cat_cols.append(c)

    # One-hot encode categoricals (baseline, robust)
    # Note: keep dummy_na=False because EDA indicated no missingness
    X = pd.get_dummies(df_parsed.drop(columns=[TARGET_COL]), columns=cat_cols, drop_first=False, dummy_na=False)
    y = df_parsed[TARGET_COL]

    # Combine back for saving (features + target)
    full = X.copy()
    full[TARGET_COL] = y.values

    # Stratified split
    train_df, valid_df = _stratified_split(full, y=y, test_size=test_size, random_state=random_state)

    # Target rates
    pos_all = float(y.mean())
    pos_train = float(train_df[TARGET_COL].mean())
    pos_valid = float(valid_df[TARGET_COL].mean())

    report = FeatureReport(
        dataset="auto_insurance_claims",
        rows_in=int(rows_in),
        cols_in=int(cols_in),
        rows_train=int(len(train_df)),
        rows_valid=int(len(valid_df)),
        target_positive_rate_all=pos_all,
        target_positive_rate_train=pos_train,
        target_positive_rate_valid=pos_valid,
        categorical_cols_encoded=cat_cols,
        numeric_cols_used=num_cols,
        parsed_columns_created=created_cols,
        final_feature_count=int(X.shape[1]),
        notes=notes,
    )

    return train_df, valid_df, report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build model features from cleaned insurance dataset.")
    p.add_argument(
        "--in_path",
        type=str,
        default="data/processed/insurance_model_input.parquet",
        help="Input cleaned model dataset path.",
    )
    p.add_argument(
        "--out_train",
        type=str,
        default="data/processed/features_train.parquet",
        help="Output path for training features parquet.",
    )
    p.add_argument(
        "--out_valid",
        type=str,
        default="data/processed/features_valid.parquet",
        help="Output path for validation features parquet.",
    )
    p.add_argument(
        "--report_path",
        type=str,
        default="reports/feature_report.json",
        help="Output path for feature report JSON.",
    )
    p.add_argument("--test_size", type=float, default=0.2, help="Validation split fraction.")
    p.add_argument("--random_state", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--drop_raw_torque_power",
        action="store_true",
        help="If set, drop original max_torque/max_power string columns after parsing.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = get_paths()

    in_path = paths.root / args.in_path
    out_train = paths.root / args.out_train
    out_valid = paths.root / args.out_valid
    report_path = paths.root / args.report_path

    ensure_dir(out_train.parent)
    ensure_dir(out_valid.parent)
    ensure_dir(report_path.parent)

    train_df, valid_df, report = build_features(
        in_path=in_path,
        test_size=args.test_size,
        random_state=args.random_state,
        drop_raw_torque_power=args.drop_raw_torque_power,
    )

    write_df(train_df, out_train)
    write_df(valid_df, out_valid)
    write_json(asdict(report), report_path)

    print(f"[OK] Train features saved: {out_train}")
    print(f"[OK] Valid features saved: {out_valid}")
    print(f"[OK] Feature report saved: {report_path}")
    print(
        f"Rows(train/valid)={report.rows_train}/{report.rows_valid} | "
        f"PosRate(all/train/valid)={report.target_positive_rate_all:.4f}/"
        f"{report.target_positive_rate_train:.4f}/{report.target_positive_rate_valid:.4f} | "
        f"Features={report.final_feature_count}"
    )


if __name__ == "__main__":
    main()
