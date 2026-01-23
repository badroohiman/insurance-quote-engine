# src/features/runtime.py
"""
Runtime feature builder for API inference.

Goal:
- Convert a raw policy record (dict) into the engineered feature vector
  expected by the trained model (aligning to model.feature_columns).

Assumptions:
- Training produced a model artifact containing feature_columns.
- Your feature set includes:
  - numeric columns (subscription_length, vehicle_age, etc.)
  - binary indicators (is_esc, is_tpms, ...)
  - one-hot columns for categoricals (region_code, segment, model, fuel_type, engine_type, etc.)
  - parsed columns: torque_value, torque_rpm, power_value, power_rpm

This builder implements minimal transforms consistent with your EDA/clean/build decisions:
- Map Yes/No -> 1/0 for known boolean fields
- Parse max_torque/max_power strings into numeric components
- One-hot encode categoricals
- Align to training feature_columns
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Columns expected as raw inputs (policy_id optional; not used for modeling)
RAW_CATEGORICAL = [
    "region_code",
    "segment",
    "model",
    "fuel_type",
    "engine_type",
    "rear_brakes_type",
    "transmission_type",
    "steering_type",
]

# Raw Yes/No columns from your dataset (adjust if your build.py differs)
RAW_BOOLEAN_YN = [
    "is_esc",
    "is_adjustable_steering",
    "is_tpms",
    "is_parking_sensors",
    "is_parking_camera",
    "is_front_fog_lights",
    "is_rear_window_wiper",
    "is_rear_window_washer",
    "is_rear_window_defogger",
    "is_brake_assist",
    "is_power_door_locks",
    "is_central_locking",
    "is_power_steering",
    "is_driver_seat_height_adjustable",
    "is_day_night_rear_view_mirror",
    "is_ecw",
    "is_speed_alert",
]

# Raw numeric columns (keep permissive; we coerce numeric at runtime)
RAW_NUMERIC = [
    "subscription_length",
    "vehicle_age",
    "customer_age",
    "region_density",
    "airbags",
    "displacement",
    "cylinder",
    "turning_radius",
    "length",
    "width",
    "gross_weight",
    "ncap_rating",
]

# Torque/power raw text columns
RAW_TORQUE_COL = "max_torque"
RAW_POWER_COL = "max_power"


@dataclass(frozen=True)
class RuntimeBuildResult:
    features: pd.DataFrame
    warnings: List[str]


_YN_MAP = {
    "yes": 1,
    "no": 0,
    "y": 1,
    "n": 0,
    True: 1,
    False: 0,
    1: 1,
    0: 0,
}


def _to_int_yn(val: Any) -> Optional[int]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, str):
        key = val.strip().lower()
        if key == "string":  # swagger placeholder
            return None
        return _YN_MAP.get(key)
    return _YN_MAP.get(val)


def _parse_torque(s: Any) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse formats like:
      - "250Nm@2750rpm"
      - "113Nm@4400rpm"
    Returns: (torque_value_nm, torque_rpm)
    """
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None, None
    if not isinstance(s, str):
        s = str(s)

    text = s.strip().lower().replace(" ", "")
    m = re.search(r"([0-9]*\.?[0-9]+)nm@([0-9]*\.?[0-9]+)rpm", text)
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


def _parse_power(s: Any) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse formats like:
      - "100.6bhp@6000rpm"
      - "83.1bhp@5000rpm"
    Returns: (power_value_bhp, power_rpm)
    """
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None, None
    if not isinstance(s, str):
        s = str(s)

    text = s.strip().lower().replace(" ", "")
    m = re.search(r"([0-9]*\.?[0-9]+)bhp@([0-9]*\.?[0-9]+)rpm", text)
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


def build_features_from_raw(
    raw: Dict[str, Any],
    feature_columns: List[str],
) -> RuntimeBuildResult:
    """
    Build a single-row feature DataFrame aligned to feature_columns.

    raw: dict of raw policy fields (from API request)
    feature_columns: training columns from the model artifact
    """
    warnings: List[str] = []

    # Build base row
    row: Dict[str, Any] = {}

    # Copy numeric-ish fields
    for col in RAW_NUMERIC:
        if col in raw:
            row[col] = raw.get(col)

    # Copy categoricals
    for col in RAW_CATEGORICAL:
        if col in raw:
            row[col] = raw.get(col)

    # Map Yes/No to 1/0
    for col in RAW_BOOLEAN_YN:
        if col in raw:
            mapped = _to_int_yn(raw.get(col))
            if mapped is None and raw.get(col) is not None:
                warnings.append(f"Could not map {col}='{raw.get(col)}' to 0/1; set to NaN.")
            row[col] = mapped

    # Torque/power parsing
    torque_val, torque_rpm = _parse_torque(raw.get(RAW_TORQUE_COL))
    power_val, power_rpm = _parse_power(raw.get(RAW_POWER_COL))

    row["torque_value"] = torque_val
    row["torque_rpm"] = torque_rpm
    row["power_value"] = power_val
    row["power_rpm"] = power_rpm

    # Build DataFrame (single row)
    df = pd.DataFrame([row])

    # Ensure numeric coercion for numeric + parsed + boolean columns
    numeric_like = set(RAW_NUMERIC) | set(RAW_BOOLEAN_YN) | {"torque_value", "torque_rpm", "power_value", "power_rpm"}
    for col in df.columns:
        if col in numeric_like:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # One-hot encode categoricals present in df
    cat_present = [c for c in RAW_CATEGORICAL if c in df.columns]
    if cat_present:
        df = pd.get_dummies(df, columns=cat_present, dummy_na=False)

    # Align to training feature columns efficiently (missing -> 0, extras dropped)
    existing = set(df.columns)
    missing = [c for c in feature_columns if c not in existing]

    if missing:
        # Add all missing columns at once to avoid DataFrame fragmentation
        missing_df = pd.DataFrame(0, index=df.index, columns=missing)
        df = pd.concat([df, missing_df], axis=1)

    # Drop extras and reorder in one step; fill_value=0 for any remaining gaps
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Final cleanup
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    return RuntimeBuildResult(features=df, warnings=warnings)
