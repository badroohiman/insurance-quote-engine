# src/api/app.py
"""
FastAPI service for the Insurance Quote Engine.

Endpoints:
- GET  /health
- POST /predict  -> returns p_claim
- POST /quote    -> returns p_claim + pricing quote

Runtime flow:
raw policy JSON -> runtime feature builder -> model.predict_proba -> quote
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.inference.predict import load_model
from src.features.runtime import build_features_from_raw
from src.pricing.quote import generate_quote
from src.pricing.config import PricingConfig
from typing import Optional, Union
from pydantic import BaseModel, Field


app = FastAPI(title="Insurance Quote Engine", version="0.1.0")

# Load model once at startup (good for API + container)
ARTIFACT = load_model()


class PolicyInput(BaseModel):
    # Core numeric
    subscription_length: Optional[float] = None
    vehicle_age: Optional[float] = None
    customer_age: Optional[int] = None
    region_density: Optional[int] = None
    airbags: Optional[int] = None
    displacement: Optional[int] = None
    cylinder: Optional[int] = None
    turning_radius: Optional[float] = None
    length: Optional[int] = None
    width: Optional[int] = None
    gross_weight: Optional[int] = None
    ncap_rating: Optional[int] = None

    # Categoricals
    region_code: Optional[str] = None
    segment: Optional[str] = None
    model: Optional[str] = None
    fuel_type: Optional[str] = None
    engine_type: Optional[str] = None
    rear_brakes_type: Optional[str] = None
    transmission_type: Optional[str] = None
    steering_type: Optional[str] = None

    # Torque/Power raw strings
    max_torque: Optional[str] = None
    max_power: Optional[str] = None

    # Yes/No flags (accept str or bool-ish)
    is_esc: Optional[Union[bool, str]] = None
    is_adjustable_steering: Optional[Union[bool, str]] = None
    is_tpms: Optional[Union[bool, str]] = None
    is_parking_sensors: Optional[Union[bool, str]] = None
    is_parking_camera: Optional[Union[bool, str]] = None
    is_front_fog_lights: Optional[Union[bool, str]] = None
    is_rear_window_wiper: Optional[Union[bool, str]] = None
    is_rear_window_washer: Optional[Union[bool, str]] = None
    is_rear_window_defogger: Optional[Union[bool, str]] = None
    is_brake_assist: Optional[Union[bool, str]] = None
    is_power_door_locks: Optional[Union[bool, str]] = None
    is_central_locking: Optional[Union[bool, str]] = None
    is_power_steering: Optional[Union[bool, str]] = None
    is_driver_seat_height_adjustable: Optional[Union[bool, str]] = None
    is_day_night_rear_view_mirror: Optional[Union[bool, str]] = None
    is_ecw: Optional[Union[bool, str]] = None
    is_speed_alert: Optional[Union[bool, str]] = None


class PredictResponse(BaseModel):
    model_name: str
    p_claim: float
    warnings: list[str] = Field(default_factory=list)


class QuoteRequest(BaseModel):
    policy: PolicyInput
    # Optional pricing overrides
    base_premium: Optional[float] = None
    risk_loading: Optional[float] = None
    min_premium: Optional[float] = None
    max_premium: Optional[float] = None
    tier_low: Optional[float] = None
    tier_high: Optional[float] = None
    currency: Optional[str] = None


class QuoteResponse(BaseModel):
    model_name: str
    p_claim: float
    warnings: list[str] = Field(default_factory=list)
    quote: Dict[str, Any]


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "model": ARTIFACT.model_name}


def _predict_from_policy_dict(policy_dict: Dict[str, Any]) -> PredictResponse:
    built = build_features_from_raw(policy_dict, ARTIFACT.feature_columns)

    # built.features is a 1-row DataFrame aligned to training columns
    p_claim = float(ARTIFACT.model.predict_proba(built.features)[:, 1][0])

    return PredictResponse(
        model_name=ARTIFACT.model_name,
        p_claim=p_claim,
        warnings=built.warnings,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(policy: PolicyInput) -> PredictResponse:
    return _predict_from_policy_dict(policy.model_dump())


@app.post("/quote", response_model=QuoteResponse)
def quote(req: QuoteRequest) -> QuoteResponse:
    pred = _predict_from_policy_dict(req.policy.model_dump())

    cfg = PricingConfig()
    # Override pricing config fields if user provided them
    cfg_dict = cfg.__dict__.copy()

    for k in ["currency", "base_premium", "risk_loading", "min_premium", "max_premium", "tier_low", "tier_high"]:
        v = getattr(req, k)
        if v is not None:
            cfg_dict[k] = v

    cfg = PricingConfig(**cfg_dict)  # type: ignore[arg-type]

    q = generate_quote(pred.p_claim, cfg=cfg)

    return QuoteResponse(
        model_name=pred.model_name,
        p_claim=pred.p_claim,
        warnings=pred.warnings,
        quote=q.to_dict(),
    )
