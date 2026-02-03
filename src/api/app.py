# src/api/app.py
"""
FastAPI service for the Insurance Quote Engine (thin API wrapper).

Endpoints:
- GET  /health
- POST /predict  -> returns p_claim (+ warnings)
- POST /quote    -> returns p_claim + quote (+ warnings)

The API layer stays thin:
- validates input
- calls src.inference.service
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.inference.service import get_artifact, predict_from_policy, quote_from_policy_dict


app = FastAPI(title="Insurance Quote Engine", version="0.1.0")


# Load model once at startup (better than module import-time for tests/reload)
@app.on_event("startup")
def _startup() -> None:
    get_artifact()  # caches model artifact


# -----------------------------
# Schemas
# -----------------------------
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

    # Yes/No flags (accept bool or strings)
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


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health() -> Dict[str, str]:
    art = get_artifact()
    return {"status": "ok", "model": art.model_name}


@app.post("/predict", response_model=PredictResponse)
def predict(policy: PolicyInput) -> PredictResponse:
    pred, warnings = predict_from_policy(policy.model_dump())
    return PredictResponse(model_name=pred.model_name, p_claim=pred.p_claim, warnings=warnings)


@app.post("/quote", response_model=QuoteResponse)
def quote(req: QuoteRequest) -> QuoteResponse:
    overrides = {
        "currency": req.currency,
        "base_premium": req.base_premium,
        "risk_loading": req.risk_loading,
        "min_premium": req.min_premium,
        "max_premium": req.max_premium,
        "tier_low": req.tier_low,
        "tier_high": req.tier_high,
    }

    out = quote_from_policy_dict(req.policy.model_dump(), pricing_overrides=overrides)

    # out is a dict; ensure warnings exists (service adds it)
    return QuoteResponse(
        model_name=str(out["model_name"]),
        p_claim=float(out["p_claim"]),
        warnings=list(out.get("warnings", [])),
        quote=dict(out["quote"]),
    )
