# src/inference/service.py
"""
End-to-end inference service for the Insurance Quote Engine.

Single source of truth:
- raw policy dict -> runtime feature builder -> p_claim
- p_claim -> pricing -> quote
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

from src.features.runtime import build_features_from_raw
from src.inference.predict import ModelArtifact, load_model
from src.inference.schemas import PredictResponse, QuoteResponse
from src.pricing.config import PricingConfig
from src.pricing.quote import generate_quote


# In-process cache (useful for FastAPI startup + AWS Lambda warm invocations)
_CACHED_ARTIFACT: Optional[ModelArtifact] = None


def get_artifact(model_path: Optional[str] = None, force_reload: bool = False) -> ModelArtifact:
    """
    Load and cache the model artifact saved by joblib.
    """
    global _CACHED_ARTIFACT
    if force_reload or _CACHED_ARTIFACT is None:
        _CACHED_ARTIFACT = load_model(model_path=model_path)
    return _CACHED_ARTIFACT


def predict_from_policy(
    policy: Dict[str, Any],
    *,
    model_path: Optional[str] = None,
    artifact: Optional[ModelArtifact] = None,
) -> Tuple[PredictResponse, list[str]]:
    """
    Predict p_claim from a raw policy dict using runtime feature builder.
    Returns (PredictResponse, warnings).
    """
    art = artifact or get_artifact(model_path=model_path)

    built = build_features_from_raw(policy, art.feature_columns)
    p_claim = float(art.model.predict_proba(built.features)[:, 1][0])

    pred = PredictResponse(model_name=art.model_name, p_claim=p_claim)
    return pred, built.warnings


def _merge_pricing_overrides(overrides: Dict[str, Any], base: PricingConfig) -> PricingConfig:
    """
    Apply user overrides to PricingConfig safely.
    Supported keys:
      currency, base_premium, risk_loading, min_premium, max_premium, tier_low, tier_high
    """
    cfg_dict = asdict(base)
    for k in ["currency", "base_premium", "risk_loading", "min_premium", "max_premium", "tier_low", "tier_high"]:
        v = overrides.get(k)
        if v is not None:
            cfg_dict[k] = v
    return PricingConfig(**cfg_dict)  # type: ignore[arg-type]


def quote_from_policy(
    policy: Dict[str, Any],
    *,
    model_path: Optional[str] = None,
    artifact: Optional[ModelArtifact] = None,
    pricing_overrides: Optional[Dict[str, Any]] = None,
    pricing_cfg: Optional[PricingConfig] = None,
) -> Tuple[QuoteResponse, list[str]]:
    """
    Full quote generation:
      raw policy -> p_claim -> pricing -> QuoteResponse
    Returns (QuoteResponse, warnings).
    """
    pred, warnings = predict_from_policy(policy, model_path=model_path, artifact=artifact)

    base_cfg = pricing_cfg or PricingConfig()
    cfg = _merge_pricing_overrides(pricing_overrides or {}, base_cfg)

    q = generate_quote(pred.p_claim, cfg=cfg)

    resp = QuoteResponse(
        model_name=pred.model_name,
        p_claim=float(pred.p_claim),
        quote=q.to_dict(),
    )
    return resp, warnings


def quote_from_policy_dict(
    policy: Dict[str, Any],
    *,
    model_path: Optional[str] = None,
    pricing_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convenience: returns a JSON-ready dict and includes warnings.
    """
    resp, warnings = quote_from_policy(policy, model_path=model_path, pricing_overrides=pricing_overrides)
    out = resp.to_dict()
    out["warnings"] = warnings
    return out
