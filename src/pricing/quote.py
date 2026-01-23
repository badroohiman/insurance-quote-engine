# src/pricing/quote.py
"""
Pricing and quote generation.

Provides:
- risk tier assignment
- premium calculation
- quote output object

Notes:
- This pricing logic is deliberately simple and explainable.
- For a production insurer, this would be replaced by a pricing model / rating engine.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import numpy as np

from src.pricing.config import PricingConfig


@dataclass(frozen=True)
class QuoteResult:
    currency: str
    p_claim: float
    risk_tier: str
    base_premium: float
    premium: float
    risk_loading: float
    notes: list[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def assign_risk_tier(p_claim: float, cfg: PricingConfig) -> str:
    """
    Simple tiering based on probability thresholds.

    - Standard: p < tier_low
    - Medium  : tier_low <= p < tier_high
    - High    : p >= tier_high
    """
    if p_claim < cfg.tier_low:
        return "standard"
    if p_claim < cfg.tier_high:
        return "medium"
    return "high"


def compute_premium(p_claim: float, cfg: PricingConfig) -> float:
    """
    Premium calculation with clamping.

    premium = base_premium * (1 + risk_loading * p_claim)
    """
    raw = cfg.base_premium * (1.0 + cfg.risk_loading * float(p_claim))
    clamped = float(np.clip(raw, cfg.min_premium, cfg.max_premium))
    return clamped


def generate_quote(
    p_claim: float,
    cfg: Optional[PricingConfig] = None,
) -> QuoteResult:
    """
    Generate a quote from a predicted probability of claim.
    """
    cfg = cfg or PricingConfig()
    tier = assign_risk_tier(p_claim, cfg)
    premium = compute_premium(p_claim, cfg)

    notes = [
        "Quote computed from baseline premium and risk loading.",
        "Risk tier assigned using probability thresholds.",
    ]

    return QuoteResult(
        currency=cfg.currency,
        p_claim=float(p_claim),
        risk_tier=tier,
        base_premium=float(cfg.base_premium),
        premium=float(premium),
        risk_loading=float(cfg.risk_loading),
        notes=notes,
    )

