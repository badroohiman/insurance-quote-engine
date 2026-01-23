# src/pricing/config.py
"""
Pricing configuration.

This is intentionally simple and explainable for a portfolio quote engine:
- base_premium: baseline premium in GBP (or any currency)
- risk_loading: how aggressively we load premium as risk increases
- min/max premium clamps
- risk tier cutoffs (based on p_claim by default)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PricingConfig:
    currency: str = "GBP"

    # Baseline premium for a policy (can later be replaced by a GLM or rules engine)
    base_premium: float = 400.0

    # Premium = base_premium * (1 + risk_loading * p_claim)
    risk_loading: float = 3.0

    # Clamp final premium
    min_premium: float = 150.0
    max_premium: float = 2000.0

    # Tiering by probability thresholds (simple default)
    tier_low: float = 0.05
    tier_high: float = 0.12
