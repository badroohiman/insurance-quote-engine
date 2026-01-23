# src/inference/schemas.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class PredictResponse:
    model_name: str
    p_claim: float

    def to_dict(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "p_claim": self.p_claim}


@dataclass(frozen=True)
class QuoteResponse:
    model_name: str
    p_claim: float
    quote: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "p_claim": self.p_claim, "quote": self.quote}
