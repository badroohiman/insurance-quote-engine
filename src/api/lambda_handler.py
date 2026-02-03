# src/api/lambda_handler.py
"""
AWS Lambda handler for FastAPI using Mangum (ASGI adapter).

How it works:
- API Gateway invokes Lambda
- Mangum translates the event into an ASGI request
- FastAPI handles routing (/health, /predict, /quote)
- Response is returned back to API Gateway

Model loading:
- We trigger get_artifact() at import time (cold start) so the model is ready.
- Service layer downloads model from S3 if needed (based on your entrypoint logic or service logic).
  If you rely on S3 download in service.py, make sure MODEL_S3_URI is set in Lambda env vars.
"""

from __future__ import annotations

import os

from mangum import Mangum

from src.api.app import app
from src.inference.service import get_artifact


# Warm up / pre-load model at cold start for lower first-request latency.
_PRELOAD_MODEL = os.getenv("PRELOAD_MODEL", "true").lower() in {"1", "true", "yes"}

if _PRELOAD_MODEL:
    # This will cache the model artifact in the service layer.
    # Ensure Lambda has permission to s3:GetObject for MODEL_S3_URI.
    get_artifact()


# Mangum handler
handler = Mangum(app)
