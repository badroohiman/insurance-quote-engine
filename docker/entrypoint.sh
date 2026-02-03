#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_LOCAL_PATH:=/app/artifacts/models/claim_risk_model.joblib}"

# Expected: MODEL_S3_URI like s3://my-bucket/insurance-quote-engine/models/claim_risk_model.joblib
if [[ -z "${MODEL_S3_URI:-}" ]]; then
  echo "[ERROR] MODEL_S3_URI is not set. Example: s3://bucket/prefix/models/claim_risk_model.joblib"
  exit 1
fi

if [[ ! -f "$MODEL_LOCAL_PATH" ]]; then
  echo "[INFO] Model not found at $MODEL_LOCAL_PATH. Downloading from $MODEL_S3_URI ..."
  python - << 'PY'
import os
from urllib.parse import urlparse

import boto3

uri = os.environ["MODEL_S3_URI"]
local_path = os.environ.get("MODEL_LOCAL_PATH", "/app/artifacts/models/claim_risk_model.joblib")

p = urlparse(uri)
if p.scheme != "s3":
    raise ValueError(f"MODEL_S3_URI must be s3://..., got: {uri}")

bucket = p.netloc
key = p.path.lstrip("/")

os.makedirs(os.path.dirname(local_path), exist_ok=True)

region = os.environ.get("AWS_REGION")
s3 = boto3.client("s3", region_name=region) if region else boto3.client("s3")

s3.download_file(bucket, key, local_path)
print(f"[OK] Downloaded model -> {local_path}")
PY
else
  echo "[INFO] Model already present at $MODEL_LOCAL_PATH"
fi

echo "[INFO] Starting API..."
exec uvicorn src.api.app:app --host 0.0.0.0 --port "${PORT:-8000}"
