# Insurance Quote Engine — End-to-End ML System (AWS-Ready)

A **production-style machine learning system** that predicts insurance claim risk and generates dynamic insurance quotes.
Built to demonstrate **real-world ML engineering** (not notebook-only modeling): reproducible training, strict training–serving feature parity, and a live cloud API.

**Tech stack:** Python · XGBoost · FastAPI · Docker · AWS (Lambda container image, API Gateway HTTP API, S3, IAM)

---

## 🚀 Why This Project

This repository showcases how to take a model **from raw data to a live, cloud-deployable API**:

- Handles **severe class imbalance (~6% positive claims)**
- Uses **probability-based pricing** (risk loading), not hard classification
- Enforces **training–serving feature parity** via a runtime feature builder
- Deployed as a **containerised** FastAPI app on **AWS Lambda**, fronted by **API Gateway**

Target audience: **Data Scientist / Machine Learning Engineer** roles.

---

## 🔍 What It Does

1. Performs EDA, data cleaning, and quality checks on auto-insurance policy data
2. Engineers structured, binary, and parsed technical features (e.g., torque/power parsing)
3. Trains and evaluates models under class imbalance
4. Selects thresholds using business-driven constraints
5. Serves predictions and pricing via a production-ready API (FastAPI)

---

## 🧠 Modeling Highlights

- **Target:** insurance claim occurrence (`claim_status`, binary 0/1)
- **Class imbalance:** ~6% positives
- **Final model:** XGBoost with imbalance handling (e.g., `scale_pos_weight`)
- **Metrics:** ROC-AUC and PR-AUC (primary for imbalance)
- **Thresholding:** precision-constrained recall maximisation (business-friendly)

> Note: Performance depends on data/version and validation design. This repository emphasises *correct methodology* and *production deployment*, not leaderboard-style metrics.

---

## 🏗 Architecture

```
Raw Policy JSON
      ↓
Runtime Feature Builder (training–serving parity)
      ↓
XGBoost Risk Model → p_claim
      ↓
Pricing Engine (rules, tiering, clamps)
      ↓
API Response (/predict, /quote)
```

**Key design choices**
- Runtime features aligned to the exact training schema (add missing cols, drop extras, reorder)
- Safe handling of unseen categories at inference time
- Model artifact stored in S3 and loaded at runtime (warm starts reuse model in-memory)

---

## 🧩 Repository Structure

```
src/
├── api/          # FastAPI app + Lambda handler (Mangum)
├── features/     # Training & runtime feature builders
├── train/        # Model training & evaluation
├── inference/    # Model loading, feature alignment, prediction utilities
├── pricing/      # Pricing configuration + quote generation
├── utils/        # Config/helpers (paths, S3 model download, etc.)
```

---

## 🌐 Live API (AWS)

Swagger UI (OpenAPI):
- **https://fab0tvk8l2.execute-api.eu-west-2.amazonaws.com/docs**

Key endpoints:
- `GET /health`
- `POST /predict`
- `POST /quote`
- `GET /openapi.json`

---

## 📬 API Examples

### 1) Health

`GET /health`

**Response (example)**
```json
{"status":"ok","model":"<model_name>"}
```

### 2) Predict

`POST /predict`

**Request (example)**
```json
{
  "vehicle_age": 5,
  "customer_age": 35,
  "airbags": 2,
  "fuel_type": "Petrol",
  "max_torque": "200Nm@1750rpm"
}
```

**Response (example)**
```json
{
  "model_name": "<model_name>",
  "p_claim": 0.073,
  "warnings": []
}
```

### 3) Quote

`POST /quote`

**Request (example)**
```json
{
  "policy": {
    "vehicle_age": 3,
    "customer_age": 40,
    "airbags": 4,
    "max_torque": "250Nm@2750rpm"
  },
  "base_premium": 400.0,
  "risk_loading": 3.0
}
```

**Response (example)**
```json
{
  "model_name": "<model_name>",
  "p_claim": 0.057,
  "warnings": [],
  "quote": {
    "currency": "GBP",
    "p_claim": 0.057,
    "risk_tier": "medium",
    "base_premium": 400.0,
    "premium": 468.4,
    "risk_loading": 3.0,
    "notes": [
      "Quote computed from baseline premium and risk loading.",
      "Risk tier assigned using probability thresholds."
    ]
  }
}
```

---

## 🐳 Run Locally (Docker)

### Build
```bash
docker build -t insurance-quote-engine:local .
```

### Run
```bash
docker run --rm -p 8000:8000 \
  -e MODEL_S3_URI="s3://insurance-quote-engine-imans-models/models/claim_risk_model.joblib" \
  -e AWS_REGION="eu-west-2" \
  insurance-quote-engine:local
```

Then open:
- http://localhost:8000/docs

> If you prefer to run purely locally (no S3), set `MODEL_LOCAL_PATH` to a file path inside the container and mount the model artifact.

---

## ☁️ AWS Deployment (Container Image on Lambda + API Gateway)

### High-level steps
1. **Build a Lambda-compatible container image** (base: `public.ecr.aws/lambda/python:3.12`)
2. **Push image to ECR**
3. **Create Lambda function** from the container image
4. Configure **environment variables**
5. Grant Lambda execution role **S3 read access** to the model artifact
6. Create **API Gateway HTTP API** routes mapping to the Lambda function

### Required environment variables
- `MODEL_S3_URI`: `s3://insurance-quote-engine-imans-models/models/claim_risk_model.joblib`
- `MODEL_LOCAL_PATH`: `/tmp/claim_risk_model.joblib`
- `PRELOAD_MODEL`: `true` (optional)

> Lambda provides `AWS_REGION` automatically; do **not** set it as a Lambda env var (reserved by AWS).

### IAM (least privilege)
Lambda execution role needs:
- `s3:GetObject` on:
  - `arn:aws:s3:::insurance-quote-engine-imans-models/models/*`

---

## 💡 What This Demonstrates

✔ End-to-end ML ownership (EDA → training → deployment)  
✔ Imbalanced classification in a production-style system  
✔ Separation of ML inference and business pricing logic  
✔ Containerised, cloud-native deployment on AWS  
✔ Modular, testable codebase with clear boundaries

---

## 👤 Author

**Iman Badrooh** — Data Scientist / Machine Learning Engineer (UK)
