# Insurance Quote Engine — End-to-End ML System (AWS-Ready)

A **production-style machine learning system** that predicts insurance claim risk and generates dynamic insurance quotes.
Built to demonstrate **real-world ML engineering**, not notebook-only modeling.

**Tech stack:** Python · XGBoost · FastAPI · Docker · AWS (SAM, Lambda, Step Functions)

---

## 🚀 Why This Project

This repository showcases how to take a model **from raw data to a live, cloud-deployable API**:

* Handles **severe class imbalance (~6% positive claims)**
* Uses **probability-based pricing**, not hard classification
* Enforces **training–serving feature parity**
* Designed for **AWS Lambda + API Gateway** deployment

Target audience: **Data Scientist / Machine Learning Engineer** roles.

---

## 🔍 What It Does

1. Performs EDA, data cleaning, and quality checks on auto-insurance policy data
2. Engineers structured, binary, and parsed technical features
3. Trains and evaluates multiple models under class imbalance
4. Selects thresholds using **business-driven precision constraints**
5. Serves predictions and pricing via a production-ready API
6. Runs locally (Docker) or in the cloud (AWS SAM)

---

## 🧠 Modeling Highlights

* **Target:** Insurance claim occurrence (`claim_status`)
* **Class imbalance:** ~6% positives
* **Final model:** XGBoost (`scale_pos_weight` applied)
* **Metrics:** ROC-AUC and PR-AUC (primary)
* **Thresholding:** Precision-constrained recall maximisation

**Typical validation performance:**

* ROC-AUC ≈ **0.65**
* PR-AUC ≈ **0.10**
* Recall ≈ **63%** at default threshold

These metrics reflect realistic expectations for imbalanced insurance risk data.

---

## 🏗 Architecture

```
Raw Policy JSON
      ↓
Runtime Feature Builder
      ↓
XGBoost Risk Model
      ↓
Pricing Engine
      ↓
API Response (Quote)
```

**Key design choices:**

* Runtime features exactly aligned with training schema
* Safe handling of unseen categories at inference time
* Model loaded once per container (warm inference)

---

## 🧩 Repository Structure

```
src/
├── api/          # FastAPI endpoints (predict / quote)
├── features/     # Training & runtime feature builders
├── train/        # Model training & evaluation
├── inference/    # Model loading & prediction
├── pricing/      # Business pricing logic
├── lambda/       # AWS Lambda handlers (SAM)
```

---

## 🌐 API Example

**POST /quote**

```json
{
  "p_claim": 0.57,
  "premium": 1085.4,
  "risk_tier": "HIGH",
  "currency": "GBP"
}
```

Interactive API documentation available at:

```
/docs
```

---

## 🐳 Run Locally (Docker)

```bash
docker build -t insurance-quote-engine .
docker run -p 8001:8000 insurance-quote-engine
```

Then open:

```
http://localhost:8001/docs
```

---

## ☁️ AWS Deployment (SAM)

The project includes AWS SAM templates for:

* API Gateway
* Lambda (Inference + Pricing)
* Step Functions (end-to-end quote workflow)
* S3-hosted model artifacts

```bash
sam build
sam deploy --guided
```

---

## 💡 What This Demonstrates

✔ End-to-end ML ownership
✔ Imbalanced classification in production
✔ ML–business logic separation
✔ Cloud-native deployment readiness
✔ Clean, modular, testable codebase

---

## 👤 Author

**Iman Badrooh**
Data Scientist / Machine Learning Engineer (UK)
