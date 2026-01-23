# Model Card — xgb_v1

## Overview
This model predicts **probability of insurance claim** (`claim_status=1`) for an auto insurance policy.
It is designed as the risk-scoring component of an insurance quote engine.

## Data
- Dataset: auto_insurance_claims
- Train rows: 46873 (pos rate: 0.0640)
- Valid rows: 11719 (pos rate: 0.0640)
- Features: 93 engineered features (one-hot + numeric + parsed torque/power).

## Evaluation (Validation)
- ROC-AUC: 0.6508
- PR-AUC (Average Precision): 0.1015
- Threshold: 0.5000

Confusion matrix (valid):
- TP: 469
- FP: 4391
- TN: 6578
- FN: 281

Precision: 0.0965
Recall: 0.6253
F1: 0.1672

## Notes
- XGBoost scale_pos_weight=14.635 (neg/pos).
- XGB wrapper does not support early stopping arguments; trained without early stopping. Consider lowering xgb_n_estimators or upgrading xgboost.
