# src/train/train.py
"""
Train and evaluate claim-risk models for the Insurance Quote Engine.

Inputs:
- data/processed/features_train.parquet
- data/processed/features_valid.parquet

Models:
- logreg: LogisticRegression with scaling + class_weight="balanced"
- gbdt : GradientBoostingClassifier (non-linear baseline)

Metrics:
- ROC-AUC
- PR-AUC (Average Precision)
- Confusion matrix + precision/recall/F1 at chosen threshold
- Optional: choose threshold to satisfy min_precision (maximize recall)

Outputs:
- artifacts/models/claim_risk_model.joblib
- reports/eval.json
- reports/model_card.md

Usage:
  python -m src.train.train --model logreg
  python -m src.train.train --model gbdt
  python -m src.train.train --model gbdt --min_precision 0.30
"""

from __future__ import annotations

import inspect

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.config import get_paths
from src.utils.io import ensure_dir, read_df, write_json
TARGET_COL = "claim_status"


@dataclass
class EvalReport:
    dataset: str
    model_name: str
    rows_train: int
    rows_valid: int
    pos_rate_train: float
    pos_rate_valid: float
    roc_auc: float
    pr_auc: float
    threshold: float
    confusion_matrix: Dict[str, int]
    precision: float
    recall: float
    f1: float
    notes: List[str]

def _fit_xgb_compat(
    clf: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    early_stopping_rounds: int,
    notes: List[str],
) -> None:
    """
    Fit XGBoost across multiple wrapper versions:
    - Some support early_stopping_rounds
    - Some support callbacks
    - Some support neither (train without early stopping)
    """
    sig = inspect.signature(clf.fit)
    params = sig.parameters

    fit_kwargs = {
        "eval_set": [(X_valid, y_valid)],
        "verbose": False,
    }

    if "early_stopping_rounds" in params:
        clf.fit(X_train, y_train, early_stopping_rounds=early_stopping_rounds, **fit_kwargs)
        notes.append(f"XGB early stopping via early_stopping_rounds={early_stopping_rounds}.")
        return

    if "callbacks" in params:
        from xgboost.callback import EarlyStopping  # type: ignore

        clf.fit(
            X_train,
            y_train,
            callbacks=[EarlyStopping(rounds=early_stopping_rounds, save_best=True)],
            **fit_kwargs,
        )
        notes.append(f"XGB early stopping via callbacks (rounds={early_stopping_rounds}).")
        return

    # Fallback: no early stopping supported
    clf.fit(X_train, y_train, **fit_kwargs)
    notes.append(
        "XGB wrapper does not support early stopping arguments; trained without early stopping. "
        "Consider lowering xgb_n_estimators or upgrading xgboost."
    )


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _prf(cm: Dict[str, int]) -> Tuple[float, float, float]:
    tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)


def _choose_threshold_for_min_precision(
    y_true: np.ndarray, y_prob: np.ndarray, min_precision: float
) -> float:
    """
    Choose the threshold that achieves precision >= min_precision and maximizes recall.
    If none satisfy, return 0.5.
    """
    thresholds = np.unique(y_prob)[::-1]  # descending

    best_thr = 0.5
    best_recall = -1.0

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        cm = _confusion(y_true, y_pred)
        precision, recall, _ = _prf(cm)
        if precision >= min_precision and recall > best_recall:
            best_recall = recall
            best_thr = float(thr)

    return best_thr


def _write_model_card(path: Path, report: EvalReport, feature_count: int) -> None:
    md = f"""# Model Card — {report.model_name}

## Overview
This model predicts **probability of insurance claim** (`{TARGET_COL}=1`) for an auto insurance policy.
It is designed as the risk-scoring component of an insurance quote engine.

## Data
- Dataset: {report.dataset}
- Train rows: {report.rows_train} (pos rate: {report.pos_rate_train:.4f})
- Valid rows: {report.rows_valid} (pos rate: {report.pos_rate_valid:.4f})
- Features: {feature_count} engineered features (one-hot + numeric + parsed torque/power).

## Evaluation (Validation)
- ROC-AUC: {report.roc_auc:.4f}
- PR-AUC (Average Precision): {report.pr_auc:.4f}
- Threshold: {report.threshold:.4f}

Confusion matrix (valid):
- TP: {report.confusion_matrix["tp"]}
- FP: {report.confusion_matrix["fp"]}
- TN: {report.confusion_matrix["tn"]}
- FN: {report.confusion_matrix["fn"]}

Precision: {report.precision:.4f}
Recall: {report.recall:.4f}
F1: {report.f1:.4f}

## Notes
{chr(10).join("- " + n for n in report.notes) if report.notes else "- None"}
"""
    path.write_text(md, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and evaluate claim-risk models.")
    p.add_argument("--train_path", type=str, default="data/processed/features_train.parquet")
    p.add_argument("--valid_path", type=str, default="data/processed/features_valid.parquet")
    p.add_argument("--model_out", type=str, default="artifacts/models/claim_risk_model.joblib")
    p.add_argument("--eval_out", type=str, default="reports/eval.json")
    p.add_argument("--model_card_out", type=str, default="reports/model_card.md")

    # Choose model
    p.add_argument("--model", type=str, default="logreg", choices=["logreg", "gbdt", "xgb"])

    # Thresholding
    p.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for classification metrics.")
    p.add_argument("--min_precision", type=float, default=None, help="Select threshold to achieve this precision.")
    p.add_argument("--random_state", type=int, default=42)

    # LogisticRegression only
    p.add_argument("--max_iter", type=int, default=5000)

    # GradientBoosting only
    p.add_argument("--n_estimators", type=int, default=300)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--max_depth", type=int, default=3)

    # XGBoost only
    p.add_argument("--xgb_n_estimators", type=int, default=2000)
    p.add_argument("--xgb_learning_rate", type=float, default=0.03)
    p.add_argument("--xgb_max_depth", type=int, default=4)
    p.add_argument("--xgb_subsample", type=float, default=0.8)
    p.add_argument("--xgb_colsample_bytree", type=float, default=0.8)
    p.add_argument("--xgb_min_child_weight", type=float, default=5.0)
    p.add_argument("--xgb_reg_lambda", type=float, default=1.0)
    p.add_argument("--early_stopping_rounds", type=int, default=100)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = get_paths()
    notes: List[str] = []

    train_path = paths.root / args.train_path
    valid_path = paths.root / args.valid_path
    model_out = paths.root / args.model_out
    eval_out = paths.root / args.eval_out
    model_card_out = paths.root / args.model_card_out

    ensure_dir(model_out.parent)
    ensure_dir(eval_out.parent)
    ensure_dir(model_card_out.parent)

    train_df = read_df(train_path)
    valid_df = read_df(valid_path)

    if TARGET_COL not in train_df.columns or TARGET_COL not in valid_df.columns:
        raise ValueError(f"Missing target column '{TARGET_COL}' in train/valid data.")

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL].astype(int).values

    X_valid = valid_df.drop(columns=[TARGET_COL])
    y_valid = valid_df[TARGET_COL].astype(int).values

    rows_train, rows_valid = len(train_df), len(valid_df)
    pos_rate_train = float(np.mean(y_train))
    pos_rate_valid = float(np.mean(y_valid))

    # Metrics
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score  # type: ignore
    except ImportError as e:
        raise ImportError("scikit-learn is required. Install with: pip install scikit-learn") from e

    # Train selected model
    if args.model == "logreg":
        from sklearn.pipeline import Pipeline  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore

        model_name = "logreg_scaled_balanced_v1"
        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=False)),
                ("model", LogisticRegression(
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=args.max_iter,
                    random_state=args.random_state,
                )),
            ]
        )
        notes.append("Scaled features with StandardScaler(with_mean=False).")
        notes.append("Used class_weight='balanced' for imbalance handling.")

    elif args.model == "xgb":
        try:
            from xgboost import XGBClassifier  # type: ignore
        except ImportError as e:
            raise ImportError("xgboost is required. Install with: pip install xgboost") from e

        model_name = "xgb_v1"

        n_pos = int(np.sum(y_train == 1))
        n_neg = int(np.sum(y_train == 0))
        spw = n_neg / max(n_pos, 1)

        clf = XGBClassifier(
            n_estimators=args.xgb_n_estimators,
            learning_rate=args.xgb_learning_rate,
            max_depth=args.xgb_max_depth,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample_bytree,
            min_child_weight=args.xgb_min_child_weight,
            reg_lambda=args.xgb_reg_lambda,
            objective="binary:logistic",
            scale_pos_weight=spw,
            eval_metric=["auc", "aucpr"],
            tree_method="hist",
            random_state=args.random_state,
            n_jobs=-1,
        )
        notes.append(f"XGBoost scale_pos_weight={spw:.3f} (neg/pos).")

        # Fit with compatibility wrapper (handles old/new xgboost APIs)
        _fit_xgb_compat(
            clf=clf,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            early_stopping_rounds=args.early_stopping_rounds,
            notes=notes,
        )

    else:
        # Gradient boosting (non-linear)
        from sklearn.ensemble import GradientBoostingClassifier  # type: ignore

        model_name = "gbdt_v1"
        clf = GradientBoostingClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            random_state=args.random_state,
        )
        notes.append(
            f"GradientBoostingClassifier(n_estimators={args.n_estimators}, "
            f"learning_rate={args.learning_rate}, max_depth={args.max_depth})."
        )
        notes.append("Note: GBDT does not use class_weight; imbalance handled via thresholding/metrics.")

    if args.model != "xgb":
        clf.fit(X_train, y_train)

    # Predict probabilities
    y_prob = clf.predict_proba(X_valid)[:, 1]

    roc_auc = float(roc_auc_score(y_valid, y_prob))
    pr_auc = float(average_precision_score(y_valid, y_prob))

    # Choose threshold
    threshold = float(args.threshold)
    if args.min_precision is not None:
        threshold = _choose_threshold_for_min_precision(y_valid, y_prob, float(args.min_precision))
        if threshold == 0.5:
            notes.append(f"No threshold achieved min_precision={args.min_precision:.3f}; fell back to 0.5.")
        else:
            notes.append(f"Threshold selected to satisfy min_precision={args.min_precision:.3f}.")

    y_pred = (y_prob >= threshold).astype(int)
    cm = _confusion(y_valid, y_pred)
    precision, recall, f1 = _prf(cm)

    report = EvalReport(
        dataset="auto_insurance_claims",
        model_name=model_name,
        rows_train=int(rows_train),
        rows_valid=int(rows_valid),
        pos_rate_train=pos_rate_train,
        pos_rate_valid=pos_rate_valid,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        threshold=threshold,
        confusion_matrix=cm,
        precision=precision,
        recall=recall,
        f1=f1,
        notes=notes,
    )

    # Save model
    try:
        import joblib  # type: ignore
    except ImportError as e:
        raise ImportError("joblib is required to save the model. Install with: pip install joblib") from e

    joblib.dump(
        {
            "model": clf,
            "model_name": model_name,
            "feature_columns": X_train.columns.tolist(),
            "target": TARGET_COL,
        },
        model_out,
    )

    # Save eval + model card
    write_json(asdict(report), eval_out)
    _write_model_card(model_card_out, report, feature_count=int(X_train.shape[1]))

    print(f"[OK] Model saved      : {model_out}")
    print(f"[OK] Eval saved       : {eval_out}")
    print(f"[OK] Model card saved  : {model_card_out}")
    print(f"Model={model_name} | Valid ROC-AUC={report.roc_auc:.4f} | PR-AUC={report.pr_auc:.4f}")
    print(
        f"Threshold={report.threshold:.4f} | "
        f"Precision={report.precision:.4f} Recall={report.recall:.4f} F1={report.f1:.4f} | "
        f"TP={cm['tp']} FP={cm['fp']} TN={cm['tn']} FN={cm['fn']}"
    )


if __name__ == "__main__":
    main()
