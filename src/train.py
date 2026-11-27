"""
HealthBridge Multi-Model Training Script
========================================
Runs three classifiers (Logistic Regression, Random Forest, XGBoost) with
MLflow experiment tracking, picks the best model by macro-F1 on the
validation set, and persists the winning pipeline + label encoder.

Usage
-----
  python src/train.py

Outputs
-------
  models/best_model.joblib        — winning sklearn Pipeline
  models/label_encoder.joblib     — fitted LabelEncoder
  reports/training_summary.json   — full metrics for all models
  reports/confusion_matrix_best.csv
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, top_k_accuracy_score, roc_auc_score
)
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parents[1]
PROC     = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
RPTS     = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
RPTS.mkdir(exist_ok=True)
mlflow.set_tracking_uri("sqlite:///mlflow.db")

FEATURE_COLS = [
    "Age", "Gender", "Fever", "Cough", "Fatigue",
    "Breathlessness", "Chest_Pain", "Nausea", "Headache",
    "BMI", "BP_Systolic", "BP_Diastolic", "SpO2",
    "Glucose_mg_dL", "HbA1c"
]
# ---------------------------------------------------------------------------


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ]), num_cols),
    ], remainder="drop")


def evaluate(pipe, X_part, y_part_enc, le, top_k=3):
    pred_enc = pipe.predict(X_part)
    pred     = le.inverse_transform(pred_enc)
    true     = le.inverse_transform(y_part_enc)
    proba    = pipe.predict_proba(X_part)
    classes  = pipe.named_steps["clf"].classes_

    metrics = {
        "accuracy":        float(accuracy_score(true, pred)),
        "macro_precision": float(precision_score(true, pred, average="macro", zero_division=0)),
        "macro_recall":    float(recall_score(true, pred, average="macro", zero_division=0)),
        "macro_f1":        float(f1_score(true, pred, average="macro", zero_division=0)),
    }
    if top_k is not None and top_k < len(le.classes_):
        metrics["top_3_accuracy"] = float(
            top_k_accuracy_score(y_part_enc, proba, k=top_k, labels=classes)
        )
        metrics["roc_auc_ovr_macro"] = float(
            roc_auc_score(y_part_enc, proba,
                          labels=classes, multi_class="ovr", average="macro")
        )
    return metrics, pred, true


def main():
    print("Loading processed data …")
    X = pd.read_csv(PROC / "features.csv")[FEATURE_COLS]
    y_raw = pd.read_csv(PROC / "labels.csv")["Diagnosis"]

    le = LabelEncoder()
    y  = le.fit_transform(y_raw)
    n_classes = len(le.classes_)
    print(f"  {len(X)} rows | {n_classes} classes | {X.shape[1]} features")

    # 70 / 15 / 15 stratified split
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    preprocessor = build_preprocessor(X_train)

    CANDIDATES = {
        "LogisticRegression": LogisticRegression(
            max_iter=5000, random_state=42, C=1.0, solver="lbfgs",
            multi_class="multinomial"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400, learning_rate=0.1, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, n_jobs=-1,
            num_class=n_classes, objective="multi:softprob",
        ),
    }

    mlflow.set_experiment("healthbridge_disease_prediction")
    summary = {}
    best_f1   = -1.0
    best_name = None
    best_pipe = None

    for name, clf in CANDIDATES.items():
        print(f"\nTraining {name} …")
        pipe = Pipeline([
            ("pre", preprocessor),
            ("clf", clf),
        ])
        with mlflow.start_run(run_name=name):
            pipe.fit(X_train, y_train)

            val_metrics,  _, _ = evaluate(pipe, X_val,  y_val,  le, top_k=None)
            test_metrics, pred, true = evaluate(pipe, X_test, y_test, le, top_k=3)

            mlflow.log_params(clf.get_params())
            mlflow.log_metrics({f"val_{k}":  v for k, v in val_metrics.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            mlflow.sklearn.log_model(
    pipe,
    artifact_path=name,
    skops_trusted_types=[
        "numpy.dtype",
        "sklearn.compose._column_transformer._RemainderColsList",
        "sklearn.tree._tree.Tree",
        "xgboost.core.Booster",
        "xgboost.sklearn.XGBClassifier"
    ]
)

            summary[name] = {"validation": val_metrics, "test": test_metrics}
            print(f"  Val macro-F1={val_metrics['macro_f1']:.4f}  "
                  f"Test acc={test_metrics['accuracy']:.4f}")

            if val_metrics["macro_f1"] > best_f1:
                best_f1   = val_metrics["macro_f1"]
                best_name = name
                best_pipe = pipe
                best_pred = pred
                best_true = true

    # -----------------------------------------------------------------------
    print(f"\nBest model: {best_name}  (val macro-F1={best_f1:.4f})")

    # Save artifacts
    joblib.dump(best_pipe, MODEL_DIR / "best_model.joblib")
    joblib.dump(le,        MODEL_DIR / "label_encoder.joblib")
    print(f"  Saved model → {MODEL_DIR / 'best_model.joblib'}")

    classes = le.classes_
    (RPTS / "classification_report_best.txt").write_text(
        classification_report(best_true, best_pred, zero_division=0), encoding="utf-8")
    pd.DataFrame(
        confusion_matrix(best_true, best_pred, labels=classes),
        index=classes, columns=classes
    ).to_csv(RPTS / "confusion_matrix_best.csv")

    summary["best_model"] = best_name
    summary["n_classes"]  = n_classes
    summary["classes"]    = classes.tolist()
    (RPTS / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  Saved reports → {RPTS}")


if __name__ == "__main__":
    main()
