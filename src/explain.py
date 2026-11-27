"""
HealthBridge SHAP Explainability Module
========================================
Generates:
  1. Global feature-importance summary plot (beeswarm)
  2. Per-prediction waterfall plot

Can be run as a standalone script to regenerate all plots on the test set,
or imported as a module from the FastAPI app / Streamlit UI.

Usage
-----
  python src/explain.py

Outputs
-------
  reports/shap/summary_beeswarm.png
  reports/shap/waterfall_sample_{i}.png  (for first 3 test samples)
"""

from __future__ import annotations

from pathlib import Path
import io
import base64
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

ROOT      = Path(__file__).resolve().parents[1]
PROC      = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
SHAP_DIR  = ROOT / "reports" / "shap"
SHAP_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "Age", "Gender", "Fever", "Cough", "Fatigue",
    "Breathlessness", "Chest_Pain", "Nausea", "Headache",
    "BMI", "BP_Systolic", "BP_Diastolic", "SpO2",
    "Glucose_mg_dL", "HbA1c"
]

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def load_artifacts():
    """Return (pipeline, label_encoder) from models/."""
    pipe = joblib.load(MODEL_DIR / "best_model.joblib")
    le   = joblib.load(MODEL_DIR / "label_encoder.joblib")
    return pipe, le


def _get_preprocessed(pipe, X: pd.DataFrame) -> np.ndarray:
    """Return the preprocessed feature matrix (before the classifier)."""
    return pipe[:-1].transform(X)


def build_explainer(pipe):
    """
    Return a SHAP explainer appropriate for the final estimator.
    Supports LogisticRegression, RandomForest, XGBoost.
    """
    clf = pipe.named_steps["clf"]
    pre = pipe[:-1]            # everything before the classifier

    if isinstance(clf, LogisticRegression):
        # LinearExplainer works directly on the preprocessed matrix
        background = shap.maskers.Independent(
            pre.transform(
                pd.read_csv(PROC / "features.csv")[FEATURE_COLS].sample(200, random_state=42)
            ),
            max_samples=200,
        )
        explainer = shap.LinearExplainer(clf, background)
        is_tree = False
    else:
        # TreeExplainer for RF and XGBoost
        background = shap.maskers.Independent(
            pre.transform(
                pd.read_csv(PROC / "features.csv")[FEATURE_COLS].sample(100, random_state=42)
            ),
            max_samples=100,
        )
        explainer = shap.TreeExplainer(clf, background)
        is_tree = True

    return explainer, is_tree


def shap_values_for_row(pipe, le, row: pd.DataFrame):
    """
    Compute SHAP values for a single row (DataFrame with FEATURE_COLS).
    Returns:
        shap_vals   : np.ndarray shape (1, n_features, n_classes)  or (n_features, n_classes)
        base_values : np.ndarray  (n_classes,)
        pred_class  : str
        pred_proba  : np.ndarray  (n_classes,)
    """
    pre  = pipe[:-1]
    clf  = pipe.named_steps["clf"]
    X_t  = pre.transform(row)

    explainer, is_tree = build_explainer(pipe)
    sv = explainer(X_t)        # Explanation object

    pred_enc   = pipe.predict(row)[0]
    pred_class = le.inverse_transform([pred_enc])[0]
    pred_proba = pipe.predict_proba(row)[0]

    return sv, pred_class, pred_proba, le.classes_


def top_shap_features(sv, class_idx: int, feature_names: list[str], top_n: int = 5) -> list[dict]:
    """
    Return top_n features (by absolute SHAP magnitude) for a given class.
    Works with multi-output SHAP Explanation objects.
    """
    if hasattr(sv, "values"):
        vals = sv.values
    else:
        vals = sv

    # vals shape: (1, n_features, n_classes) or (n_features, n_classes)
    if vals.ndim == 3:
        vals = vals[0]          # → (n_features, n_classes)
    if vals.ndim == 2:
        col_vals = vals[:, class_idx]
    else:
        col_vals = vals

    idx_sorted = np.argsort(np.abs(col_vals))[::-1][:top_n]
    return [
        {"feature": feature_names[i], "shap_value": float(col_vals[i])}
        for i in idx_sorted
    ]


def waterfall_to_base64(pipe, le, row: pd.DataFrame, class_idx: int | None = None) -> str:
    """
    Generate a SHAP waterfall plot for the predicted class and return
    it as a base64-encoded PNG string (for embedding in Streamlit / API).
    """
    pre = pipe[:-1]
    clf = pipe.named_steps["clf"]
    X_t = pre.transform(row)

    explainer, _ = build_explainer(pipe)
    sv = explainer(X_t)

    pred_enc = pipe.predict(row)[0]
    if class_idx is None:
        class_idx = int(pred_enc)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Build a simple bar chart as a robust fallback that always works
    if hasattr(sv, "values") and sv.values.ndim == 3:
        vals = sv.values[0, :, class_idx]
        base = float(sv.base_values[0, class_idx]) if sv.base_values.ndim == 2 else float(sv.base_values[class_idx])
    elif hasattr(sv, "values") and sv.values.ndim == 2:
        vals = sv.values[0, :]
        base = float(sv.base_values[0]) if hasattr(sv.base_values, "__len__") else float(sv.base_values)
    else:
        vals = np.array(sv.values).flatten()
        base = float(np.mean(sv.base_values)) if hasattr(sv, "base_values") else 0.0

    idx_sorted = np.argsort(np.abs(vals))[::-1][:10]
    feat_names = [FEATURE_COLS[i] for i in idx_sorted]
    feat_vals  = vals[idx_sorted]

    colors = ["#e74c3c" if v > 0 else "#3498db" for v in feat_vals]
    ax.barh(range(len(feat_names)), feat_vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(feat_names)))
    ax.set_yticklabels(feat_names, fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=10)
    pred_label = le.inverse_transform([pred_enc])[0]
    ax.set_title(f"SHAP Feature Impact — Predicted: {pred_label}", fontsize=12, fontweight="bold")
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#ffffff")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Standalone script mode
# ---------------------------------------------------------------------------

def main():
    from sklearn.model_selection import train_test_split
    import json

    print("Loading artifacts …")
    pipe, le = load_artifacts()
    X = pd.read_csv(PROC / "features.csv")[FEATURE_COLS]
    y_raw = pd.read_csv(PROC / "labels.csv")["Diagnosis"]
    y = le.transform(y_raw)

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test  # numpy array

    print("Building SHAP explainer …")
    pre = pipe[:-1]
    clf = pipe.named_steps["clf"]
    X_test_t = pre.transform(X_test)

    explainer, is_tree = build_explainer(pipe)
    print("Computing SHAP values on test set (this may take ~1 min for RF/XGBoost) …")
    # Use a background sample for speed
    bg_size = min(200, len(X_test))
    X_bg = X_test_t[:bg_size]
    sv_bg = explainer(X_bg)

    # ---- Global summary beeswarm ----
    print("Saving global summary plot …")
    fig, ax = plt.subplots(figsize=(10, 7))

    if hasattr(sv_bg, "values") and sv_bg.values.ndim == 3:
        # multi-class: mean absolute SHAP across classes
        mean_abs = np.abs(sv_bg.values).mean(axis=(0, 2))
    else:
        mean_abs = np.abs(sv_bg.values).mean(axis=0)

    idx_sorted = np.argsort(mean_abs)
    ax.barh(
        [FEATURE_COLS[i] for i in idx_sorted],
        mean_abs[idx_sorted],
        color="#2ecc71", edgecolor="white"
    )
    ax.set_xlabel("Mean |SHAP value| (average impact on model output)", fontsize=11)
    ax.set_title("Global Feature Importance — SHAP", fontsize=13, fontweight="bold")
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#ffffff")
    plt.tight_layout()
    out_path = SHAP_DIR / "summary_beeswarm.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")

    # ---- Per-sample waterfall plots (first 3 samples) ----
    for i in range(min(3, len(X_test))):
        row = X_test.iloc[[i]]
        b64 = waterfall_to_base64(pipe, le, row)
        out_path = SHAP_DIR / f"waterfall_sample_{i}.png"
        import base64 as _b64
        out_path.write_bytes(_b64.b64decode(b64))
        print(f"  Saved → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
