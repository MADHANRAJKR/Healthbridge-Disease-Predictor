"""
HealthBridge FastAPI Backend
=============================
Endpoints
---------
  GET  /                 — health check
  GET  /diseases         — list all supported disease classes
  POST /predict          — predict disease from symptoms + return SHAP explanation
  GET  /docs             — Swagger UI (auto-generated)
  GET  /redoc            — ReDoc UI

Run
---
  uvicorn api.main:app --reload --port 8000
  (from repo root)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from src/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import SymptomInput, PredictionResponse, DiseaseProba, ShapFeature
from explain import (
    load_artifacts, top_shap_features, build_explainer,
    waterfall_to_base64, FEATURE_COLS
)

# ---------------------------------------------------------------------------
# App state — loaded once at startup
# ---------------------------------------------------------------------------
_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model artifacts …")
    pipe, le = load_artifacts()
    _state["pipe"] = pipe
    _state["le"]   = le
    _state["diseases"] = sorted(le.classes_.tolist())
    print(f"  Loaded model: {type(pipe.named_steps['clf']).__name__}")
    print(f"  Disease classes ({len(_state['diseases'])}): {_state['diseases'][:5]} …")
    yield
    _state.clear()


# ---------------------------------------------------------------------------
app = FastAPI(
    title="HealthBridge Disease Prediction API",
    description=(
        "AI-powered symptom-to-disease predictor with SHAP explainability. "
        "Built for HealthBridge Hospitals — educational prototype only."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def health_check():
    """API liveness probe."""
    return {
        "status": "ok",
        "service": "HealthBridge Disease Prediction API",
        "model": type(_state.get("pipe", {}).named_steps.get("clf")).__name__
                  if _state else "not loaded",
        "n_diseases": len(_state.get("diseases", [])),
    }


@app.get("/diseases", tags=["Info"])
def list_diseases():
    """Return the list of all disease classes the model can predict."""
    return {
        "n_diseases": len(_state["diseases"]),
        "diseases":   _state["diseases"],
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(payload: SymptomInput):
    """
    Predict the most likely disease and return:
    - Top-3 predicted diseases with probabilities
    - SHAP explanation (top-5 features + base64 waterfall chart)
    """
    pipe = _state.get("pipe")
    le   = _state.get("le")
    if pipe is None:
        raise HTTPException(503, "Model not loaded")

    # Build input DataFrame
    row = pd.DataFrame([payload.model_dump()])
    row = row[FEATURE_COLS]  # ensure column order

    try:
        proba      = pipe.predict_proba(row)[0]
        pred_enc   = pipe.predict(row)[0]
        pred_class = le.inverse_transform([pred_enc])[0]
        confidence = float(proba[pred_enc])

        # Top-3
        top3_idx = np.argsort(proba)[::-1][:3]
        top3     = [
            DiseaseProba(disease=le.classes_[i], probability=float(proba[i]))
            for i in top3_idx
        ]

        # SHAP
        explainer, _ = build_explainer(pipe)
        X_t          = pipe[:-1].transform(row)
        sv           = explainer(X_t)
        shap5        = top_shap_features(sv, int(pred_enc), FEATURE_COLS, top_n=5)
        shap_top5    = [ShapFeature(**f) for f in shap5]

        # Waterfall chart
        chart_b64 = waterfall_to_base64(pipe, le, row)

    except Exception as exc:
        raise HTTPException(500, f"Prediction error: {exc}") from exc

    return PredictionResponse(
        predicted_disease=pred_class,
        confidence=confidence,
        top_3=top3,
        shap_top5=shap_top5,
        shap_chart_b64=chart_b64,
    )
