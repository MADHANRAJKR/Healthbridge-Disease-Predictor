"""
HealthBridge FastAPI — Pydantic schemas
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class SymptomInput(BaseModel):
    """All 15 features expected by the model."""
    Age:            float = Field(30,    ge=1,   le=120, description="Patient age in years")
    Gender:         int   = Field(1,     ge=0,   le=1,   description="0 = Female, 1 = Male")
    Fever:          int   = Field(0,     ge=0,   le=1,   description="Fever present (0/1)")
    Cough:          int   = Field(0,     ge=0,   le=1,   description="Cough present (0/1)")
    Fatigue:        int   = Field(0,     ge=0,   le=1,   description="Fatigue present (0/1)")
    Breathlessness: int   = Field(0,     ge=0,   le=1,   description="Breathlessness present (0/1)")
    Chest_Pain:     int   = Field(0,     ge=0,   le=1,   description="Chest pain present (0/1)")
    Nausea:         int   = Field(0,     ge=0,   le=1,   description="Nausea present (0/1)")
    Headache:       int   = Field(0,     ge=0,   le=1,   description="Headache present (0/1)")
    BMI:            float = Field(22.0,  ge=10,  le=70,  description="Body Mass Index")
    BP_Systolic:    float = Field(120.0, ge=60,  le=250, description="Systolic blood pressure (mmHg)")
    BP_Diastolic:   float = Field(80.0,  ge=40,  le=150, description="Diastolic blood pressure (mmHg)")
    SpO2:           float = Field(98.0,  ge=50,  le=100, description="Oxygen saturation (%)")
    Glucose_mg_dL:  float = Field(90.0,  ge=40,  le=600, description="Blood glucose (mg/dL)")
    HbA1c:          float = Field(5.0,   ge=3,   le=20,  description="Glycated haemoglobin (%)")

    model_config = {"json_schema_extra": {
        "example": {
            "Age": 45, "Gender": 1,
            "Fever": 1, "Cough": 0, "Fatigue": 1,
            "Breathlessness": 0, "Chest_Pain": 1, "Nausea": 0, "Headache": 0,
            "BMI": 27.5, "BP_Systolic": 148, "BP_Diastolic": 92,
            "SpO2": 97, "Glucose_mg_dL": 110, "HbA1c": 6.1
        }
    }}


class DiseaseProba(BaseModel):
    disease:     str
    probability: float


class ShapFeature(BaseModel):
    feature:    str
    shap_value: float


class PredictionResponse(BaseModel):
    predicted_disease: str
    confidence:        float
    top_3:             list[DiseaseProba]
    shap_top5:         list[ShapFeature]
    shap_chart_b64:    Optional[str] = Field(
        None, description="Base64-encoded PNG of the SHAP waterfall chart"
    )
    disclaimer: str = (
        "This prediction is an AI prototype for educational purposes only. "
        "It is NOT a clinical diagnosis. Always consult a qualified healthcare professional."
    )
