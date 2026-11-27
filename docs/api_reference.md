# HealthBridge Disease Prediction API — Reference

Base URL: `http://localhost:8000`  
OpenAPI (Swagger) UI: `http://localhost:8000/docs`  
ReDoc UI: `http://localhost:8000/redoc`

---

## Endpoints

### `GET /`
Health check — confirms the API is running and the model is loaded.

**Response `200 OK`**
```json
{
  "status": "ok",
  "service": "HealthBridge Disease Prediction API",
  "model": "XGBClassifier",
  "n_diseases": 30
}
```

**cURL**
```bash
curl http://localhost:8000/
```

---

### `GET /diseases`
Returns the full list of disease classes the model can predict.

**Response `200 OK`**
```json
{
  "n_diseases": 30,
  "diseases": [
    "Anaemia", "Appendicitis", "Arthritis", "Asthma", "COPD",
    "ChickenPox", "Dengue", "Gastroenteritis", "HeartDisease",
    "Healthy", "Hypertension", "Hyperthyroidism", "Hypothyroidism",
    "IrritableBowelSyndrome", "Influenza", "Jaundice", "KidneyDisease",
    "LiverDisease", "Malaria", "Migraine", "Osteoporosis",
    "PepticUlcer", "Pneumonia", "Psoriasis", "Sinusitis",
    "Stroke", "Tuberculosis", "Type2Diabetes", "Typhoid",
    "Urinary_Tract_Infection"
  ]
}
```

**cURL**
```bash
curl http://localhost:8000/diseases
```

---

### `POST /predict`
Predict the most likely disease from symptom/vital inputs and return a SHAP explanation.

**Request Body** (`application/json`)

| Field | Type | Range | Default | Description |
|---|---|---|---|---|
| `Age` | float | 1–120 | 30 | Patient age (years) |
| `Gender` | int | 0 or 1 | 1 | 0 = Female, 1 = Male |
| `Fever` | int | 0 or 1 | 0 | Fever present |
| `Cough` | int | 0 or 1 | 0 | Cough present |
| `Fatigue` | int | 0 or 1 | 0 | Fatigue present |
| `Breathlessness` | int | 0 or 1 | 0 | Breathlessness present |
| `Chest_Pain` | int | 0 or 1 | 0 | Chest pain present |
| `Nausea` | int | 0 or 1 | 0 | Nausea present |
| `Headache` | int | 0 or 1 | 0 | Headache present |
| `BMI` | float | 10–70 | 22.0 | Body Mass Index |
| `BP_Systolic` | float | 60–250 | 120 | Systolic blood pressure (mmHg) |
| `BP_Diastolic` | float | 40–150 | 80 | Diastolic blood pressure (mmHg) |
| `SpO2` | float | 50–100 | 98 | Blood oxygen saturation (%) |
| `Glucose_mg_dL` | float | 40–600 | 90 | Blood glucose (mg/dL) |
| `HbA1c` | float | 3–20 | 5.0 | Glycated haemoglobin (%) |

**Example Request**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 45, "Gender": 1,
    "Fever": 1, "Cough": 0, "Fatigue": 1,
    "Breathlessness": 0, "Chest_Pain": 1, "Nausea": 0, "Headache": 0,
    "BMI": 27.5, "BP_Systolic": 148, "BP_Diastolic": 92,
    "SpO2": 97, "Glucose_mg_dL": 110, "HbA1c": 6.1
  }'
```

**Response `200 OK`**
```json
{
  "predicted_disease": "HeartDisease",
  "confidence": 0.72,
  "top_3": [
    { "disease": "HeartDisease",  "probability": 0.72 },
    { "disease": "Hypertension",  "probability": 0.15 },
    { "disease": "Stroke",        "probability": 0.07 }
  ],
  "shap_top5": [
    { "feature": "BP_Systolic",   "shap_value":  0.412 },
    { "feature": "Chest_Pain",    "shap_value":  0.318 },
    { "feature": "BP_Diastolic",  "shap_value":  0.208 },
    { "feature": "Age",           "shap_value":  0.117 },
    { "feature": "HbA1c",         "shap_value":  0.085 }
  ],
  "shap_chart_b64": "<base64-encoded PNG string>",
  "disclaimer": "This prediction is an AI prototype for educational purposes only. It is NOT a clinical diagnosis. Always consult a qualified healthcare professional."
}
```

**Error Responses**

| Code | Meaning |
|---|---|
| `422 Unprocessable Entity` | Invalid or missing fields in request body |
| `500 Internal Server Error` | Prediction failed (check server logs) |
| `503 Service Unavailable` | Model not loaded at startup |

---

## Python Client Example

```python
import httpx

payload = {
    "Age": 50, "Gender": 0,
    "Fever": 0, "Cough": 0, "Fatigue": 1,
    "Breathlessness": 0, "Chest_Pain": 0, "Nausea": 0, "Headache": 0,
    "BMI": 30.0, "BP_Systolic": 115, "BP_Diastolic": 75,
    "SpO2": 98, "Glucose_mg_dL": 240, "HbA1c": 8.5
}

resp = httpx.post("http://localhost:8000/predict", json=payload)
result = resp.json()
print(f"Predicted: {result['predicted_disease']} ({result['confidence']*100:.1f}%)")
for item in result["top_3"]:
    print(f"  {item['disease']}: {item['probability']*100:.1f}%")
```

---

## Running the API

```bash
# From repository root
uvicorn api.main:app --reload --port 8000

# Production (no reload)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2
```

The model is loaded **once at startup**; each `/predict` call is fast (~10–50 ms per request).
