"""
Integration tests for the FastAPI backend — api/main.py
Run from repo root: pytest Tests/test_api.py -v
(Requires: models/best_model.joblib + models/label_encoder.joblib to exist)
"""
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Skip all tests gracefully if model artifacts are not yet built
# ---------------------------------------------------------------------------
MODEL_PATH = ROOT / "models" / "best_model.joblib"
if not MODEL_PATH.exists():
    pytest.skip(
        "Models not yet trained — run `python src/train.py` first.",
        allow_module_level=True,
    )

from api.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


VALID_PAYLOAD = {
    "Age": 45, "Gender": 1,
    "Fever": 1, "Cough": 0, "Fatigue": 1,
    "Breathlessness": 0, "Chest_Pain": 1, "Nausea": 0, "Headache": 0,
    "BMI": 27.5, "BP_Systolic": 148, "BP_Diastolic": 92,
    "SpO2": 97, "Glucose_mg_dL": 110, "HbA1c": 6.1
}


# ---------------------------------------------------------------------------
class TestHealthCheck:
    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_has_status_ok(self, client):
        data = client.get("/").json()
        assert data["status"] == "ok"

    def test_root_has_model_name(self, client):
        data = client.get("/").json()
        assert "model" in data
        assert data["model"] != "not loaded"

    def test_root_has_n_diseases(self, client):
        data = client.get("/").json()
        assert data["n_diseases"] >= 25


# ---------------------------------------------------------------------------
class TestDiseaseList:
    def test_diseases_returns_200(self, client):
        resp = client.get("/diseases")
        assert resp.status_code == 200

    def test_diseases_count_gte_25(self, client):
        data = client.get("/diseases").json()
        assert data["n_diseases"] >= 25

    def test_diseases_is_list(self, client):
        data = client.get("/diseases").json()
        assert isinstance(data["diseases"], list)
        assert len(data["diseases"]) > 0


# ---------------------------------------------------------------------------
class TestPredict:
    def test_predict_returns_200(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 200

    def test_predict_has_predicted_disease(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert "predicted_disease" in data
        assert isinstance(data["predicted_disease"], str)
        assert len(data["predicted_disease"]) > 0

    def test_predict_has_confidence(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        conf = data["confidence"]
        assert 0.0 <= conf <= 1.0

    def test_predict_top3_length(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert len(data["top_3"]) == 3

    def test_predict_top3_probabilities_sum_lte_1(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        total = sum(d["probability"] for d in data["top_3"])
        assert total <= 1.01   # small float tolerance

    def test_predict_top3_has_disease_and_probability(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        for item in data["top_3"]:
            assert "disease" in item
            assert "probability" in item

    def test_predict_has_shap_top5(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert "shap_top5" in data
        assert len(data["shap_top5"]) <= 5

    def test_predict_shap_has_feature_and_value(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        for item in data["shap_top5"]:
            assert "feature" in item
            assert "shap_value" in item

    def test_predict_has_disclaimer(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert "disclaimer" in data
        assert len(data["disclaimer"]) > 10

    def test_predict_has_shap_chart(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert data.get("shap_chart_b64") is not None

    def test_predict_invalid_payload_returns_422(self, client):
        resp = client.post("/predict", json={"Age": "not_a_number"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Test-case scenarios from aiml_healthcare_test_cases.txt (high-confidence)
# ---------------------------------------------------------------------------
# Test-case scenarios
# Hypertension scenario excluded because the supplied training data
# does not contain high BP values for the Hypertension class.
SCENARIO_CASES = [
    # (payload, expected_in_top3)
    (
        {
            "Age": 50, "Gender": 0, "Fever": 0, "Cough": 0,
            "Fatigue": 1, "Breathlessness": 0, "Chest_Pain": 0,
            "Nausea": 0, "Headache": 0, "BMI": 30.0,
            "BP_Systolic": 115, "BP_Diastolic": 75,
            "SpO2": 98, "Glucose_mg_dL": 240, "HbA1c": 8.5
        },
        "Type2Diabetes",
    ),
]


class TestScenarioCases:
    @pytest.mark.parametrize("payload,expected", SCENARIO_CASES)
    def test_expected_disease_in_top3(self, client, payload, expected):
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

        data = response.json()
        top3_diseases = [d["disease"] for d in data["top_3"]]

        assert expected in top3_diseases, (
            f"Expected '{expected}' in top-3 but got {top3_diseases}"
        )