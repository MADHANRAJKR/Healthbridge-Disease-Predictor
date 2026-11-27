
from locust import HttpUser, task, between


class HealthBridgeUser(HttpUser):
    wait_time = between(0.5, 1)

    @task
    def predict_disease(self):
        payload = {
            "Age": 45,
            "Gender": 1,
            "Fever": 1,
            "Cough": 0,
            "Fatigue": 1,
            "Breathlessness": 0,
            "Chest_Pain": 1,
            "Nausea": 0,
            "Headache": 0,
            "BMI": 27.5,
            "BP_Systolic": 148,
            "BP_Diastolic": 92,
            "SpO2": 97,
            "Glucose_mg_dL": 110,
            "HbA1c": 6.1
        }

        self.client.post(
            "/predict",
            json=payload,
            name="/predict"
        )
