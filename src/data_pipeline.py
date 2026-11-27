"""
HealthBridge Data Pipeline
==========================
Loads the supplied aiml_healthcare_symptoms.csv (8 diseases, 800 rows, 15 features),
validates it, then augments it with a programmatically-generated synthetic dataset
that covers an additional 22 disease classes — bringing the total to 30 diseases
and ~2 600 rows, all using the SAME 15-feature schema.

Output
------
  data/processed/features.csv   — feature matrix (15 columns)
  data/processed/labels.csv     — target series (single column "Diagnosis")
  reports/data_audit_full.json  — extended audit stats

Usage
-----
  python src/data_pipeline.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW   = ROOT / "data" / "raw"   / "aiml_healthcare_symptoms.csv"
OUT   = ROOT / "data" / "processed"
RPTS  = ROOT / "reports"
OUT.mkdir(parents=True, exist_ok=True)
RPTS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Synthetic disease profiles
# Each dict encodes the *expected distribution* for every feature.
# Format: { feature: (mean_or_prob, std_or_None) }
# Binary features (Fever, Cough …) use probability of 1; continuous features
# use (mean, std).
# ---------------------------------------------------------------------------
DISEASE_PROFILES = {
    # --- already in supplied CSV (kept for reference, NOT regenerated) ------
    # Influenza, Hypertension, Type2Diabetes, Pneumonia,
    # Healthy, Anaemia, Gastroenteritis, Migraine
    # -----------------------------------------------------------------------

    # Additional 22 diseases
    "Asthma": {
        "Age": (38, 15), "Gender_F_prob": 0.55,
        "Fever": 0.3, "Cough": 0.9, "Fatigue": 0.6,
        "Breathlessness": 0.85, "Chest_Pain": 0.5, "Nausea": 0.2, "Headache": 0.2,
        "BMI": (26, 5), "BP_Systolic": (118, 12), "BP_Diastolic": (76, 8),
        "SpO2": (93, 3), "Glucose_mg_dL": (90, 15), "HbA1c": (5.2, 0.4),
    },
    "COPD": {
        "Age": (63, 10), "Gender_F_prob": 0.40,
        "Fever": 0.2, "Cough": 0.95, "Fatigue": 0.75,
        "Breathlessness": 0.90, "Chest_Pain": 0.4, "Nausea": 0.2, "Headache": 0.15,
        "BMI": (24, 5), "BP_Systolic": (125, 15), "BP_Diastolic": (80, 10),
        "SpO2": (88, 4), "Glucose_mg_dL": (95, 18), "HbA1c": (5.5, 0.5),
    },
    "Tuberculosis": {
        "Age": (35, 15), "Gender_F_prob": 0.45,
        "Fever": 0.95, "Cough": 0.95, "Fatigue": 0.90,
        "Breathlessness": 0.70, "Chest_Pain": 0.60, "Nausea": 0.40, "Headache": 0.30,
        "BMI": (19, 3), "BP_Systolic": (110, 12), "BP_Diastolic": (70, 8),
        "SpO2": (91, 3), "Glucose_mg_dL": (88, 15), "HbA1c": (5.0, 0.4),
    },
    "Dengue": {
        "Age": (28, 15), "Gender_F_prob": 0.50,
        "Fever": 0.98, "Cough": 0.30, "Fatigue": 0.90,
        "Breathlessness": 0.25, "Chest_Pain": 0.20, "Nausea": 0.80, "Headache": 0.90,
        "BMI": (23, 4), "BP_Systolic": (100, 12), "BP_Diastolic": (65, 8),
        "SpO2": (96, 2), "Glucose_mg_dL": (82, 12), "HbA1c": (4.9, 0.3),
    },
    "Malaria": {
        "Age": (30, 15), "Gender_F_prob": 0.45,
        "Fever": 0.98, "Cough": 0.20, "Fatigue": 0.85,
        "Breathlessness": 0.30, "Chest_Pain": 0.15, "Nausea": 0.75, "Headache": 0.80,
        "BMI": (21, 4), "BP_Systolic": (105, 12), "BP_Diastolic": (68, 8),
        "SpO2": (94, 3), "Glucose_mg_dL": (79, 12), "HbA1c": (4.8, 0.3),
    },
    "Typhoid": {
        "Age": (25, 12), "Gender_F_prob": 0.50,
        "Fever": 0.97, "Cough": 0.25, "Fatigue": 0.85,
        "Breathlessness": 0.20, "Chest_Pain": 0.15, "Nausea": 0.70, "Headache": 0.70,
        "BMI": (21, 4), "BP_Systolic": (102, 10), "BP_Diastolic": (66, 8),
        "SpO2": (96, 2), "Glucose_mg_dL": (85, 14), "HbA1c": (4.9, 0.3),
    },
    "ChickenPox": {
        "Age": (12, 10), "Gender_F_prob": 0.50,
        "Fever": 0.90, "Cough": 0.40, "Fatigue": 0.80,
        "Breathlessness": 0.15, "Chest_Pain": 0.10, "Nausea": 0.50, "Headache": 0.60,
        "BMI": (20, 4), "BP_Systolic": (108, 10), "BP_Diastolic": (68, 7),
        "SpO2": (97, 2), "Glucose_mg_dL": (84, 10), "HbA1c": (4.8, 0.3),
    },
    "Jaundice": {
        "Age": (38, 18), "Gender_F_prob": 0.45,
        "Fever": 0.60, "Cough": 0.15, "Fatigue": 0.85,
        "Breathlessness": 0.20, "Chest_Pain": 0.20, "Nausea": 0.80, "Headache": 0.40,
        "BMI": (22, 4), "BP_Systolic": (108, 10), "BP_Diastolic": (70, 8),
        "SpO2": (96, 2), "Glucose_mg_dL": (90, 15), "HbA1c": (5.0, 0.4),
    },
    "HeartDisease": {
        "Age": (58, 12), "Gender_F_prob": 0.35,
        "Fever": 0.15, "Cough": 0.30, "Fatigue": 0.75,
        "Breathlessness": 0.80, "Chest_Pain": 0.90, "Nausea": 0.45, "Headache": 0.20,
        "BMI": (28, 5), "BP_Systolic": (142, 18), "BP_Diastolic": (90, 12),
        "SpO2": (94, 3), "Glucose_mg_dL": (110, 20), "HbA1c": (6.0, 0.6),
    },
    "Stroke": {
        "Age": (65, 12), "Gender_F_prob": 0.45,
        "Fever": 0.10, "Cough": 0.10, "Fatigue": 0.70,
        "Breathlessness": 0.40, "Chest_Pain": 0.30, "Nausea": 0.50, "Headache": 0.70,
        "BMI": (27, 5), "BP_Systolic": (155, 20), "BP_Diastolic": (95, 12),
        "SpO2": (95, 3), "Glucose_mg_dL": (120, 25), "HbA1c": (6.3, 0.7),
    },
    "KidneyDisease": {
        "Age": (52, 15), "Gender_F_prob": 0.40,
        "Fever": 0.30, "Cough": 0.20, "Fatigue": 0.85,
        "Breathlessness": 0.55, "Chest_Pain": 0.30, "Nausea": 0.70, "Headache": 0.50,
        "BMI": (25, 5), "BP_Systolic": (145, 18), "BP_Diastolic": (92, 12),
        "SpO2": (95, 3), "Glucose_mg_dL": (115, 25), "HbA1c": (6.2, 0.6),
    },
    "LiverDisease": {
        "Age": (48, 15), "Gender_F_prob": 0.40,
        "Fever": 0.45, "Cough": 0.15, "Fatigue": 0.90,
        "Breathlessness": 0.25, "Chest_Pain": 0.30, "Nausea": 0.85, "Headache": 0.35,
        "BMI": (24, 5), "BP_Systolic": (112, 12), "BP_Diastolic": (72, 8),
        "SpO2": (96, 2), "Glucose_mg_dL": (95, 18), "HbA1c": (5.5, 0.5),
    },
    "Arthritis": {
        "Age": (58, 15), "Gender_F_prob": 0.65,
        "Fever": 0.25, "Cough": 0.10, "Fatigue": 0.70,
        "Breathlessness": 0.15, "Chest_Pain": 0.15, "Nausea": 0.25, "Headache": 0.30,
        "BMI": (27, 5), "BP_Systolic": (125, 15), "BP_Diastolic": (80, 10),
        "SpO2": (97, 2), "Glucose_mg_dL": (95, 15), "HbA1c": (5.5, 0.4),
    },
    "Osteoporosis": {
        "Age": (65, 12), "Gender_F_prob": 0.75,
        "Fever": 0.10, "Cough": 0.10, "Fatigue": 0.55,
        "Breathlessness": 0.15, "Chest_Pain": 0.20, "Nausea": 0.15, "Headache": 0.20,
        "BMI": (22, 4), "BP_Systolic": (122, 14), "BP_Diastolic": (78, 9),
        "SpO2": (97, 2), "Glucose_mg_dL": (90, 12), "HbA1c": (5.3, 0.4),
    },
    "Hypothyroidism": {
        "Age": (45, 15), "Gender_F_prob": 0.75,
        "Fever": 0.10, "Cough": 0.10, "Fatigue": 0.90,
        "Breathlessness": 0.25, "Chest_Pain": 0.15, "Nausea": 0.30, "Headache": 0.40,
        "BMI": (29, 5), "BP_Systolic": (118, 12), "BP_Diastolic": (76, 8),
        "SpO2": (97, 2), "Glucose_mg_dL": (92, 14), "HbA1c": (5.4, 0.4),
    },
    "Hyperthyroidism": {
        "Age": (40, 14), "Gender_F_prob": 0.70,
        "Fever": 0.30, "Cough": 0.15, "Fatigue": 0.65,
        "Breathlessness": 0.45, "Chest_Pain": 0.30, "Nausea": 0.40, "Headache": 0.45,
        "BMI": (20, 3), "BP_Systolic": (130, 15), "BP_Diastolic": (80, 10),
        "SpO2": (97, 2), "Glucose_mg_dL": (88, 12), "HbA1c": (5.1, 0.3),
    },
    "Appendicitis": {
        "Age": (25, 12), "Gender_F_prob": 0.45,
        "Fever": 0.80, "Cough": 0.15, "Fatigue": 0.70,
        "Breathlessness": 0.20, "Chest_Pain": 0.30, "Nausea": 0.85, "Headache": 0.30,
        "BMI": (23, 4), "BP_Systolic": (108, 10), "BP_Diastolic": (70, 7),
        "SpO2": (97, 2), "Glucose_mg_dL": (90, 12), "HbA1c": (5.0, 0.3),
    },
    "PepticUlcer": {
        "Age": (42, 15), "Gender_F_prob": 0.45,
        "Fever": 0.30, "Cough": 0.15, "Fatigue": 0.65,
        "Breathlessness": 0.20, "Chest_Pain": 0.60, "Nausea": 0.80, "Headache": 0.30,
        "BMI": (22, 4), "BP_Systolic": (112, 10), "BP_Diastolic": (72, 7),
        "SpO2": (97, 2), "Glucose_mg_dL": (88, 12), "HbA1c": (5.1, 0.3),
    },
    "IrritableBowelSyndrome": {
        "Age": (35, 14), "Gender_F_prob": 0.60,
        "Fever": 0.15, "Cough": 0.10, "Fatigue": 0.70,
        "Breathlessness": 0.15, "Chest_Pain": 0.30, "Nausea": 0.75, "Headache": 0.45,
        "BMI": (22, 4), "BP_Systolic": (115, 10), "BP_Diastolic": (74, 7),
        "SpO2": (97, 2), "Glucose_mg_dL": (88, 12), "HbA1c": (5.1, 0.3),
    },
    "Sinusitis": {
        "Age": (32, 14), "Gender_F_prob": 0.55,
        "Fever": 0.60, "Cough": 0.70, "Fatigue": 0.65,
        "Breathlessness": 0.25, "Chest_Pain": 0.15, "Nausea": 0.30, "Headache": 0.90,
        "BMI": (24, 4), "BP_Systolic": (115, 10), "BP_Diastolic": (74, 7),
        "SpO2": (97, 2), "Glucose_mg_dL": (88, 12), "HbA1c": (5.0, 0.3),
    },
    "Urinary_Tract_Infection": {
        "Age": (38, 18), "Gender_F_prob": 0.75,
        "Fever": 0.75, "Cough": 0.10, "Fatigue": 0.65,
        "Breathlessness": 0.15, "Chest_Pain": 0.10, "Nausea": 0.55, "Headache": 0.40,
        "BMI": (24, 4), "BP_Systolic": (112, 10), "BP_Diastolic": (72, 7),
        "SpO2": (97, 2), "Glucose_mg_dL": (90, 12), "HbA1c": (5.1, 0.3),
    },
    "Psoriasis": {
        "Age": (40, 15), "Gender_F_prob": 0.50,
        "Fever": 0.20, "Cough": 0.10, "Fatigue": 0.55,
        "Breathlessness": 0.10, "Chest_Pain": 0.10, "Nausea": 0.20, "Headache": 0.25,
        "BMI": (25, 5), "BP_Systolic": (118, 12), "BP_Diastolic": (76, 8),
        "SpO2": (97, 2), "Glucose_mg_dL": (92, 12), "HbA1c": (5.2, 0.3),
    },
}

BINARY_FEATURES = ["Fever", "Cough", "Fatigue", "Breathlessness",
                   "Chest_Pain", "Nausea", "Headache"]
CONTINUOUS_FEATURES = ["Age", "BMI", "BP_Systolic", "BP_Diastolic",
                       "SpO2", "Glucose_mg_dL", "HbA1c"]
FEATURE_COLS = ["Age", "Gender", "Fever", "Cough", "Fatigue",
                "Breathlessness", "Chest_Pain", "Nausea", "Headache",
                "BMI", "BP_Systolic", "BP_Diastolic", "SpO2",
                "Glucose_mg_dL", "HbA1c"]
SAMPLES_PER_DISEASE = 100


def generate_synthetic(rng: np.random.Generator) -> pd.DataFrame:
    """Return a DataFrame with SAMPLES_PER_DISEASE rows per extra disease."""
    frames = []
    for disease, prof in DISEASE_PROFILES.items():
        n = SAMPLES_PER_DISEASE
        rows = {}

        # Age
        rows["Age"] = rng.normal(prof["Age"][0], prof["Age"][1], n).clip(1, 99).astype(int)

        # Gender
        rows["Gender"] = rng.choice(
            ["F", "M"], size=n,
            p=[prof["Gender_F_prob"], 1 - prof["Gender_F_prob"]]
        )

        # Binary symptoms
        for feat in BINARY_FEATURES:
            prob = prof[feat]
            rows[feat] = rng.binomial(1, prob, n)

        # Continuous vitals
        for feat in CONTINUOUS_FEATURES:
            mean, std = prof[feat]
            rows[feat] = rng.normal(mean, std, n)

        # Clip physiological ranges
        rows["Age"] = np.clip(rows["Age"], 1, 99)
        rows["BMI"] = np.clip(rows["BMI"], 12, 60).round(1)
        rows["BP_Systolic"] = np.clip(rows["BP_Systolic"], 70, 220).astype(int)
        rows["BP_Diastolic"] = np.clip(rows["BP_Diastolic"], 40, 140).astype(int)
        rows["SpO2"] = np.clip(rows["SpO2"], 70, 100).astype(int)
        rows["Glucose_mg_dL"] = np.clip(rows["Glucose_mg_dL"], 50, 500).astype(int)
        rows["HbA1c"] = np.clip(rows["HbA1c"], 3.5, 15.0).round(1)

        df = pd.DataFrame(rows)
        df["Diagnosis"] = disease
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def load_supplied(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert "Diagnosis" in df.columns, "Target column 'Diagnosis' not found"
    # Drop Patient_ID if present
    df = df.drop(columns=["Patient_ID"], errors="ignore")
    return df[FEATURE_COLS + ["Diagnosis"]]


def run_audit(df: pd.DataFrame) -> dict:
    return {
        "total_rows": int(len(df)),
        "n_disease_classes": int(df["Diagnosis"].nunique()),
        "disease_classes": sorted(df["Diagnosis"].unique().tolist()),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_distribution": df["Diagnosis"].value_counts().sort_index().to_dict(),
    }


def main():
    rng = np.random.default_rng(42)

    print("Loading supplied data …")
    supplied = load_supplied(RAW)
    print(f"  Supplied rows: {len(supplied)}, classes: {supplied['Diagnosis'].nunique()}")

    print("Generating synthetic augmentation …")
    synthetic = generate_synthetic(rng)
    print(f"  Synthetic rows: {len(synthetic)}, classes: {synthetic['Diagnosis'].nunique()}")

    combined = pd.concat([supplied, synthetic], ignore_index=True).sample(
        frac=1, random_state=42).reset_index(drop=True)

    # Encode Gender as binary (M=1, F=0) for sklearn compatibility
    combined["Gender"] = (combined["Gender"] == "M").astype(int)

    X = combined[FEATURE_COLS].copy()
    # Gender is now int; convert any remaining object cols
    for col in FEATURE_COLS:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    y = combined["Diagnosis"]

    audit = run_audit(combined)
    print(f"\nAudit — total rows: {audit['total_rows']}, "
          f"disease classes: {audit['n_disease_classes']}")
    print("  Classes:", audit["disease_classes"])

    X.to_csv(OUT / "features.csv", index=False)
    y.to_csv(OUT / "labels.csv", index=False, header=True)
    (RPTS / "data_audit_full.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8")

    print(f"\nSaved:\n  {OUT / 'features.csv'}\n  {OUT / 'labels.csv'}")
    print(f"  {RPTS / 'data_audit_full.json'}")


if __name__ == "__main__":
    main()
