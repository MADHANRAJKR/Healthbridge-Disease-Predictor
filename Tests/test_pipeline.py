"""
Tests for the data pipeline — src/data_pipeline.py
Run from repo root: pytest Tests/test_pipeline.py -v
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_pipeline import (
    load_supplied, generate_synthetic, run_audit,
    FEATURE_COLS, BINARY_FEATURES, CONTINUOUS_FEATURES, DISEASE_PROFILES, RAW
)


# ---------------------------------------------------------------------------
# load_supplied
# ---------------------------------------------------------------------------
class TestLoadSupplied:
    def test_loads_dataframe(self):
        df = load_supplied(RAW)
        assert isinstance(df, pd.DataFrame)

    def test_has_correct_feature_columns(self):
        df = load_supplied(RAW)
        for col in FEATURE_COLS:
            assert col in df.columns, f"Missing feature column: {col}"

    def test_has_diagnosis_column(self):
        df = load_supplied(RAW)
        assert "Diagnosis" in df.columns

    def test_no_patient_id_column(self):
        df = load_supplied(RAW)
        assert "Patient_ID" not in df.columns

    def test_no_missing_values(self):
        df = load_supplied(RAW)
        assert df[FEATURE_COLS].isnull().sum().sum() == 0

    def test_row_count(self):
        df = load_supplied(RAW)
        assert len(df) == 800

    def test_eight_disease_classes(self):
        df = load_supplied(RAW)
        assert df["Diagnosis"].nunique() == 8


# ---------------------------------------------------------------------------
# generate_synthetic
# ---------------------------------------------------------------------------
class TestGenerateSynthetic:
    @pytest.fixture(scope="class")
    def synth(self):
        rng = np.random.default_rng(42)
        return generate_synthetic(rng)

    def test_returns_dataframe(self, synth):
        assert isinstance(synth, pd.DataFrame)

    def test_has_all_feature_columns(self, synth):
        for col in FEATURE_COLS:
            assert col in synth.columns

    def test_has_diagnosis_column(self, synth):
        assert "Diagnosis" in synth.columns

    def test_disease_count_matches_profiles(self, synth):
        assert synth["Diagnosis"].nunique() == len(DISEASE_PROFILES)

    def test_rows_per_disease(self, synth):
        counts = synth["Diagnosis"].value_counts()
        # Each disease should have 100 rows
        assert (counts == 100).all(), f"Uneven row counts: {counts.to_dict()}"

    def test_binary_features_are_binary(self, synth):
        for feat in BINARY_FEATURES:
            unique = set(synth[feat].unique())
            assert unique.issubset({0, 1}), f"{feat} has non-binary values: {unique}"

    def test_age_in_range(self, synth):
        assert synth["Age"].between(1, 99).all()

    def test_spo2_in_range(self, synth):
        assert synth["SpO2"].between(70, 100).all()

    def test_bmi_in_range(self, synth):
        assert synth["BMI"].between(12, 60).all()

    def test_gender_values(self, synth):
        # Gender is stored as "M" or "F" before pipeline encoding
        assert set(synth["Gender"].unique()).issubset({"M", "F"})


# ---------------------------------------------------------------------------
# run_audit
# ---------------------------------------------------------------------------
class TestRunAudit:
    @pytest.fixture(scope="class")
    def combined(self):
        rng = np.random.default_rng(42)
        supplied = load_supplied(RAW)
        synthetic = generate_synthetic(rng)
        # Encode Gender for consistency
        for df in [supplied, synthetic]:
            df["Gender"] = df["Gender"].map({"M": 1, "F": 0}).fillna(df["Gender"])
        import pandas as _pd
        return _pd.concat([supplied, synthetic], ignore_index=True)

    def test_audit_keys(self, combined):
        audit = run_audit(combined)
        for key in ["total_rows", "n_disease_classes", "disease_classes",
                    "missing_values", "duplicate_rows", "class_distribution"]:
            assert key in audit

    def test_total_rows(self, combined):
        audit = run_audit(combined)
        assert audit["total_rows"] == len(combined)

    def test_disease_count_gte_25(self, combined):
        audit = run_audit(combined)
        assert audit["n_disease_classes"] >= 25, (
            f"Only {audit['n_disease_classes']} diseases — need at least 25"
        )
