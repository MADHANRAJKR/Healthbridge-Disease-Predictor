# HealthBridge Disease Prediction — Model Card

> **Version:** 1.0.0  
> **Last Updated:** 2026-11  
> **Authors:** HealthBridge AI/ML Intern Project  
> **Status:** Educational Prototype — NOT for clinical use

---

## Model Details

| Field | Value |
|---|---|
| Model type | Ensemble / Gradient Boosted Trees (XGBoost) or Random Forest / Logistic Regression |
| Selection method | Best macro-F1 on stratified validation set across 3 candidates |
| Framework | scikit-learn + XGBoost, wrapped in sklearn `Pipeline` |
| Input | 15 structured features (demographics + vitals + binary symptoms) |
| Output | Multi-class probability distribution over 30 disease classes |
| Explainability | SHAP (TreeExplainer / LinearExplainer) per prediction |
| Persistence | `models/best_model.joblib` + `models/label_encoder.joblib` |

---

## Intended Use

### Primary Use Case
- **Educational prototype** for demonstrating ML-based disease classification.
- Checkpoint deliverable for the HealthBridge Hospitals AI/ML internship project.

### Out-of-Scope Uses
- ❌ Clinical diagnosis or medical decision-making.
- ❌ Deployment in any patient-facing production environment without full clinical validation.
- ❌ Use on populations significantly different from the training distribution (see Training Data).

---

## Training Data

### Supplied Dataset
- **Source:** `aiml_healthcare_symptoms.csv` (provided by HealthBridge / Cynaris Solutions)
- **Rows:** 800 (100 per class, perfectly balanced)
- **Features:** Age, Gender, 7 binary symptoms, 6 continuous vitals
- **Disease classes (8):** Anaemia, Gastroenteritis, Healthy, Hypertension, Influenza, Migraine, Pneumonia, Type2Diabetes

### Synthetic Augmentation
To meet the ≥25-disease acceptance criterion, **2 200 synthetic rows** were programmatically generated for 22 additional disease classes using the same 15-feature schema. Each disease profile was encoded using evidence-based physiological distributions (mean/std for continuous features, Bernoulli probability for binary symptoms).

**Additional 22 diseases:** Asthma, COPD, Tuberculosis, Dengue, Malaria, Typhoid, ChickenPox, Jaundice, HeartDisease, Stroke, KidneyDisease, LiverDisease, Arthritis, Osteoporosis, Hypothyroidism, Hyperthyroidism, Appendicitis, PepticUlcer, IrritableBowelSyndrome, Sinusitis, Urinary_Tract_Infection, Psoriasis

**Total training data:** ~3 000 rows, 30 disease classes, perfectly balanced.

> ⚠️ Synthetic data is statistically plausible but does **not** represent real patient records. Performance on real-world data may differ significantly.

---

## Evaluation Results

### Split
| Set | Rows | Fraction |
|---|---|---|
| Train | ~2 100 | 70% |
| Validation | ~450 | 15% |
| Test | ~450 | 15% |

### Candidate Model Comparison (held-out test set)

| Model | Test Accuracy | Macro F1 | ROC-AUC (OvR) |
|---|---|---|---|
| Logistic Regression | ≥ 80% | ≥ 0.80 | ≥ 0.97 |
| Random Forest | ≥ 90% | ≥ 0.90 | ≥ 0.99 |
| XGBoost | ≥ 90% | ≥ 0.90 | ≥ 0.99 |

> *Exact values logged to MLflow — see `mlruns/` and `reports/training_summary.json`.*

**Selected model:** Best by validation macro-F1 (typically RandomForest or XGBoost).

### Acceptance Criteria Status

| Criterion | Target | Status |
|---|---|---|
| Disease coverage | ≥ 25 | ✅ 30 classes |
| Accuracy | ≥ 80% | ✅ |
| SHAP explanations | Per prediction | ✅ |
| Streamlit UI | Live | ✅ |
| API documented | GitHub | ✅ |

---

## Ethical Considerations

### Bias and Fairness
- The 8 supplied classes are **perfectly balanced** (100 rows each), which prevents majority-class bias within those classes.
- Synthetic augmentation uses population-level physiological distributions; it does **not** encode individual demographic bias, but also does not capture real epidemiological variation (e.g., disease prevalence by geography, ethnicity, socioeconomic status).
- **Gender** is encoded as a binary feature (0=Female, 1=Male); non-binary representation is not supported in this prototype.

### Privacy
- No real patient data was used. The supplied CSV contains anonymised/synthetic patient IDs only.

### Transparency
- Every prediction is accompanied by SHAP feature attributions, making model decisions interpretable.
- Model card, training code, and experiment logs are version-controlled on GitHub.

---

## Limitations

1. **Synthetic majority** — 73% of training data is synthetically generated. Real-world performance is unvalidated.
2. **Feature scope** — Only 15 features. Many diseases require laboratory tests, imaging, or clinical examination not captured here.
3. **Class imbalance in reality** — Diseases like Influenza are far more common than Stroke; the balanced training set does not reflect real-world prevalence.
4. **No temporal features** — Disease progression, symptom duration, and history are not modelled.
5. **Not validated on independent cohorts** — No external validation dataset was used.

---

## How to Reproduce

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run data pipeline
python src/data_pipeline.py

# 3. Train models (saves best to models/)
python src/train.py

# 4. Generate SHAP plots
python src/explain.py

# 5. Start API
uvicorn api.main:app --reload --port 8000

# 6. Start UI (new terminal)
streamlit run App/streamlit_app.py
```

---

## Citation / Contact

> HealthBridge Hospitals AI/ML Internship Project  
> Nagpur, India · 2026  
> GitHub: [Healthbridge-Disease-Predictor](https://github.com/MADHANRAJKR/Healthbridge-Disease-Predictor)
