# HealthBridge Disease Prediction – Model Card

**Version:** 1.0.0  
**Last Updated:** November 2026  
**Authors:** HealthBridge AI/ML Intern Project  
**Status:** Educational Prototype – NOT for Clinical Use

---

## 1. Model Details

| Field | Value |
|---|---|
| Model type | Multi-class classification |
| Selected model | Logistic Regression (C=10) |
| Selection method | Highest validation Macro-F1 |
| Framework | Scikit-learn |
| Input | 15 structured features |
| Output | Top-3 disease predictions with probabilities |
| Explainability | SHAP |
| Model persistence | `models/best_model.joblib` |
| Label encoder | `models/label_encoder.joblib` |

## 2. Intended Use

### Primary Use Case

- Educational prototype for demonstrating machine-learning-based disease prediction.
- Demonstration of explainable AI in healthcare.
- HealthBridge Hospitals AI/ML internship capstone project.

### Out-of-Scope Uses

- Clinical diagnosis or medical decision-making.
- Replacing qualified healthcare professionals.
- Patient-facing production deployment without clinical validation.
- Emergency medical assessment or treatment recommendations.

---

## 3. Training Data

### Supplied Dataset

**Source:** `aiml_healthcare_symptoms.csv`

- Original records: 800
- Original disease classes: 8
- Records per original class: 100

Original disease classes:

1. Anaemia
2. Gastroenteritis
3. Healthy
4. Hypertension
5. Influenza
6. Migraine
7. Pneumonia
8. Type2Diabetes

### Synthetic Data Augmentation

To expand disease coverage, 2,200 synthetic records were generated programmatically for 22 additional disease classes.

Additional classes:

Asthma, COPD, Tuberculosis, Dengue, Malaria, Typhoid, ChickenPox, Jaundice, HeartDisease, Stroke, KidneyDisease, LiverDisease, Arthritis, Osteoporosis, Hypothyroidism, Hyperthyroidism, Appendicitis, PepticUlcer, IrritableBowelSyndrome, Sinusitis, Urinary_Tract_Infection, and Psoriasis.

### Combined Dataset

| Description | Value |
|---|---:|
| Original records | 800 |
| Synthetic records | 2,200 |
| Total records | 3,000 |
| Total disease classes | 30 |
| Input features | 15 |

### Input Features

- Age and Gender
- Fever, Cough, Fatigue
- Breathlessness and Chest Pain
- Nausea and Headache
- BMI
- BP_Systolic and BP_Diastolic
- SpO2
- Glucose_mg_dL
- HbA1c

`Patient_ID` is excluded from model training. `Diagnosis` is the target variable.

**Data limitation:** Approximately 73.3% of the combined dataset is synthetically generated. Synthetic records are not real patient observations and have not been clinically validated.

---

## 4. Data Preprocessing and Splitting

The data pipeline loads the original dataset, generates additional synthetic records, combines the datasets, and prepares the features and target labels.

| Dataset | Records | Percentage |
|---|---:|---:|
| Training | 2,100 | 70% |
| Validation | 450 | 15% |
| Testing | 450 | 15% |
| Total | 3,000 | 100% |

The dataset is divided into training, validation, and testing sets for model development and evaluation.

---

## 5. Model Training and Evaluation

The following configurations were evaluated:

- Logistic Regression (C=0.1)
- Logistic Regression (C=1.0)
- Logistic Regression (C=10.0)
- Random Forest
- XGBoost

### Candidate Model Results

| Model | Validation Macro-F1 | Test Accuracy |
|---|---:|---:|
| Logistic Regression (C=0.1) | 0.5120 | 47.78% |
| Logistic Regression (C=1.0) | 0.5591 | 48.89% |
| Logistic Regression (C=10.0) | 0.5634 | 49.11% |
| Random Forest | 0.4895 | 49.78% |
| XGBoost | 0.5025 | 47.78% |

**Selected model:** Logistic Regression (C=10.0).

The model was selected based on the highest validation Macro-F1 score of 0.5634. Random Forest had a slightly higher test accuracy, but model selection was based on validation performance.

ROC-AUC values are not included because verified results are not currently available.

### Acceptance Criteria Status

| Criterion | Target | Current Status |
|---|---|---|
| Disease coverage | At least 25 | 30 classes achieved |
| Test accuracy | At least 80% | Not achieved (49.11%) |
| SHAP explanations | Per prediction | Implemented |
| Streamlit interface | Working UI | Implemented |
| API documentation | GitHub | Available |

The current model requires further improvement before it can be considered suitable for real-world healthcare applications.

---

## 6. Explainability

SHAP (SHapley Additive exPlanations) is integrated to explain individual predictions.

The system provides:

- Top five influential features.
- SHAP feature contribution values.
- Waterfall explanation visualizations.
- Feature-level explanations through the prediction API.

SHAP explains how model features contribute to a prediction. It does not establish medical causation or guarantee prediction correctness.

---

## 7. Ethical Considerations and Bias Audit

### Bias and Fairness

The following limitations have been identified:

- The original eight disease classes contain 100 records each.
- Additional classes are represented using synthetic data.
- Synthetic data may not reflect real patient populations.
- Disease prevalence and demographic distributions may differ in real healthcare settings.
- Gender is represented using a binary encoding in this prototype.
- Performance may vary across disease classes and demographic groups.

**Bias audit status:** Preliminary dataset and fairness limitations review completed. A complete demographic fairness audit has not yet been performed.

Further evaluation should include:

- Class-wise precision, recall, and F1-score.
- Confusion matrix analysis.
- Performance comparison across age groups.
- Performance comparison across gender groups.
- Evaluation on independent, clinically validated datasets.

### Privacy

The project uses the supplied project dataset and programmatically generated synthetic records. The privacy characteristics of the supplied dataset should be verified with its data provider before any external or production use.

### Transparency

- Prediction results include SHAP-based feature explanations.
- Training and testing code is maintained in the GitHub repository.
- Model limitations and evaluation results are documented in this model card.

---

## 8. Limitations

1. **Synthetic data:** Approximately 73.3% of the combined dataset is synthetic.
2. **Low accuracy:** Current test accuracy is 49.11%, below the original 80% target.
3. **Limited features:** Only 15 input features are used.
4. **No external validation:** The model has not been evaluated on an independent clinical dataset.
5. **Real-world prevalence:** The balanced dataset does not represent actual disease prevalence.
6. **Clinical limitations:** Many diseases require laboratory tests, imaging, medical history, and physical examination.
7. **No temporal information:** Symptom duration and disease progression are not modelled.

The system is an educational prototype and must not be used as a substitute for professional medical diagnosis.

---

## 9. Deployment Guide

### Requirements

- Python 3.12
- Scikit-learn
- FastAPI
- Uvicorn
- Streamlit
- SHAP
- MLflow

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Prepare the Dataset

```bash
python src/data_pipeline.py
```

### Step 3: Train the Model

```bash
python src/train.py
```

### Step 4: Generate SHAP Explanations

```bash
python src/explain.py
```

### Step 5: Start FastAPI

```bash
uvicorn api.main:app --reload --port 8000
```

API documentation:

`http://127.0.0.1:8000/docs`

### Step 6: Start Streamlit

Open another terminal and run:

```bash
streamlit run App/streamlit_app.py
```

### Step 7: Run Automated Tests

```bash
pytest Tests/ -v
```

Previously recorded test result: 39 passed, 0 failed. Tests should be rerun after final modifications.

---

## 10. Performance and Future Improvements

Locust load testing was conducted with 50 concurrent users for 60 seconds at a spawn rate of 10 users per second.

### 10.1 Before and After Optimization

| Metric | Before Optimization | After Optimization |
|---|---:|---:|
| Concurrent users | 50 | 50 |
| Total requests | 238 | 363 |
| Failed requests | 0 | 0 |
| Average response time | 10,205.25 ms | 6,642.03 ms |
| Median response time | 10,000 ms | 6,700 ms |
| Minimum response time | 7,030.63 ms | 3,403.60 ms |
| Maximum response time | 16,700.74 ms | 10,459.24 ms |
| Throughput | 4.15 req/s | 6.33 req/s |

### 10.2 Optimization Results

The following improvements were achieved:

- Average response time reduced by approximately 34.9%.
- Throughput increased by approximately 52.5%.
- Total successful requests increased from 238 to 363.
- No request failures were recorded in either test.

The optimization focused on reusing the SHAP explainer and reducing SHAP chart generation overhead.

Although performance improved, the average response time remains approximately 6.64 seconds. Further optimization is required before production deployment.

### 10.3 Future Improvements

- Further optimize SHAP explanation generation.
- Improve prediction processing and API response time.
- Increase the amount of clinically validated training data.
- Improve model accuracy and generalization.
- Conduct a complete demographic fairness audit.
- Perform additional load testing under different concurrency levels.
- Evaluate deployment-level performance and scalability.



## 11. Project Conclusion

The HealthBridge Disease Prediction System demonstrates an end-to-end machine-learning workflow involving data preprocessing, model training, evaluation, explainable AI, FastAPI backend integration, and Streamlit application development.

The project provides an educational prototype for exploring machine learning in healthcare. Further data validation, model improvement, fairness evaluation, and performance optimization are required before real-world healthcare use.

---

## 12. Project Repository

**Project:** HealthBridge Disease Prediction System  
**Organization:** HealthBridge Hospitals  
**Location:** Nagpur, India  
**Year:** 2026  

**GitHub Repository:**  
https://github.com/MADHANRAJKR/Healthbridge-Disease-Predictor