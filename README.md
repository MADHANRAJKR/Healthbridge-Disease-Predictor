# HealthBridge Disease Prediction & Symptom Checker

## 🏥 Project Overview

This project is an AI/ML-based disease prediction and symptom-checking prototype developed for HealthBridge Hospitals.

The system uses machine learning models trained on public symptom-disease data to predict possible diseases based on user-selected symptoms.

The project includes:

* Machine learning disease classification
* 25+ disease classes
* Symptom-based prediction
* Model evaluation
* SHAP-based explainability
* FastAPI backend
* Streamlit user interface
* API documentation

## 🎯 Project Objective

The objective is to develop a beginner-friendly healthcare AI prototype that can:

1. Accept symptoms from a user.
2. Process the symptoms using a machine-learning pipeline.
3. Predict the most likely disease class.
4. Provide an explanation of important symptoms using SHAP.
5. Expose predictions through a FastAPI REST API.
6. Provide an interactive Streamlit interface.

> **Disclaimer:** This project is an educational and technical prototype. It is not intended to provide medical diagnosis or replace a qualified healthcare professional.

## 🏗️ Planned Architecture

```text
Symptom-Disease Dataset
          ↓
Data Validation
          ↓
Preprocessing
          ↓
Feature Encoding
          ↓
Model Training
          ↓
Model Evaluation
          ↓
Best Model
          ↓
SHAP Explainability
          ↓
FastAPI Backend
          ↓
Streamlit Interface
```

## 🤖 Models

The initial models planned for comparison are:

* Logistic Regression
* Random Forest
* XGBoost

The final model will be selected based on performance on the test dataset.

## 📊 Evaluation Metrics

The following metrics will be used:

* Accuracy
* Precision
* Recall
* F1-score
* Macro F1-score
* Confusion matrix
* Top-3 accuracy where appropriate

The target acceptance criterion is:

**Accuracy ≥ 80%**

## 🔍 Explainability

SHAP will be used to explain individual predictions and identify the symptoms that contributed most strongly to a prediction.

## 🛠️ Technology Stack

| Technology   | Purpose              |
| ------------ | -------------------- |
| Python       | Programming          |
| Pandas       | Data processing      |
| NumPy        | Numerical operations |
| Scikit-learn | Machine learning     |
| XGBoost      | Gradient boosting    |
| SHAP         | Model explainability |
| FastAPI      | REST API             |
| Uvicorn      | API server           |
| Streamlit    | Web interface        |
| Jupyter      | Experimentation      |

## 📁 Project Structure

```text
healthbridge-disease-predictor/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
├── src/
├── models/
├── api/
├── app/
├── tests/
├── docs/
│
├── README.md
├── requirements.txt
└── .gitignore
```

## 🚧 Project Status

### Checkpoint 1 — Research + Architecture

* [x] GitHub repository created
* [x] Project architecture defined
* [x] Evaluation metrics defined
* [x] Research references identified
* [ ] Data exploration
* [ ] Data preprocessing
* [ ] Model training
* [ ] Model evaluation
* [ ] SHAP integration
* [ ] FastAPI implementation
* [ ] Streamlit implementation

## ⚠️ Limitations

The model will be trained using public symptom-disease datasets. Therefore, its predictions may not represent real-world clinical diagnosis.

The system should be treated as a machine-learning demonstration and decision-support prototype rather than a medical diagnostic tool.
