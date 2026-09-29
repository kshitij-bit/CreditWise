# CreditWise AI – Agent Guidelines

## Project Purpose

CreditWise AI is a credit risk assessment and loan decision platform built as a placement-ready end-to-end machine learning project. The goal is to predict the probability of default for loan applicants and generate a credit risk score, risk classification, and loan decision recommendation.

## Context

- This is a student project designed to be explainable during placement interviews.
- Keep it simple. Do not over-engineer.
- Every design choice should be justifiable in an interview.

## Planned Stack

**ML/Data:** Python, Pandas, NumPy, Scikit-learn, XGBoost, CatBoost, SHAP

**Backend:** Flask, SQLAlchemy

**Database:** SQLite

**Frontend:** React, Tailwind CSS, Recharts

## Hard Constraints

- Do NOT add Kafka, Kubernetes, Spark, Airflow, Docker, microservices, or any infrastructure beyond what the stack lists above.
- The project must remain a single, monolithic Flask app serving a React frontend.
- No unnecessary abstractions, message queues, or distributed systems.
- If you are unsure whether a technology is needed, the answer is no.

## Development Phases (Build Order)

The phases must be executed in this order. Do not skip ahead.

1. **Phase 1 – Project setup:** Folder structure, `requirements.txt`, `.gitignore`, `README.md`
2. **Phase 2 – Data & EDA:** Load a public credit dataset (Kaggle/UCI), explore distributions, missingness, correlations, class balance
3. **Phase 3 – Preprocessing:** Cleaning, encoding, imputation, feature scaling, class imbalance handling (SMOTE/ADASYN)
4. **Phase 4 – Feature engineering:** Create domain-relevant features, save fitted featurizer alongside the model
5. **Phase 5 – Model training:** Train baseline logistic regression, then XGBoost and CatBoost; compare with cross-validation; pick threshold based on business trade-off (defaults caught vs. good loans rejected)
6. **Phase 6 – Explainability:** SHAP global feature importance + per-prediction waterfall explanations
7. **Phase 7 – Decision logic:** Pure function that maps predicted probability → score (0–100) → risk band (Low/Moderate/High/Very High) → recommendation (Approve/Review/Reject)
8. **Phase 8 – Flask API:** `/predict`, `/history`, `/analytics`, `/explain` endpoints; SQLite persistence via SQLAlchemy
9. **Phase 9 – React frontend:** Prediction form, results with SHAP explanation, history, portfolio analytics
10. **Phase 10 – Tests & polish:** pytest for core logic and API; README with demo walkthrough

## Key Architecture Rules

- The **featurizer (scaler/encoder) must be saved with the model** (`ml/artifacts/`). Never apply preprocessing without fitting the same featurizer used in training.
- The **decision logic** (probability → score → band → recommendation) lives in a standalone pure function with no framework dependencies. Test it independently.
- **No ML logic lives in the frontend.** The frontend calls Flask endpoints only via HTTP.
- **No synthetic/fake data.** Use a real public dataset.
- Processed data and model artifacts are in `.gitignore` and not committed.

## Interview Readiness

When making changes, consider: "Can I explain this clearly in 2 minutes to an interviewer?" If the answer is no, simplify it.

Focus areas that impress in interviews:
- Reproducible pipeline (train once, serve many times)
- Class imbalance handling with documented trade-off
- SHAP explainability (most student projects lack this)
- Test coverage on business-critical logic
- Clean separation between ML, API, and UI layers

## Single Test Run

```bash
# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_decision_service.py -v
```

## Linting (Phase 8+)

No linting setup until Flask backend exists. When added, run:
```bash
flake8 src/ app/ tests/
```
