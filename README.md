# CreditWise AI

## Problem Statement

Lending institutions face significant financial risk from loan defaults. Manual credit evaluation is slow, inconsistent, and difficult to explain. There is a need for a transparent, data-driven credit risk assessment system that can predict the likelihood of default and support loan approval decisions.

## Project Objective

Build an end-to-end machine learning system that:

1. Accepts borrower and loan application details.
2. Predicts the probability of default.
3. Generates an internal credit risk score.
4. Classifies applicants into risk bands.
5. Recommends a loan decision.
6. Explains every prediction using SHAP.
7. Logs all predictions and provides portfolio analytics.

## Planned Features

- Borrower input form with validation.
- Probability of default prediction.
- Credit risk score (0–100).
- Risk classification: Low, Moderate, High, Very High.
- Loan decision recommendation: Approve, Review, Reject.
- SHAP waterfall chart for per-applicant explanation.
- Prediction history stored in SQLite.
- Portfolio analytics dashboard with Recharts.

## ML Models to Compare

- Logistic Regression (baseline)
- XGBoost
- CatBoost

All compared using stratified cross-validation.

## Technology Stack

| Layer       | Tools                                                  |
|-------------|--------------------------------------------------------|
| ML / Data   | Python, Pandas, NumPy, Scikit-learn, XGBoost, CatBoost, SHAP, Imbalanced-learn |
| Backend     | Flask, SQLAlchemy                                      |
| Database    | SQLite                                                 |
| Frontend    | React, Tailwind CSS, Recharts                          |

## Development Phases

| Phase | Description                                   |
|-------|-----------------------------------------------|
| 1     | Project setup, folder structure, dependencies  |
| 2     | Data loading, EDA, dataset selection            |
| 3     | Preprocessing, encoding, class imbalance        |
| 4     | Feature engineering, featurizer persistence     |
| 5     | Model training and cross-validation             |
| 6     | SHAP explainability                             |
| 7     | Score/band/recommendation decision logic        |
| 8     | Flask API with persistence                      |
| 9     | React frontend                                  |
| 10    | Tests, polish, documentation                    |

## Current Status

**Phase:** 1 — Project setup (in progress)

## How to Run

Instructions will be added as each phase is completed.

## License

This project is for educational and placement preparation purposes.
