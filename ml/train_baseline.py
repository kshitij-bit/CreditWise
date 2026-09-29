"""
Phase 5 - Baseline Logistic Regression for CreditWise AI.

Trains TWO interpretable baseline classifiers on the exact same
train/test split so the comparison is fair:

    Model A - FULL          : includes loan_grade and loan_int_rate
    Model B - APPLICATION-ONLY : excludes loan_grade and loan_int_rate

loan_grade and loan_int_rate may carry lender-assigned information, so
Model B tells us how strong the model is using only applicant-supplied
information.

Run with:
    python ml/train_baseline.py

Design rules (deliberately enforced)
------------------------------------
    - One stratified train/test split, shared by both configurations
    - Preprocessor is fitted on X_train ONLY (no test-set leakage)
    - Default probability threshold 0.50
    - No SMOTE / ADASYN / class_weight / tuning / other models / SHAP
"""

import sys
from pathlib import Path

# Make the project root importable so `ml.*` modules work no matter from
# which folder this script is executed (e.g. python ml/train_baseline.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml.data_cleaning import clean_credit_data
from ml.evaluate import (
    evaluate_classifier,
    print_comparison_table,
    print_confusion_matrix,
)
from ml.feature_engineering import engineer_features
from ml.preprocessing import TARGET, build_logistic_preprocessor

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "credit_risk_dataset.csv"

TEST_SIZE = 0.20
RANDOM_STATE = 42

# The two configurations Phase 4 defined for the leakage comparison.
CONFIGS = ("full", "application_only")


# ---------------------------------------------------------------------------
# Reusable building blocks (small, testable functions)
# ---------------------------------------------------------------------------
def load_data(path=RAW_DATA_PATH):
    """Load the raw CSV into a DataFrame."""
    return pd.read_csv(path)


def get_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    ONE stratified train/test split, shared by both model configurations.

    Returns (X_train, X_test, y_train, y_test).
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def build_model(config, estimator=None):
    """
    Build (unfitted) a baseline model Pipeline.

    Preprocessing is the Phase 4 ColumnTransformer for the given config,
    followed by a classifier. The default classifier is a simple
    LogisticRegression (Phase 5 baseline); passing `estimator` lets Phase 7
    swap in other fixed-baseline estimators (RF / XGBoost / CatBoost) while
    reusing the EXACT same preprocessor and feature configuration.
    """
    preprocessor, features = build_logistic_preprocessor(config)

    if estimator is None:
        estimator = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )
    return model, features


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def print_dataset_facts(X_train, X_test, y_train, y_test):
    """Print dataset split facts and target distributions."""
    separator = "=" * 60
    print(separator)
    print("DATASET SPLIT FACTS")
    print(separator)

    def print_target_distribution(y, label):
        counts = y.value_counts().sort_index()
        percents = y.value_counts(normalize=True).sort_index() * 100
        for status, count in counts.items():
            print(f"    {label}: loan_status={status}: {count:,} ({percents[status]:.2f}%)")

    print(f"Cleaned dataset size : {len(X_train) + len(X_test):,} rows")
    print(f"Training samples     : {len(X_train):,}")
    print(f"Testing samples      : {len(X_test):,}")
    print("Training target distribution:")
    print_target_distribution(y_train, "  train")
    print("Test target distribution:")
    print_target_distribution(y_test, "  test")
    print(separator)


def print_interpretation(full_metrics, app_only_metrics):
    """
    Print an automated, cautious interpretation of the comparison.

    We quantify the gaps but never claim leakage is proven - only that
    loan_grade / loan_int_rate need deployment consideration.
    """
    roc_gap = full_metrics["roc_auc"] - app_only_metrics["roc_auc"]
    recall_gap = full_metrics["recall"] - app_only_metrics["recall"]
    f1_gap = full_metrics["f1"] - app_only_metrics["f1"]

    if roc_gap >= 0.05:
        gain_text = "substantial gain"
    elif roc_gap >= 0.01:
        gain_text = "moderate gain"
    else:
        gain_text = "small gain"

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print(f"Difference in ROC-AUC (FULL - APPLICATION-ONLY): {roc_gap:+.4f}")
    print(f"Difference in Recall  (FULL - APPLICATION-ONLY): {recall_gap:+.4f}")
    print(f"Difference in F1      (FULL - APPLICATION-ONLY): {f1_gap:+.4f}")
    print(f"-> The FULL model shows a {gain_text} in ROC-AUC from including "
          "loan_grade and loan_int_rate.")
    print("-> These two fields are lender-assigned/priced. This does NOT prove")
    print("   data leakage - it means they require deployment consideration:")
    print("   if the goal is to screen applicants BEFORE a grade/rate exists,")
    print("   the APPLICATION-ONLY model is the realistic one.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    print("Loading raw dataset...")
    df = load_data()
    print(f"  Loaded {len(df):,} rows.")

    # 1) Deterministic cleaning (duplicates + impossible records).
    cleaned, cleaning_summary = clean_credit_data(df)

    # 2) Deterministic feature engineering (ratios, log, binary flags).
    engineered, _ = engineer_features(cleaned)

    # 3) Separate features from the target.
    X = engineered.drop(columns=[TARGET])
    y = engineered[TARGET]

    # 4) ONE stratified split, reused for both configurations.
    X_train, X_test, y_train, y_test = get_split(X, y)
    print_dataset_facts(X_train, X_test, y_train, y_test)

    # 5) Train and evaluate both configurations.
    results = {}
    for config in CONFIGS:
        print(f"\nTraining MODEL ({config})...")

        model, features = build_model(config)
        # Fitting the Pipeline fits the preprocessor AND the model on the
        # training split only - the test set stays unseen.
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)               # default threshold 0.50
        y_proba = model.predict_proba(X_test)[:, 1]  # P(default)

        metrics = evaluate_classifier(y_test, y_pred, y_proba)
        results[config] = {"model": model, "features": features, "metrics": metrics}

        print(f"  Config fitted with {len(features['numerical'])} numerical, "
              f"{len(features['categorical'])} categorical, "
              f"{len(features['ordinal'])} ordinal features.")

    full_metrics = results["full"]["metrics"]
    app_only_metrics = results["application_only"]["metrics"]

    # 6) Comparison output.
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (test set, threshold 0.50)")
    print("=" * 60)
    print_comparison_table(full_metrics, app_only_metrics)

    print_confusion_matrix(full_metrics["confusion_matrix"], "CONFUSION MATRIX - FULL")
    print_confusion_matrix(
        app_only_metrics["confusion_matrix"], "CONFUSION MATRIX - APPLICATION-ONLY"
    )

    print_interpretation(full_metrics, app_only_metrics)

    print("\nPhase 5 complete. No production model was saved - results are printed only.")


if __name__ == "__main__":
    main()