"""
Phase 5 tests - baseline model training and evaluation.

Run with:
    python -m pytest tests/test_phase5.py -v

Covers:
    - train/test split is stratified
    - both configurations share the exact same row split
    - the preprocessor fits on training data only (no leakage)
    - predicted probabilities stay within [0, 1]
    - evaluate.py returns every expected metric
    - both model pipelines train and predict without errors
"""

import numpy as np
import pandas as pd
import pytest

from ml.data_cleaning import clean_credit_data
from ml.evaluate import evaluate_classifier
from ml.feature_engineering import engineer_features
from ml.preprocessing import TARGET, build_logistic_preprocessor
from ml.train_baseline import (
    RANDOM_STATE,
    TEST_SIZE,
    build_model,
    get_split,
    load_data,
)

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def prepared_data():
    """Cleaned + engineered full dataset with X / y separation."""
    df = load_data()
    cleaned, _ = clean_credit_data(df)
    engineered, _ = engineer_features(cleaned)
    X = engineered.drop(columns=[TARGET])
    y = engineered[TARGET]
    return X, y, engineered


# --------------------------------------------------------------------------
# SPLIT CORRECTNESS
# --------------------------------------------------------------------------
def test_split_is_stratified(prepared_data):
    X, y, _ = prepared_data
    X_train, X_test, y_train, y_test = get_split(X, y)

    overall_default_rate = y.mean()
    # Stratification must keep both splits close to the overall default rate.
    assert abs(y_train.mean() - overall_default_rate) < 0.01
    assert abs(y_test.mean() - overall_default_rate) < 0.01

    # Row counts must match test_size.
    assert len(X_test) == int(round(len(X) * TEST_SIZE))
    assert len(X_train) == len(X) - len(X_test)


def test_split_counts_match_requested_sizes(prepared_data):
    X, y, _ = prepared_data
    X_train, X_test, y_train, y_test = get_split(X, y)
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)


def test_both_configurations_use_same_row_split(prepared_data):
    X, y, _ = prepared_data
    # Called twice with the same seed the split is identical, guaranteeing
    # that every configuration trains on the exact same rows.
    # NOTE: .equals() is used instead of == because the engineered features
    # contain NaN, and NaN == NaN is False with plain numpy comparisons.
    s1 = get_split(X, y)
    s2 = get_split(X, y)
    for a, b in zip(s1, s2):
        assert a.equals(b)


# --------------------------------------------------------------------------
# PREPROCESSING LEAKAGE
# --------------------------------------------------------------------------
def test_preprocessor_fits_on_training_data_only(prepared_data):
    X, y, _ = prepared_data
    X_train, X_test, y_train, y_test = get_split(X, y)

    model, feats = build_model("full")
    model.fit(X_train, y_train)

    # Fitting the Pipeline fits the preprocessor on X_train only. Verify the
    # numerical imputer's learned medians equal the medians computed from the
    # TRAINING data alone, for every numerical column in feature order.
    numerical_pipeline = model["preprocessor"].named_transformers_["numerical"]
    imputer = numerical_pipeline.named_steps["imputer"]

    train_medians = X_train[feats["numerical"]].median().to_numpy()
    np.testing.assert_allclose(imputer.statistics_, train_medians)


def test_transform_leaves_no_errors_on_test(prepared_data):
    X, y, _ = prepared_data
    X_train, X_test, y_train, y_test = get_split(X, y)
    model, _ = build_model("full")
    model.fit(X_train.head(2000), y_train.head(2000))
    # Must transform unseen test rows without an exception.
    model.predict(X_test.head(20))
    model.predict_proba(X_test.head(20))


# --------------------------------------------------------------------------
# MODEL PIPELINES
# --------------------------------------------------------------------------
def test_full_and_application_only_feature_sets():
    _, full_features = build_model("full")
    _, app_features = build_model("application_only")

    assert "loan_grade" in full_features["ordinal"]
    assert "loan_int_rate" in full_features["numerical"]
    assert app_features["ordinal"] == {}
    assert "loan_int_rate" not in app_features["numerical"]


def test_pipelines_train_and_predict(prepared_data):
    X, y, _ = prepared_data
    X_train, X_test, y_train, y_test = get_split(X, y)

    for config in ("full", "application_only"):
        model, _ = build_model(config)
        # Train on a representative subset to keep the test fast.
        model.fit(X_train.head(3000), y_train.head(3000))
        y_pred = model.predict(X_test.head(100))
        y_proba = model.predict_proba(X_test.head(100))[:, 1]

        assert y_pred.shape == (100,)
        assert y_proba.shape == (100,)


def test_probabilities_are_in_unit_range(prepared_data):
    X, y, _ = prepared_data
    X_train, X_test, y_train, y_test = get_split(X, y)

    model, _ = build_model("full")
    model.fit(X_train.head(3000), y_train.head(3000))
    y_proba = model.predict_proba(X_test.head(100))[:, 1]

    assert y_proba.min() >= 0.0
    assert y_proba.max() <= 1.0


# --------------------------------------------------------------------------
# EVALUATION MODULE
# --------------------------------------------------------------------------
def test_evaluate_classifier_returns_all_expected_metrics():
    y_true = np.array([0, 1, 1, 0, 1, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 1])
    y_proba = np.array([0.1, 0.9, 0.6, 0.2, 0.8, 0.3, 0.4, 0.1, 0.7, 0.95])

    metrics = evaluate_classifier(y_true, y_pred, y_proba)

    expected_keys = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "confusion_matrix",
    }
    assert expected_keys.issubset(metrics.keys())

    # Probability-based metrics are valid probabilities in [0, 1].
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    # Confusion matrix is presented as a labelled DataFrame.
    assert isinstance(metrics["confusion_matrix"], pd.DataFrame)
    assert metrics["confusion_matrix"].shape == (2, 2)
    # Matrix cells sum to the number of samples.
    assert metrics["confusion_matrix"].to_numpy().sum() == len(y_true)


def test_evaluate_classifier_handles_zero_division():
    # Model predicts every sample as negative -> no predicted positives.
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 0, 0, 0])
    y_proba = np.array([0.1, 0.2, 0.3, 0.2])

    metrics = evaluate_classifier(y_true, y_pred, y_proba)
    # zero_division=0 => precision/f1 collapse to 0.0 instead of raising.
    assert metrics["precision"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["recall"] == 0.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))