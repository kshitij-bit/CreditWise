"""
Phase 11 tests - final frozen holdout evaluation.

Run with:
    python -m pytest tests/test_phase11.py -v

Covers:
    - exact locked threshold is 0.325 and lives on the Phase 10 grid
    - Phase 9 best XGBoost parameters are reused (canonical XGBOOST_PARAMS)
    - XGBoost never receives class weighting
    - APPLICATION-ONLY features are used (loan_grade / loan_int_rate excluded)
    - the locked pipeline fits ONLY on the training set
    - the holdout set is never passed to model fitting
    - holdout probabilities stay in [0, 1] and are finite
    - predicted labels match the locked-threshold rule
    - confusion-matrix totals equal the holdout sample count
    - all required metrics are finite
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix

import ml.final_evaluation as fe
from ml.threshold_analysis import THRESHOLDS, XGBOOST_PARAMS, threshold_metrics
from ml.data_cleaning import clean_credit_data
from ml.feature_engineering import engineer_features
from ml.preprocessing import TARGET
from ml.train_baseline import get_split, load_data


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def holdout_split():
    """Same Phase 5-10 stratified holdout split used across the project."""
    df = load_data()
    cleaned, _ = clean_credit_data(df)
    engineered, _ = engineer_features(cleaned)
    X = engineered.drop(columns=[TARGET])
    y = engineered[TARGET]
    return get_split(X, y)


@pytest.fixture(scope="module")
def holdout_evaluation(holdout_split):
    """Mini end-to-end: fit on a 400-row training subset, evaluate the full holdout."""
    X_train, X_test, y_train, y_test = holdout_split
    X_sub, y_sub = X_train.head(400), y_train.head(400)
    results, y_proba = fe.run_final_evaluation(X_sub, y_sub, X_test, y_test)
    return results, y_proba, y_test


# --------------------------------------------------------------------------
# LOCKED CONFIGURATION
# --------------------------------------------------------------------------
def test_locked_threshold_is_0325():
    assert fe.LOCKED_THRESHOLD == pytest.approx(0.325)
    assert fe.LOCKED_THRESHOLD in THRESHOLDS


def test_phase9_params_reused_correctly():
    pipeline = fe.build_final_pipeline()
    classifier = pipeline.named_steps["classifier"]
    params = classifier.get_params()
    for key in ("n_estimators", "max_depth", "learning_rate",
                "subsample", "colsample_bytree"):
        assert params[key] == XGBOOST_PARAMS[key]
    # XGBoost candidate never receives class weighting.
    assert params.get("scale_pos_weight") in (None, 1)


def test_application_only_features_used():
    pipeline = fe.build_final_pipeline()
    preprocessor = pipeline.named_steps["preprocessor"]
    transformer_names = [t for t, _, _ in preprocessor.transformers]
    assert "ordinal_loan_grade" not in transformer_names
    for _, _, columns in preprocessor.transformers:
        if isinstance(columns, list):
            assert "loan_grade" not in columns
            assert "loan_int_rate" not in columns


# --------------------------------------------------------------------------
# TRAINING-SET-ONLY FITTING
# --------------------------------------------------------------------------
def test_model_fits_only_on_training_set(holdout_split, monkeypatch):
    X_train, X_test, y_train, y_test = holdout_split
    X_sub, y_sub = X_train.head(300), y_train.head(300)

    import sklearn.pipeline as pl

    fit_calls = []
    original_fit = pl.Pipeline.fit

    def spy_fit(self, X, y=None, **params):
        fit_calls.append((X, y))
        return original_fit(self, X, y, **params)

    monkeypatch.setattr(pl.Pipeline, "fit", spy_fit)
    fe.run_final_evaluation(X_sub, y_sub, X_test, y_test)

    assert len(fit_calls) == 1
    fit_x, fit_y = fit_calls[0]
    assert fit_x is X_sub and fit_y is y_sub
    assert fit_x is not X_test and fit_y is not y_test


# --------------------------------------------------------------------------
# HOLDOUT PREDICTIONS
# --------------------------------------------------------------------------
def test_holdout_probabilities_in_unit_interval(holdout_evaluation):
    results, y_proba, _ = holdout_evaluation
    assert len(y_proba) == results["counts"]["total"]
    assert np.isfinite(y_proba).all()
    assert (y_proba >= 0.0).all()
    assert (y_proba <= 1.0).all()


def test_locked_threshold_predictions_match_rule(holdout_evaluation):
    results, y_proba, _ = holdout_evaluation
    expected = (y_proba >= fe.LOCKED_THRESHOLD).astype(int)
    assert np.array_equal(results["y_pred"], expected)


# --------------------------------------------------------------------------
# CONFUSION MATRIX CONSISTENCY
# --------------------------------------------------------------------------
def test_confusion_totals_equal_holdout_size(holdout_evaluation):
    results, _, _ = holdout_evaluation
    c = results["classification"]
    assert c["tn"] + c["fp"] + c["fn"] + c["tp"] == results["counts"]["total"]
    assert c["tn"] + c["fp"] + c["fn"] + c["tp"] == len(results["y_pred"])


def test_confusion_counts_match_sklearn_recompute(holdout_evaluation):
    # Recompute the confusion matrix directly from scikit-learn using the real
    # holdout labels + locked-threshold predictions.
    results, y_proba, y_test = holdout_evaluation
    tn, fp, fn, tp = confusion_matrix(
        y_test, results["y_pred"], labels=[0, 1]
    ).ravel()
    c = results["classification"]
    assert (c["tn"], c["fp"], c["fn"], c["tp"]) == (int(tn), int(fp), int(fn), int(tp))


def test_synthetic_evaluate_holdout_confusion_totals():
    manual = fe.evaluate_holdout(
        np.array([0, 1, 1, 0, 1]),
        np.array([0.2, 0.9, 0.6, 0.1, 0.4]),
        threshold=0.325,
    )
    c = manual["classification"]
    assert c["tp"] + c["fn"] == 3  # all actual positives accounted for
    assert c["tn"] + c["fp"] == 2  # all actual negatives accounted for
    assert c["precision"] == pytest.approx(1.0)  # no false positives predicted
    assert c["recall"] == pytest.approx(1.0)  # all positives correctly caught


def test_classification_matches_threshold_metrics(holdout_evaluation):
    # The shared metric helper must produce identical classification numbers.
    results, y_proba, y_test = holdout_evaluation
    c = results["classification"]
    shared = threshold_metrics(y_test, y_proba, fe.LOCKED_THRESHOLD)
    for key in ("precision", "recall", "f1", "f2", "specificity", "fpr", "fnr",
                "tn", "fp", "fn", "tp"):
        assert c[key] == pytest.approx(shared[key])


# --------------------------------------------------------------------------
# METRIC FINITENESS
# --------------------------------------------------------------------------
def test_all_metrics_finite(holdout_evaluation):
    results, _, _ = holdout_evaluation
    for key, value in results["ranking"].items():
        assert np.isfinite(value), key
    for key, value in results["classification"].items():
        assert np.isfinite(value), key
    for key, value in results["counts"].items():
        assert value >= 0, key


# --------------------------------------------------------------------------
# OOF REFERENCE HELPER
# --------------------------------------------------------------------------
def test_oof_reference_structure(holdout_split, monkeypatch):
    X_train, _, y_train, _ = holdout_split

    y = np.array([0, 1, 1, 0, 1, 0])
    p = np.array([0.1, 0.9, 0.6, 0.2, 0.7, 0.3])
    fake_df = pd.DataFrame({"fold": [1, 2, 3]})

    class FakeOof:
        def __call__(self, *args, **kwargs):
            return y, p, fake_df

    monkeypatch.setattr(fe, "run_oof_predictions", FakeOof())
    ref = fe.recompute_oof_reference(X_train, y_train)
    for key in ("roc_auc", "pr_auc", "precision", "recall", "f1",
                "oof_best_f1_threshold"):
        assert np.isfinite(ref[key])
    assert ref["oof_best_f1_threshold"] in THRESHOLDS


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))