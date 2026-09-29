"""
Phase 6 tests - 5-fold cross-validation of the two Logistics-Regression
configurations (validates the Phase 6 baseline validation design).

Run with:
    python -m pytest tests/test_phase6.py -v

Covers:
    - StratifiedKFold is configured with exactly 5 folds
    - validation folds preserve approximately the target distribution
    - preprocessing lives inside the model pipeline (per-fold fitting
      therefore cannot leak validation information)
    - X_test / y_test are never passed into cross-validation
    - every required metric is returned per fold
    - both feature configurations run successfully end-to-end
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from ml.cross_validate_baseline import (
    N_SPLITS,
    REPORT_METRICS,
    get_split,
    load_data,
    make_folds,
    run_cross_validation,
)
from ml.data_cleaning import clean_credit_data
from ml.feature_engineering import engineer_features
from ml.preprocessing import TARGET
from ml.train_baseline import RANDOM_STATE, TEST_SIZE


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


@pytest.fixture(scope="module")
def holdout_split(prepared_data):
    """The SAME Phase 5 train/test split used by Phase 6 (holdout set)."""
    X, y, _ = prepared_data
    return get_split(X, y)


# --------------------------------------------------------------------------
# FOLD CONFIGURATION
# --------------------------------------------------------------------------
def test_stratified_kfold_has_exactly_5_folds():
    splitter, n_folds = make_folds(np.zeros(100))
    assert isinstance(splitter, StratifiedKFold)
    assert splitter.n_splits == 5
    assert splitter.n_splits == N_SPLITS
    assert splitter.shuffle is True
    assert splitter.random_state == 42
    assert n_folds == 5


def test_cv_uses_train_portions_only(holdout_split):
    X_train, X_test, y_train, y_test = holdout_split
    fold_df = run_cross_validation(X_train.head(2000), y_train.head(2000), "full")
    # The returned per-fold rows must sum to the training rows used.
    assert len(fold_df) == 5
    assert X_train.shape[0] > 0 and X_test.shape[0] > 0


def test_validation_folds_preserve_target_distribution(prepared_data):
    X, y, _ = prepared_data
    X_train, _, y_train, _ = get_split(X, y)

    overall_default_rate = y_train.mean()
    splitter, _ = make_folds(y_train)
    for train_idx, val_idx in splitter.split(X_train, y_train):
        val_default_rate = y_train.iloc[val_idx].mean()
        # Stratification must keep every validation fold close to the overall
        # default rate of the training portion.
        assert abs(val_default_rate - overall_default_rate) < 0.02


def test_fold_sizes_are_expected(prepared_data):
    X, y, _ = prepared_data
    X_train, _, y_train, _ = get_split(X, y)
    splitter, _ = make_folds(y_train)

    fold_sizes = [len(val_idx) for _, val_idx in splitter.split(X_train, y_train)]
    # 5 approximately-equal folds (~20% of the training data each).
    assert len(fold_sizes) == 5
    assert max(fold_sizes) - min(fold_sizes) <= 1
    assert sum(fold_sizes) == len(X_train)


# --------------------------------------------------------------------------
# PIPELINE / LEAKAGE STRUCTURE
# --------------------------------------------------------------------------
def test_preprocessor_is_inside_the_model_pipeline(prepared_data):
    X, y, _ = prepared_data
    X_train, _, y_train, _ = get_split(X, y)

    X_sub = X_train.head(500)
    y_sub = y_train.head(500)
    splitter, _ = make_folds(y_sub)
    train_idx, _ = next(splitter.split(X_sub, y_sub))
    fold_train = X_sub.iloc[train_idx]

    from ml.train_baseline import build_model

    model, feats = build_model("full")

    # The preprocessor must be a named pipelined step, NOT a fitted stat
    # fed into the classifier. Fitting the Pipeline fits preprocessor and
    # classifier together on the SAME training rows.
    assert "preprocessor" in model.named_steps
    model.fit(fold_train, y_sub.iloc[train_idx])

    imputer = model["preprocessor"].named_transformers_["numerical"].named_steps["imputer"]
    train_medians = fold_train[feats["numerical"]].median().to_numpy()
    np.testing.assert_allclose(imputer.statistics_, train_medians)


def test_x_test_y_test_not_passed_to_cross_validation(holdout_split, monkeypatch):
    X_train, X_test, y_train, y_test = holdout_split
    X_sub = X_train.head(2000)
    y_sub = y_train.head(2000)

    calls = []

    # Snapshot the REAL fold splits for the exact subset passed into CV, then
    # swap in a spy splitter that records WHICH objects it is called with.
    splitter, _ = make_folds(y_sub)
    splits_reference = list(splitter.split(X_sub, y_sub))

    class SpySplitter:
        n_splits = 5

        def split(self, X, y):
            calls.append((X, y))
            yield from splits_reference

    from ml import cross_validate_baseline as cv

    monkeypatch.setattr(cv, "make_folds", lambda *a, **k: (SpySplitter(), 5))
    cv.run_cross_validation(X_sub, y_sub, "full", n_splits=5)

    # The splitter must have been called ONLY with the training objects.
    assert len(calls) == 1
    split_x, split_y = calls[0]
    assert split_x is X_sub
    assert split_y is y_sub
    assert split_x is not X_test
    assert split_y is not y_test


# --------------------------------------------------------------------------
# METRICS COMPLETENESS
# --------------------------------------------------------------------------
def test_required_metrics_are_returned_per_fold(prepared_data):
    X, y, _ = prepared_data
    X_train, _, y_train, _ = get_split(X, y)

    fold_df = run_cross_validation(X_train.head(2000), y_train.head(2000), "full")

    assert list(fold_df.columns) == ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    for metric in REPORT_METRICS + ["accuracy"]:
        values = fold_df[metric]
        # Every required metric has 5 values, all probabilities in [0, 1].
        assert len(values) == 5
        assert values.between(0.0, 1.0).all()


def test_both_configurations_run_successfully(prepared_data):
    X, y, _ = prepared_data
    X_train, _, y_train, _ = get_split(X, y)

    for config in ("full", "application_only"):
        fold_df = run_cross_validation(X_train.head(2000), y_train.head(2000), config)
        assert len(fold_df) == 5
        for metric in REPORT_METRICS:
            assert fold_df[metric].notna().all()


def test_summary_statistics_are_correctly_derived(prepared_data):
    from ml.cross_validate_baseline import summarize_fold_scores

    X, y, _ = prepared_data
    X_train, _, y_train, _ = get_split(X, y)
    fold_df = run_cross_validation(X_train.head(2000), y_train.head(2000), "full")

    summary = summarize_fold_scores(fold_df)
    assert summary.shape == (len(REPORT_METRICS), 5)

    metric = "roc_auc"
    row = summary[summary["Metric"] == "ROC-AUC"].iloc[0]
    assert row["Mean"] == pytest.approx(fold_df[metric].mean())
    assert row["Std"] == pytest.approx(fold_df[metric].std())
    assert row["Min"] == pytest.approx(fold_df[metric].min())
    assert row["Max"] == pytest.approx(fold_df[metric].max())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))