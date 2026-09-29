"""
Phase 7 tests - four-model comparison on the APPLICATION-ONLY feature set.

Run with:
    python -m pytest tests/test_phase7.py -v

Covers:
    - loan_grade / loan_int_rate are excluded from every application-only
      candidate model
    - all four models share the exact same folds
    - preprocessing stays INSIDE the model pipeline (per-fold fitting cannot
      leak validation information)
    - cross-validation never touches the holdout X_test / y_test
    - all reported metrics are finite
    - every predicted probability falls inside [0, 1]
"""

import numpy as np
import pandas as pd
import pytest

from ml.compare_models import (
    CANDIDATE_NAMES,
    MODEL_REGISTRY,
    build_model_pipeline,
    make_comparison_table,
)
from ml.cross_validate_baseline import make_folds
from ml.data_cleaning import clean_credit_data
from ml.feature_engineering import engineer_features
from ml.preprocessing import TARGET, feature_lists
from ml.train_baseline import build_model, get_split, load_data


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
    """The SAME Phase 5/6 train/test split used across the project."""
    X, y, _ = prepared_data
    return get_split(X, y)


@pytest.fixture(scope="module")
def results_small(holdout_split):
    """Full 5-fold CV of all four models on a small subset (fast)."""
    X_train, _, y_train, _ = holdout_split
    from ml.compare_models import run_comparison

    return run_comparison(X_train.head(1200), y_train.head(1200))


# --------------------------------------------------------------------------
# FEATURE EXCLUSION
# --------------------------------------------------------------------------
def test_application_only_features_exclude_loan_grade_and_loan_int_rate():
    features = feature_lists("application_only")
    assert features["ordinal"] == {}
    assert "loan_grade" not in [
        c for cols in features.values() if isinstance(cols, list) for c in cols
    ]
    assert "loan_int_rate" not in features["numerical"]
    assert "loan_int_rate" not in [
        c for cols in features.values() if isinstance(cols, list) for c in cols
    ]


def test_every_candidate_pipeline_excludes_leakage_sensitive_features():
    for name in CANDIDATE_NAMES:
        model = build_model_pipeline(name)
        assert set(model.named_steps) == {"preprocessor", "classifier"}
        preprocessor = model["preprocessor"]
        # Application-only has no ordinal encoder (loan_grade is dropped).
        assert "ordinal_loan_grade" not in [
            t for t, _, _ in preprocessor.transformers
        ]
        # Neither lender-assigned field may appear in any transformer's columns.
        for _, _, columns in preprocessor.transformers:
            if isinstance(columns, list):
                assert "loan_grade" not in columns
                assert "loan_int_rate" not in columns


def test_build_model_with_estimator_keeps_feature_groups():
    from sklearn.ensemble import RandomForestClassifier

    model, features = build_model("application_only", estimator=RandomForestClassifier())
    assert "preprocessor" in model.named_steps
    assert "loan_grade" not in features["ordinal"]
    assert "loan_int_rate" not in features["numerical"]


# --------------------------------------------------------------------------
# SHARED FOLDS / TRAIN-ONLY DATA
# --------------------------------------------------------------------------
def test_all_models_share_the_same_folds_and_train_only_data(holdout_split, monkeypatch):
    """The exact same fold indices are used for every model, on train data only."""
    import ml.cross_validate_baseline as cv
    from ml.compare_models import run_comparison

    X_train, X_test, y_train, y_test = holdout_split
    X_sub = X_train.head(800)
    y_sub = y_train.head(800)

    used_splits = []
    used_objects = []

    real_splitter, _ = make_folds(y_sub)
    reference_splits = list(real_splitter.split(X_sub, y_sub))

    class SpySplitter:
        n_splits = 5

        def split(self, X, y):
            used_objects.append((X, y))
            used_splits.append([(a.tolist(), b.tolist()) for a, b in reference_splits])
            yield from reference_splits

    monkeypatch.setattr(cv, "make_folds", lambda *a, **k: (SpySplitter(), 5))
    results = run_comparison(X_sub, y_sub)

    # One CV pass per model, all on the SAME 5 fold assignments.
    assert len(results) == len(CANDIDATE_NAMES) == len(used_splits)
    for used in used_splits[1:]:
        assert used == used_splits[0]

    # The splitter was only ever handed the training objects.
    for X_seen, y_seen in used_objects:
        assert X_seen is X_sub
        assert y_seen is y_sub
        assert X_seen is not X_test
        assert y_seen is not y_test


# --------------------------------------------------------------------------
# PREPROCESSING INSIDE THE PIPELINE
# --------------------------------------------------------------------------
def test_preprocessed_fitted_inside_pipeline_on_fold_train_only(holdout_split):
    X_train, _, y_train, _ = holdout_split
    X_sub = X_train.head(800)
    y_sub = y_train.head(800)

    splitter, _ = make_folds(y_sub)
    train_idx, _ = next(splitter.split(X_sub, y_sub))
    fold_train = X_sub.iloc[train_idx]

    model = build_model_pipeline("logistic_regression")
    model.fit(fold_train, y_sub.iloc[train_idx])

    # The preprocessor must be a named pipeline step fitted on the fold
    # training rows alone (imputer medians come from those rows only).
    imputer = model["preprocessor"].named_transformers_["numerical"].named_steps["imputer"]
    _, features = build_model("application_only")
    train_medians = fold_train[features["numerical"]].median().to_numpy()
    np.testing.assert_allclose(imputer.statistics_, train_medians)


# --------------------------------------------------------------------------
# METRIC AND PROBABILITY SANITY
# --------------------------------------------------------------------------
def test_all_reported_metrics_are_finite(results_small):
    for name in CANDIDATE_NAMES:
        fold_df = results_small[name]
        assert list(fold_df.columns) == [
            "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fit_time_s",
        ]
        for column in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
            values = fold_df[column]
            assert len(values) == 5
            assert np.isfinite(values).all()
            assert values.between(0.0, 1.0).all()
        # Fit time must be a positive, finite number.
        assert (fold_df["fit_time_s"] > 0).all()
        assert np.isfinite(fold_df["fit_time_s"]).all()


def test_predicted_probabilities_are_in_unit_range(holdout_split):
    X_train, _, y_train, _ = holdout_split
    X_fit = X_train.head(1000)
    y_fit = y_train.head(1000)
    X_score = X_train.iloc[1000:1200]
    y_score = y_train.iloc[1000:1200]

    for name in CANDIDATE_NAMES:
        model = build_model_pipeline(name)
        model.fit(X_fit, y_fit)
        proba = model.predict_proba(X_score)[:, 1]
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0
        assert np.isfinite(proba).all()
        np.testing.assert_allclose(proba, model.predict_proba(X_score)[:, 1])


# --------------------------------------------------------------------------
# COMPARISON TABLE
# --------------------------------------------------------------------------
def test_comparison_table_sorted_by_pr_auc():
    def fake_folds(pr, roc, rec, f1, prec):
        return pd.DataFrame(
            {
                "fold": [1, 2, 3, 4, 5],
                "pr_auc": [pr] * 5,
                "roc_auc": [roc] * 5,
                "precision": [prec] * 5,
                "recall": [rec] * 5,
                "f1": [f1] * 5,
                "fit_time_s": [1.0] * 5,
            }
        ).set_index("fold")

    results = {
        "logistic_regression": fake_folds(0.64, 0.82, 0.43, 0.54, 0.73),
        "random_forest": fake_folds(0.78, 0.88, 0.52, 0.62, 0.72),
        "xgboost": fake_folds(0.80, 0.90, 0.55, 0.63, 0.74),
        "catboost": fake_folds(0.82, 0.91, 0.56, 0.64, 0.75),
    }
    table = make_comparison_table(results)
    pr_auc_order = table["PR-AUC Mean"].tolist()
    assert pr_auc_order == sorted(pr_auc_order, reverse=True)
    # Reference row appears in the table.
    assert "Logistic Regression (FULL ref)" in table["Model"].tolist()
    assert table.shape[0] == len(CANDIDATE_NAMES) + 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))