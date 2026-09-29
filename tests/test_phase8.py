"""
Phase 8 tests - class imbalance strategy comparison (XGBoost & CatBoost).

Run with:
    python -m pytest tests/test_phase8.py -v

Covers:
    - class-weight helpers derive weights from labels only (correct formulas)
    - all six experiment pipelines exclude loan_grade / loan_int_rate
    - the class_weighted builder receives fold-TRAINING labels only
    - SMOTE lives inside an imblearn Pipeline and never resamples validation
    - exactly six experiments run with identical 5-fold CV splits
    - all metrics are finite and probabilities stay in [0, 1]
    - the frozen test set is never passed to cross-validation
    - primary table has the 6 expected rows; delta table matches baselines
"""

import numpy as np
import pandas as pd
import pytest

from ml.compare_models import prepare_data
from ml.cross_validate_baseline import make_folds, run_cv_scores
from ml.data_cleaning import clean_credit_data
from ml.feature_engineering import engineer_features
from ml.imbalance_experiment import (
    EXPERIMENTS,
    MODELS,
    STRATEGIES,
    STRATEGY_LABELS,
    build_experiment_pipeline,
    catboost_balanced_weights,
    make_delta_table,
    make_primary_table,
    run_experiments,
    xgb_scale_pos_weight,
)
from ml.preprocessing import TARGET
from ml.train_baseline import get_split, load_data


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
    """The SAME Phase 5/6/7 train/test split used across the project."""
    X, y, _ = prepared_data
    return get_split(X, y)


@pytest.fixture(scope="module")
def results_small(holdout_split):
    """All six experiments on a small subset (fast)."""
    X_train, _, y_train, _ = holdout_split
    return run_experiments(X_train.head(800), y_train.head(800))


# --------------------------------------------------------------------------
# CLASS-WEIGHT HELPERS
# --------------------------------------------------------------------------
def test_xgb_scale_pos_weight_formula():
    y = pd.Series([0] * 60 + [1] * 40)
    assert xgb_scale_pos_weight(y) == pytest.approx(60 / 40)


def test_xgb_scale_pos_weight_uses_given_labels_only():
    y = np.array([0] * 9 + [1] * 1)
    assert xgb_scale_pos_weight(y) == pytest.approx(9.0)


def test_catboost_balanced_weights_formula():
    y = pd.Series([0] * 60 + [1] * 40)
    w0, w1 = catboost_balanced_weights(y)
    assert w0 == pytest.approx(100 / (2 * 60))
    assert w1 == pytest.approx(100 / (2 * 40))
    # Balanced touches every sample once on average.
    assert 60 * w0 + 40 * w1 == pytest.approx(100.0)


def test_weight_helpers_reject_single_class():
    with pytest.raises(ValueError):
        xgb_scale_pos_weight(np.array([0, 0, 0]))
    with pytest.raises(ValueError):
        catboost_balanced_weights(np.array([1, 1, 1]))


# --------------------------------------------------------------------------
# FEATURE EXCLUSION FOR ALL SIX EXPERIMENTS
# --------------------------------------------------------------------------
def test_all_six_pipelines_exclude_leakage_sensitive_features(holdout_split):
    X_train, _, y_train, _ = holdout_split
    y_ft = y_train.head(200)

    for model, strategy in EXPERIMENTS:
        pipeline = build_experiment_pipeline(model, strategy, y_fold_train=y_ft)
        # Every pipeline has a preprocessor + classifier step.
        assert "classifier" in pipeline.named_steps

        preprocessor = pipeline["preprocessor"]
        transformer_names = [t for t, _, _ in preprocessor.transformers]
        assert "ordinal_loan_grade" not in transformer_names
        for _, _, columns in preprocessor.transformers:
            if isinstance(columns, list):
                assert "loan_grade" not in columns
                assert "loan_int_rate" not in columns


# --------------------------------------------------------------------------
# CLASS-WEIGHTED: TRAINING LABELS ONLY
# --------------------------------------------------------------------------
def test_class_weighted_builder_uses_fold_training_labels(holdout_split):
    X_train, _, y_train, _ = holdout_split
    y_ft = y_train.head(200)

    for model in MODELS:
        pipeline = build_experiment_pipeline(model, "class_weighted", y_fold_train=y_ft)
        estimator = pipeline["classifier"]
        if model == "xgboost":
            assert estimator.get_params()["scale_pos_weight"] == pytest.approx(
                xgb_scale_pos_weight(y_ft)
            )
        else:  # catboost
            assert estimator.get_params()["class_weights"] == pytest.approx(
                catboost_balanced_weights(y_ft)
            )


def test_fold_labels_passed_to_builder_are_train_only(holdout_split, monkeypatch):
    """The builder receives the fold's TRAINING labels, never validation."""
    import ml.cross_validate_baseline as cv

    X_train, X_test, y_train, y_test = holdout_split
    X_sub = X_train.head(400)
    y_sub = y_train.head(400)

    seen_labels = []
    real_splitter, _ = make_folds(y_sub)
    reference_splits = list(real_splitter.split(X_sub, y_sub))

    class SpySplitter:
        n_splits = 5

        def split(self, X, y):
            yield from reference_splits

    def builder(X_ft, y_ft, seen=seen_labels):
        seen.append((X_ft.index.tolist(), y_ft.index.tolist()))
        return build_experiment_pipeline("xgboost", "baseline")

    monkeypatch.setattr(cv, "make_folds", lambda *a, **k: (SpySplitter(), 5))
    run_cv_scores(X_sub, y_sub, builder, n_splits=5)

    # The builder saw exactly the 5 fold-TRAINING label sets.
    assert len(seen_labels) == 5
    for train_idx, val_idx in reference_splits:
        expected_index = y_sub.index[train_idx].tolist()
        seen_x_index, seen_y_index = seen_labels.pop(0)
        assert seen_y_index == expected_index
        assert seen_x_index == expected_index  # same row alignment as labels


# --------------------------------------------------------------------------
# SMOTE SAFETY
# --------------------------------------------------------------------------
def test_smote_is_inside_pipeline_and_after_preprocessing():
    from imblearn.pipeline import Pipeline as ImbPipeline

    for model in MODELS:
        pipeline = build_experiment_pipeline(model, "smote")
        assert isinstance(pipeline, ImbPipeline)
        step_names = [name for name, _ in pipeline.steps]
        assert step_names == ["preprocessor", "smote", "classifier"]
        assert "smote" in pipeline.named_steps


def test_smote_predict_does_not_resample_validation(holdout_split):
    """Validation is transformed but NEVER upsampled by SMOTE."""
    X_train, _, y_train, _ = holdout_split
    X_fit = X_train.head(800)
    y_fit = y_train.head(800)
    X_val = X_train.iloc[800:1000]
    y_val = y_train.iloc[800:1000]

    for model in MODELS:
        pipeline = build_experiment_pipeline(model, "smote")
        pipeline.fit(X_fit, y_fit)
        # One prediction per validation row - validation was not resampled.
        preds = pipeline.predict(X_val)
        proba = pipeline.predict_proba(X_val)[:, 1]
        assert len(preds) == len(X_val)
        assert len(proba) == len(X_val)
        assert proba.min() >= 0.0 and proba.max() <= 1.0
        np.testing.assert_allclose(proba, pipeline.predict_proba(X_val)[:, 1])


# --------------------------------------------------------------------------
# EXPERIMENT RUN + METRIC SANITY
# --------------------------------------------------------------------------
def test_exactly_six_experiments_expected():
    assert EXPERIMENTS == [
        ("xgboost", "baseline"),
        ("xgboost", "class_weighted"),
        ("xgboost", "smote"),
        ("catboost", "baseline"),
        ("catboost", "class_weighted"),
        ("catboost", "smote"),
    ]


def test_all_experiments_report_finite_metrics(results_small):
    assert len(results_small) == 6
    for key, fold_df in results_small.items():
        assert fold_df.shape[0] == 5  # 5 folds per experiment
        for column in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
            values = fold_df[column]
            assert len(values) == 5
            assert np.isfinite(values).all()
            assert values.between(0.0, 1.0).all()
        assert (
            fold_df["fit_time_s"] > 0
        ).all() and np.isfinite(fold_df["fit_time_s"]).all()


def test_frozen_test_set_never_used_in_cv(holdout_split, monkeypatch):
    """The splitter (and therefore the models) only sees training objects."""
    import ml.cross_validate_baseline as cv

    X_train, X_test, y_train, y_test = holdout_split
    X_sub = X_train.head(400)
    y_sub = y_train.head(400)

    seen_objects = []
    real_splitter, _ = make_folds(y_sub)
    reference_splits = list(real_splitter.split(X_sub, y_sub))

    class SpySplitter:
        n_splits = 5

        def split(self, X, y):
            seen_objects.append((X, y))
            yield from reference_splits

    monkeypatch.setattr(cv, "make_folds", lambda *a, **k: (SpySplitter(), 5))

    def builder(X_ft, y_ft):
        return build_experiment_pipeline("xgboost", "baseline")

    run_cv_scores(X_sub, y_sub, builder, n_splits=5)

    assert len(seen_objects) == 1
    split_x, split_y = seen_objects[0]
    assert split_x is X_sub
    assert split_y is y_sub
    assert split_x is not X_test
    assert split_y is not y_test


# --------------------------------------------------------------------------
# TABLES
# --------------------------------------------------------------------------
def test_primary_table_has_expected_rows_and_order(results_small):
    table = make_primary_table(results_small)
    assert table.shape[0] == 6
    assert table["Strategy"].tolist() == [
        "Baseline", "Class Weighted", "SMOTE", "Baseline", "Class Weighted", "SMOTE",
    ]
    assert table["Model"].tolist() == ["XGBoost"] * 3 + ["CatBoost"] * 3


def test_delta_table_matches_baselines(results_small):
    table = make_delta_table(results_small)
    assert table.shape[0] == 6
    for _, row in table.iterrows():
        model = "xgboost" if row["Model"] == "XGBoost" else "catboost"
        strategy = next(
            s for s in STRATEGIES if STRATEGY_LABELS[s] == row["Strategy"]
        )
        base = results_small[(model, "baseline")]
        fold = results_small[(model, strategy)]
        assert row["dPR-AUC"] == pytest.approx(
            fold["pr_auc"].mean() - base["pr_auc"].mean()
        )
        assert row["dRecall"] == pytest.approx(
            fold["recall"].mean() - base["recall"].mean()
        )
    baseline_rows = table[table["Strategy"] == "Baseline"]
    assert (baseline_rows[["dPrecision", "dRecall", "dF1", "dPR-AUC"]] == 0.0).all().all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))