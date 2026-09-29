"""
Phase 10 tests - threshold optimization and operating-point analysis.

Run with:
    python -m pytest tests/test_phase10.py -v

Covers:
    - threshold grid: 0.20-0.70 step 0.025, 0.50 present, all within (0, 1)
    - threshold_metrics purity: counts sum, specificity+FPR=1, F2 formula,
      precision/recall from counts, metrics finite
    - best-F1 / best-F2 selection with ties preferring the higher threshold
    - OOF runner: exactly one prediction per training row, no row predicted
      by a model trained on itself, alignment with y_train, finite proba
    - confusion-matrix totals equal the number of OOF samples
    - CatBoost fold class weights derive from fold-TRAINING labels only
    - XGBoost never receives class weighting
    - application-only features remain enforced (no loan_grade/loan_int_rate)
    - the frozen holdout set is never passed to the OOF runner
    - mini end-to-end OOF runs for both Phase 9 finalists
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, confusion_matrix, fbeta_score, roc_auc_score

import ml.threshold_analysis as ta
from ml.cross_validate_baseline import make_folds
from ml.threshold_analysis import (
    DEFAULT_THRESHOLD,
    THRESHOLDS,
    best_f1_threshold,
    best_f2_threshold,
    build_pipeline,
    operating_point_rows,
    run_oof_predictions,
    sweep_thresholds,
    threshold_metrics,
)
from ml.data_cleaning import clean_credit_data
from ml.feature_engineering import engineer_features
from ml.imbalance_experiment import catboost_balanced_weights
from ml.preprocessing import TARGET
from ml.train_baseline import get_split, load_data


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def holdout_split():
    """Same Phase 5/6/7/8/9 holdout split used across the project."""
    df = load_data()
    cleaned, _ = clean_credit_data(df)
    engineered, _ = engineer_features(cleaned)
    X = engineered.drop(columns=[TARGET])
    y = engineered[TARGET]
    return get_split(X, y)


@pytest.fixture(scope="module")
def subset(holdout_split):
    """Small contiguous training subset so mini OOF runs stay fast."""
    X_train, _, y_train, _ = holdout_split
    return X_train.head(500), y_train.head(500)


@pytest.fixture(scope="module")
def oof_results(subset):
    """Mini out-of-fold runs for both Phase 9 finalists on the subset."""
    X_sub, y_sub = subset
    out = {}
    for model in ("xgboost", "catboost"):
        oof_y, oof_proba, fold_df = run_oof_predictions(
            X_sub,
            y_sub,
            lambda X_ft, y_ft, m=model: build_pipeline(m, y_fold_train=y_ft),
            n_splits=5,
        )
        out[model] = (oof_y, oof_proba, fold_df)
    return out


# Synthetic data for the pure metric helpers.
Y_TRUE = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0])
Y_PROBA = np.array([0.15, 0.85, 0.65, 0.30, 0.55, 0.40, 0.10, 0.75, 0.90, 0.05])


# --------------------------------------------------------------------------
# THRESHOLD GRID
# --------------------------------------------------------------------------
def test_threshold_grid_bounds_and_step():
    assert THRESHOLDS[0] == pytest.approx(0.20)
    assert THRESHOLDS[-1] == pytest.approx(0.70)
    steps = np.diff(np.asarray(THRESHOLDS))
    assert np.allclose(steps, 0.025)
    assert THRESHOLDS == sorted(THRESHOLDS)


def test_threshold_grid_includes_default():
    assert DEFAULT_THRESHOLD in THRESHOLDS


def test_threshold_grid_within_unit_interval():
    assert all(0.0 < t < 1.0 for t in THRESHOLDS)


# --------------------------------------------------------------------------
# PURE THRESHOLD METRICS
# --------------------------------------------------------------------------
@pytest.mark.parametrize("t", [0.20, 0.50, 0.70])
def test_threshold_metrics_counts_sum_to_total(t):
    metrics = threshold_metrics(Y_TRUE, Y_PROBA, t)
    assert metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"] == len(Y_TRUE)

    y_pred = (Y_PROBA >= t).astype(int)
    cm = confusion_matrix(Y_TRUE, y_pred, labels=[0, 1]).ravel()
    assert [metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]] == cm.tolist()


@pytest.mark.parametrize("t", [0.20, 0.50, 0.70])
def test_threshold_metrics_specificity_fpr_complement(t):
    metrics = threshold_metrics(Y_TRUE, Y_PROBA, t)
    assert metrics["specificity"] + metrics["fpr"] == pytest.approx(1.0)


@pytest.mark.parametrize("t", [0.20, 0.50, 0.70])
def test_threshold_metrics_f2_matches_fbeta(t):
    metrics = threshold_metrics(Y_TRUE, Y_PROBA, t)
    y_pred = (Y_PROBA >= t).astype(int)
    assert metrics["f2"] == pytest.approx(
        fbeta_score(Y_TRUE, y_pred, beta=2, zero_division=0)
    )


@pytest.mark.parametrize("t", [0.20, 0.50, 0.70])
def test_threshold_metrics_precision_recall_from_counts(t):
    metrics = threshold_metrics(Y_TRUE, Y_PROBA, t)
    tp, fp, fn = metrics["tp"], metrics["fp"], metrics["fn"]
    denom_prec = tp + fp
    denom_rec = tp + fn
    expected_precision = tp / denom_prec if denom_prec else 0.0
    expected_recall = tp / denom_rec if denom_rec else 0.0
    assert metrics["precision"] == pytest.approx(expected_precision)
    assert metrics["recall"] == pytest.approx(expected_recall)


@pytest.mark.parametrize("t", [0.20, 0.50, 0.70])
def test_threshold_metrics_values_finite(t):
    metrics = threshold_metrics(Y_TRUE, Y_PROBA, t)
    for key, value in metrics.items():
        if key in ("tn", "fp", "fn", "tp"):
            assert isinstance(value, (int, np.integer))
        else:
            assert np.isfinite(value)


def test_threshold_metrics_zero_division_does_not_crash():
    # A threshold above all probabilities -> zero positives -> 0/0 safe.
    metrics = threshold_metrics(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]), 0.95)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["f2"] == 0.0
    assert metrics["tn"] == 3 and metrics["fp"] == 0
    assert metrics["tp"] == 0 and metrics["fn"] == 0


def test_sweep_thresholds_columns_and_structure():
    sweep = sweep_thresholds(Y_TRUE, Y_PROBA)
    expected_columns = [
        "threshold", "precision", "recall", "f1", "f2", "specificity",
        "fpr", "fnr", "tn", "fp", "fn", "tp",
    ]
    assert list(sweep.columns) == expected_columns
    assert len(sweep) == len(THRESHOLDS)
    assert np.isclose(sweep["threshold"], DEFAULT_THRESHOLD).any()
    assert (sweep["threshold"].diff().iloc[1:] > 0).all()


# --------------------------------------------------------------------------
# OPERATING POINT SELECTION
# --------------------------------------------------------------------------
def _tie_sweep():
    return pd.DataFrame(
        {
            "threshold": [0.20, 0.30, 0.40, 0.50],
            "f1": [0.70, 0.80, 0.80, 0.70],
            "f2": [0.71, 0.81, 0.79, 0.75],
        }
    )


def test_best_f1_tie_prefers_higher_threshold():
    threshold, row = best_f1_threshold(_tie_sweep())
    assert threshold == pytest.approx(0.40)
    assert row["f1"] == pytest.approx(0.80)


def test_best_f2_tie_prefers_higher_threshold():
    threshold, row = best_f2_threshold(_tie_sweep())
    assert threshold == pytest.approx(0.30)
    assert row["f2"] == pytest.approx(0.81)


def test_best_threshold_unique_maximum():
    sweep = pd.DataFrame(
        {
            "threshold": [0.20, 0.25, 0.30, 0.35, 0.40],
            "f1": [0.60, 0.70, 0.75, 0.72, 0.68],
            "f2": [0.62, 0.71, 0.74, 0.72, 0.70],
        }
    )
    t_f1, row_f1 = best_f1_threshold(sweep)
    t_f2, row_f2 = best_f2_threshold(sweep)
    assert t_f1 == pytest.approx(0.30)
    assert t_f2 == pytest.approx(0.30)
    assert row_f1["f1"] == pytest.approx(0.75)
    assert row_f2["f2"] == pytest.approx(0.74)


def test_operating_point_rows_contains_three_points():
    sweep = sweep_thresholds(Y_TRUE, Y_PROBA)
    op = operating_point_rows(sweep)
    assert set(op.keys()) == {"DEFAULT", "BALANCED", "RISK-SENSITIVE"}
    assert op["DEFAULT"][0] == pytest.approx(DEFAULT_THRESHOLD)
    assert op["DEFAULT"][1]["threshold"] == pytest.approx(DEFAULT_THRESHOLD)
    assert op["BALANCED"][1]["f1"] == sweep["f1"].max()
    assert op["RISK-SENSITIVE"][1]["f2"] == sweep["f2"].max()


# --------------------------------------------------------------------------
# OOF PREDICTION INTEGRITY
# --------------------------------------------------------------------------
def test_real_folds_partition_without_self_overlap(subset):
    X_sub, y_sub = subset
    splitter, _ = make_folds(y_sub)
    val_chunks = []
    for _, (train_idx, val_idx) in enumerate(splitter.split(X_sub, y_sub)):
        assert len(set(train_idx) & set(val_idx)) == 0
        val_chunks.append(val_idx)
    covered = np.sort(np.concatenate(val_chunks))
    assert np.array_equal(covered, np.arange(len(y_sub)))


@pytest.mark.parametrize("model", ["xgboost", "catboost"])
def test_oof_length_and_alignment(oof_results, subset, model):
    oof_y, oof_proba, fold_df = oof_results[model]
    _, y_sub = subset
    assert len(oof_y) == len(y_sub)
    assert len(oof_proba) == len(y_sub)
    assert np.array_equal(oof_y, y_sub.to_numpy())
    assert fold_df.shape[0] == 5
    for col in ["roc_auc", "pr_auc"]:
        assert np.isfinite(fold_df[col]).all()


@pytest.mark.parametrize("model", ["xgboost", "catboost"])
def test_oof_probabilities_finite_and_in_unit_interval(oof_results, model):
    _, oof_proba, _ = oof_results[model]
    assert np.isfinite(oof_proba).all()
    assert (oof_proba >= 0.0).all()
    assert (oof_proba <= 1.0).all()


@pytest.mark.parametrize("model", ["xgboost", "catboost"])
def test_confusion_totals_equal_oof_samples(oof_results, model):
    oof_y, oof_proba, _ = oof_results[model]
    sweep = sweep_thresholds(oof_y, oof_proba)
    for _, row in sweep.iterrows():
        assert row["tn"] + row["fp"] + row["fn"] + row["tp"] == len(oof_y)


@pytest.mark.parametrize("model", ["xgboost", "catboost"])
def test_sweep_on_oof_metrics_finite(oof_results, model):
    oof_y, oof_proba, _ = oof_results[model]
    sweep = sweep_thresholds(oof_y, oof_proba)
    for _, row in sweep.iterrows():
        for col in ["precision", "recall", "f1", "f2", "specificity", "fpr", "fnr"]:
            assert np.isfinite(row[col])


@pytest.mark.parametrize("model", ["xgboost", "catboost"])
def test_sweep_on_oof_contains_default_row(oof_results, model):
    oof_y, oof_proba, _ = oof_results[model]
    sweep = sweep_thresholds(oof_y, oof_proba)
    default = sweep.loc[np.isclose(sweep["threshold"], DEFAULT_THRESHOLD)]
    assert len(default) == 1
    assert default.iloc[0]["threshold"] == pytest.approx(DEFAULT_THRESHOLD)


@pytest.mark.parametrize("model", ["xgboost", "catboost"])
def test_oof_aucs_plausible_and_consistent(oof_results, model):
    oof_y, oof_proba, fold_df = oof_results[model]
    roc = roc_auc_score(oof_y, oof_proba)
    pr = average_precision_score(oof_y, oof_proba)
    assert 0.5 < roc <= 1.0
    assert 0.0 < pr <= 1.0
    assert roc == pytest.approx(fold_df["roc_auc"].mean(), abs=0.05)
    assert pr == pytest.approx(fold_df["pr_auc"].mean(), abs=0.05)


def test_xgboost_never_gets_class_weighting():
    pipeline = build_pipeline("xgboost", y_fold_train=pd.Series([0] * 6 + [1] * 4))
    assert pipeline["classifier"].get_params().get("scale_pos_weight") in (None, 1)


def test_catboost_requires_fold_training_labels():
    with pytest.raises(ValueError):
        build_pipeline("catboost")


def test_build_pipeline_rejects_unknown_model():
    with pytest.raises(ValueError):
        build_pipeline("random_forest")


def test_catboost_fold_weights_use_train_labels_only(subset, monkeypatch):
    """Mirror of the Phase 9 spy-splitter test: weights match the fold's
    TRAINING labels exactly and never touch validation rows."""
    X_sub, y_sub = subset
    real_splitter, _ = make_folds(y_sub)
    reference_splits = list(real_splitter.split(X_sub, y_sub))

    class SpySplitter:
        n_splits = 5

        def split(self, X, y):
            yield from reference_splits

    captured = []

    def track_weights(X_ft, y_ft):
        pipeline = build_pipeline("catboost", y_fold_train=y_ft)
        weights = pipeline["classifier"].get_params()["class_weights"]
        captured.append((y_ft.index.tolist(), list(weights)))
        return pipeline

    monkeypatch.setattr(ta, "make_folds", lambda *a, **k: (SpySplitter(), 5))
    run_oof_predictions(X_sub, y_sub, track_weights, n_splits=5)

    assert len(captured) == 5
    for (train_idx, _), (y_idx, weights) in zip(reference_splits, captured):
        expected_index = y_sub.index[train_idx].tolist()
        assert y_idx == expected_index
        assert weights == pytest.approx(catboost_balanced_weights(y_sub.iloc[train_idx]))


def test_holdout_set_never_used_in_oof(holdout_split, monkeypatch):
    X_train, X_test, y_train, y_test = holdout_split
    X_sub, y_sub = X_train.head(300), y_train.head(300)
    real_splitter, _ = make_folds(y_sub)
    reference_splits = list(real_splitter.split(X_sub, y_sub))

    seen = []

    class SpySplitter:
        n_splits = 5

        def split(self, X, y):
            seen.append((X, y))
            yield from reference_splits

    monkeypatch.setattr(ta, "make_folds", lambda *a, **k: (SpySplitter(), 5))

    def builder(X_ft, y_ft):
        return build_pipeline("xgboost")

    run_oof_predictions(X_sub, y_sub, builder, n_splits=5)

    split_x, split_y = seen[0]
    assert split_x is X_sub and split_y is y_sub
    assert split_x is not X_test and split_y is not y_test


# --------------------------------------------------------------------------
# APPLICATION-ONLY FEATURE GUARD
# --------------------------------------------------------------------------
@pytest.mark.parametrize("model", ["xgboost", "catboost"])
def test_application_only_features_enforced(model):
    y_fold = pd.Series([0] * 6 + [1] * 4)
    pipeline = build_pipeline(model, y_fold_train=y_fold)
    preprocessor = pipeline.named_steps["preprocessor"]
    transformer_names = [t for t, _, _ in preprocessor.transformers]
    assert "ordinal_loan_grade" not in transformer_names
    for _, _, columns in preprocessor.transformers:
        if isinstance(columns, list):
            assert "loan_grade" not in columns
            assert "loan_int_rate" not in columns


def test_verify_application_only_features_helper():
    feats = ta.verify_application_only_features()
    assert feats["config"] == "application_only"
    assert "loan_grade" not in feats["ordinal"]
    assert "loan_int_rate" not in feats["numerical"]


# --------------------------------------------------------------------------
# MINI END-TO-END
# --------------------------------------------------------------------------
@pytest.mark.parametrize("model", ["xgboost", "catboost"])
def test_mini_oof_end_to_end(oof_results, model):
    oof_y, oof_proba, fold_df = oof_results[model]
    sweep = sweep_thresholds(oof_y, oof_proba)
    op = operating_point_rows(sweep)
    for key in ("DEFAULT", "BALANCED", "RISK-SENSITIVE"):
        threshold, row = op[key]
        assert threshold in THRESHOLDS
        for col in ["precision", "recall", "f1", "f2", "specificity", "fpr", "fnr"]:
            assert np.isfinite(row[col])
    assert fold_df[["roc_auc", "pr_auc"]].notna().all().all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))