"""
Phase 9 tests - light hyperparameter tuning (XGBoost baseline, CatBoost weighted).

Run with:
    python -m pytest tests/test_phase9.py -v

Covers:
    - both staged grids: size (18), allowed keys/values, no duplicates,
      no forbidden params (scale_pos_weight / class_weights never in space)
    - every tuned pipeline excludes loan_grade / loan_int_rate
    - XGBoost never receives scale_pos_weight (default 1.0 preserved)
    - CatBoost balanced weights derive from fold-TRAINING labels only
    - identical 5-fold splits across repeated runs
    - the frozen holdout set is never passed to cross-validation
    - mini end-to-end tuning: 18 combos each, finite metrics, proba in [0,1]
    - selection rule: PR-AUC -> ROC-AUC -> F1 (never accuracy)
    - delta + leader/backup helpers on synthetic fold data
"""

import numpy as np
import pandas as pd
import pytest

from ml.cross_validate_baseline import run_cv_scores
from ml.imbalance_experiment import catboost_balanced_weights
from ml.compare_models import prepare_data
from ml.tune_models import (
    CAT_PARAM_SPACE,
    CAT_SUPPORTED_VALUES,
    CAT_TUNED_PARAMS,
    MODEL_LABELS,
    XGB_PARAM_SPACE,
    XGB_SUPPORTED_VALUES,
    XGB_TUNED_PARAMS,
    baseline_params,
    baseline_pipeline,
    build_tuned_pipeline,
    choose_leader,
    combo_label,
    make_deltas,
    make_tuning_table,
    run_tuning,
    select_best_label,
)
from ml.train_baseline import get_split, load_data
from ml.data_cleaning import clean_credit_data
from ml.feature_engineering import engineer_features
from ml.preprocessing import TARGET


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def holdout_split():
    """Same Phase 5/6/7/8 holdout split used across the project."""
    df = load_data()
    cleaned, _ = clean_credit_data(df)
    engineered, _ = engineer_features(cleaned)
    X = engineered.drop(columns=[TARGET])
    y = engineered[TARGET]
    return get_split(X, y)


@pytest.fixture(scope="module")
def subset_xgboost_results(holdout_split):
    """Full 18-combo XGBoost tuning on a small subset (fast)."""
    X_train, _, y_train, _ = holdout_split
    return run_tuning(X_train.head(500), y_train.head(500), "xgboost")


@pytest.fixture(scope="module")
def subset_catboost_results(holdout_split):
    """Full 18-combo CatBoost tuning on a small subset (fast)."""
    X_train, _, y_train, _ = holdout_split
    return run_tuning(X_train.head(500), y_train.head(500), "catboost")


# --------------------------------------------------------------------------
# SEARCH SPACES
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "space, supported, tuned_keys",
    [
        (XGB_PARAM_SPACE, XGB_SUPPORTED_VALUES, XGB_TUNED_PARAMS),
        (CAT_PARAM_SPACE, CAT_SUPPORTED_VALUES, CAT_TUNED_PARAMS),
    ],
)
def test_space_has_expected_size(space, supported, tuned_keys):
    assert 15 <= len(space) <= 20
    assert len(space) == 18


@pytest.mark.parametrize(
    "space, supported, tuned_keys",
    [
        (XGB_PARAM_SPACE, XGB_SUPPORTED_VALUES, XGB_TUNED_PARAMS),
        (CAT_PARAM_SPACE, CAT_SUPPORTED_VALUES, CAT_TUNED_PARAMS),
    ],
)
def test_space_keys_and_values_within_supported_ranges(space, supported, tuned_keys):
    for params in space:
        assert set(params) == set(tuned_keys)
        for key, value in params.items():
            assert value in supported[key]


def test_xgb_space_never_contains_scale_pos_weight():
    for params in XGB_PARAM_SPACE:
        assert "scale_pos_weight" not in params


def test_cat_space_never_contains_class_weights():
    for params in CAT_PARAM_SPACE:
        assert "class_weights" not in params


@pytest.mark.parametrize("space", [XGB_PARAM_SPACE, CAT_PARAM_SPACE])
def test_spaces_contain_no_duplicate_configs(space):
    signatures = [tuple(sorted(params.items())) for params in space]
    assert len(signatures) == len(set(signatures))


# --------------------------------------------------------------------------
# PIPELINE BUILDING
# --------------------------------------------------------------------------
@pytest.mark.parametrize("model", ["xgboost", "catboost"])
def test_every_tuned_pipeline_excludes_leakage_sensitive_features(model, holdout_split):
    X_train, _, y_train, _ = holdout_split
    y_ft = y_train.head(200)
    space = XGB_PARAM_SPACE if model == "xgboost" else CAT_PARAM_SPACE

    for params in space:
        pipeline = build_tuned_pipeline(model, params, y_fold_train=y_ft)
        assert "classifier" in pipeline.named_steps
        preprocessor = pipeline["preprocessor"]
        transformer_names = [t for t, _, _ in preprocessor.transformers]
        assert "ordinal_loan_grade" not in transformer_names
        for _, _, columns in preprocessor.transformers:
            if isinstance(columns, list):
                assert "loan_grade" not in columns
                assert "loan_int_rate" not in columns


def test_xgboost_never_sets_scale_pos_weight_across_grid():
    for params in XGB_PARAM_SPACE:
        pipeline = build_tuned_pipeline("xgboost", params)
        # Absent (None) or default (1) both mean "no class weighting applied".
        assert pipeline["classifier"].get_params().get("scale_pos_weight") in (None, 1)


def test_build_tuned_pipeline_rejects_forbidden_params():
    with pytest.raises(ValueError):
        build_tuned_pipeline("xgboost", {"scale_pos_weight": 2.0})
    with pytest.raises(ValueError):
        build_tuned_pipeline("catboost", {"class_weights": [1.0, 1.0]}, y_fold_train=np.array([0, 1, 0]))


def test_xgboost_baseline_pipeline_has_no_class_weighting():
    pipeline = baseline_pipeline("xgboost")
    # Absent (None) or default (1) both mean "no class weighting applied".
    assert pipeline["classifier"].get_params().get("scale_pos_weight") in (None, 1)


def test_catboost_baseline_pipeline_is_class_weighted_from_fold_train_labels():
    """Phase 8 candidate for CatBoost was CLASS WEIGHTED: the Phase 9 baseline
    reference must rebuild weights from fold TRAINING labels only."""
    y_fold = pd.Series([0] * 60 + [1] * 40)
    pipeline = baseline_pipeline("catboost", y_fold_train=y_fold)
    weights = pipeline["classifier"].get_params()["class_weights"]
    assert list(weights) == pytest.approx(catboost_balanced_weights(y_fold))


def test_catboost_baseline_pipeline_requires_fold_labels():
    with pytest.raises(ValueError):
        baseline_pipeline("catboost")


def test_catboost_requires_fold_training_labels():
    with pytest.raises(ValueError):
        build_tuned_pipeline("catboost", CAT_PARAM_SPACE[0])


def test_catboost_weights_use_fold_train_labels_only(holdout_split, monkeypatch):
    """The CatBoost builder receives fold-TRAINING labels and weights match them exactly."""
    import ml.cross_validate_baseline as cv

    X_train, _, y_train, _ = holdout_split
    X_sub = X_train.head(400)
    y_sub = y_train.head(400)

    real_splitter, _ = cv.make_folds(y_sub)
    reference_splits = list(real_splitter.split(X_sub, y_sub))

    class SpySplitter:
        n_splits = 5

        def split(self, X, y):
            yield from reference_splits

    captured = []  # (X_index, y_index, class_weights)

    def builder(X_ft, y_ft):
        pipeline = build_tuned_pipeline("catboost", CAT_PARAM_SPACE[0], y_fold_train=y_ft)
        weights = pipeline["classifier"].get_params()["class_weights"]
        captured.append((X_ft.index.tolist(), y_ft.index.tolist(), weights))
        return pipeline

    monkeypatch.setattr(cv, "make_folds", lambda *a, **k: (SpySplitter(), 5))
    run_cv_scores(X_sub, y_sub, builder, n_splits=5)

    assert len(captured) == 5
    for (train_idx, _val_idx), (x_idx, y_idx, weights) in zip(reference_splits, captured):
        expected = y_sub.index[train_idx].tolist()
        assert y_idx == expected
        assert x_idx == expected
        assert list(weights) == pytest.approx(catboost_balanced_weights(y_sub.iloc[train_idx]))


def test_identical_folds_across_repeated_runs(holdout_split):
    """Same combo, same splitter seed -> identical per-fold metrics on two runs."""
    X_train, _, y_train, _ = holdout_split
    X_sub = X_train.head(400)
    y_sub = y_train.head(400)
    params = XGB_PARAM_SPACE[0]

    run1 = run_cv_scores(
        X_sub, y_sub,
        lambda X_ft, y_ft: build_tuned_pipeline("xgboost", params),
        n_splits=5,
    )
    run2 = run_cv_scores(
        X_sub, y_sub,
        lambda X_ft, y_ft: build_tuned_pipeline("xgboost", params),
        n_splits=5,
    )
    pd.testing.assert_frame_equal(run1[["roc_auc", "pr_auc"]], run2[["roc_auc", "pr_auc"]])


def test_holdout_set_never_used_in_tuning(holdout_split, monkeypatch):
    import ml.cross_validate_baseline as cv

    X_train, X_test, y_train, y_test = holdout_split
    X_sub = X_train.head(400)
    y_sub = y_train.head(400)

    seen_objects = []
    real_splitter, _ = cv.make_folds(y_sub)
    reference_splits = list(real_splitter.split(X_sub, y_sub))

    class SpySplitter:
        n_splits = 5

        def split(self, X, y):
            seen_objects.append((X, y))
            yield from reference_splits

    monkeypatch.setattr(cv, "make_folds", lambda *a, **k: (SpySplitter(), 5))
    params = CAT_PARAM_SPACE[0]

    def builder(X_ft, y_ft):
        return build_tuned_pipeline("catboost", params, y_fold_train=y_ft)

    run_cv_scores(X_sub, y_sub, builder, n_splits=5)

    split_x, split_y = seen_objects[0]
    assert split_x is X_sub and split_y is y_sub
    assert split_x is not X_test and split_y is not y_test


# --------------------------------------------------------------------------
# END-TO-END MINI TUNING
# --------------------------------------------------------------------------
def _assert_finite_tuning(results, baseline_df):
    assert len(results) == 18
    for label, (params, fold_df) in results.items():
        assert fold_df.shape[0] == 5
        for column in ["roc_auc", "pr_auc", "precision", "recall", "f1"]:
            values = fold_df[column]
            assert np.isfinite(values).all()
            assert values.between(0.0, 1.0).all()
        assert (fold_df["fit_time_s"] > 0).all()
    assert baseline_df.shape[0] == 5


def test_mini_tuning_xgboost_all_combos(subset_xgboost_results):
    results, baseline_df = subset_xgboost_results
    _assert_finite_tuning(results, baseline_df)


def test_mini_tuning_catboost_all_combos(subset_catboost_results):
    results, baseline_df = subset_catboost_results
    _assert_finite_tuning(results, baseline_df)


def test_mini_tuning_xgboost_best_matches_selection(subset_xgboost_results):
    results, _ = subset_xgboost_results
    table = make_tuning_table(results)
    assert table.shape[0] == 18
    assert (table["PR-AUC"].diff().iloc[1:] <= 0).all()  # sorted descending
    best_label = select_best_label(results)
    assert table.iloc[0]["Config"] == best_label
    # never selected by accuracy
    assert "Accuracy" not in table.columns


def test_mini_tuning_catboost_best_matches_selection(subset_catboost_results):
    results, _ = subset_catboost_results
    table = make_tuning_table(results)
    assert table.shape[0] == 18
    assert (table["PR-AUC"].diff().iloc[1:] <= 0).all()
    best_label = select_best_label(results)
    assert table.iloc[0]["Config"] == best_label


# --------------------------------------------------------------------------
# SELECTION, DELTAS, LEADER
# --------------------------------------------------------------------------
def _fake_fold_df(pr_auc, roc_auc, f1, precision=0.5, recall=0.5):
    return pd.DataFrame(
        {
            "pr_auc": [pr_auc] * 5,
            "roc_auc": [roc_auc] * 5,
            "f1": [f1] * 5,
            "precision": [precision] * 5,
            "recall": [recall] * 5,
            "fit_time_s": [1.0] * 5,
        },
        index=pd.RangeIndex(start=1, stop=6, name="fold"),
    )


def test_selection_uses_pr_auc_then_roc_auc_then_f1():
    results = {
        "lower_pr": ({}, _fake_fold_df(0.70, 0.90, 0.65)),
        "tie_pr_lower_roc": ({}, _fake_fold_df(0.80, 0.89, 0.68)),
        "tie_pr_better_roc": ({}, _fake_fold_df(0.80, 0.91, 0.66)),
    }
    assert select_best_label(results) == "tie_pr_better_roc"

    results = {
        "tie_all_but_f1": ({}, _fake_fold_df(0.80, 0.90, 0.66)),
        "tie_pr_roc_better_f1": ({}, _fake_fold_df(0.80, 0.90, 0.70)),
    }
    assert select_best_label(results) == "tie_pr_roc_better_f1"


def test_deltas_are_best_minus_baseline():
    baseline = _fake_fold_df(pr_auc=0.75, roc_auc=0.88, f1=0.62, precision=0.70, recall=0.55)
    best = _fake_fold_df(pr_auc=0.78, roc_auc=0.90, f1=0.64, precision=0.68, recall=0.61)
    deltas = make_deltas(best, baseline)
    assert deltas["dPR-AUC"] == pytest.approx(0.03)
    assert deltas["dROC-AUC"] == pytest.approx(0.02)
    assert deltas["dRecall"] == pytest.approx(0.06)
    assert deltas["dPrecision"] == pytest.approx(-0.02)
    assert deltas["dF1"] == pytest.approx(0.02)
    self_deltas = make_deltas(baseline, baseline)
    assert all(v == pytest.approx(0.0) for v in self_deltas.values())


def test_choose_leader_by_pr_auc_then_roc_then_f1():
    first = ("XGBoost", {"PR-AUC": 0.80, "ROC-AUC": 0.90, "F1": 0.68})
    second = ("CatBoost", {"PR-AUC": 0.81, "ROC-AUC": 0.89, "F1": 0.67})
    leader, backup = choose_leader(first, second)
    assert leader[0] == "CatBoost"
    assert backup[0] == "XGBoost"

    first = ("XGBoost", {"PR-AUC": 0.80, "ROC-AUC": 0.90, "F1": 0.68})
    second = ("CatBoost", {"PR-AUC": 0.80, "ROC-AUC": 0.89, "F1": 0.70})
    leader, backup = choose_leader(first, second)
    assert leader[0] == "XGBoost"  # PR-AUC tie -> ROC-AUC decides

    first = ("XGBoost", {"PR-AUC": 0.80, "ROC-AUC": 0.90, "F1": 0.68})
    second = ("CatBoost", {"PR-AUC": 0.80, "ROC-AUC": 0.90, "F1": 0.70})
    leader, backup = choose_leader(first, second)
    assert leader[0] == "CatBoost"  # PR + ROC tie -> F1 decides


def test_baseline_params_captured_match_expected_defaults():
    xgb = baseline_params("xgboost")
    assert xgb == {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.3,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
    }
    cat = baseline_params("catboost")
    assert cat["iterations"] == 100
    assert cat["depth"] == 6
    assert cat["l2_leaf_reg"] == 3.0
    assert cat["learning_rate"] is None  # CatBoost auto


def test_combo_labels_are_unique_within_each_space():
    xgb_labels = [combo_label("xgboost", p) for p in XGB_PARAM_SPACE]
    cat_labels = [combo_label("catboost", p) for p in CAT_PARAM_SPACE]
    assert len(set(xgb_labels)) == len(xgb_labels) == 18
    assert len(set(cat_labels)) == len(cat_labels) == 18


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))