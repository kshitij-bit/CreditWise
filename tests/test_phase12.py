"""
Phase 12 tests - SHAP explainability for the locked final XGBoost model.

Run with:
    python -m pytest tests/test_phase12.py -v

Covers:
    - locked model parameters unchanged (Phase 9 canonical values)
    - application-only features remain used
    - loan_grade / loan_int_rate excluded from preprocessing and SHAP names
    - transformed feature count matches the columns seen by XGBoost
    - SHAP output dimensions and finiteness
    - explanation generation does not modify model predictions
    - global importance table sorted descending and reproducible
    - feature directions valid (-1 / 0 / +1) and aligned with feature names
    - example selection is confined to the correct confusion cells
    - generated figures exist as valid PNG files
"""

import numpy as np
import pandas as pd
import pytest

import ml.explain_model as em
from ml.final_evaluation import LOCKED_THRESHOLD, build_final_pipeline
from ml.threshold_analysis import XGBOOST_PARAMS


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_context():
    """Tiny locked pipeline + SHAP context (also exercises the sampler)."""
    from ml.compare_models import prepare_data

    X_train, X_test, y_train, y_test = prepare_data()
    X_sub, y_sub = X_train.head(400), y_train.head(400)
    X_te, y_te = X_test.head(80), y_test.head(80)

    pipe = build_final_pipeline()
    pipe.fit(X_sub, y_sub)
    return em.explain_holdout(pipe, X_te, y_te, max_rows=40, seed=42)





# --------------------------------------------------------------------------
# Locked model / feature configuration
# --------------------------------------------------------------------------
def test_locked_params_unchanged():
    pipeline = build_final_pipeline()
    params = pipeline.named_steps["classifier"].get_params()
    for key in ("n_estimators", "max_depth", "learning_rate",
                "subsample", "colsample_bytree"):
        assert params[key] == XGBOOST_PARAMS[key]
    assert params.get("scale_pos_weight") in (None, 1)


def test_application_only_features_used():
    pipeline = build_final_pipeline()
    preprocessor = pipeline.named_steps["preprocessor"]
    transformer_names = [t for t, _, _ in preprocessor.transformers]
    assert "ordinal_loan_grade" not in transformer_names
    for _, _, columns in preprocessor.transformers:
        if isinstance(columns, list):
            assert "loan_grade" not in columns
            assert "loan_int_rate" not in columns


def test_shap_names_exclude_leakage_features(tiny_context):
    names = tiny_context["feature_names"]
    assert "loan_grade" not in names
    assert "loan_int_rate" not in names


# --------------------------------------------------------------------------
# Transformed feature names
# --------------------------------------------------------------------------
def test_transformed_feature_count_matches_columns(tiny_context):
    assert len(tiny_context["feature_names"]) == tiny_context["n_features"]
    assert tiny_context["n_features"] == tiny_context["Xt"].shape[1]


def test_feature_names_are_human_readable(tiny_context):
    names = tiny_context["feature_names"]
    for name in names:
        assert not name.startswith(("numerical__", "categorical__", "ordinal_")), name
    expected = {
        "person_age", "person_income", "loan_amnt", "loan_to_income_ratio",
        "log_person_income", "person_home_ownership_RENT",
        "loan_intent_MEDICAL", "person_emp_length_missing",
    }
    assert expected.issubset(set(names))


def test_clean_feature_name_mapping():
    assert em.clean_feature_name("numerical__person_age") == "person_age"
    assert (em.clean_feature_name("numerical__missingindicator_person_emp_length")
            == "person_emp_length_missing")
    assert em.clean_feature_name("categorical__loan_intent_MEDICAL") == "loan_intent_MEDICAL"
    assert em.clean_feature_name("ordinal_loan_grade") == "loan_grade"


# --------------------------------------------------------------------------
# SHAP outputs
# --------------------------------------------------------------------------
def test_shap_output_dimensions(tiny_context):
    sv = np.asarray(tiny_context["shap_values"])
    assert sv.shape == (tiny_context["n_rows"], tiny_context["n_features"])


def test_shap_values_finite(tiny_context):
    sv = np.asarray(tiny_context["shap_values"])
    assert np.isfinite(sv).all()
    assert np.isfinite(tiny_context["expected_value"])


def test_explanation_does_not_modify_predictions(tiny_context):
    clf = tiny_context["pipeline"].named_steps["classifier"]
    recomputed = clf.predict_proba(tiny_context["Xt"])[:, 1]
    np.testing.assert_allclose(recomputed, tiny_context["y_proba"], atol=1e-12)


def test_shap_reconstruction_error_small(tiny_context):
    err = em.shap_reconstruction_error(tiny_context)
    assert err < 0.01, f"base + sum(SHAP) should reproduce log-odds within 1% probability"


# --------------------------------------------------------------------------
# Global importance / directions
# --------------------------------------------------------------------------
def test_importance_table_sorted_descending(tiny_context):
    table = em.importance_table(np.asarray(tiny_context["shap_values"]),
                                tiny_context["feature_names"])
    values = table["mean_abs_shap"].tolist()
    assert values == sorted(values, reverse=True)
    assert table.iloc[0]["feature"] == tiny_context["feature_names"][
        np.argmax(np.abs(np.asarray(tiny_context["shap_values"])).mean(axis=0))]


def test_feature_directions_aligned_and_valid(tiny_context):
    directions = em.feature_directions(tiny_context["Xt"],
                                       tiny_context["shap_values"],
                                       tiny_context["feature_names"])
    assert set(directions.keys()) == set(tiny_context["feature_names"])
    assert set(directions.values()) <= {-1, 0, 1}


# --------------------------------------------------------------------------
# Local example selection
# --------------------------------------------------------------------------
def test_select_examples_uses_correct_cells():
    y = np.array([1, 0, 0, 1, 0, 1, 1, 0])
    p = np.array([0.8, 0.2, 0.4, 0.1, 0.9, 0.6, 0.5, 0.3])
    pred = (p >= 0.325).astype(int)
    labels = em.select_examples(y, p, pred)
    assert labels["true_positive"] in np.where((y == 1) & (pred == 1))[0]
    assert labels["true_negative"] in np.where((y == 0) & (pred == 0))[0]
    assert labels["false_positive"] in np.where((y == 0) & (pred == 1))[0]
    assert labels["false_negative"] in np.where((y == 1) & (pred == 0))[0]


def test_select_examples_returns_none_for_missing_cells():
    y = np.array([1, 1, 1, 1])
    p = np.array([0.9, 0.8, 0.7, 0.6])
    pred = (p >= 0.325).astype(int)
    labels = em.select_examples(y, p, pred)
    assert labels["false_positive"] is None
    assert labels["true_negative"] is None


def test_example_summary_structure(tiny_context):
    labels = em.select_examples(tiny_context["y_true"],
                                tiny_context["y_proba"],
                                tiny_context["y_pred"])
    cells = ["true_positive", "true_negative", "false_positive", "false_negative"]
    present = [c for c in cells if labels[c] is not None]
    assert present, "tiny sample should contain at least one example cell"
    info = em.example_summary(labels[present[0]], tiny_context)
    assert 0 <= info["probability"] <= 1
    assert info["threshold"] == LOCKED_THRESHOLD
    assert len(info["top_increasing"]) <= 5
    assert len(info["top_decreasing"]) <= 5
    assert info["row_id"] in np.asarray(tiny_context["row_ids"])


# --------------------------------------------------------------------------
# Figure generation
# --------------------------------------------------------------------------
def test_generated_plots_exist_as_png(tmp_path, tiny_context):
    label_cells = ["true_positive", "true_negative",
                   "false_positive", "false_negative"]
    labels = em.select_examples(tiny_context["y_true"],
                                tiny_context["y_proba"],
                                tiny_context["y_pred"])
    position = next((labels[c] for c in label_cells if labels[c] is not None), 0)

    bar = tmp_path / "shap_global_bar.png"
    summary = tmp_path / "shap_summary.png"
    waterfall = tmp_path / "shap_example.png"

    em.save_shap_bar(tiny_context, bar, top_n=10)
    em.save_shap_summary(tiny_context, summary, top_n=10)
    em.save_waterfall(tiny_context, position, waterfall)

    for path in (bar, summary, waterfall):
        assert path.exists() and path.stat().st_size > 0
        with open(path, "rb") as fh:
            assert fh.read(4) == b"\x89PNG", f"{path.name} is not a PNG"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))