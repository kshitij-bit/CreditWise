"""
Phase 15 tests - production model artifact and inference layer.

Run with:
    python -m pytest tests/test_phase15.py -v

Covers:
    - exact locked XGBoost parameters (unchanged, no class weighting)
    - threshold remains exactly 0.325
    - application-only schema (loan_grade / loan_int_rate excluded)
    - loan_grade / loan_int_rate / loan_status never required / accepted
    - input validation: valid, missing, numeric, categorical, impossible
      values, and person_emp_length=None (supported missing value)
    - artifact + metadata existence and correctness
    - serialization round-trip (pre-save vs post-load probabilities match)
    - predict_applicant reuses Phase 14 build_risk_assessment()
    - production training fits ONCE on the full cleaned labelled dataset
    - no production-fit evaluation metrics are generated
"""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

import ml.predict as pr
import ml.risk_engine as re
import ml.train_production as tp
from ml.final_evaluation import LOCKED_THRESHOLD
from ml.feature_engineering import ENGINEERED_FEATURES
from ml.preprocessing import feature_lists
from ml.threshold_analysis import XGBOOST_PARAMS


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def full_data():
    """The complete cleaned labelled dataset (canonical, no split)."""
    return tp.prepare_full_data()


@pytest.fixture(scope="module")
def tmp_fitted_pipeline(full_data):
    """Locked pipeline fitted on a small subset for behavior/round-trip tests."""
    X_all, y_all = full_data
    pipeline = tp.build_production_pipeline()
    pipeline.fit(X_all.head(400), y_all.head(400))
    return pipeline


@pytest.fixture(scope="module")
def real_artifacts():
    """Loaded production artifacts; skipped when not present (fresh clone)."""
    if not pr.MODEL_PATH.exists() or not pr.MODEL_METADATA_PATH.exists():
        pytest.skip("production artifacts not present - run "
                    "python ml/train_production.py first")
    return pr.load_model_artifacts()


def _demo_applicant():
    return {
        "person_age": 35,
        "person_income": 60000,
        "person_home_ownership": "RENT",
        "person_emp_length": 8.0,
        "loan_intent": "PERSONAL",
        "loan_amnt": 15000,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 5,
    }


# --------------------------------------------------------------------------
# Locked configuration
# --------------------------------------------------------------------------
def test_locked_model_parameters_exact():
    pipeline = tp.build_production_pipeline()
    params = pipeline.named_steps["classifier"].get_params()
    for key, value in XGBOOST_PARAMS.items():
        assert params[key] == value, key
    # The Phase 9 XGBoost winner uses the BASELINE strategy (no weighting).
    assert params.get("scale_pos_weight") in (None, 1)


def test_threshold_is_0325():
    assert re.CLASSIFICATION_THRESHOLD == pytest.approx(0.325)
    assert re.CLASSIFICATION_THRESHOLD == pytest.approx(LOCKED_THRESHOLD)
    assert re.MODEL_VERSION == "creditwise-xgb-v1"


def test_application_only_schema():
    feats = feature_lists("application_only")
    assert "loan_grade" not in feats["numerical"]
    assert "loan_grade" not in feats["categorical"]
    assert "loan_grade" not in feats["ordinal"]
    assert "loan_int_rate" not in feats["numerical"]
    assert "loan_status" not in feats["numerical"]
    assert "loan_status" not in feats["categorical"]

    schema = pr.build_input_schema()
    assert "loan_grade" not in schema["model_features"]
    assert "loan_int_rate" not in schema["model_features"]
    assert "loan_status" not in schema["model_features"]
    assert schema["config"] == "application_only"


def test_pipeline_excludes_leakage_sensitive_features():
    pipeline = tp.build_production_pipeline()
    transformer_names = [n for n, _, _ in pipeline["preprocessor"].transformers]
    assert "ordinal_loan_grade" not in transformer_names
    for _, _, columns in pipeline["preprocessor"].transformers:
        if isinstance(columns, list):
            assert "loan_grade" not in columns
            assert "loan_int_rate" not in columns


def test_engineered_feature_definitions_match_metadata():
    engineered = pr.SCHEMA["engineered_features"]
    assert set(engineered) == set(ENGINEERED_FEATURES)
    assert len(engineered) == 5


# --------------------------------------------------------------------------
# Schema + input validation
# --------------------------------------------------------------------------
def test_raw_input_fields_are_required_not_generated():
    expected = {
        "person_age", "person_income", "person_emp_length", "loan_amnt",
        "cb_person_cred_hist_length", "person_home_ownership", "loan_intent",
        "cb_person_default_on_file",
    }
    assert set(pr.RAW_INPUT_FIELDS) == expected
    # loan_grade / loan_int_rate / loan_status are explicitly excluded.
    assert "loan_grade" not in pr.RAW_INPUT_FIELDS
    assert "loan_int_rate" not in pr.RAW_INPUT_FIELDS
    assert "loan_status" not in pr.RAW_INPUT_FIELDS
    assert pr.RAW_INPUT_FIELDS == pr.SCHEMA["raw_input_features"]


def test_valid_input_accepted_and_normalized():
    normalized = pr.validate_applicant_input(_demo_applicant())
    assert set(normalized) == set(pr.RAW_INPUT_FIELDS)
    assert normalized["person_age"] == 35.0
    assert normalized["person_emp_length"] == 8.0
    assert normalized["cb_person_default_on_file"] == "N"


def test_valid_input_accepted_from_dataframe():
    frame = pd.DataFrame([_demo_applicant()], columns=pr.RAW_INPUT_FIELDS)
    normalized = pr.validate_applicant_input(frame)
    assert normalized["person_age"] == 35.0


def test_multi_row_dataframe_rejected():
    frame = pd.DataFrame([_demo_applicant(), _demo_applicant()],
                         columns=pr.RAW_INPUT_FIELDS)
    with pytest.raises(ValueError, match="single applicant"):
        pr.validate_applicant_input(frame)


@pytest.mark.parametrize("field", pr.RAW_INPUT_FIELDS)
def test_missing_mandatory_field_rejected(field):
    applicant = _demo_applicant()
    del applicant[field]
    with pytest.raises(ValueError) as excinfo:
        pr.validate_applicant_input(applicant)
    assert "Missing required raw applicant fields" in str(excinfo.value)
    assert field in str(excinfo.value)


def test_invalid_numeric_types_rejected():
    for field, bad in [
        ("person_age", "thirty"),
        ("person_income", []),
        ("loan_amnt", None),
        ("cb_person_cred_hist_length", {"x": 1}),
    ]:
        applicant = _demo_applicant()
        applicant[field] = bad
        with pytest.raises(ValueError, match=f"'{field}'"):
            pr.validate_applicant_input(applicant)


def test_nan_and_infinite_rejected():
    for field, bad in [
        ("person_income", float("nan")),
        ("person_age", float("inf")),
        ("loan_amnt", float("-inf")),
    ]:
        applicant = _demo_applicant()
        applicant[field] = bad
        with pytest.raises(ValueError, match=f"'{field}'"):
            pr.validate_applicant_input(applicant)


@pytest.mark.parametrize(
    "field,value",
    [
        ("person_home_ownership", "UNKNOWN"),
        ("loan_intent", "GAMBLING"),
        ("cb_person_default_on_file", "1"),
        ("person_home_ownership", "rent"),
    ],
)
def test_invalid_categorical_rejected(field, value):
    applicant = _demo_applicant()
    applicant[field] = value
    with pytest.raises(ValueError, match=f"'{field}'"):
        pr.validate_applicant_input(applicant)


@pytest.mark.parametrize(
    "field,value",
    [
        ("person_age", 101),
        ("person_age", 0),
        ("person_emp_length", 61),
        ("person_emp_length", -1),
        ("person_income", 0),
        ("loan_amnt", -5),
    ],
)
def test_impossible_values_rejected(field, value):
    applicant = _demo_applicant()
    applicant[field] = value
    with pytest.raises(ValueError, match=f"'{field}'"):
        pr.validate_applicant_input(applicant)


def test_employment_longer_than_age_rejected():
    applicant = _demo_applicant()
    applicant["person_age"] = 35
    applicant["person_emp_length"] = 40.0
    with pytest.raises(ValueError, match="person_emp_length"):
        pr.validate_applicant_input(applicant)


def test_person_emp_length_none_accepted():
    applicant = _demo_applicant()
    applicant["person_emp_length"] = None
    normalized = pr.validate_applicant_input(applicant)
    assert normalized["person_emp_length"] is None


def test_forbidden_fields_rejected():
    for field in ("loan_grade", "loan_int_rate", "loan_status"):
        applicant = _demo_applicant()
        applicant[field] = "A" if field == "loan_grade" else 1
        with pytest.raises(ValueError, match=field):
            pr.validate_applicant_input(applicant)


def test_loan_percent_income_rejected():
    applicant = _demo_applicant()
    applicant["loan_percent_income"] = 0.3
    with pytest.raises(ValueError, match="loan_percent_income"):
        pr.validate_applicant_input(applicant)


def test_invalid_top_level_type_rejected():
    with pytest.raises(ValueError, match="dictionary or a one-row"):
        pr.validate_applicant_input([1, 2, 3])


# --------------------------------------------------------------------------
# Prediction behavior (self-contained, small fitted pipeline)
# --------------------------------------------------------------------------
def test_predict_probability_in_unit_interval(tmp_fitted_pipeline):
    p = pr.predict_default_probability(_demo_applicant(), pipeline=tmp_fitted_pipeline)
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0
    assert np.isfinite(p)


def test_predict_with_emp_length_none(tmp_fitted_pipeline):
    applicant = _demo_applicant()
    applicant["person_emp_length"] = None
    p = pr.predict_default_probability(applicant, pipeline=tmp_fitted_pipeline)
    assert 0.0 <= p <= 1.0


def test_predict_applicant_output_structure(tmp_fitted_pipeline):
    assessment = pr.predict_applicant(_demo_applicant(), pipeline=tmp_fitted_pipeline)
    assert set(assessment) >= {
        "default_risk_probability", "risk_score", "risk_category", "decision",
        "classification_threshold", "model_version", "message",
    }
    assert assessment["classification_threshold"] == pytest.approx(0.325)
    assert assessment["model_version"] == "creditwise-xgb-v1"
    # Phase 14 rounds the raw probability to 4dp in the assessment object.
    raw = pr.predict_default_probability(_demo_applicant(), pipeline=tmp_fitted_pipeline)
    assert assessment["default_risk_probability"] == pytest.approx(raw, abs=5e-5)


def test_predict_applicant_reuses_build_risk_assessment(tmp_fitted_pipeline, monkeypatch):
    calls = []
    original = pr.build_risk_assessment

    def spy(probability, model_version=None):
        calls.append((probability, model_version))
        return original(probability, model_version=model_version)

    monkeypatch.setattr(pr, "build_risk_assessment", spy)
    assessment = pr.predict_applicant(_demo_applicant(), pipeline=tmp_fitted_pipeline)
    assert len(calls) == 1
    proba, version = calls[0]
    assert 0.0 <= proba <= 1.0
    assert version == "creditwise-xgb-v1"
    # The Phase 14 engine produced the complete assessment untouched.
    assert assessment == re.build_risk_assessment(proba, model_version=version)


def test_predictions_deterministic(tmp_fitted_pipeline):
    a = pr.predict_default_probability(_demo_applicant(), pipeline=tmp_fitted_pipeline)
    b = pr.predict_default_probability(_demo_applicant(), pipeline=tmp_fitted_pipeline)
    assert a == b


# --------------------------------------------------------------------------
# Serialization round-trip (pre-save vs post-load)
# --------------------------------------------------------------------------
def test_serialization_round_trip_probabilities_match(tmp_path, full_data,
                                                      tmp_fitted_pipeline):
    X_all, _ = full_data
    sample = X_all.head(10)
    path = tmp_path / "roundtrip.joblib"

    result = tp.validate_serialization_round_trip(tmp_fitted_pipeline, sample, path)
    assert result["serialization_ok"] is True
    assert result["sample_size"] == 10
    assert result["max_absolute_difference"] <= tp.SERIALIZATION_ATOL
    assert result["probabilities_in_unit_interval"] is True

    # Explicit direct comparison as well.
    before = tmp_fitted_pipeline.predict_proba(sample)[:, 1]
    reloaded = joblib.load(path)
    after = reloaded.predict_proba(sample)[:, 1]
    np.testing.assert_allclose(before, after, atol=1e-12)


# --------------------------------------------------------------------------
# Production training behavior
# --------------------------------------------------------------------------
class _FakeFittedPipeline:
    """Records the training call; never actually fits XGBoost."""

    def __init__(self):
        self.fit_calls = []

    def fit(self, X, y=None, **params):
        self.fit_calls.append((X, y))
        return self


def test_production_training_fits_once_on_full_dataset(full_data, monkeypatch):
    X_all, y_all = full_data
    fake = _FakeFittedPipeline()
    monkeypatch.setattr(tp, "build_production_pipeline", lambda: fake)

    trained = tp.train_production_model(X_all, y_all)
    assert trained is fake
    assert len(fake.fit_calls) == 1
    fit_x, fit_y = fake.fit_calls[0]
    assert fit_x is X_all
    assert fit_y is y_all
    assert len(fit_x) == len(X_all)


def test_phase11_reference_capture_is_labelled(monkeypatch):
    canned = {
        "ranking": {"roc_auc": 0.91, "pr_auc": 0.83, "brier": 0.08},
        "classification": {"precision": 0.78, "recall": 0.68, "f1": 0.72,
                           "tn": 100, "fp": 10, "fn": 5, "tp": 20},
        "counts": {"total": 135, "actual_defaults": 25, "actual_non_defaults": 110,
                   "predicted_defaults": 30},
    }

    def fake_prepare():
        return ("X_train", "X_test", "y_train", "y_test")

    def fake_eval(*args):
        results = {
            "ranking": canned["ranking"],
            "classification": canned["classification"],
            "counts": canned["counts"],
        }
        return results, None

    monkeypatch.setattr(tp, "prepare_data", fake_prepare)
    monkeypatch.setattr(tp, "run_final_evaluation", fake_eval)

    ref = tp.capture_phase11_reference_metrics()
    assert ref["evaluation_source"] == "Phase 11 frozen holdout"
    assert ref["threshold"] == pytest.approx(0.325)
    assert ref["ranking_metrics"]["roc_auc"] == pytest.approx(0.91)
    assert ref["classification_metrics"]["recall"] == pytest.approx(0.68)
    assert ref["counts"]["total"] == 135


def test_no_production_fit_evaluation_metrics(tmp_fitted_pipeline):
    n = 100
    X_sample = pd.DataFrame(np.zeros((n, len(pr.MODEL_FEATURES))),
                            columns=pr.MODEL_FEATURES)
    y_sample = pd.Series(np.zeros(n))
    ref = {
        "evaluation_source": "Phase 11 frozen holdout",
        "threshold": 0.325,
        "ranking_metrics": {"roc_auc": 0.91},
        "classification_metrics": {"precision": 0.78},
        "counts": {"total": 100},
    }
    meta = tp.build_model_metadata(tmp_fitted_pipeline, X_sample, y_sample, ref)

    # The ONLY evaluation block is the frozen Phase 11 reference.
    assert meta["evaluation"]["evaluation_source"] == "Phase 11 frozen holdout"
    production_metric_keys = {
        "production_roc_auc", "production_pr_auc", "training_roc_auc",
        "training_pr_auc", "train_accuracy", "production_f1",
    }
    flat = json.dumps(meta).lower().replace("_", "").replace(" ", "")
    for key in production_metric_keys:
        assert key.lower().replace("_", "") not in flat, key
    # Evaluation is labelled and lives ONLY under the "evaluation" block.
    assert "evaluation_source" not in meta
    assert list(meta["evaluation"]) == ["evaluation_source", "threshold",
                                        "ranking_metrics", "classification_metrics",
                                        "counts"]
    assert meta["evaluation"]["evaluation_source"] == "Phase 11 frozen holdout"


# --------------------------------------------------------------------------
# Real production artifacts (present in this repo after Phase 15 run)
# --------------------------------------------------------------------------
def test_artifact_exists():
    assert pr.MODEL_PATH.exists(), "creditwise_model.joblib missing"
    assert pr.MODEL_PATH.stat().st_size > 0


def test_metadata_exists_and_is_valid_json():
    assert pr.MODEL_METADATA_PATH.exists(), "model_metadata.json missing"
    metadata = json.loads(pr.MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    assert metadata["model_version"] == "creditwise-xgb-v1"


def test_metadata_fields_are_correct(real_artifacts):
    _, metadata = real_artifacts
    assert metadata["model_name"] == "CreditWise XGBoost"
    assert metadata["model_version"] == "creditwise-xgb-v1"
    assert metadata["algorithm"] == "XGBClassifier"
    assert metadata["feature_configuration"] == "application_only"
    assert metadata["classification_threshold"] == pytest.approx(0.325)
    assert "RAW" in metadata["probability_representation"].upper()
    assert metadata["risk_score_range"] == [0, 100]
    assert metadata["raw_input_features"] == pr.RAW_INPUT_FIELDS
    assert set(metadata["engineered_features"]) == set(ENGINEERED_FEATURES)
    assert "loan_grade" in metadata["excluded_features"]
    assert "loan_int_rate" in metadata["excluded_features"]
    assert "loan_status" in metadata["excluded_features"]
    assert metadata["model_parameters"] == XGBOOST_PARAMS
    assert metadata["class_weighting"] == "none"
    assert metadata["production_training_row_count"] > 0


def test_metadata_phase11_reference_block(real_artifacts):
    _, metadata = real_artifacts
    evaluation = metadata["evaluation"]
    assert evaluation["evaluation_source"] == "Phase 11 frozen holdout"
    assert evaluation["threshold"] == pytest.approx(0.325)
    for block in ("ranking_metrics", "classification_metrics", "counts"):
        assert block in evaluation
        assert len(evaluation[block]) > 0
    # No production-fit performance is ever stored as evaluation.
    for key in evaluation:
        assert "production" not in key


def test_real_artifact_predicts_in_unit_interval(real_artifacts):
    pipeline, _ = real_artifacts
    p = pr.predict_default_probability(_demo_applicant(), pipeline=pipeline)
    assert 0.0 <= p <= 1.0
    assessment = pr.predict_applicant(_demo_applicant(), pipeline=pipeline)
    # Phase 14 rounds the raw probability to 4dp in the assessment object.
    assert assessment["default_risk_probability"] == pytest.approx(p, abs=5e-5)
    assert 0 <= assessment["risk_score"] <= 100
    assert assessment["risk_category"] in re.RISK_CATEGORY_NAMES
    assert assessment["decision"] in ("APPROVE", "REVIEW", "REJECT")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))