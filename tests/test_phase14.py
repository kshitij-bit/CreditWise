"""
Phase 14 tests - risk score, risk bands and decision engine.

Run with:
    python -m pytest tests/test_phase14.py -v

Covers:
    - risk scores in [0, 100] with exact mapping (round-half-up ties)
    - probability validation (out-of-range / NaN / inf / None rejected)
    - exact band boundaries: 0.00, 0.149, 0.150, 0.324, 0.325, 0.599, 0.600, 1.00
    - 0.325 maps to HIGH (locked classifier threshold edge included)
    - decisions map correctly and deterministically
    - threshold value remains exactly 0.325
    - assessment object structure contains all expected fields
    - raw probability is returned unchanged (never modified by the engine)
    - internal risk score is not represented as a bureau score
    - output is JSON-serializable for the future API layer
"""

import json

import numpy as np
import pytest

import ml.risk_engine as re
from ml.final_evaluation import LOCKED_THRESHOLD


# --------------------------------------------------------------------------
# Locked configuration
# --------------------------------------------------------------------------
def test_threshold_remains_0325():
    assert re.CLASSIFICATION_THRESHOLD == pytest.approx(0.325)
    assert re.CLASSIFICATION_THRESHOLD == LOCKED_THRESHOLD


def test_pinned_boundary_constants():
    assert re.LOW_RISK_MAX == pytest.approx(0.15)
    assert re.MODERATE_RISK_MAX == pytest.approx(0.325)
    assert re.HIGH_RISK_MAX == pytest.approx(0.60)


# --------------------------------------------------------------------------
# Risk score
# --------------------------------------------------------------------------
def test_risk_score_range_across_sweep():
    for p in np.arange(0.0, 1.0001, 0.013):
        score = re.probability_to_risk_score(p)
        assert 0 <= score <= 100


def test_risk_score_exact_mapping():
    expected = {
        0.00: 0,
        0.08: 8,
        0.149: 15,
        0.150: 15,
        0.274: 27,
        0.324: 32,
        0.325: 33,  # round-half-up: plain round() would give 32
        0.599: 60,
        0.600: 60,
        0.76: 76,
        1.00: 100,
    }
    for p, score in expected.items():
        assert re.probability_to_risk_score(p) == score, p


# --------------------------------------------------------------------------
# Probability validation
# --------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [-0.01, 1.01, float("nan"), float("inf"),
                                 float("-inf"), None])
def test_invalid_probabilities_rejected(bad):
    with pytest.raises(ValueError):
        re.validate_probability(bad)
    with pytest.raises(ValueError):
        re.probability_to_risk_score(bad)
    with pytest.raises(ValueError):
        re.get_risk_category(bad)
    with pytest.raises(ValueError):
        re.build_risk_assessment(bad)


# --------------------------------------------------------------------------
# Risk category boundaries
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "p,category",
    [
        (0.00, "LOW"),
        (0.149, "LOW"),
        (0.150, "MODERATE"),
        (0.324, "MODERATE"),
        (0.325, "HIGH"),
        (0.599, "HIGH"),
        (0.600, "VERY_HIGH"),
        (1.00, "VERY_HIGH"),
    ],
)
def test_category_boundaries(p, category):
    assert re.get_risk_category(p) == category


def test_locked_threshold_maps_to_high():
    assert re.get_risk_category(0.325) == "HIGH"
    assert re.get_risk_category(re.CLASSIFICATION_THRESHOLD) == "HIGH"


def test_categories_cover_unit_interval():
    for p in np.arange(0.0, 1.0001, 0.001):
        assert re.get_risk_category(p) in re.RISK_CATEGORY_NAMES


# --------------------------------------------------------------------------
# Decisions
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "p,decision",
    [
        (0.00, "APPROVE"),
        (0.149, "APPROVE"),
        (0.150, "REVIEW"),
        (0.324, "REVIEW"),
        (0.325, "REVIEW"),
        (0.599, "REVIEW"),
        (0.600, "REJECT"),
        (1.00, "REJECT"),
    ],
)
def test_decisions_map_correctly(p, decision):
    assert re.get_decision(p) == decision


# --------------------------------------------------------------------------
# Assessment object
# --------------------------------------------------------------------------
def test_assessment_structure_contains_all_fields():
    result = re.build_risk_assessment(0.274)
    assert set(result) >= {
        "default_risk_probability", "risk_score", "risk_category",
        "decision", "classification_threshold", "model_version", "message",
    }
    assert result["default_risk_probability"] == pytest.approx(0.274)
    assert result["risk_score"] == 27
    assert result["risk_category"] == "MODERATE"
    assert result["decision"] == "REVIEW"
    assert result["classification_threshold"] == pytest.approx(0.325)
    assert result["model_version"] == re.MODEL_VERSION
    assert result["message"] == re.MESSAGES["MODERATE"]
    assert "top_risk_increasing_factors" in result
    assert "top_risk_reducing_factors" in result
    assert result["top_risk_increasing_factors"] == []
    assert result["top_risk_reducing_factors"] == []


def test_models_all_produce_expected_messages():
    assert re.MESSAGES["LOW"] == "Low predicted credit risk."
    assert (re.MESSAGES["MODERATE"]
            == "Moderate predicted credit risk. Manual review recommended.")
    assert (re.MESSAGES["HIGH"]
            == "High predicted credit risk. Detailed review recommended.")
    assert (re.MESSAGES["VERY_HIGH"]
            == "Very high predicted credit risk. Application rejection recommended.")


def test_custom_model_version_honored():
    result = re.build_risk_assessment(0.5, model_version="custom-v2")
    assert result["model_version"] == "custom-v2"


def test_raw_probability_is_never_modified():
    for p in (0.0, 0.274, 0.325, 0.599, 0.6, 1.0):
        result = re.build_risk_assessment(p)
        assert result["default_risk_probability"] == pytest.approx(p)


def test_assessment_is_deterministic():
    assert re.build_risk_assessment(0.41) == re.build_risk_assessment(0.41)


def test_assessment_is_json_serializable():
    payload = json.dumps(re.build_risk_assessment(0.274))
    assert re.build_risk_assessment(0.274) == json.loads(payload)


# --------------------------------------------------------------------------
# Internal score - NOT a bureau score
# --------------------------------------------------------------------------
def test_risk_score_disclaimer_clear():
    assert "CreditWise" in re.RISK_SCORE_DISCLAIMER
    assert "NOT" in re.RISK_SCORE_DISCLAIMER
    assert "bureau" in re.RISK_SCORE_DISCLAIMER.lower()
    assert "300-900" not in re.RISK_SCORE_DISCLAIMER


def test_note_states_model_based_support():
    result = re.build_risk_assessment(0.325)
    assert "decision support" in result["note"].lower()
    assert "underwriting" in result["note"].lower()
    assert "CreditWise" in re.RISK_SCORE_DISCLAIMER


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))