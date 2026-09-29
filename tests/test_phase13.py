"""
Phase 13 tests - probability calibration analysis.

Run with:
    python -m pytest tests/test_phase13.py -v

Covers:
    - holdout labels are never used to fit calibration
    - application-only features remain enforced
    - probabilities stay in [0, 1] and are finite
    - all methods return the same number of predictions
    - Brier scores are finite
    - reliability bins cover all observations
    - final XGBoost parameters remain unchanged
    - classification threshold remains exactly 0.325
    - ECE / bin-table / decision-rule behavior on synthetic inputs
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV

import ml.calibration_analysis as ca
from ml.final_evaluation import LOCKED_THRESHOLD, build_final_pipeline
from ml.threshold_analysis import XGBOOST_PARAMS


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def calib_set():
    """Tiny locked fit + raw/sigmoid/isotonic hold-out probabilities."""
    from ml.compare_models import prepare_data

    X_train, X_test, y_train, y_test = prepare_data()
    Xtr, ytr = X_train.head(600), y_train.head(600)
    Xte, yte = X_test.head(250), np.asarray(y_test.head(250))

    pipeline = ca.fit_locked_pipeline(Xtr, ytr)
    raw = ca.predict_raw(pipeline, Xte)
    sigmoid = ca.fit_calibrated("SIGMOID", Xtr, ytr).predict_proba(Xte)[:, 1]
    isotonic = ca.fit_calibrated("ISOTONIC", Xtr, ytr).predict_proba(Xte)[:, 1]
    return yte, raw, sigmoid, isotonic


# --------------------------------------------------------------------------
# Locked configuration guards
# --------------------------------------------------------------------------
def test_locked_params_unchanged():
    pipeline = build_final_pipeline()
    params = pipeline.named_steps["classifier"].get_params()
    for key in ("n_estimators", "max_depth", "learning_rate",
                "subsample", "colsample_bytree"):
        assert params[key] == XGBOOST_PARAMS[key]
    assert params.get("scale_pos_weight") in (None, 1)


def test_threshold_still_0325():
    assert LOCKED_THRESHOLD == pytest.approx(0.325)


def test_application_only_features_enforced():
    pipeline = build_final_pipeline()
    preprocessor = pipeline.named_steps["preprocessor"]
    transformer_names = [t for t, _, _ in preprocessor.transformers]
    assert "ordinal_loan_grade" not in transformer_names
    for _, _, columns in preprocessor.transformers:
        if isinstance(columns, list):
            assert "loan_grade" not in columns
            assert "loan_int_rate" not in columns


# --------------------------------------------------------------------------
# Leakage safety
# --------------------------------------------------------------------------
def test_calibration_never_fits_holdout(monkeypatch):
    from ml.compare_models import prepare_data

    X_train, X_test, y_train, _ = prepare_data()
    Xtr, ytr = X_train.head(300), y_train.head(300)

    calls = []
    original_fit = CalibratedClassifierCV.fit

    def spy_fit(self, X, y=None, **params):
        calls.append((X, y))
        return original_fit(self, X, y, **params)

    monkeypatch.setattr(CalibratedClassifierCV, "fit", spy_fit)
    calibrator = ca.fit_calibrated("SIGMOID", Xtr, ytr)
    calibrator.predict_proba(X_test.head(50))

    assert len(calls) == 1
    fitted_x, fitted_y = calls[0]
    assert fitted_x is Xtr, "calibration.fit must receive the training DataFrame"
    assert fitted_y is ytr, "calibration.fit must receive the training labels"


# --------------------------------------------------------------------------
# Probability outputs
# --------------------------------------------------------------------------
def test_probabilities_in_unit_interval_and_finite(calib_set):
    _, raw, sigmoid, isotonic = calib_set
    for proba in (raw, sigmoid, isotonic):
        assert np.isfinite(proba).all()
        assert (proba >= 0.0).all()
        assert (proba <= 1.0).all()


def test_all_methods_return_same_number_of_predictions(calib_set):
    yte, raw, sigmoid, isotonic = calib_set
    assert len(raw) == len(sigmoid) == len(isotonic) == len(yte)


def test_brier_scores_finite(calib_set):
    _, raw, sigmoid, isotonic = calib_set
    yte, *_ = calib_set
    for proba in (raw, sigmoid, isotonic):
        assert np.isfinite(ca.evaluate_probabilities(yte, proba)["brier"])


def test_ranking_metrics_stable_across_monotone_methods(calib_set):
    yte, raw, sigmoid, isotonic = calib_set
    raw_m = ca.evaluate_probabilities(yte, raw)
    sig_m = ca.evaluate_probabilities(yte, sigmoid)
    iso_m = ca.evaluate_probabilities(yte, isotonic)
    # Monotone recalibration preserves ranking in theory; small deviations on
    # a 600-row calibration sample come from OOF-fit noise and ties.
    for metric in ("roc_auc", "pr_auc"):
        assert abs(raw_m[metric] - sig_m[metric]) < 0.03, metric
        assert abs(raw_m[metric] - iso_m[metric]) < 0.03, metric


# --------------------------------------------------------------------------
# Reliability bins / ECE
# --------------------------------------------------------------------------
def test_bins_cover_all_observations(calib_set):
    yte, raw, sigmoid, isotonic = calib_set
    for proba in (raw, sigmoid, isotonic):
        table = ca.bin_calibration_table(yte, proba)
        assert int(table["count"].sum()) == len(yte)
        assert (table["count"] >= 0).all()


def test_bin_table_rows_have_sane_values(calib_set):
    yte, raw, *_ = calib_set
    table = ca.bin_calibration_table(yte, raw)
    cols = {"bin", "count", "mean_predicted", "observed_rate", "abs_gap"}
    assert cols <= set(table.columns)
    present = table[~table["mean_predicted"].isna()]
    assert ((present["mean_predicted"] >= 0.0) & (present["mean_predicted"] <= 1.0)).all()
    assert ((present["observed_rate"] >= 0.0) & (present["observed_rate"] <= 1.0)).all()
    assert (present["abs_gap"] >= 0.0).all()


def test_ece_zero_for_perfect_calibration():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.5, 0.5, 0.5, 0.5])
    assert ca.expected_calibration_error(y, p) == pytest.approx(0.0, abs=1e-12)


def test_ece_positive_for_miscalibration():
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    p = np.full(10, 0.9)
    assert ca.expected_calibration_error(y, p) == pytest.approx(0.4, abs=1e-9)


def test_signed_bias_sign():
    assert ca.signed_bias([0, 0, 0], [0.9, 0.8, 0.7]) == pytest.approx(0.8)


# --------------------------------------------------------------------------
# Decision rule
# --------------------------------------------------------------------------
def test_decision_rule_returns_valid_choice():
    raw = {"ece": 0.02, "brier": 0.10}
    sigmoid = {"ece": 0.09, "brier": 0.11}
    isotonic = {"ece": 0.08, "brier": 0.11}
    assert ca.recommend_calibration(raw, sigmoid, isotonic) == "RAW"


def test_decision_rule_prefers_sigmoid_on_tie():
    raw = {"ece": 0.20, "brier": 0.12}
    sigmoid = {"ece": 0.05, "brier": 0.10}
    isotonic = {"ece": 0.049, "brier": 0.099}
    assert ca.recommend_calibration(raw, sigmoid, isotonic) == "SIGMOID"


def test_decision_rule_prefers_isotonic_when_meaningfully_better():
    raw = {"ece": 0.20, "brier": 0.12}
    sigmoid = {"ece": 0.10, "brier": 0.105}
    isotonic = {"ece": 0.02, "brier": 0.08}
    assert ca.recommend_calibration(raw, sigmoid, isotonic) == "ISOTONIC"


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def test_generated_plots_exist_as_png(tmp_path, calib_set):
    yte, raw, sigmoid, isotonic = calib_set
    results = {
        "RAW": {"_proba": raw, "brier": 0.0, "ece": 0.0, "roc_auc": 0.0,
                "pr_auc": 0.0, "bias": 0.0},
        "SIGMOID": {"_proba": sigmoid, "brier": 0.0, "ece": 0.0, "roc_auc": 0.0,
                    "pr_auc": 0.0, "bias": 0.0},
        "ISOTONIC": {"_proba": isotonic, "brier": 0.0, "ece": 0.0,
                     "roc_auc": 0.0, "pr_auc": 0.0, "bias": 0.0},
    }
    curve = tmp_path / "calibration_curve.png"
    dist = tmp_path / "probability_distribution.png"
    ca.save_calibration_curve(yte, results, curve)
    ca.save_probability_distribution(results, dist)
    for path in (curve, dist):
        assert path.exists() and path.stat().st_size > 0
        with open(path, "rb") as fh:
            assert fh.read(4) == b"\x89PNG", f"{path.name} is not a PNG"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))