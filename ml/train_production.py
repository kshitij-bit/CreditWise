"""
Phase 15 - Production model artifact training for CreditWise AI.

All experimentation is complete and every decision is FROZEN. This script
builds the single DEPLOYMENT artifact by fitting the locked XGBoost model
once on ALL cleaned labelled rows (no train/test split).

Frozen configuration (do NOT change)
------------------------------------
    Model      : Tuned XGBoost (exact Phase 9 best parameters)
    Features   : APPLICATION-ONLY (loan_grade / loan_int_rate excluded)
    Threshold  : 0.325
    Probability: RAW XGBoost probability
    Version    : creditwise-xgb-v1

Official evaluation metrics remain the PHASE 11 frozen holdout results.
This fit is a deployment artifact only - its performance is NEVER reported
as an evaluation. No CV, no tuning, no threshold change, no metrics.

Artifacts produced
------------------
    ml/artifacts/creditwise_model.joblib
    ml/artifacts/model_metadata.json

Run with:
    python ml/train_production.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from ml.compare_models import prepare_data
from ml.data_cleaning import clean_credit_data
from ml.feature_engineering import engineer_features
from ml.final_evaluation import run_final_evaluation
from ml.predict import (
    MODEL_METADATA_PATH,
    MODEL_PATH,
    build_input_schema,
    predict_default_probability,
)
from ml.preprocessing import TARGET
from ml.risk_engine import CLASSIFICATION_THRESHOLD, MODEL_VERSION
from ml.threshold_analysis import XGBOOST_PARAMS, verify_application_only_features
from ml.train_baseline import RANDOM_STATE, load_data

MODEL_NAME = "CreditWise XGBoost"
ALGORITHM = "XGBClassifier"
FEATURE_CONFIGURATION = "application_only"
PROBABILITY_REPRESENTATION = "RAW XGBoost probability"
RISK_SCORE_RANGE = [0, 100]
SERIALIZATION_ATOL = 1e-12


# ---------------------------------------------------------------------------
# Data preparation (canonical, no split for the deployment fit)
# ---------------------------------------------------------------------------
def prepare_full_data():
    """
    All cleaned + engineered labelled rows (train and holdout combined),
    in original dataset order. Deterministic - does not split.
    """
    df = load_data()
    cleaned, _ = clean_credit_data(df)
    engineered, _ = engineer_features(cleaned)
    X = engineered.drop(columns=[TARGET])
    y = engineered[TARGET]
    return X, y


# ---------------------------------------------------------------------------
# Locked pipeline
# ---------------------------------------------------------------------------
def build_production_pipeline():
    """
    Unfitted LOCKED application-only XGBoost pipeline with hard guards:
        - exact Phase 9 best parameters
        - no class weighting
        - application-only features (loan_grade / loan_int_rate excluded)
    """
    from ml.final_evaluation import build_final_pipeline

    pipeline = build_final_pipeline()
    assert_parameters_locked(pipeline)
    assert_application_only(pipeline)
    return pipeline


def assert_parameters_locked(pipeline):
    """Fail loudly if any locked hyperparameter ever drifts."""
    params = pipeline.named_steps["classifier"].get_params()
    for key, value in XGBOOST_PARAMS.items():
        assert params[key] == value, (
            f"Locked parameter '{key}' drifted: {params[key]} != {value}"
        )
    assert params.get("scale_pos_weight") in (None, 1), (
        "The tuned XGBoost winner must never receive class weighting."
    )


def assert_application_only(pipeline):
    """Fail loudly if a leakage-sensitive feature ever re-enters the model."""
    verify_application_only_features()
    transformer_names = [name for name, _, _ in pipeline["preprocessor"].transformers]
    assert "ordinal_loan_grade" not in transformer_names
    for _, _, columns in pipeline["preprocessor"].transformers:
        if isinstance(columns, list):
            assert "loan_grade" not in columns
            assert "loan_int_rate" not in columns


def train_production_model(X_all, y_all):
    """
    Fit the locked pipeline exactly once on the COMPLETE cleaned dataset.
    No CV, no splitting, no metrics - deployment fit only.
    """
    pipeline = build_production_pipeline()
    pipeline.fit(X_all, y_all)
    return pipeline


# ---------------------------------------------------------------------------
# Phase 11 frozen reference metrics (captured, not recomputed as evaluation)
# ---------------------------------------------------------------------------
def capture_phase11_reference_metrics():
    """
    Reproduce the EXACT Phase 11 frozen holdout results with the existing
    final-evaluation implementation. Stored ONLY as reference metadata, never
    as a new evaluation of the production fit.
    """
    X_train, X_test, y_train, y_test = prepare_data()
    results, _ = run_final_evaluation(X_train, y_train, X_test, y_test)
    return {
        "evaluation_source": "Phase 11 frozen holdout",
        "threshold": float(CLASSIFICATION_THRESHOLD),
        "ranking_metrics": results["ranking"],
        "classification_metrics": results["classification"],
        "counts": results["counts"],
    }


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
def _json_safe(value):
    """Recursively convert numpy scalars to plain JSON types."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    return value


def build_model_metadata(pipeline, X_all, y_all, phase11_evaluation):
    """All metadata for model_metadata.json - no invented values."""
    y_all = np.asarray(y_all)
    schema = build_input_schema()
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "algorithm": ALGORITHM,
        "feature_configuration": FEATURE_CONFIGURATION,
        "classification_threshold": float(CLASSIFICATION_THRESHOLD),
        "probability_representation": PROBABILITY_REPRESENTATION,
        "risk_score_range": RISK_SCORE_RANGE,
        "raw_input_features": schema["raw_input_features"],
        "engineered_features": schema["engineered_features"],
        "excluded_features": schema["excluded_features"],
        "model_parameters": dict(XGBOOST_PARAMS),
        "class_weighting": "none",
        "random_state": RANDOM_STATE,
        "production_training_row_count": int(len(X_all)),
        "production_training_default_counts": {
            "default": int((y_all == 1).sum()),
            "non_default": int((y_all == 0).sum()),
        },
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluation": _json_safe(phase11_evaluation),
    }


def save_model_artifact(pipeline, path=MODEL_PATH):
    """Persist the fitted production pipeline with joblib."""
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def save_metadata(metadata, path=MODEL_METADATA_PATH):
    """Persist model metadata as pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


# ---------------------------------------------------------------------------
# Serialization validation (pre-save vs post-load equivalence)
# ---------------------------------------------------------------------------
def validate_serialization_round_trip(pipeline, X_sample, path=MODEL_PATH):
    """
    Predict a deterministic small sample BEFORE saving, then reload the
    artifact and predict the SAME sample again. Asserts numerical equality
    within SERIALIZATION_ATOL and returns a summary dict.
    """
    before = pipeline.predict_proba(X_sample)[:, 1]
    save_model_artifact(pipeline, path)
    import joblib

    reloaded = joblib.load(path)
    after = reloaded.predict_proba(X_sample)[:, 1]

    assert len(before) == len(after) == len(X_sample)
    assert np.isfinite(before).all() and np.isfinite(after).all()
    max_diff = float(np.max(np.abs(before - after)))
    assert max_diff <= SERIALIZATION_ATOL, (
        f"Serialization round-trip drifted by {max_diff} > {SERIALIZATION_ATOL}."
    )
    return {
        "sample_size": int(len(X_sample)),
        "max_absolute_difference": max_diff,
        "probabilities_in_unit_interval": bool(
            (before >= 0.0).all() and (before <= 1.0).all()
        ),
        "serialization_ok": True,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    separator = "=" * 72
    print("\n" + separator)
    print("PHASE 15 - PRODUCTION MODEL ARTIFACT")
    print(separator)
    print(f"  Model        : {MODEL_NAME} ({MODEL_VERSION})")
    print(f"  Features     : {FEATURE_CONFIGURATION}")
    print(f"  Threshold    : {CLASSIFICATION_THRESHOLD}")
    print(f"  Probability  : {PROBABILITY_REPRESENTATION}")

    print("\n1. Loading the complete cleaned labelled dataset...")
    X_all, y_all = prepare_full_data()
    n_default = int((y_all == 1).sum())
    n_non_default = int((y_all == 0).sum())
    print(f"    Complete cleaned labelled rows : {len(X_all):,}")
    print(f"    Defaults   : {n_default:,}")
    print(f"    Non-defaults: {n_non_default:,}")

    print("\n2. Building the locked application-only XGBoost pipeline...")
    pipeline = build_production_pipeline()
    assert_parameters_locked(pipeline)
    assert_application_only(pipeline)
    print("    Locked params verified, loan_grade / loan_int_rate excluded.")

    print("\n3. Fitting EXACTLY ONCE on all cleaned labelled rows...")
    pipeline = train_production_model(X_all, y_all)
    print(f"    Production fit complete ({len(X_all):,} rows).")

    print("\n4. Capturing the frozen PHASE 11 holdout reference metrics...")
    phase11_evaluation = capture_phase11_reference_metrics()
    print(f"    evaluation_source: {phase11_evaluation['evaluation_source']}")
    print("    (reference metadata only - not a new evaluation)")

    print("\n5. Serialization validation (pre-save vs post-load)...")
    sample = X_all.head(5)
    validation = validate_serialization_round_trip(pipeline, sample, MODEL_PATH)
    print(f"    Deterministic sample size      : {validation['sample_size']}")
    print(f"    Max |before - after|           : {validation['max_absolute_difference']:.2e}")
    print(f"    Probabilities in [0, 1]        : "
          f"{validation['probabilities_in_unit_interval']}")
    print("    SUCCESS - serialization round-trip identical within "
          f"{SERIALIZATION_ATOL}.")

    print("\n6. Writing model_metadata.json...")
    metadata = build_model_metadata(pipeline, X_all, y_all, phase11_evaluation)
    save_metadata(metadata, MODEL_METADATA_PATH)
    print(f"    Saved : {MODEL_METADATA_PATH}")

    print("\n" + separator)
    print("PHASE 15 COMPLETE")
    print(separator)
    print(f"  Artifact  : {MODEL_PATH}")
    print(f"  Metadata  : {MODEL_METADATA_PATH}")
    print("  Official evaluation metrics remain the Phase 11 frozen holdout.")
    print("  No production-fit performance was reported as evaluation.")
    print(separator)


if __name__ == "__main__":
    main()