"""
Phase 15 - Production inference service for CreditWise AI.

Accepts RAW applicant information and returns a complete CreditWise risk
assessment for the LOCKED production model.

Frozen configuration (do not change)
------------------------------------
    Model      : Tuned XGBoost (exact Phase 9 best parameters)
    Features   : APPLICATION-ONLY (loan_grade / loan_int_rate excluded)
    Threshold  : 0.325
    Probability: RAW XGBoost predict_proba[:, 1]
    Score/band : Phase 14 risk_engine.build_risk_assessment()

Flow
----
    Raw applicant dict / DataFrame
        -> validate_applicant_input()
        -> canonical feature engineering (ml.feature_engineering)
        -> fitted production pipeline (ml/artifacts/creditwise_model.joblib)
        -> predict_proba()[:, 1]
        -> ml.risk_engine.build_risk_assessment()
        -> complete assessment object

Guarantees
----------
    - loan_grade / loan_int_rate are NEVER required inputs.
    - loan_status is NEVER accepted as a model feature.
    - person_emp_length may be None (the preprocessing pipeline imputes it),
      all other mandatory fields may not be missing.
    - The internal risk score is NOT a CIBIL or bureau score.

Run with:
    python ml/predict.py     (loads the artifact and prints a DEMONSTRATION)
"""

import json
import math
import numbers
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from ml.feature_engineering import ENGINEERED_FEATURES, engineer_features
from ml.preprocessing import feature_lists
from ml.risk_engine import MODEL_VERSION, build_risk_assessment

# ---------------------------------------------------------------------------
# Artifact locations (owned by the inference layer)
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "creditwise_model.joblib"
MODEL_METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
SHAP_SUMMARY_PATH = ARTIFACTS_DIR / "shap_summary.json"

# ---------------------------------------------------------------------------
# Locked feature schema - derived from the canonical project definitions.
# ---------------------------------------------------------------------------
# Lender-assigned / excluded fields that must NEVER enter the model.
FORBIDDEN_FIELDS = ["loan_grade", "loan_int_rate", "loan_status"]
# Extra raw column dropped during feature engineering (rebuilt from raw input).
DROPPED_DURING_ENGINEERING = "loan_percent_income"

# Allowed raw categorical values (from the cleaned dataset).
HOME_OWNERSHIP_VALUES = ("RENT", "OWN", "MORTGAGE", "OTHER")
LOAN_INTENT_VALUES = (
    "DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT",
    "MEDICAL", "PERSONAL", "VENTURE",
)
DEFAULT_ON_FILE_VALUES = ("Y", "N")

# Cleaning bounds (data_cleaning.py MAX_VALID_AGE / MAX_VALID_EMP_LENGTH).
MAX_VALID_AGE = 100
MAX_VALID_EMP_LENGTH = 60


def build_input_schema():
    """
    Canonical application-only input schema.

    Derives raw / engineered / model feature lists from `preprocessing`
    and `feature_engineering` constants - never a manually invented list.
    """
    feats = feature_lists("application_only")
    model_features = list(feats["numerical"]) + list(feats["categorical"])
    if feats["ordinal"]:
        model_features.extend(list(feats["ordinal"].keys()))

    # Raw applicant fields = model features minus the engineered ones, plus
    # cb_person_default_on_file (consumed purely to build has_prior_default).
    raw_input_features = [
        f for f in model_features if f not in ENGINEERED_FEATURES
    ]
    raw_input_features.append("cb_person_default_on_file")

    excluded_features = list(FORBIDDEN_FIELDS) + [DROPPED_DURING_ENGINEERING]

    return {
        "config": "application_only",
        "model_features": model_features,
        "raw_input_features": raw_input_features,
        "engineered_features": list(ENGINEERED_FEATURES),
        "excluded_features": excluded_features,
    }


SCHEMA = build_input_schema()
RAW_INPUT_FIELDS = SCHEMA["raw_input_features"]
MODEL_FEATURES = SCHEMA["model_features"]

NUMERIC_FIELDS = {
    "person_age", "person_income", "person_emp_length", "loan_amnt",
    "cb_person_cred_hist_length",
}
OPTIONAL_NUMERIC_FIELDS = {"person_emp_length"}
CATEGORICAL_ALLOWED = {
    "person_home_ownership": HOME_OWNERSHIP_VALUES,
    "loan_intent": LOAN_INTENT_VALUES,
    "cb_person_default_on_file": DEFAULT_ON_FILE_VALUES,
}

_ARTIFACT_CACHE = {}


def _is_missing(value):
    """True for None / NaN / pd.NA - i.e. a field the applicant did not give."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _require_numeric(applicant, field, *, minimum=None, maximum=None,
                     allow_none=False):
    """Validate one numeric field; returns the float value (or None)."""
    value = applicant[field]
    if allow_none and _is_missing(value):
        return None
    if _is_missing(value):
        raise ValueError(
            f"'{field}' is mandatory and was not provided (got {value!r})."
        )
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(
            f"'{field}' must be a real number, got {type(value).__name__} "
            f"({value!r})."
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"'{field}' must be finite, got {number!r}.")
    if minimum is not None and number <= minimum:
        raise ValueError(
            f"'{field}' must be greater than {minimum}, got {number:g}."
        )
    if maximum is not None and number > maximum:
        raise ValueError(
            f"'{field}' must be at most {maximum}, got {number:g}."
        )
    return number


def validate_applicant_input(data):
    """
    Validate one RAW applicant and return a normalized dict.

    Accepts a dict or a one-row pandas DataFrame. Raises ValueError with a
    clear message for any invalid / missing / forbidden input. Optional
    person_emp_length=None is preserved (model supports it).
    """
    applicant = dict(data) if isinstance(data, dict) else None

    if applicant is None and isinstance(data, pd.DataFrame):
        if len(data) != 1:
            raise ValueError(
                "A single applicant is required; got a DataFrame with "
                f"{len(data)} rows."
            )
        applicant = data.iloc[0].to_dict()

    if not isinstance(applicant, dict):
        raise ValueError(
            "Input must be a dictionary or a one-row pandas DataFrame, got "
            f"{type(data).__name__}."
        )

    present = {key for key in applicant.keys() if not _is_missing(applicant[key])}

    forbidden = [f for f in FORBIDDEN_FIELDS if f in present]
    if forbidden:
        raise ValueError(
            "Forbidden model inputs were provided: " + ", ".join(sorted(forbidden)) +
            ". These fields are excluded from the application-only model."
        )
    if DROPPED_DURING_ENGINEERING in present:
        raise ValueError(
            f"'{DROPPED_DURING_ENGINEERING}' is rebuilt from raw fields during "
            "feature engineering and must not be supplied."
        )

    missing = [f for f in RAW_INPUT_FIELDS if f not in applicant]
    if missing:
        raise ValueError(
            "Missing required raw applicant fields: " + ", ".join(missing) + "."
        )

    normalized = {}
    for field in RAW_INPUT_FIELDS:
        if field in NUMERIC_FIELDS:
            lower = None
            if field == "person_age":
                lower, upper = 0, MAX_VALID_AGE
            elif field == "person_income":
                lower, upper = 0, None
            elif field == "person_emp_length":
                lower, upper = 0, MAX_VALID_EMP_LENGTH
            elif field == "loan_amnt":
                lower, upper = 0, None
            else:  # cb_person_cred_hist_length
                lower, upper = None, None
            value = _require_numeric(
                applicant, field,
                minimum=lower if lower is not None else None,
                maximum=upper,
                allow_none=field in OPTIONAL_NUMERIC_FIELDS,
            )
            normalized[field] = value
        else:
            value = applicant[field]
            if _is_missing(value):
                raise ValueError(
                    f"'{field}' is mandatory and was not provided."
                )
            allowed = CATEGORICAL_ALLOWED[field]
            if not isinstance(value, str) or value not in allowed:
                raise ValueError(
                    f"'{field}' must be one of {sorted(allowed)}, got {value!r}."
                )
            normalized[field] = str(value)

    # Employment cannot exceed age / the 60-year cap (data_cleaning rules).
    if normalized["person_emp_length"] is not None:
        if normalized["person_emp_length"] > normalized["person_age"]:
            raise ValueError(
                "'person_emp_length' cannot exceed 'person_age' "
                f"({normalized['person_emp_length']:g} > "
                f"{normalized['person_age']:g})."
            )

    return normalized


def build_engineered_frame(normalized):
    """
    One-row DataFrame -> canonical feature engineering -> model-feature frame.

    The fitted pipeline expects the ENGINEERED feature columns; feature
    engineering is always called before predict_proba and is never
    duplicated inside the pipeline.
    """
    raw_df = pd.DataFrame([normalized], columns=RAW_INPUT_FIELDS)
    engineered, _ = engineer_features(raw_df)
    return engineered[MODEL_FEATURES]


def load_model_artifacts():
    """
    Load the fitted production pipeline and metadata from disk (cached).

    Returns (pipeline, metadata). Raises FileNotFoundError with guidance if
    the artifact has not been generated yet.
    """
    if _ARTIFACT_CACHE:
        return _ARTIFACT_CACHE["pipeline"], _ARTIFACT_CACHE["metadata"]
    if not MODEL_PATH.exists() or not MODEL_METADATA_PATH.exists():
        raise FileNotFoundError(
            "Production artifacts missing. Run first: "
            "python ml/train_production.py"
        )
    import joblib

    pipeline = joblib.load(MODEL_PATH)
    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    _ARTIFACT_CACHE["pipeline"] = pipeline
    _ARTIFACT_CACHE["metadata"] = metadata
    return pipeline, metadata


def load_shap_summary():
    """
    Load the PRECOMPUTED global SHAP explainability artifact from disk.

    Returns a dict with model_version / top_features / plots / disclaimer.
    Raises FileNotFoundError with guidance if it has not been generated yet.
    No SHAP values are recomputed at request time.
    """
    if not SHAP_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            "SHAP explainability artifact missing. Run first: "
            "python ml/explain_model.py"
        )
    return json.loads(SHAP_SUMMARY_PATH.read_text(encoding="utf-8"))


def predict_default_probability(applicant, pipeline=None):
    """
    RAW default-risk probability (0..1) for one applicant.

    pipeline: optional explicitly-fitted pipeline (default = production
    artifact). Always returns a float between 0 and 1.
    """
    if pipeline is None:
        pipeline, _ = load_model_artifacts()
    normalized = validate_applicant_input(applicant)
    X = build_engineered_frame(normalized)
    probability = float(pipeline.predict_proba(X)[:, 1][0])
    if not math.isfinite(probability) or not (0.0 <= probability <= 1.0):
        raise ValueError(
            f"Model returned an invalid probability: {probability!r}."
        )
    return probability


def predict_applicant(applicant, pipeline=None):
    """
    Complete CreditWise assessment for one applicant.

    1. validates raw input
    2. builds a one-row DataFrame
    3. runs canonical feature engineering
    4. predicts with the fitted production pipeline
    5. passes the RAW probability to the Phase 14 build_risk_assessment()

    All score / band / decision logic lives in ml.risk_engine - it is never
    duplicated here.
    """
    if pipeline is None:
        pipeline, _ = load_model_artifacts()
    probability = predict_default_probability(applicant, pipeline=pipeline)
    return build_risk_assessment(probability, model_version=MODEL_VERSION)


# ---------------------------------------------------------------------------
# CLI demonstration (inference only - never used for training/evaluation)
# ---------------------------------------------------------------------------
DEMO_APPLICANT = {
    "person_age": 35,
    "person_income": 60000,
    "person_home_ownership": "RENT",
    "person_emp_length": 8.0,
    "loan_intent": "PERSONAL",
    "loan_amnt": 15000,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 5,
}


def print_assessment(assessment):
    separator = "=" * 60
    print(separator)
    print("CreditWise AI Assessment")
    print(separator)
    print(f"  Default-risk probability estimate : {assessment['default_risk_probability']:.4f}")
    print(f"  CreditWise Risk Score             : {assessment['risk_score']} / 100")
    print(f"  Risk Category                     : {assessment['risk_category']}")
    print(f"  Decision                          : {assessment['decision']}")
    print(f"  Classification Threshold          : {assessment['classification_threshold']}")
    print(f"  Model Version                     : {assessment['model_version']}")
    print(separator)
    print(f"  {assessment['message']}")
    print(f"  {assessment['note']}")
    print("  NOTE: This internal CreditWise risk score is NOT a CIBIL score.")
    print(separator)


def main():
    print("\nLoading production artifacts...")
    pipeline, metadata = load_model_artifacts()
    print(f"  Pipeline        : {MODEL_PATH.name}")
    print(f"  Model version   : {metadata.get('model_version')}")

    print("\nRunning DEMONSTRATION applicant (inference only)...")
    assessment = predict_applicant(DEMO_APPLICANT, pipeline=pipeline)
    print_assessment(assessment)


if __name__ == "__main__":
    main()