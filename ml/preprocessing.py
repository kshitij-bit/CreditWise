"""
Phase 4 - Reusable Scikit-learn preprocessing components for CreditWise AI.

This module only DEFINES and BUILDS preprocessing pipelines. It never fits
them to the dataset. Fitting must happen inside Phase 5 on the **training
split only** to avoid data leakage.

Pipeline for the baseline Logistic Regression
----------------------------------------------
    numerical  -> SimpleImputer(strategy="median", add_indicator=True)
                  -> StandardScaler()
    categorical-> SimpleImputer(strategy="most_frequent")
                  -> OneHotEncoder(handle_unknown="ignore")
    ordinal    -> SimpleImputer(strategy="most_frequent")
                  -> OrdinalEncoder(A=1 ... G=7)     [loan_grade only]
    combined   -> ColumnTransformer

Two feature configurations
--------------------------
    "full"             : includes loan_grade and loan_int_rate
    "application_only" : excludes loan_grade and loan_int_rate

The "application_only" config lets us later measure how much apparent
predictive power comes from lender-assigned/priced fields (possible
leakage discussion), with no changes needed elsewhere.

Why each feature belongs in its group
-------------------------------------
    numerical   : continuous / count values that need scaling for linear models
    categorical : nominal strings with no order -> one-hot encoded
    ordinal     : loan_grade has a strict order A..G -> ordinal encoding
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Constant definitions
# ---------------------------------------------------------------------------
TARGET = "loan_status"

# Columns that are assigned/priced by the lender. Keep them in FULL config,
# drop them in APPLICATION_ONLY config to investigate possible leakage.
LEAKAGE_SENSITIVE_FEATURES = ["loan_grade", "loan_int_rate"]

NUMERICAL_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_to_income_ratio",
    "log_person_income",
    "employment_to_age_ratio",
    "credit_history_to_age_ratio",
    "cb_person_cred_hist_length",
    "has_prior_default",  # binary 0/1 - safe to treat as numeric
]

# Nominal categorical features (no intrinsic order) -> one-hot encoded.
CATEGORICAL_FEATURES = [
    "person_home_ownership",
    "loan_intent",
]

# Ordinal categorical features -> mapped to 1..N (A is best / lowest risk).
ORDINAL_FEATURES_ORDER = {
    "loan_grade": ["A", "B", "C", "D", "E", "F", "G"],
}

ALL_FEATURES = (
    NUMERICAL_FEATURES + CATEGORICAL_FEATURES + list(ORDINAL_FEATURES_ORDER.keys())
)

# Safety: the target must never become a model feature.
assert TARGET not in ALL_FEATURES, f"{TARGET} must not be used as a feature!"
assert "loan_status" not in NUMERICAL_FEATURES and "loan_status" not in CATEGORICAL_FEATURES


# ---------------------------------------------------------------------------
# Feature list selection
# ---------------------------------------------------------------------------
def feature_lists(config="full"):
    """
    Return {"numerical": [...], "categorical": [...], "ordinal": {...}}
    for the given configuration.

    config="full"             -> every feature
    config="application_only" -> excludes loan_grade and loan_int_rate
    """
    if config not in ("full", "application_only"):
        raise ValueError(f"Unknown config '{config}'. Use 'full' or 'application_only'.")

    if config == "application_only":
        numerical = [f for f in NUMERICAL_FEATURES if f not in LEAKAGE_SENSITIVE_FEATURES]
        ordinal = {k: v for k, v in ORDINAL_FEATURES_ORDER.items() if k not in LEAKAGE_SENSITIVE_FEATURES}
    else:
        numerical = list(NUMERICAL_FEATURES)
        ordinal = {k: v for k, v in ORDINAL_FEATURES_ORDER.items()}

    return {
        "config": config,
        "numerical": numerical,
        "categorical": list(CATEGORICAL_FEATURES),
        "ordinal": ordinal,
    }


# ---------------------------------------------------------------------------
# Preprocessing pipeline builders (define only - NO fitting here)
# ---------------------------------------------------------------------------
def build_logistic_preprocessor(config="full"):
    """
    Build (but do NOT fit) the Logistic Regression preprocessing pipeline
    as a ColumnTransformer for the requested feature configuration.

        categorical   -> most_frequent imputation + one-hot (handle unknown)
        ordinal       -> most_frequent imputation + ordinal encoding (1..N)
        numerical     -> median imputation (+ missing indicator) + scaling
    """
    features = feature_lists(config)

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    ordinal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    transformers = [("numerical", numerical_pipeline, features["numerical"])]

    if features["categorical"]:
        transformers.append(("categorical", categorical_pipeline, features["categorical"]))

    if features["ordinal"]:
        # Each ordinal column gets its own encoder so the category order is
        # explicit and unambiguous.
        for column, categories in features["ordinal"].items():
            transformers.append(
                (f"ordinal_{column}", ordinal_pipeline, [column])
            )

    preprocessor = ColumnTransformer(transformers)
    return preprocessor, features


def print_feature_schema(features):
    """Print the feature groups for the current configuration (readable)."""
    separator = "-" * 60
    print(separator)
    print(f"FEATURE SCHEMA - config: {features['config']}")
    print(separator)
    print(f"Numerical ({len(features['numerical'])}):")
    for f in features["numerical"]:
        print(f"    - {f}")
    print(f"Categorical ({len(features['categorical'])}):")
    for f in features["categorical"]:
        print(f"    - {f}")
    print(f"Ordinal ({len(features['ordinal'])}):")
    for column, order in features["ordinal"].items():
        print(f"    - {column} "
              f"(ordered {' > '.join(order)})")
    print(separator)


if __name__ == "__main__":
    # Quick manual verification (builds only - never fits):
    #   python ml/preprocessing.py
    print("Verbose column counts per configuration:")
    for config in ("full", "application_only"):
        _, feats = build_logistic_preprocessor(config)
        n_lists = [feats["numerical"], feats["categorical"], list(feats["ordinal"])]
        total = sum(len(x) for x in n_lists)
        print(f"  {config:18s}: {total} input features "
              f"(num={len(feats['numerical'])}, cat={len(feats['categorical'])}, "
              f"ord={len(feats['ordinal'])})")
    print("\nFull schema for reference:")
    print_feature_schema(feature_lists("full"))