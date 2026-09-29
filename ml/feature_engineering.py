"""
Phase 4 - Deterministic feature engineering for CreditWise AI.

Creates domain-relevant features directly from the cleaned raw columns.

Features created
----------------
    loan_to_income_ratio        = loan_amnt / person_income
    log_person_income           = log1p(person_income)
    employment_to_age_ratio     = person_emp_length / person_age
    credit_history_to_age_ratio = cb_person_cred_hist_length / person_age
    has_prior_default           = (cb_person_default_on_file == "Y").astype(int)

Design rules
------------
    - loan_status is NEVER used to construct a feature (no target leakage).
    - Every division is guarded so a zero denominator produces NaN (which the
      preprocessing pipeline will impute) instead of an error.
    - loan_percent_income is dropped: EDA showed it disagreed with the raw
      ratio in ~1,000 rows, so we rebuild it from the raw columns instead of
      trusting the stored value.
    - cb_person_default_on_file is replaced by has_prior_default (0/1).
    - Acting on a copy - the input DataFrame is never modified.
"""

import numpy as np
import pandas as pd

# Dropped columns replaced by cleaner engineered versions.
COLUMNS_REPLACED = ["loan_percent_income", "cb_person_default_on_file"]

# Canonical list of engineered feature names (single source of truth for
# downstream phases: metadata, input schema, tests).
ENGINEERED_FEATURES = [
    "loan_to_income_ratio",
    "log_person_income",
    "employment_to_age_ratio",
    "credit_history_to_age_ratio",
    "has_prior_default",
]


def engineer_features(df):
    """
    Return (engineered_df, engineered_feature_names) where engineered_df is a
    COPY of the input with the engineered features added.

    The input DataFrame is never modified.
    """
    result = df.copy()

    # Safe division: a zero denominator (or missing numerator) becomes NaN
    # instead of raising or producing infinity.
    def safe_divide(numerator, denominator):
        denominator = denominator.replace(0, np.nan)
        return numerator / denominator

    # 1) Rebuild the loan-to-income ratio from raw columns.
    result["loan_to_income_ratio"] = safe_divide(result["loan_amnt"], result["person_income"])

    # 2) Log transform for the heavily right-skewed income.
    result["log_person_income"] = np.log1p(result["person_income"])

    # 3) Employment stability relative to career length.
    result["employment_to_age_ratio"] = safe_divide(result["person_emp_length"], result["person_age"])

    # 4) Credit maturity relative to age.
    result["credit_history_to_age_ratio"] = safe_divide(
        result["cb_person_cred_hist_length"], result["person_age"]
    )

    # 5) Explicit binary flag for a previous default.
    result["has_prior_default"] = (result["cb_person_default_on_file"] == "Y").astype(int)

    # Remove the columns we have replaced/rebuild.
    result = result.drop(columns=[c for c in COLUMNS_REPLACED if c in result.columns])

    engineered_names = list(ENGINEERED_FEATURES)
    return result, engineered_names


if __name__ == "__main__":
    # Allow quick manual verification:
    #   python ml/feature_engineering.py
    from pathlib import Path

    from data_cleaning import clean_credit_data

    raw_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "credit_risk_dataset.csv"
    df = pd.read_csv(raw_path)
    cleaned, _ = clean_credit_data(df)
    engineered, names = engineer_features(cleaned)
    print(f"\nEngineered feature columns created: {names}")
    print(f"\nResulting columns:\n{list(engineered.columns)}")
    print(f"\nShape: {engineered.shape}")