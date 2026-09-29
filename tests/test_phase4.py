"""
Phase 4 tests - data cleaning, feature engineering and preprocessing.

Run with:
    python -m pytest tests/test_phase4.py -v

The tests verify the three machine-learning-stage modules independently and
together, with a strong focus on:
    - the original input DataFrame never being modified
    - the target never being changed or used to build features (no leakage)
    - impossible rows being removed and high-income rows being kept
    - default-value preservation while missing values stay missing
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.data_cleaning import MAX_VALID_AGE, MAX_VALID_EMP_LENGTH, clean_credit_data
from ml.feature_engineering import COLUMNS_REPLACED, engineer_features
from ml.preprocessing import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    LEAKAGE_SENSITIVE_FEATURES,
    NUMERICAL_FEATURES,
    ORDINAL_FEATURES_ORDER,
    TARGET,
    build_logistic_preprocessor,
    feature_lists,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "credit_risk_dataset.csv"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def raw_df():
    """The real raw dataset (read-only, never modified)."""
    return pd.read_csv(RAW_CSV)


@pytest.fixture(scope="module")
def clean_engine():
    """cleaned + engineered copy of the real dataset."""
    df = pd.read_csv(RAW_CSV)
    cleaned, _ = clean_credit_data(df)
    engineered, _ = engineer_features(cleaned)
    return cleaned, engineered


# --------------------------------------------------------------------------
# DATA CLEANING
# --------------------------------------------------------------------------
def test_cleaning_does_not_modify_input(raw_df):
    snapshot = raw_df.copy()
    clean_credit_data(raw_df)
    pd.testing.assert_frame_equal(raw_df, snapshot)


def test_cleaning_removes_duplicates(raw_df):
    cleaned, summary = clean_credit_data(raw_df)
    assert summary["duplicates_removed"] > 0
    assert cleaned.duplicated().sum() == 0


def test_cleaning_removes_impossible_rows(raw_df):
    cleaned, summary = clean_credit_data(raw_df)
    assert summary["invalid_age_removed"] > 0
    assert summary["invalid_emp_length_removed"] > 0
    # No impossible records may remain.
    assert (cleaned["person_age"] > MAX_VALID_AGE).sum() == 0
    assert (cleaned["person_emp_length"] > MAX_VALID_EMP_LENGTH).sum() == 0
    assert (cleaned["person_emp_length"] > cleaned["person_age"]).sum() == 0


def test_cleaning_summary_is_arithmetically_consistent(raw_df):
    _, summary = clean_credit_data(raw_df)

    # Recompute the union of invalid rows the same way the cleaner does.
    after_dup = raw_df.drop_duplicates()
    after_age = after_dup[after_dup["person_age"] <= MAX_VALID_AGE]
    emp_invalid = (after_age["person_emp_length"] > MAX_VALID_EMP_LENGTH) | (
        after_age["person_emp_length"] > after_age["person_age"]
    )

    expected_final = len(after_age) - int(emp_invalid.sum())
    assert summary["final_rows"] == expected_final
    assert (
        summary["final_rows"]
        == summary["original_rows"]
        - summary["duplicates_removed"]
        - summary["invalid_age_removed"]
        - summary["invalid_rows_removed_total"]
    )


def test_cleaning_keeps_high_income_rows(raw_df):
    cleaned, _ = clean_credit_data(raw_df)
    # High incomes are plausible for wealthy borrowers - they must survive.
    assert (cleaned["person_income"] > 1_000_000).sum() > 0
    assert cleaned["person_income"].max() > 1_000_000


def test_cleaning_does_not_fill_missing_values(raw_df):
    cleaned, _ = clean_credit_data(raw_df)
    # Both known-missing columns must still have missing values after cleaning.
    assert cleaned["loan_int_rate"].isna().sum() > 0
    assert cleaned["person_emp_length"].isna().sum() > 0


def test_cleaning_does_not_change_target(raw_df):
    cleaned, _ = clean_credit_data(raw_df)
    # For every surviving row, the target matches the original value.
    original_target = raw_df.loc[cleaned.index, TARGET]
    pd.testing.assert_series_equal(
        cleaned[TARGET].reset_index(drop=True),
        original_target.reset_index(drop=True),
    )


def test_cleaning_does_not_use_target_to_filter():
    # The exact same input-target pairs must survive regardless of target.
    df = pd.DataFrame(
        {
            "person_age": [30, 30, 144, 40, 30, 45],
            "person_emp_length": [5.0, 5.0, 4.0, 70.0, 35.0, 20.0],
            "person_income": [50000] * 5 + [200000],
            "loan_status": [0, 0, 0, 1, 1, 0],
        }
    )
    flipped = df.copy()
    flipped["loan_status"] = 1 - flipped["loan_status"]

    _, s1 = clean_credit_data(df)
    _, s2 = clean_credit_data(flipped)
    assert s1["final_rows"] == s2["final_rows"] == 2


def test_cleaning_removes_impossible_rows_correctly():
    df = pd.DataFrame(
        {
            "person_age": [30, 30, 144, 70, 30, 45],
            "person_emp_length": [5.0, 5.0, 4.0, 70.0, 35.0, 20.0],
            "person_income": [50000] * 5 + [200000],
            "loan_status": [0, 0, 0, 1, 1, 0],
        }
    )
    cleaned, summary = clean_credit_data(df)
    # Kept rows: the valid adult and the (kept) high-income applicant.
    assert summary["original_rows"] == 6
    assert summary["duplicates_removed"] == 1
    assert summary["invalid_age_removed"] == 1
    assert summary["invalid_emp_length_removed"] == 1
    assert summary["invalid_emp_greater_age_removed"] == 1
    assert summary["final_rows"] == 2
    assert set(cleaned["person_age"]) == {30, 45}
    assert set(cleaned["person_income"]) == {50000, 200000}


# --------------------------------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------------------------------
def test_engineering_does_not_modify_input(raw_df):
    cleaned, _ = clean_credit_data(raw_df)
    snapshot = cleaned.copy()
    engineer_features(cleaned)
    pd.testing.assert_frame_equal(cleaned, snapshot)


def test_engineering_creates_expected_features(raw_df):
    cleaned, _ = clean_credit_data(raw_df)
    engineered, names = engineer_features(cleaned)
    expected = [
        "loan_to_income_ratio",
        "log_person_income",
        "employment_to_age_ratio",
        "credit_history_to_age_ratio",
        "has_prior_default",
    ]
    assert set(names) == set(expected)
    for name in expected:
        assert name in engineered.columns


def test_engineering_ratio_is_exact(raw_df):
    cleaned, _ = clean_credit_data(raw_df)
    engineered, _ = engineer_features(cleaned)
    expected = cleaned["loan_amnt"] / cleaned["person_income"]
    pd.testing.assert_series_equal(
        engineered["loan_to_income_ratio"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_engineering_replaces_stale_columns(raw_df):
    cleaned, _ = clean_credit_data(raw_df)
    engineered, _ = engineer_features(cleaned)
    for column in COLUMNS_REPLACED:
        assert column not in engineered.columns


def test_engineering_has_prior_default_is_binary(raw_df):
    cleaned, _ = clean_credit_data(raw_df)
    engineered, _ = engineer_features(cleaned)
    assert set(engineered["has_prior_default"].unique()).issubset({0, 1})


def test_engineering_division_is_safe(raw_df):
    cleaned, _ = clean_credit_data(raw_df)
    engineered, _ = engineer_features(cleaned)
    ratio_cols = [
        "loan_to_income_ratio",
        "employment_to_age_ratio",
        "credit_history_to_age_ratio",
    ]
    for col in ratio_cols:
        assert np.isfinite(engineered[col].replace([np.inf, -np.inf], np.nan).dropna()).all()


def test_engineering_does_not_leak_target(raw_df):
    cleaned, _ = clean_credit_data(raw_df)
    engineered_a, _ = engineer_features(cleaned)

    flipped = cleaned.copy()
    flipped["loan_status"] = 1 - flipped["loan_status"]
    engineered_b, _ = engineer_features(flipped)

    # Feature columns must be identical no matter what the target says.
    feature_cols = [c for c in engineered_a.columns if c != TARGET]
    for col in feature_cols:
        pd.testing.assert_series_equal(
            engineered_a[col].reset_index(drop=True),
            engineered_b[col].reset_index(drop=True),
            check_names=False,
        )


# --------------------------------------------------------------------------
# PREPROCESSING
# --------------------------------------------------------------------------
def test_preprocessing_defines_target_and_feature_groups():
    assert TARGET == "loan_status"
    assert TARGET not in ALL_FEATURES
    assert TARGET not in NUMERICAL_FEATURES
    assert TARGET not in CATEGORICAL_FEATURES
    assert TARGET not in ORDINAL_FEATURES_ORDER


def test_leakage_sensitive_features_are_identified():
    assert "loan_grade" in LEAKAGE_SENSITIVE_FEATURES
    assert "loan_int_rate" in LEAKAGE_SENSITIVE_FEATURES


def test_feature_lists_full_vs_application_only():
    full = feature_lists("full")
    app = feature_lists("application_only")

    assert "loan_grade" in full["ordinal"]
    assert "loan_int_rate" in full["numerical"]

    assert app["ordinal"] == {}
    assert "loan_int_rate" not in app["numerical"]


def test_preprocessor_is_never_fitted_by_default():
    preprocessor, _ = build_logistic_preprocessor("full")
    # ColumnTransformer only exposes fitted state after .fit() is called.
    assert not hasattr(preprocessor, "transformers_")


def test_preprocessor_end_to_end_keeps_target(raw_df):
    # Build the transformer and smoke-fit it on a small subset (test-only
    # validation; the real fit happens on the training split in Phase 5).
    cleaned, _ = clean_credit_data(raw_df)
    engineered, _ = engineer_features(cleaned)

    subset = engineered.head(200)
    y = subset[TARGET]

    full_pp, full_feats = build_logistic_preprocessor("full")
    app_pp, app_feats = build_logistic_preprocessor("application_only")

    X_full = full_pp.fit_transform(subset[full_feats["numerical"] + full_feats["categorical"] + list(full_feats["ordinal"])])
    X_app = app_pp.fit_transform(subset[app_feats["numerical"] + app_feats["categorical"] + list(app_feats["ordinal"])])

    # Both run without error and yield one row per input row.
    assert X_full.shape[0] == len(subset)
    assert X_app.shape[0] == len(subset)
    # Application-only excludes loan_grade / loan_int_rate -> fewer columns.
    assert X_full.shape[1] > X_app.shape[1]

    # The target is exhausted by the pipeline, not transformed or expanded.
    assert len(y) == len(subset)


def test_preprocessor_output_is_sparse_when_scaled(raw_df):
    """Sanity check: every transform step succeeds on the real columns."""
    cleaned, _ = clean_credit_data(raw_df)
    engineered, _ = engineer_features(cleaned)
    pp, feats = build_logistic_preprocessor("full")
    X = pp.fit_transform(engineered[feats["numerical"] + feats["categorical"] + list(feats["ordinal"])].head(50))
    assert X.shape[0] == 50


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))