"""
Phase 4 - Deterministic data cleaning for CreditWise AI.

What this module does
---------------------
    - Removes exact duplicate rows
    - Removes clearly impossible records:
        * person_age > 100
        * person_emp_length > 60
        * person_emp_length > person_age
    - Keeps high-income observations (they are not necessarily errors)

What this module deliberately does NOT do
------------------------------------------
    - It never touches the raw CSV on disk
    - It never fills missing values (that belongs to preprocessing, using
      statistics learned on the training split only)
    - It never uses loan_status when deciding which rows to drop

Determinism / leakage
---------------------
    All rules here are deterministic (no statistics are learned from the
    data), so cleaning can safely run *before* the train/test split.
"""

import pandas as pd

# Sanity bounds discovered during EDA (Phase 3).
MAX_VALID_AGE = 100
MAX_VALID_EMP_LENGTH = 60


def clean_credit_data(df):
    """
    Return (cleaned_df, summary) where cleaned_df is a COPY of the input
    DataFrame with duplicates and impossible records removed.

    The input DataFrame is never modified - everything works on a copy.
    """
    original_rows = len(df)

    # 1) Work on a copy so the caller's DataFrame is never modified.
    working = df.copy()

    # 2) Remove exact duplicate rows.
    duplicates_removed = int(working.duplicated().sum())
    working = working.drop_duplicates()

    # 3) Remove impossible age records.
    invalid_age_mask = working["person_age"] > MAX_VALID_AGE
    invalid_age_removed = int(invalid_age_mask.sum())
    working = working[~invalid_age_mask]

    # 4) Remove impossible employment records.
    #    NaN employment length simply fails every comparison, so those rows
    #    are kept and handled later by median imputation.
    invalid_emp_length_mask = working["person_emp_length"] > MAX_VALID_EMP_LENGTH
    invalid_emp_greater_age_mask = working["person_emp_length"] > working["person_age"]
    invalid_emp_removed_total = int(
        (invalid_emp_length_mask | invalid_emp_greater_age_mask).sum()
    )
    working = working[~(invalid_emp_length_mask | invalid_emp_greater_age_mask)]

    final_rows = len(working)

    summary = {
        "original_rows": original_rows,
        "duplicates_removed": duplicates_removed,
        "invalid_age_removed": invalid_age_removed,
        "invalid_emp_length_removed": int(invalid_emp_length_mask.sum()),
        "invalid_emp_greater_age_removed": int(invalid_emp_greater_age_mask.sum()),
        "invalid_rows_removed_total": invalid_emp_removed_total,
        "final_rows": final_rows,
    }

    _print_summary(summary)
    return working, summary


def _print_summary(summary):
    """Print a human-readable cleaning report."""
    separator = "=" * 60
    print(separator)
    print("DATA CLEANING SUMMARY")
    print(separator)
    print(f"Original rows                          : {summary['original_rows']:,}")
    print(f"Duplicate rows removed                 : {summary['duplicates_removed']:,}")
    print(f"Invalid age rows removed (> {MAX_VALID_AGE})            : {summary['invalid_age_removed']:,}")
    print(f"Invalid employment rows removed (> {MAX_VALID_EMP_LENGTH}): {summary['invalid_emp_length_removed']:,}")
    print(f"Employment > age rows removed          : {summary['invalid_emp_greater_age_removed']:,}")
    print(f"Total invalid rows removed             : {summary['invalid_rows_removed_total']:,}")
    print(f"Final row count                        : {summary['final_rows']:,}")
    print(separator)


if __name__ == "__main__":
    # Allow quick manual verification:
    #   python ml/data_cleaning.py
    from pathlib import Path

    raw_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "credit_risk_dataset.csv"
    df = pd.read_csv(raw_path)
    clean_credit_data(df)