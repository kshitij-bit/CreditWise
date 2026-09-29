"""
Phase 2 - Dataset Inspection for CreditWise AI.

Reads the raw credit risk dataset and prints:
    - Dataset shape, head, columns, dtypes
    - Missing values, duplicates
    - Descriptive statistics, categorical unique values
    - Target class counts / percentages
    - Data-quality sanity checks

Purely diagnostic. Does NOT modify, clean, or transform the data.
"""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
# Path(__file__).resolve() -> absolute path of THIS script, regardless of how
# and from where it is invoked. .parent climbs to the project root folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "credit_risk_dataset.csv"

# ---------------------------------------------------------------------------
# Helpers for nicely formatted output
# ---------------------------------------------------------------------------
def print_section(title):
    """Print a clear section banner so the console output is easy to scan."""
    separator = "=" * 70
    print(f"\n{separator}")
    print(title)
    print(separator)


def print_subsection(title):
    """Print a smaller banner for sub-sections inside a section."""
    separator = "-" * 70
    print(f"\n{separator}")
    print(title)
    print(separator)


# ---------------------------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------------------------
df = pd.read_csv(RAW_DATA_PATH)

# ---------------------------------------------------------------------------
# 2. Basic structure
# ---------------------------------------------------------------------------
print_section("1. DATASET SHAPE")
print(f"Number of rows    : {df.shape[0]}")
print(f"Number of columns : {df.shape[1]}")

print_section("2. FIRST 5 ROWS")
print(df.head().to_string())

print_section("3. COLUMN NAMES")
for column in df.columns:
    print(f"- {column}")

print_section("4. DATA TYPES")
print(df.dtypes.to_string())

# ---------------------------------------------------------------------------
# 3. Missing values
# ---------------------------------------------------------------------------
print_section("5. MISSING VALUES")

# Total missing per column and the corresponding percentage of the dataset.
missing_counts = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100

# Build a small summary table from the two series.
missing_summary = pd.DataFrame(
    {
        "missing_count": missing_counts,
        "missing_percent": missing_percent.round(2),
    }
)

if missing_counts.sum() == 0:
    print("No missing values in the dataset.")
else:
    print("Columns with missing values:")
    print(missing_summary[missing_summary["missing_count"] > 0].to_string())
    print(f"\nTotal missing cells: {missing_counts.sum()}")

# ---------------------------------------------------------------------------
# 4. Duplicate rows
# ---------------------------------------------------------------------------
print_section("6. DUPLICATE ROWS")

n_duplicates = df.duplicated().sum()
print(f"Duplicate rows: {n_duplicates} ({n_duplicates / len(df) * 100:.2f}%)")

# ---------------------------------------------------------------------------
# 5. Numerical descriptive statistics
# ---------------------------------------------------------------------------
print_section("7. NUMERICAL DESCRIPTIVE STATISTICS")
print(df.describe().to_string())

# ---------------------------------------------------------------------------
# 6. Categorical columns - unique values
# ---------------------------------------------------------------------------
print_section("8. UNIQUE VALUES FOR CATEGORICAL COLUMNS")

# Select columns that hold text (categorical) data.
categorical_columns = df.select_dtypes(include=["object", "str"]).columns

for column in categorical_columns:
    print_subsection(f"COLUMN: {column} ({df[column].nunique()} unique values)")
    # value_counts shows every category together with its frequency.
    print(df[column].value_counts(dropna=False).to_string())

# ---------------------------------------------------------------------------
# 7. Target variable distribution
# ---------------------------------------------------------------------------
print_section("9. TARGET VARIABLE: loan_status")

# loan_status is our target: 1 = default, 0 = no default.
status_counts = df["loan_status"].value_counts().sort_index()
status_percent = (status_counts / len(df)) * 100

print("Class counts:")
for status_value, count in status_counts.items():
    label = "DEFAULT" if int(status_value) == 1 else "NO DEFAULT"
    print(f"  loan_status = {status_value} ({label:10s}): {count} rows")

print("Class percentages:")
for status_value, percent in status_percent.items():
    print(f"  loan_status = {status_value}: {percent:.2f}%")

# ---------------------------------------------------------------------------
# 8. Data-quality / sanity checks
#      These only COUNT suspicious rows. No rows are removed or changed.
# ---------------------------------------------------------------------------
print_section("10. DATA-QUALITY SANITY CHECKS (diagnostic only)")

# Each check prints how many rows violate the expected/mostly-sane range.


def report_check(check_name, mask, expected_clean=True):
    """Count rows flagged by a boolean mask and printed verdict."""
    n_flags = int(mask.sum())
    status = "OK" if n_flags == 0 else "FLAGGED"
    print(f"[{status:7s}] {check_name}: {n_flags} rows")
    return n_flags


# --- person_age checks ---
report_check(
    "person_age < 18",
    df["person_age"] < 18,
)
report_check(
    "person_age > 100",
    df["person_age"] > 100,
)

# --- person_emp_length checks (missing values are handled separately) ---
report_check(
    "person_emp_length < 0",
    df["person_emp_length"] < 0,
)
report_check(
    "person_emp_length > 60",
    df["person_emp_length"] > 60,
)
# It is impossible to have worked MORE years than one has been alive.
report_check(
    "person_emp_length > person_age",
    df["person_emp_length"] > df["person_age"],
)

# --- person_income checks ---
report_check(
    "person_income <= 0",
    df["person_income"] <= 0,
)

# --- loan_amnt checks ---
report_check(
    "loan_amnt <= 0",
    df["loan_amnt"] <= 0,
)

# --- loan_int_rate checks ---
report_check(
    "loan_int_rate <= 0",
    df["loan_int_rate"] <= 0,
)

# --- loan_percent_income range checks ---
# loan_percent_income = loan_amnt / person_income.
# A value of 1 means the whole yearly income is needed to repay the loan;
# anything above that amount is financially implausible in this dataset.
print_subsection("loan_percent_income range checks")
# Reasonable band for a consumer-loan dataset (most values sit around 0.1-0.5).
report_check(
    "loan_percent_income < 0 (negative share)",
    df["loan_percent_income"] < 0,
)
report_check(
    "loan_percent_income > 1 (loan larger than yearly income)",
    df["loan_percent_income"] > 1,
)
# Values in the top ~1% of the distribution are suspicious but still valid.
high_share_threshold = 0.75
report_check(
    f"loan_percent_income > {high_share_threshold} (very high share)",
    df["loan_percent_income"] > high_share_threshold,
)

# ---------------------------------------------------------------------------
# Final summary line
# ---------------------------------------------------------------------------
print_section("INSPECTION COMPLETE")
print("This script is read-only: the original CSV was not modified.")