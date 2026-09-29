"""
Phase 13 - Probability calibration analysis for the locked final XGBoost model.

ANALYSIS ONLY. The predictive configuration is FROZEN from Phase 11:

    Model      : Tuned XGBoost (exact Phase 9 best parameters)
    Features   : APPLICATION-ONLY (loan_grade / loan_int_rate excluded)
    Threshold  : 0.325

Question to answer
------------------
    "When the model predicts roughly 0.60 risk, do approximately 60% of
     similar applicants actually default?"

Guards
------
    - No threshold change, no hyperparameter tuning, no feature change.
    - The locked pipeline is fitted ONLY on the complete X_train / y_train.
    - Calibration (sigmoid / Platt, isotonic) is learned on TRAINING data only
      using leakage-safe cross-validation (CalibratedClassifierCV with the
      project's 5-fold StratifiedKFold). y_test is NEVER used to fit anything.
    - The locked classification model itself is NOT modified.
    - Probabilities are described as "default-risk probability estimates",
      never as guaranteed real-world default rates.
    - No risk-score bands, no production artifacts, no backend/frontend.

Run with:
    python ml/calibration_analysis.py

Output files
------------
    reports/figures/calibration_curve.png      (reliability diagram)
    reports/figures/probability_distribution.png
"""

import sys
from pathlib import Path

# Make the project root importable so `ml.*` modules work no matter from
# which folder this script is executed (e.g. python ml/calibration_analysis.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")  # headless figure generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from ml.compare_models import prepare_data
from ml.cross_validate_baseline import make_folds
from ml.final_evaluation import LOCKED_THRESHOLD, build_final_pipeline
from ml.threshold_analysis import XGBOOST_PARAMS, verify_application_only_features

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
N_BINS = 10
BIN_EDGES = np.linspace(0.0, 1.0, N_BINS + 1)
METHODS = ("RAW", "SIGMOID", "ISOTONIC")
CALIBRATED_METHODS = ("SIGMOID", "ISOTONIC")


# ---------------------------------------------------------------------------
# Pure calibration helpers (independently unit-tested)
# ---------------------------------------------------------------------------
def expected_calibration_error(y_true, y_proba, n_bins=N_BINS):
    """
    Binned ECE over uniform [0, 1] bins.

    ECE = sum over bins ( fraction of samples in bin *
                         |mean predicted prob - observed default rate| ).
    Empty bins contribute zero and never raise.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_proba, edges[1:-1], right=False)

    total = len(y_proba)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        weight = mask.sum() / total
        gap = abs(float(np.mean(y_proba[mask])) - float(np.mean(y_true[mask])))
        ece += weight * gap
    return float(ece)


def signed_bias(y_true, y_proba):
    """Mean signed difference (predicted - observed). + => overconfident."""
    return float(np.mean(np.asarray(y_proba) - np.asarray(y_true)))


def bin_calibration_table(y_true, y_proba, edges=BIN_EDGES):
    """
    Per-bin calibration report over uniform-width bins.

    Columns: Bin | Count | Mean Predicted Probability | Observed Default Rate
             | Absolute Calibration Gap. Every observation falls into exactly
             one bin, so the counts always sum to len(y_proba).
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    bin_ids = np.digitize(y_proba, edges[1:-1], right=False)

    rows = []
    for i in range(len(edges) - 1):
        mask = bin_ids == i
        label = f"{edges[i]:.1f}-{edges[i + 1]:.1f}"
        count = int(mask.sum())
        if count == 0:
            rows.append({"bin": label, "count": 0, "mean_predicted": np.nan,
                         "observed_rate": np.nan, "abs_gap": np.nan})
            continue
        mean_pred = float(np.mean(y_proba[mask]))
        obs_rate = float(np.mean(y_true[mask]))
        rows.append({"bin": label, "count": count, "mean_predicted": mean_pred,
                     "observed_rate": obs_rate, "abs_gap": abs(mean_pred - obs_rate)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Prediction generation
# ---------------------------------------------------------------------------
def fit_locked_pipeline(X_train, y_train):
    """Fit the locked final XGBoost pipeline on the training portion ONLY."""
    pipeline = build_final_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline


def predict_raw(pipeline, X_test):
    """Raw locked-model default-risk probabilities for hold-out observations."""
    return pipeline.predict_proba(X_test)[:, 1]


def fit_calibrated(method, X_train, y_train):
    """
    Fit a leakage-safe calibrated predictor on the training portion ONLY.

    Uses CalibratedClassifierCV with the project's 5-fold StratifiedKFold
    (shuffle=True, random_state=42): calibration is estimated from
    out-of-fold training predictions, never from the holdout set.
    """
    if method not in CALIBRATED_METHODS:
        raise ValueError(f"Unknown calibration method '{method}'.")
    splitter, _ = make_folds(y_train)
    calibrator = CalibratedClassifierCV(
        estimator=build_final_pipeline(),
        method="sigmoid" if method == "SIGMOID" else "isotonic",
        cv=splitter,
        ensemble="auto",
    )
    calibrator.fit(X_train, y_train)
    return calibrator


def evaluate_probabilities(y_true, y_proba):
    """Brier / ROC-AUC / PR-AUC / ECE for one probability source."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    return {
        "brier": float(brier_score_loss(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "ece": expected_calibration_error(y_true, y_proba),
        "bias": signed_bias(y_true, y_proba),
    }


def comparison_table(results):
    """Summary DataFrame across the RAW / SIGMOID / ISOTONIC methods."""
    rows = []
    for method in METHODS:
        r = results[method]
        rows.append({
            "method": method,
            "brier": r["brier"],
            "ece": r["ece"],
            "roc_auc": r["roc_auc"],
            "pr_auc": r["pr_auc"],
            "bias": r.get("bias", np.nan),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Decision rule (pure, testable)
# ---------------------------------------------------------------------------
def recommend_calibration(raw, sigmoid, isotonic):
    """
    Choose the probability representation for reporting.

    RAW      : keep raw probabilities if already reasonably calibrated and
               no materially better option exists.
    SIGMOID  : one-parameter monotone recalibration (stable, low variance).
    ISOTONIC : flexible monotone recalibration (best fit, higher variance).
    """
    methods = {"RAW": raw, "SIGMOID": sigmoid, "ISOTONIC": isotonic}
    if raw["ece"] <= 0.03 and raw["brier"] <= min(
            m["brier"] for m in methods.values()) + 0.005:
        return "RAW"
    best_name = min(methods, key=lambda name: methods[name]["ece"])
    # Stability tie-break: prefer sigmoid over isotonic unless isotonic is
    # meaningfully better (>= 1 ECE point of improvement).
    if best_name == "ISOTONIC" and (isotonic["ece"] - sigmoid["ece"]) >= -0.01:
        return "SIGMOID"
    return best_name


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def save_calibration_curve(y_test, results, path, n_bins=N_BINS):
    """Reliability diagram: ideal diagonal + RAW + SIGMOID + ISOTONIC."""
    from sklearn.calibration import calibration_curve as _curve

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Ideal calibration")
    style = {
        "RAW": ("#1f77b4", "o", "Raw XGBoost"),
        "SIGMOID": ("#2ca02c", "s", "Sigmoid (Platt)"),
        "ISOTONIC": ("#d62728", "^", "Isotonic"),
    }
    for method in METHODS:
        proba = results[method]["_proba"]
        prob_true, prob_pred = _curve(
            np.asarray(y_test), proba, n_bins=n_bins, strategy="uniform"
        )
        color, marker, label = style[method]
        plt.plot(prob_pred, prob_true, marker + "-", color=color, label=label)
    plt.xlabel("Mean predicted default-risk probability")
    plt.ylabel("Observed default rate")
    plt.title("Calibration curve - locked XGBoost, hold-out set")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.gcf().savefig(str(path), bbox_inches="tight", dpi=150)
    plt.close()


def save_probability_distribution(results, path, n_bins=30):
    """Overlaid histograms of predicted probabilities per method."""
    plt.figure(figsize=(8, 5))
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for method in METHODS:
        proba = np.asarray(results[method]["_proba"])
        plt.hist(proba, bins=bin_edges, alpha=0.45, density=True,
                 label=method if method != "RAW" else "Raw XGBoost")
    plt.xlabel("Predicted default-risk probability")
    plt.ylabel("Density")
    plt.title("Probability distribution by calibration method (hold-out set)")
    plt.legend()
    plt.tight_layout()
    plt.gcf().savefig(str(path), bbox_inches="tight", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------
def print_locked_configuration():
    separator = "=" * 72
    print("\n" + separator)
    print("LOCKED MODEL CONFIGURATION (calibration analysis ONLY)")
    print(separator)
    print(f"  Model        : Tuned XGBoost, n_estimators={XGBOOST_PARAMS['n_estimators']}, "
          f"max_depth={XGBOOST_PARAMS['max_depth']}, "
          f"learning_rate={XGBOOST_PARAMS['learning_rate']}")
    print(f"  Features     : APPLICATION-ONLY "
          f"(loan_grade / loan_int_rate excluded - verified)")
    print(f"  Threshold    : {LOCKED_THRESHOLD:.3f} (locked, not changed)")


def print_comparison_table(table):
    separator = "=" * 72
    print("\n" + separator)
    print("COMPARISON - RAW vs SIGMOID vs ISOTONIC (hold-out set)")
    print(separator)
    print(table.to_string(index=False))
    print("\n  Note: sigmoid and isotonic calibration are MONOTONE transforms.")
    print("  They re-map probabilities but preserve the ranking of applicants,")
    print("  so ROC-AUC / PR-AUC are expected to stay essentially unchanged.")
    print("  Calibration is about probability INTERPRETATION, not ranking.")


def print_bin_table(df, label):
    separator = "=" * 72
    print("\n" + separator)
    print(f"CALIBRATION BINS - {label}")
    print(separator)
    for _, row in df.iterrows():
        if np.isnan(row["mean_predicted"]):
            print(f"  {row['bin']:10s} count {int(row['count']):6d}  (empty bin)")
        else:
            print(f"  {row['bin']:10s} count {int(row['count']):6d}  "
                  f"mean_pred={row['mean_predicted']:.3f}  "
                  f"obs_rate={row['observed_rate']:.3f}  "
                  f"gap={row['abs_gap']:.3f}")


def print_interpretation(table, results):
    separator = "=" * 72
    print("\n" + separator)
    print("INTERPRETATION")
    print(separator)
    raw = results["RAW"]
    print("\n  - Raw probabilities are ")
    if raw["bias"] > 0.005:
        print("    OVERCONFIDENT: the model predicts slightly higher default-risk")
        print("    probabilities than the observed default rates overall.")
    elif raw["bias"] < -0.005:
        print("    UNDERCONFIDENT: the model predicts slightly lower default-risk")
        print("    probabilities than the observed default rates overall.")
    else:
        print("    essentially UNBIASED overall (predicted vs observed mean "
              "default rate).")

    brier_row = table.loc[table["brier"].idxmin()]
    print(f"\n  - Lowest Brier Score : {brier_row['method']} "
          f"({brier_row['brier']:.4f})")
    print("\n  - ROC-AUC / PR-AUC across methods:")
    for _, row in table.iterrows():
        print(f"      {row['method']:9s} ROC-AUC={row['roc_auc']:.4f}  "
              f"PR-AUC={row['pr_auc']:.4f}")
    print("    (Monotone recalibration does not materially change ranking "
          "metrics.)")

    decision = recommend_calibration(
        results["RAW"], results["SIGMOID"], results["ISOTONIC"]
    )
    print(f"\n  - RECOMMENDED probability representation for reporting: "
          f"{decision}.")
    print("\n  IMPORTANT:")
    print("    - Probabilities are 'default-risk probability estimates' and")
    print("      are not guaranteed real-world default rates.")
    print("    - The classification threshold 0.325 is NOT changed by this")
    print("      phase and none of the calibrated methods replace the locked")
    print("      classification model.")
    print("    - If a calibrated probability layer is used for reporting, its")
    print("      integration with score bands / decision logic is a separate,")
    print("      deliberate step for a later phase.")
    return decision


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    print("Phase 13 - probability calibration analysis for the locked XGBoost.")
    print("Loading data and applying the same deterministic cleaning + split...")
    X_train, X_test, y_train, y_test = prepare_data()
    y_test_arr = np.asarray(y_test)
    print(f"  X_train / y_train : {len(X_train):,} rows (fitted here only)")
    print(f"  X_test  / y_test  : {len(X_test):,} rows (frozen hold-out)")

    verify_application_only_features()
    print_locked_configuration()

    print("\nFitting the locked pipeline on X_train / y_train ONLY...")
    pipeline = fit_locked_pipeline(X_train, y_train)
    raw_proba = predict_raw(pipeline, X_test)
    print(f"  Raw hold-out default-risk probabilities: {len(raw_proba):,}")

    results = {
        "RAW": evaluate_probabilities(y_test_arr, raw_proba),
        "SIGMOID": None,
        "ISOTONIC": None,
    }
    results["RAW"]["_proba"] = raw_proba

    for method in CALIBRATED_METHODS:
        print(f"Fitting leakage-safe {method} calibration on X_train only "
              f"(5-fold out-of-fold)...")
        calibrator = fit_calibrated(method, X_train, y_train)
        cal_proba = calibrator.predict_proba(X_test)[:, 1]
        results[method] = evaluate_probabilities(y_test_arr, cal_proba)
        results[method]["_proba"] = cal_proba
        print(f"  {method} hold-out probabilities: {len(cal_proba):,}")

    table = comparison_table(results)
    print_comparison_table(table)
    print_bin_table(bin_calibration_table(y_test_arr, raw_proba), "RAW")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    curve_path = FIGURES_DIR / "calibration_curve.png"
    dist_path = FIGURES_DIR / "probability_distribution.png"
    save_calibration_curve(y_test_arr, results, curve_path)
    save_probability_distribution(results, dist_path)
    print(f"\n  Saved: {curve_path}")
    print(f"  Saved: {dist_path}")

    decision = print_interpretation(table, results)

    print("\n" + "=" * 72)
    print(f"RESULT: use {decision} probabilities for default-risk probability "
          "reporting.")
    print("=" * 72)
    print("Phase 13 complete. No threshold change, no retuning, no risk-score")
    print("bands, no production artifacts, no backend/frontend. The locked")
    print("classification model remains frozen at threshold 0.325.")


if __name__ == "__main__":
    main()