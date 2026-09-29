"""
Phase 11 - Final frozen holdout evaluation for CreditWise AI.

Evaluates the LOCKED model exactly once on the frozen 20% holdout set. All
model-development decisions are frozen from Phases 9/10 and are NOT changed
based on holdout performance.

Locked configuration (frozen - do not optimize or modify)
---------------------------------------------------------
    Model      : Tuned XGBoost (exact Phase 9 best parameters)
    Features   : APPLICATION-ONLY (loan_grade / loan_int_rate excluded)
    Threshold  : 0.325 (selected on out-of-fold training predictions in Phase 10)

Run with:
    python ml/final_evaluation.py

Methodology
-----------
    - Exact same deterministic cleaning and stratified split as Phases 5-10
      (test_size=0.20, random_state=42, stratify=y).
    - The locked pipeline is fitted ONLY on the complete X_train / y_train.
    - X_test / y_test is used for prediction exactly once.
    - Threshold-0.50 metrics are printed REFERENCE ONLY - they never influence
      the locked threshold, which remains 0.325.

Guards
------
    - No threshold change, no hyperparameter tuning, no SHAP.
    - No risk-score bands, no backend / frontend work.
    - No production model artifacts are saved.
"""

import sys
from pathlib import Path

# Make the project root importable so `ml.*` modules work no matter from
# which folder this script is executed (e.g. python ml/final_evaluation.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from ml.compare_models import prepare_data
from ml.evaluate import CONFUSION_LABELS, print_confusion_matrix
from ml.threshold_analysis import (
    DEFAULT_THRESHOLD,
    XGBOOST_PARAMS,
    best_f1_threshold,
    build_pipeline,
    run_oof_predictions,
    sweep_thresholds,
    threshold_metrics,
    verify_application_only_features,
)

# ---------------------------------------------------------------------------
# Locked configuration (frozen)
# ---------------------------------------------------------------------------
# Phase 10 selected 0.325 as the BALANCED (best-F1) operating threshold for the
# tuned XGBoost candidate using out-of-fold training predictions only.
LOCKED_THRESHOLD = 0.325

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "f2": "F2",
    "specificity": "Specificity",
    "fpr": "False Positive Rate",
    "fnr": "False Negative Rate",
}


# ---------------------------------------------------------------------------
# Locked pipeline
# ---------------------------------------------------------------------------
def build_final_pipeline():
    """
    Unfitted locked XGBoost pipeline.

    Reuses threshold_analysis.build_pipeline("xgboost"), which wraps
    build_tuned_pipeline("xgboost", XGBOOST_PARAMS) - the exact canonical
    Phase 9 best-parameter definition (n_estimators=200, max_depth=4,
    learning_rate=0.10, subsample=1.0, colsample_bytree=1.0) with NO class
    weighting.
    """
    return build_pipeline("xgboost")


# ---------------------------------------------------------------------------
# Holdout evaluation (pure helpers - independently unit-tested)
# ---------------------------------------------------------------------------
def build_confusion_frame(metrics):
    """Labeled TN/FP/FN/TP confusion matrix from one threshold_metrics dict."""
    return pd.DataFrame(
        [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
        index=pd.Index(CONFUSION_LABELS, name="Actual"),
        columns=pd.Index(CONFUSION_LABELS, name="Predicted"),
    )


def evaluate_holdout(y_true, y_proba, threshold=LOCKED_THRESHOLD):
    """
    All final metrics for the holdout set at one locked threshold.

    Returns a dict with:
        ranking        : ROC-AUC, PR-AUC / average precision, Brier score
        classification : accuracy + every threshold_metrics key
        counts         : total, actual defaults/non-defaults, predicted defaults
        y_pred         : hard predictions at `threshold`
    """
    y_true_arr = np.asarray(y_true)
    y_proba_arr = np.asarray(y_proba)
    y_pred = (y_proba_arr >= float(threshold)).astype(int)

    classification = threshold_metrics(y_true_arr, y_proba_arr, threshold)
    classification["accuracy"] = accuracy_score(y_true_arr, y_pred)

    counts = {
        "total": int(len(y_true_arr)),
        "actual_defaults": int((y_true_arr == 1).sum()),
        "actual_non_defaults": int((y_true_arr == 0).sum()),
        "predicted_defaults": int(y_pred.sum()),
    }

    ranking = {
        "roc_auc": roc_auc_score(y_true_arr, y_proba_arr),
        "pr_auc": average_precision_score(y_true_arr, y_proba_arr),
        "brier": brier_score_loss(y_true_arr, y_proba_arr),
    }

    return {"ranking": ranking, "classification": classification,
            "counts": counts, "y_pred": y_pred}


def run_final_evaluation(X_train, y_train, X_test, y_test,
                         threshold=LOCKED_THRESHOLD):
    """
    Fits the locked pipeline on X_train/y_train ONLY, then evaluates the
    holdout set exactly once. Returns (holdout_results, y_proba).
    """
    pipeline = build_final_pipeline()
    pipeline.fit(X_train, y_train)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return evaluate_holdout(y_test, y_proba, threshold), y_proba


def recompute_oof_reference(X_train, y_train):
    """
    Phase 10 OOF reference regenerated with the EXACT same OOF routine and
    locked pipeline used in Phase 10 (canonical rather than hard-coded).
    """
    oof_y, oof_proba, _ = run_oof_predictions(
        X_train,
        y_train,
        lambda X_ft, y_ft: build_pipeline("xgboost"),
        n_splits=5,
    )
    oof_sweep = sweep_thresholds(oof_y, oof_proba)
    oof_best_f1_threshold, _ = best_f1_threshold(oof_sweep)
    locked = threshold_metrics(oof_y, oof_proba, LOCKED_THRESHOLD)
    return {
        "roc_auc": roc_auc_score(oof_y, oof_proba),
        "pr_auc": average_precision_score(oof_y, oof_proba),
        "precision": locked["precision"],
        "recall": locked["recall"],
        "f1": locked["f1"],
        "oof_best_f1_threshold": oof_best_f1_threshold,
    }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def print_locked_configuration():
    separator = "=" * 72
    print("\n" + separator)
    print("1. LOCKED MODEL CONFIGURATION (frozen)")
    print(separator)
    print("  Model           : Tuned XGBoost (Phase 9 best parameters)")
    print(f"  n_estimators    : {XGBOOST_PARAMS['n_estimators']}")
    print(f"  max_depth       : {XGBOOST_PARAMS['max_depth']}")
    print(f"  learning_rate   : {XGBOOST_PARAMS['learning_rate']}")
    print(f"  subsample       : {XGBOOST_PARAMS['subsample']}")
    print(f"  colsample_bytree: {XGBOOST_PARAMS['colsample_bytree']}")
    print("  Class weighting : none (baseline strategy)")
    print("\n  2. LOCKED FEATURE CONFIGURATION")
    print("     Feature set   : APPLICATION-ONLY (loan_grade and loan_int_rate")
    print("                     excluded - verified above)")
    print("\n  3. LOCKED THRESHOLD")
    print(f"     Threshold     : {LOCKED_THRESHOLD:.3f} (Phase 10 OOF best-F1 point)")


def print_ranking_section(results):
    separator = "=" * 72
    print("\n" + separator)
    print("4. HOLDOUT RANKING METRICS (threshold-independent)")
    print(separator)
    r = results["ranking"]
    print(f"  ROC-AUC                       : {r['roc_auc']:.4f}")
    print(f"  PR-AUC / Average Precision    : {r['pr_auc']:.4f}")
    print(f"  Brier score                   : {r['brier']:.4f}")


def print_classification_section(results, threshold):
    separator = "=" * 72
    print("\n" + separator)
    print(f"5. HOLDOUT CLASSIFICATION METRICS (threshold {threshold:.3f})")
    print(separator)
    c = results["classification"]
    for key in METRIC_LABELS:
        print(f"  {METRIC_LABELS[key]:20s}: {c[key]:.4f}")
    print("\n  TN | FP | FN | TP : "
          f"{c['tn']} | {c['fp']} | {c['fn']} | {c['tp']}")


def print_counts_section(results):
    separator = "=" * 72
    print("\n" + separator)
    print("TEST SET COUNTS")
    print(separator)
    c = results["counts"]
    print(f"  Total test samples          : {c['total']:,}")
    print(f"  Actual defaults             : {c['actual_defaults']:,}")
    print(f"  Actual non-defaults         : {c['actual_non_defaults']:,}")
    print(f"  Predicted defaults/high-risk: {c['predicted_defaults']:,} "
          f"(threshold {LOCKED_THRESHOLD:.3f})")


def print_confusion_section(results):
    separator = "=" * 72
    print("\n" + separator)
    print("6. CONFUSION MATRIX (threshold 0.325)")
    print(separator)
    print_confusion_matrix(build_confusion_frame(results["classification"]),
                           "CONFUSION MATRIX - FINAL HOLDOUT")


def print_reference_050(results, y_proba, y_test):
    separator = "=" * 72
    print("\n" + separator)
    print("REFERENCE ONLY - THRESHOLD 0.50 (NOT USED FOR MODEL SELECTION)")
    print(separator)
    ref = evaluate_holdout(y_test, y_proba, DEFAULT_THRESHOLD)["classification"]
    for key in METRIC_LABELS:
        print(f"  {METRIC_LABELS[key]:20s}: {ref[key]:.4f}")
    print("  The official final operating threshold remains 0.325. This block")
    print("  is descriptive only and never feeds back into any decision.")


def print_generalization_table(oof_ref, results):
    separator = "=" * 72
    print("\n" + separator)
    print("7. GENERALIZATION CHECK - OOF (Phase 10) vs HOLDOUT (Phase 11)")
    print(separator)
    h = results["classification"]
    rows = [
        ("ROC-AUC", oof_ref["roc_auc"], results["ranking"]["roc_auc"]),
        ("PR-AUC", oof_ref["pr_auc"], results["ranking"]["pr_auc"]),
        ("Precision @ 0.325", oof_ref["precision"], h["precision"]),
        ("Recall @ 0.325", oof_ref["recall"], h["recall"]),
        ("F1 @ 0.325", oof_ref["f1"], h["f1"]),
    ]
    table = pd.DataFrame(
        [
            {
                "Metric": label,
                "OOF": f"{oof_value:.4f}",
                "Holdout": f"{holdout_value:.4f}",
                "Difference": f"{holdout_value - oof_value:+.4f}",
            }
            for label, oof_value, holdout_value in rows
        ]
    )
    print(table.to_string(index=False))
    print("\n  OOF Phase 10 reminders: ROC-AUC ~0.898, PR-AUC ~0.811, and the OOF")
    print(f"  best-F1 threshold was {oof_ref['oof_best_f1_threshold']:.3f} "
          f"(locked threshold {LOCKED_THRESHOLD:.3f}).")
    return rows


def print_interpretation(rows):
    separator = "=" * 72
    print("\n" + separator)
    print("8. FINAL INTERPRETATION")
    print(separator)
    max_abs = max(abs(holdout - oof) for _, oof, holdout in rows)

    if max_abs < 0.03:
        verdict = ("All HOLDOUT metrics land within ~3 points of the Phase 10 "
                   "out-of-fold estimates.\n"
                   "  Generalization to unseen applicants appears REASONABLE.")
    else:
        verdict = ("One or more HOLDOUT metrics diverge from the OOF estimates by "
                   "more than 3 points.\n"
                   "  The shift is noted for discussion; a modest gap is not "
                   "automatically overfitting,\n"
                   "  and random fold/holdout variance can explain part of it.")

    print(f"  Largest |Holdout - OOF| difference : {max_abs:.4f}")
    print("  " + verdict)
    print("\n  ACTION - NONE. The model, features and threshold are frozen.")
    print("  The holdout result does NOT feed back into model selection, and")
    print("  no further tuning runs. The locked threshold remains 0.325.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    print("Loading data and applying the same deterministic cleaning + split...")
    X_train, X_test, y_train, y_test = prepare_data()
    print(f"  X_train / y_train : {len(X_train):,} rows (fitted here only)")
    print(f"  X_test  / y_test  : {len(X_test):,} rows (frozen holdout)")

    verify_application_only_features()
    print("\n  Verified: 'loan_grade' and 'loan_int_rate' are excluded "
          "(APPLICATION-ONLY).")

    print("\nRegenerating the Phase 10 OOF reference with the identical OOF")
    print("routine and locked pipeline (canonical values, not hard-coded)...")
    oof_ref = recompute_oof_reference(X_train, y_train)

    print_locked_configuration()

    print("\nFitting the locked pipeline on X_train / y_train only...")
    results, y_proba = run_final_evaluation(X_train, y_train, X_test, y_test)
    print("  Final holdout prediction generated exactly once with "
          f"y_pred = (probability >= {LOCKED_THRESHOLD}).")

    print_ranking_section(results)
    print_counts_section(results)
    print_classification_section(results, LOCKED_THRESHOLD)
    print_confusion_section(results)
    print_reference_050(results, y_proba, y_test)
    rows = print_generalization_table(oof_ref, results)
    print_interpretation(rows)

    print("\nPhase 11 complete. No production model was saved; the locked")
    print("threshold remains 0.325 and the model is frozen for downstream phases.")


if __name__ == "__main__":
    main()