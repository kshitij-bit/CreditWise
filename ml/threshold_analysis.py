"""
Phase 10 - Threshold optimization and operating-point analysis for CreditWise AI.

Evaluates classification thresholds on OUT-OF-FOLD probabilities for the two
Phase 9 finalists (frozen parameters - no further tuning):

    1. Tuned XGBoost        - baseline strategy (no class weighting)
    2. Tuned CatBoost       - class weighted from fold-training labels only

Run with:
    python ml/threshold_analysis.py

Methodology
-----------
    - 5-fold StratifiedKFold (shuffle=True, random_state=42): the exact same
      splitter used in Phases 6-9.
    - Every training row receives exactly ONE out-of-fold probability from a
      model that never trained on that row.
    - Preprocessing stays inside the pipeline and is refitted per fold.
    - CatBoost class weights are recomputed from the current fold's TRAINING
      labels only.
    - ROC-AUC / PR-AUC are threshold-independent and reported once from the
      combined out-of-fold probabilities.

Threshold grid
--------------
    0.20 to 0.70 in steps of 0.025 (0.50 explicitly included). For every
    threshold: precision, recall, F1, F2 (beta=2), specificity, FPR, FNR and
    the TN/FP/FN/TP counts. zero_division=0 is used everywhere.

Operating points
----------------
    DEFAULT        : threshold = 0.50
    BALANCED       : threshold maximizing F1
    RISK-SENSITIVE : threshold maximizing F2 (recall-weighted trade-off)
    Ties prefer the HIGHER threshold (fewer avoidable false positives).

Guards
------
    - The frozen 20% holdout set is never evaluated.
    - No hyperparameter tuning, no SHAP, no model artifacts saved.
    - Results are printed only.
"""

import sys
from pathlib import Path

# Make the project root importable so `ml.*` modules work no matter from
# which folder this script is executed (e.g. python ml/threshold_analysis.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.compare_models import prepare_data
from ml.cross_validate_baseline import make_folds
from ml.evaluate import SCALAR_METRICS, evaluate_classifier
from ml.preprocessing import build_logistic_preprocessor
from ml.tune_models import build_tuned_pipeline

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
CV_CONFIG = "application_only"
N_SPLITS = 5
CV_SEED = 42

DEFAULT_THRESHOLD = 0.50
THRESHOLD_MIN = 0.20
THRESHOLD_MAX = 0.70
THRESHOLD_STEP = 0.025
THRESHOLDS = np.round(
    np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP / 2, THRESHOLD_STEP),
    3,
).tolist()

MODELS = ("xgboost", "catboost")
MODEL_LABELS = {
    "xgboost": "Tuned XGBoost (baseline)",
    "catboost": "Tuned CatBoost (class-weighted)",
}

# Phase 9 best parameters - frozen for Phase 10, no further tuning.
XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.10,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
}
CATBOOST_PARAMS = {
    "iterations": 200,
    "depth": 6,
    "learning_rate": 0.10,
    "l2_leaf_reg": 3,
}

# Phase 9 CV means (reported in that phase) - informational reproducibility
# reference for the OOF AUC prints. Never used for selection.
PHASE9_CV_REFERENCE = {
    "xgboost": {"roc_auc": 0.8984, "pr_auc": 0.8115},
    "catboost": {"roc_auc": 0.8958, "pr_auc": 0.8077},
}

OPERATING_POINT_NAMES = ("DEFAULT", "BALANCED", "RISK-SENSITIVE")

# Columns formatted as 4-decimal floats in the printed threshold table.
FLOAT_METRIC_COLS = ["precision", "recall", "f1", "f2", "specificity", "fpr", "fnr"]


# ---------------------------------------------------------------------------
# Model / pipeline construction (reuses Phase 9 tuned pipelines)
# ---------------------------------------------------------------------------
def build_pipeline(model, y_fold_train=None):
    """
    Build (unfitted) the application-only pipeline for ONE Phase 9 finalist.

    xgboost : baseline strategy, so it NEVER receives class weights.
    catboost: class_weights are recomputed from the CURRENT fold's TRAINING
              labels only (via build_tuned_pipeline).
    """
    if model == "xgboost":
        return build_tuned_pipeline("xgboost", XGBOOST_PARAMS)
    if model == "catboost":
        if y_fold_train is None:
            raise ValueError("CatBoost needs fold-training labels for class weights.")
        return build_tuned_pipeline(
            "catboost", CATBOOST_PARAMS, y_fold_train=y_fold_train
        )
    raise ValueError(f"Unknown model '{model}'.")


# ---------------------------------------------------------------------------
# Out-of-fold prediction engine (core of Phase 10)
# ---------------------------------------------------------------------------
def run_oof_predictions(X_train, y_train, build_pipeline_fn, n_splits=N_SPLITS):
    """
    Generate one out-of-fold probability for every training row.

    build_pipeline_fn : callable(x_fold_train, y_fold_train) returning a
        FRESH unpipeline for the fold (the preprocessor is refitted inside
        the pipeline, so validation never leaks into fitting).

    Returns (oof_y, oof_proba, fold_metrics_df):
        oof_y      : true labels aligned positionally with X_train/y_train
        oof_proba  : out-of-fold P(default); one entry per training row
        fold_df    : per-fold scalar metrics (AUC sanity roll-up)

    Structural guards enforced here:
        - fold train/validation indices never overlap (no self-prediction)
        - the validation chunks partition all rows exactly once
        - no unfilled / non-finite probability entries survive
    """
    splitter, n_folds = make_folds(y_train, n_splits=n_splits)
    if n_folds != n_splits:
        raise ValueError(f"Expected {n_splits} folds, got splitter with {n_folds}.")

    n_rows = len(X_train)
    oof_y = np.empty(n_rows, dtype=int)
    oof_proba = np.empty(n_rows, dtype=float)
    val_chunks = []
    fold_rows = []

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(X_train, y_train), start=1
    ):
        if len(set(train_idx) & set(val_idx)) > 0:
            raise AssertionError("Fold training and validation indices overlapped.")

        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        model = build_pipeline_fn(X_fold_train, y_fold_train)
        model.fit(X_fold_train, y_fold_train)
        proba = model.predict_proba(X_fold_val)[:, 1]

        oof_y[val_idx] = y_fold_val.to_numpy()
        oof_proba[val_idx] = proba
        val_chunks.append(val_idx)

        metrics = evaluate_classifier(
            y_fold_val, (proba >= DEFAULT_THRESHOLD).astype(int), proba
        )
        row = {"fold": fold}
        row.update({metric: metrics[metric] for metric in SCALAR_METRICS})
        fold_rows.append(row)

    covered = np.sort(np.concatenate(val_chunks))
    if not np.array_equal(covered, np.arange(n_rows)):
        raise AssertionError("OOF coverage is not exactly one prediction per row.")
    if not np.isfinite(oof_proba).all():
        raise AssertionError("OOF probability vector contains non-finite entries.")

    fold_df = pd.DataFrame(fold_rows).set_index("fold")
    return oof_y, oof_proba, fold_df


# ---------------------------------------------------------------------------
# Threshold evaluation (pure helpers - independently unit-tested)
# ---------------------------------------------------------------------------
def threshold_metrics(y_true, y_proba, threshold):
    """
    All threshold-dependent metrics at one operating threshold.

    y_pred = (P(default) >= threshold). Returns a dict with precision, recall,
    f1, f2 (beta=2), specificity, fpr, fnr and the counts tn/fp/fn/tp.
    zero_division=0 is used so degenerate folds do not crash.
    """
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_proba) >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

    denom_negative = tn + fp
    denom_positive = fn + tp
    specificity = tn / denom_negative if denom_negative else 0.0
    fpr = fp / denom_negative if denom_negative else 0.0
    fnr = fn / denom_positive if denom_positive else 0.0

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f2": float(f2),
        "specificity": float(specificity),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def sweep_thresholds(y_true, y_proba, thresholds=THRESHOLDS):
    """One row per threshold of threshold_metrics over the configured grid."""
    return pd.DataFrame([threshold_metrics(y_true, y_proba, t) for t in thresholds])


def row_at_threshold(sweep, threshold):
    """Return the sweep row for an exact threshold value."""
    matches = sweep.loc[np.isclose(sweep["threshold"], threshold, atol=1e-9)]
    if matches.empty:
        raise ValueError(f"Threshold {threshold} is not present in the sweep.")
    return matches.iloc[0]


def best_threshold_by_metric(sweep, metric):
    """
    Best threshold for one metric. Ties prefer the HIGHER threshold so we do
    not inflate false positives unnecessarily.
    """
    best_value = sweep[metric].max()
    candidates = sweep.loc[np.isclose(sweep[metric], best_value, atol=1e-12)]
    threshold = float(candidates["threshold"].max())
    row = candidates.loc[
        np.isclose(candidates["threshold"], threshold, atol=1e-9)
    ].iloc[0]
    return threshold, row


def best_f1_threshold(sweep):
    """(threshold, row) maximizing F1 (BALANCED operating point)."""
    return best_threshold_by_metric(sweep, "f1")


def best_f2_threshold(sweep):
    """(threshold, row) maximizing F2 (RISK-SENSITIVE operating point)."""
    return best_threshold_by_metric(sweep, "f2")


def operating_point_rows(sweep, default_threshold=DEFAULT_THRESHOLD):
    """
    The three operating points as a dict:
        DEFAULT        -> (0.50, row)
        BALANCED       -> best-F1 (threshold, row)
        RISK-SENSITIVE -> best-F2 (threshold, row)
    """
    return {
        "DEFAULT": (default_threshold, row_at_threshold(sweep, default_threshold)),
        "BALANCED": best_f1_threshold(sweep),
        "RISK-SENSITIVE": best_f2_threshold(sweep),
    }


def recall_matched_row(sweep, target_recall):
    """Sweep row whose recall is closest to the target (for the key comparison)."""
    work = sweep.assign(recall_gap=(sweep["recall"] - target_recall).abs())
    return work.loc[work["recall_gap"].idxmin()]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def verify_application_only_features():
    """
    Confirm the pipeline really excludes loan_grade and loan_int_rate.
    Returns the feature dict so callers can print a confirmation.
    """
    _, feats = build_logistic_preprocessor(CV_CONFIG)
    leaky = [
        f
        for f in ("loan_grade", "loan_int_rate")
        if f in feats["numerical"] or f in feats["categorical"] or f in feats["ordinal"]
    ]
    if leaky:
        raise AssertionError(f"Application-only config leaked features: {leaky}")
    return feats


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def format_threshold_table(sweep):
    """4-decimal formatting for the float metric columns of the threshold table."""
    display = sweep.copy()
    for col in FLOAT_METRIC_COLS:
        display[col] = display[col].map(lambda v: f"{v:.4f}")
    return display


def print_model_section(name, data):
    """One model's full report: AUCs, complete threshold table, operating points."""
    separator = "=" * 76
    print("\n" + separator)
    print(f"MODEL - {data['label']}")
    print(separator)
    ref = PHASE9_CV_REFERENCE[name]
    print(f"  OOF ROC-AUC  : {data['roc_auc']:.4f}   "
          f"(Phase 9 CV mean {ref['roc_auc']:.4f})")
    print(f"  OOF PR-AUC   : {data['pr_auc']:.4f}   "
          f"(Phase 9 CV mean {ref['pr_auc']:.4f})")

    print(f"\n  THRESHOLD TABLE (out-of-fold estimates; counts over "
          f"{len(data['oof_y']):,} OOF rows):")
    print(format_threshold_table(data["sweep"]).to_string(index=False))

    print("\n  OPERATING POINTS")
    print("  " + "-" * 72)
    for op_name in OPERATING_POINT_NAMES:
        t, row = data["op"][op_name]
        print(f"    {op_name:15s}: threshold {t:.3f} | Precision {row['precision']:.4f} "
              f"| Recall {row['recall']:.4f} | F1 {row['f1']:.4f} | F2 {row['f2']:.4f} "
              f"| Specificity {row['specificity']:.4f} | FP {int(row['fp'])} "
              f"| FN {int(row['fn'])}")


def make_operating_point_summary(model_data):
    """Compact 6-row summary: every model at all three operating points."""
    rows = []
    for name, data in model_data.items():
        for op_name in OPERATING_POINT_NAMES:
            t, row = data["op"][op_name]
            rows.append(
                {
                    "Model": data["label"],
                    "Operating Point": op_name,
                    "Threshold": float(t),
                    "Precision": row["precision"],
                    "Recall": row["recall"],
                    "F1": row["f1"],
                    "F2": row["f2"],
                    "Specificity": row["specificity"],
                    "FP": int(row["fp"]),
                    "FN": int(row["fn"]),
                }
            )
    return pd.DataFrame(rows)


def format_operating_point_summary(summary):
    display = summary.copy()
    for col in ["Threshold", "Precision", "Recall", "F1", "F2", "Specificity"]:
        display[col] = display[col].map(lambda v: f"{v:.4f}")
    return display


def make_auc_summary(model_data):
    """OOF ROC-AUC / PR-AUC per model with the Phase 9 CV reference."""
    rows = []
    for name, data in model_data.items():
        ref = PHASE9_CV_REFERENCE[name]
        rows.append(
            {
                "Model": data["label"],
                "OOF ROC-AUC": data["roc_auc"],
                "OOF PR-AUC": data["pr_auc"],
                "Phase 9 CV ROC-AUC": ref["roc_auc"],
                "Phase 9 CV PR-AUC": ref["pr_auc"],
            }
        )
    return pd.DataFrame(rows)


def print_key_comparison(model_data):
    """Answer: can threshold-adjusted XGBoost match weighted CatBoost recall
    at comparable (or better) precision?"""
    separator = "=" * 76
    print("\n" + separator)
    print("KEY COMPARISON - THRESHOLD-ADJUSTED XGBOOST vs WEIGHTED CATBOOST")
    print(separator)

    xgb = model_data["xgboost"]
    cat = model_data["catboost"]
    cat_default = cat["op"]["DEFAULT"][1]
    target_recall = float(cat_default["recall"])
    xgb_matched = recall_matched_row(xgb["sweep"], target_recall)
    xgb_default = xgb["op"]["DEFAULT"][1]

    def row_dict(label, row):
        return {
            "Config": label,
            "Threshold": float(row["threshold"]),
            "Precision": row["precision"],
            "Recall": row["recall"],
            "F1": row["f1"],
            "F2": row["f2"],
            "FP": int(row["fp"]),
            "FN": int(row["fn"]),
        }

    table = pd.DataFrame(
        [
            row_dict("XGBoost @ default (0.50)", xgb_default),
            row_dict(
                f"XGBoost @ recall-matched ({xgb_matched['threshold']:.3f})",
                xgb_matched,
            ),
            row_dict("CatBoost (class-weighted) @ default (0.50)", cat_default),
        ]
    )
    display = table.copy()
    for col in ["Threshold", "Precision", "Recall", "F1", "F2"]:
        display[col] = display[col].map(lambda v: f"{v:.4f}")
    print("\n  Matched-recall comparison:")
    print("  " + "-" * 72)
    print(display.to_string(index=False))

    print("\n  CatBoost recall @ 0.50            : "
          f"{target_recall:.4f}")
    print(f"  XGBoost threshold reaching it     : "
          f"{xgb_matched['threshold']:.3f} (recall {xgb_matched['recall']:.4f})")
    print(f"  Precision at that point           : "
          f"XGBoost {xgb_matched['precision']:.4f} vs CatBoost {cat_default['precision']:.4f}")
    print(f"  F1 at that point                  : "
          f"XGBoost {xgb_matched['f1']:.4f} vs CatBoost {cat_default['f1']:.4f}")
    print(f"  OOF PR-AUC                         : "
          f"XGBoost {xgb['pr_auc']:.4f} vs CatBoost {cat['pr_auc']:.4f}")
    print(f"  OOF ROC-AUC                        : "
          f"XGBoost {xgb['roc_auc']:.4f} vs CatBoost {cat['roc_auc']:.4f}")

    precision_edge = xgb_matched["precision"] - cat_default["precision"]
    pr_edge = xgb["pr_auc"] - cat["pr_auc"]
    print("\n  Verdict (out-of-fold estimates):")
    if precision_edge >= 0 and pr_edge >= 0:
        print(f"    YES - by lowering its threshold to {xgb_matched['threshold']:.3f}, "
              "XGBoost reaches")
        print("    class-weighted CatBoost recall while keeping equal or higher")
        print("    precision and OOF PR-AUC.")
    elif precision_edge >= 0:
        print(f"    PARTLY - XGBoost matches recall with better precision "
              f"({precision_edge:+.4f}),")
        print(f"    but its overall ranking (PR-AUC {pr_edge:+.4f}) trails CatBoost.")
    else:
        print(f"    NO - XGBoost reaches the recall only by giving up precision "
              f"({precision_edge:+.4f})")
        print("    and still does not beat CatBoost on ranking quality (PR-AUC).")
    print("    No monetary costs are assumed and no single threshold is claimed")
    print("    universally correct.")


def choose_leader_model(model_data):
    """Primary PR-AUC, then ROC-AUC, then F1 at the BALANCED point."""

    def score(name):
        data = model_data[name]
        return (data["pr_auc"], data["roc_auc"], data["op"]["BALANCED"][1]["f1"])

    return max(model_data, key=score)


def print_recommendation(model_data, leader_name):
    separator = "=" * 76
    print("\n" + separator)
    print("RECOMMENDATION")
    print(separator)
    leader = model_data[leader_name]
    backup_name = "catboost" if leader_name == "xgboost" else "xgboost"
    backup = model_data[backup_name]

    print(f"  Leading model : {leader['label']}")
    print(f"    OOF PR-AUC  : {leader['pr_auc']:.4f} | "
          f"OOF ROC-AUC : {leader['roc_auc']:.4f}")
    for op_name in OPERATING_POINT_NAMES:
        t, row = leader["op"][op_name]
        print(f"    {op_name:15s}: threshold {t:.3f} | P {row['precision']:.4f} "
              f"| R {row['recall']:.4f} | F1 {row['f1']:.4f} | F2 {row['f2']:.4f} "
              f"| SP {row['specificity']:.4f} | FP {int(row['fp'])} | FN {int(row['fn'])}")

    print(f"\n  Backup model  : {backup['label']}")
    print(f"    OOF PR-AUC  : {backup['pr_auc']:.4f} | "
          f"OOF ROC-AUC : {backup['roc_auc']:.4f}")
    for op_name in OPERATING_POINT_NAMES:
        t, row = backup["op"][op_name]
        print(f"    {op_name:15s}: threshold {t:.3f} | P {row['precision']:.4f} "
              f"| R {row['recall']:.4f} | F1 {row['f1']:.4f} | F2 {row['f2']:.4f} "
              f"| SP {row['specificity']:.4f} | FP {int(row['fp'])} | FN {int(row['fn'])}")

    bal_t, bal_row = leader["op"]["BALANCED"]
    risk_t, risk_row = leader["op"]["RISK-SENSITIVE"]
    print(f"\n  Case for {leader['label']}:")
    print(f"    - Ranking quality  : PR-AUC {leader['pr_auc']:.4f} vs "
          f"{backup['pr_auc']:.4f}; ROC-AUC {leader['roc_auc']:.4f} vs "
          f"{backup['roc_auc']:.4f}.")
    print(f"    - Balanced point   : threshold {bal_t:.3f} gives F1 "
          f"{bal_row['f1']:.4f} with recall {bal_row['recall']:.4f} and "
          f"precision {bal_row['precision']:.4f}.")
    print(f"    - False positives  : {bal_row['fp']} vs "
          f"{backup['op']['BALANCED'][1]['fp']} (OOF).")
    print(f"    - False negatives  : {bal_row['fn']} vs "
          f"{backup['op']['BALANCED'][1]['fn']} (OOF).")
    print(f"  Preferred operating threshold : {bal_t:.3f} (BALANCED, max F1).")
    print(f"  Alternative (risk-sensitive) : {risk_t:.3f} (max F2).")


def print_final_report(model_data, leader_name):
    separator = "=" * 76
    print("\n" + separator)
    print("FINAL REPORT")
    print(separator)

    def threshold_for(name, key):
        return model_data[name]["op"][key][0]

    leader = model_data[leader_name]
    print(f"  1. XGBoost default threshold      : {threshold_for('xgboost', 'DEFAULT'):.3f}")
    print(f"  2. XGBoost best-F1 threshold      : {threshold_for('xgboost', 'BALANCED'):.3f}")
    print(f"  3. XGBoost best-F2 threshold      : {threshold_for('xgboost', 'RISK-SENSITIVE'):.3f}")
    print(f"  4. CatBoost default threshold     : {threshold_for('catboost', 'DEFAULT'):.3f}")
    print(f"  5. CatBoost best-F1 threshold     : {threshold_for('catboost', 'BALANCED'):.3f}")
    print(f"  6. CatBoost best-F2 threshold     : {threshold_for('catboost', 'RISK-SENSITIVE'):.3f}")
    print(f"  7. Recommended model + threshold  : {leader['label']} at "
          f"{leader['op']['BALANCED'][0]:.3f} (BALANCED / max F1); "
          f"risk-sensitive alternative at {leader['op']['RISK-SENSITIVE'][0]:.3f}")
    print(separator)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    print("Loading raw dataset and preparing the shared APPLICATION-ONLY split...")
    X_train, X_test, y_train, y_test = prepare_data()
    print(f"  X_train / y_train : {len(X_train):,} rows (used by out-of-fold CV)")
    print(f"  X_test  / y_test  : {len(X_test):,} rows (frozen holdout - never evaluated)")

    verify_application_only_features()
    print("\n  Application-only features verified: 'loan_grade' and 'loan_int_rate'"
          " are excluded.")

    model_data = {}
    for name in MODELS:

        def builder(X_fold_train, y_fold_train, m=name):
            return build_pipeline(m, y_fold_train=y_fold_train)

        oof_y, oof_proba, fold_df = run_oof_predictions(
            X_train, y_train, builder, n_splits=N_SPLITS
        )
        model_data[name] = {
            "label": MODEL_LABELS[name],
            "oof_y": oof_y,
            "oof_proba": oof_proba,
            "fold_df": fold_df,
            "roc_auc": roc_auc_score(oof_y, oof_proba),
            "pr_auc": average_precision_score(oof_y, oof_proba),
            "sweep": sweep_thresholds(oof_y, oof_proba),
            "op": None,
        }
        model_data[name]["op"] = operating_point_rows(model_data[name]["sweep"])
        print_model_section(name, model_data[name])

    print("\n" + "=" * 76)
    print("COMPACT OPERATING-POINT SUMMARY (out-of-fold estimates)")
    print("=" * 76)
    summary = make_operating_point_summary(model_data)
    print(format_operating_point_summary(summary).to_string(index=False))

    print("\n" + "=" * 76)
    print("OUT-OF-FOLD AUC SUMMARY")
    print("=" * 76)
    auc_summary = make_auc_summary(model_data)
    for col in ["OOF ROC-AUC", "OOF PR-AUC", "Phase 9 CV ROC-AUC", "Phase 9 CV PR-AUC"]:
        auc_summary[col] = auc_summary[col].map(lambda v: f"{v:.4f}")
    print(auc_summary.to_string(index=False))

    print_key_comparison(model_data)

    leader_name = choose_leader_model(model_data)
    print_recommendation(model_data, leader_name)
    print_final_report(model_data, leader_name)

    print("\nPhase 10 complete. The frozen holdout set was NOT evaluated and no")
    print("production model was saved - all numbers come from out-of-fold folds.")


if __name__ == "__main__":
    main()