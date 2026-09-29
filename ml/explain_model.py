"""
Phase 12 - SHAP explainability for the locked final XGBoost credit-risk model.

Explanation-only phase. The predictive configuration is FROZEN from Phase 11:

    Model      : Tuned XGBoost (exact Phase 9 best parameters)
    Features   : APPLICATION-ONLY (loan_grade / loan_int_rate excluded)
    Threshold  : 0.325

Guards
------
    - No hyperparameter tuning, no threshold change, no calibration.
    - The locked pipeline is fitted ONLY on the complete X_train / y_train.
    - SHAP is computed on HOLD-OUT observations (X_test) only.
    - SHAP is used to explain the model's prediction behavior. It does NOT
      prove that a feature causes default - no causal claims are made.
    - No risk-score bands and no production model artifacts are saved.

Run with:
    python ml/explain_model.py

Outputs
-------
    reports/figures/shap_global_bar.png        (mean |SHAP| top-15 bar)
    reports/figures/shap_summary.png           (beeswarm / summary plot)
    reports/figures/shap_high_risk_example.png (waterfall, true positive)
    reports/figures/shap_low_risk_example.png  (waterfall, true negative)
    reports/figures/shap_false_positive_example.png   (if available)
    reports/figures/shap_false_negative_example.png   (if available)
"""

import json
import sys
from pathlib import Path

# Make the project root importable so `ml.*` modules work no matter from
# which folder this script is executed (e.g. python ml/explain_model.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")  # headless figure generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from ml.compare_models import prepare_data
from ml.final_evaluation import LOCKED_THRESHOLD, build_final_pipeline
from ml.predict import SHAP_SUMMARY_PATH
from ml.risk_engine import MODEL_VERSION
from ml.threshold_analysis import (
    XGBOOST_PARAMS,
    verify_application_only_features,
)

# ---------------------------------------------------------------------------
# Configuration (explanation-only)
# ---------------------------------------------------------------------------
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Deterministic representative SHAP sample (random_state=42) is used ONLY if
# the holdout set exceeds this row count. The holdout has 6,482 rows, so the
# full holdout is used by default.
MAX_SHAP_ROWS = 10_000
SHAP_SAMPLE_SEED = 42
TOP_N = 15

DISCLAIMER = (
    "SHAP explains the model's prediction behavior; it does not establish "
    "causality. Contributions describe how each feature pushes a prediction "
    "toward or away from default risk, not why a person defaults."
)


# ---------------------------------------------------------------------------
# Transformed feature names (human-readable, verified)
# ---------------------------------------------------------------------------
def clean_feature_name(raw):
    """
    Turn a ColumnTransformer output name into a human-readable label.

    Examples:
        numerical__person_age                       -> person_age
        numerical__missingindicator_person_emp_length -> person_emp_length_missing
        categorical__loan_intent_MEDICAL            -> loan_intent_MEDICAL
        ordinal_loan_grade                          -> loan_grade
    """
    name = str(raw)
    for prefix in ("numerical__", "categorical__", "ordinal_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    if name.startswith("missingindicator_"):
        name = name[len("missingindicator_"):] + "_missing"
    return name


def transformed_feature_names(preprocessor, X_data):
    """
    Recover EXACT transformed feature names from a fitted preprocessor.

    `X_data` must be a pandas DataFrame whose column order matches the
    preprocessor training order. Raises a ValueError if the name count does
    not match the number of columns the preprocessor produces (i.e. what
    XGBoost actually sees).
    """
    if not hasattr(X_data, "columns"):
        raise TypeError("transformed_feature_names expects a DataFrame.")
    feature_columns = list(X_data.columns)
    raw_names = preprocessor.get_feature_names_out(feature_columns)
    names = [clean_feature_name(n) for n in raw_names]
    hint = preprocessor.transform(X_data.head(100))
    if hasattr(hint, "toarray"):
        n_columns = hint.toarray().shape[1]
    else:
        n_columns = np.asarray(hint).shape[1]
    if len(names) != n_columns:
        raise ValueError(
            f"Feature-name mismatch: {len(names)} names vs {n_columns} columns. "
            "SHAP explanations must use human-readable names aligned to XGBoost."
        )
    return names


# ---------------------------------------------------------------------------
# SHAP computation on holdout observations
# ---------------------------------------------------------------------------
def _as_dense(matrix):
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray(), dtype=float)
    return np.asarray(matrix, dtype=float)


def _class1_shap_values(explainer, Xt):
    """Extract class-1 (default) SHAP values for a binary XGBClassifier."""
    values = explainer.shap_values(Xt)
    if isinstance(values, list):
        return np.asarray(values[1], dtype=float), float(np.asarray(explainer.expected_value)[1])
    return np.asarray(values, dtype=float), float(np.asarray(explainer.expected_value))


def explain_holdout(pipeline, X_test, y_test, threshold=LOCKED_THRESHOLD,
                    max_rows=MAX_SHAP_ROWS, seed=SHAP_SAMPLE_SEED):
    """
    Compute SHAP values for hold-out observations WITHOUT refitting anything.

    A deterministic representative sample (random_state=42) is used only when
    max_rows is exceeded. Returns a context dict with aligned arrays, IDS,
    predictions and the fitted explainer. Predictions are made BEFORE any
    SHAP computation so explanation can never change model outputs.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Keep the original DataFrame index: it is the applicant's row identifier.
    n = len(X_test)
    positions = np.arange(n)
    if n > max_rows:
        rng = np.random.RandomState(seed)
        positions = np.sort(rng.choice(n, size=max_rows, replace=False))

    X_sel = X_test.iloc[positions] if isinstance(X_test, pd.DataFrame) else X_test[positions]
    row_ids = np.asarray(X_sel.index if isinstance(X_sel, pd.DataFrame) else positions)

    Xt = _as_dense(preprocessor.transform(X_sel))
    y_proba = classifier.predict_proba(Xt)[:, 1]
    y_pred = (y_proba >= float(threshold)).astype(int)
    y_true = np.asarray(y_test)[positions]

    feature_names = transformed_feature_names(preprocessor, X_sel)
    if len(feature_names) != Xt.shape[1]:
        raise ValueError("Transformed feature names must match XGBoost input columns.")

    explainer = shap.TreeExplainer(classifier)
    sv, base = _class1_shap_values(explainer, Xt)

    return {
        "pipeline": pipeline,
        "explainer": explainer,
        "Xt": Xt,
        "feature_names": feature_names,
        "shap_values": sv,
        "expected_value": base,
        "y_proba": y_proba,
        "y_pred": y_pred,
        "y_true": y_true,
        "row_ids": row_ids,
        "positions": positions,
        "threshold": float(threshold),
        "n_features": int(Xt.shape[1]),
        "n_rows": int(Xt.shape[0]),
    }


# ---------------------------------------------------------------------------
# Global SHAP (mean |SHAP| importance + directions)
# ---------------------------------------------------------------------------
def importance_table(shap_values, feature_names, top_n=None):
    """Mean absolute SHAP value per feature, sorted descending."""
    sv = np.asarray(shap_values)
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    rows = [
        {
            "rank": i + 1,
            "feature": feature_names[order[i]],
            "mean_abs_shap": float(mean_abs[order[i]]),
        }
        for i in range(len(order))
    ]
    df = pd.DataFrame(rows)
    return df.head(top_n) if top_n else df


def feature_directions(Xt, shap_values, feature_names):
    """
    Direction of each feature's contribution to the model's predicted risk.

    "-1" = higher feature value reduces predicted default risk
    "+1" = higher feature value increases predicted default risk
    " 0" = constant / no stable direction on this sample

    This describes the model's behavior - it is NOT a causal statement.
    """
    Xt = np.asarray(Xt)
    sv = np.asarray(shap_values)
    directions = {}
    for i, name in enumerate(feature_names):
        col = Xt[:, i]
        svcol = sv[:, i]
        if np.std(col) < 1e-12 or np.std(svcol) < 1e-12:
            directions[name] = 0
            continue
        unique = np.unique(col)
        if unique.size == 2 and np.array_equal(unique, [0.0, 1.0]):
            present = col > 0.5
            if present.sum() == 0 or (~present).sum() == 0:
                directions[name] = 0
            else:
                directions[name] = int(
                    np.sign(svcol[present].mean() - svcol[~present].mean())
                )
        else:
            corr = np.corrcoef(col, svcol)[0, 1]
            directions[name] = 0 if not np.isfinite(corr) else int(np.sign(corr))
    return directions


def shap_reconstruction_error(ctx):
    """Outputs that are unchanged by explanation: base + sum(SHAP) = log-odds."""
    log_odds = ctx["expected_value"] + np.asarray(ctx["shap_values"]).sum(axis=1)
    proba_from_shap = 1.0 / (1.0 + np.exp(-log_odds))
    return np.max(np.abs(proba_from_shap - np.asarray(ctx["y_proba"])))


# ---------------------------------------------------------------------------
# Local explanations (example applicants)
# ---------------------------------------------------------------------------
def select_examples(y_true, y_proba, y_pred):
    """
    Pick sample rows per confusion cell (median-probability member).

    Returns {"true_positive": idx_or_None, ...} where idx is a row index into
    the SHAP context arrays. None -> cell not present in the sample.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = np.asarray(y_pred)

    def pick(mask):
        idx = np.where(mask)[0]
        if idx.size == 0:
            return None
        median = np.median(y_proba[idx])
        return int(idx[np.argmin(np.abs(y_proba[idx] - median))])

    return {
        "true_positive": pick((y_true == 1) & (y_pred == 1)),
        "true_negative": pick((y_true == 0) & (y_pred == 0)),
        "false_positive": pick((y_true == 0) & (y_pred == 1)),
        "false_negative": pick((y_true == 1) & (y_pred == 0)),
    }


def example_summary(context_row, ctx):
    """
    Full local explanation for one applicant: labels, probability, threshold,
    and the top five features increasing / decreasing the model's risk score.

    `context_row` is a row index into the SHAP context arrays (as returned by
    select_examples).
    """
    r = int(context_row)
    names = ctx["feature_names"]
    sv_row = np.asarray(ctx["shap_values"][r])

    pairs = sorted(zip(names, sv_row.tolist()), key=lambda kv: kv[1])
    increasing = [(n, float(v)) for n, v in reversed(pairs) if v > 0][:5]
    if not increasing:
        increasing = [(n, float(v)) for n, v in reversed(pairs)][:5]
    decreasing = [(n, float(v)) for n, v in pairs if v < 0][:5]
    if not decreasing:
        decreasing = [(n, float(v)) for n, v in pairs][:5]

    return {
        "row_id": ctx["row_ids"][r],
        "probability": float(ctx["y_proba"][r]),
        "threshold": float(ctx["threshold"]),
        "predicted_class": int(ctx["y_pred"][r]),
        "actual_class": int(ctx["y_true"][r]),
        "top_increasing": increasing,
        "top_decreasing": decreasing,
        "context_row": r,
    }


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------
def save_shap_bar(ctx, path, top_n=TOP_N):
    """Global mean |SHAP| bar plot for the top features."""
    shap.summary_plot(
        np.asarray(ctx["shap_values"]),
        ctx["Xt"],
        feature_names=ctx["feature_names"],
        plot_type="bar",
        max_display=top_n,
        show=False,
    )
    plt.gcf().savefig(str(path), bbox_inches="tight", dpi=150)
    plt.close()


def save_shap_summary(ctx, path, top_n=TOP_N):
    """Beeswarm / summary plot (value color: red = high, blue = low)."""
    shap.summary_plot(
        np.asarray(ctx["shap_values"]),
        ctx["Xt"],
        feature_names=ctx["feature_names"],
        max_display=top_n,
        show=False,
    )
    plt.gcf().savefig(str(path), bbox_inches="tight", dpi=150)
    plt.close()


def save_waterfall(ctx, context_row, path, max_display=15):
    """Per-applicant waterfall explanation for one hold-out row."""
    r = int(context_row)
    explanation = shap.Explanation(
        values=np.asarray(ctx["shap_values"][r]),
        base_values=ctx["expected_value"],
        data=np.asarray(ctx["Xt"][r]),
        feature_names=ctx["feature_names"],
    )
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    plt.gcf().savefig(str(path), bbox_inches="tight", dpi=150)
    plt.close()


def save_shap_summary_json(table, directions):
    """Persist precomputed global SHAP features as a lightweight JSON artifact."""
    features = [
        {
            "feature": row["feature"],
            "importance": float(row["mean_abs_shap"]),
            "direction": int(directions.get(row["feature"], 0)),
        }
        for _, row in table.iterrows()
    ]
    payload = {
        "model_version": MODEL_VERSION,
        "explanation_type": "global_shap",
        "top_features": features,
        "plots": ["shap_global_bar.png", "shap_summary.png"],
        "disclaimer": DISCLAIMER,
    }
    SHAP_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SHAP_SUMMARY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------
def print_global_section(ctx):
    separator = "=" * 72
    print("\n" + separator)
    print("GLOBAL SHAP - FEATURE IMPORTANCE (mean |SHAP|)")
    print(f"{separator}\n  Sample          : {ctx['n_rows']:,} hold-out observations")
    print(f"  Features        : {ctx['n_features']}")
    print(f"  Threshold used  : {ctx['threshold']:.3f} (locked)")

    directions = feature_directions(ctx["Xt"], ctx["shap_values"], ctx["feature_names"])
    table = importance_table(ctx["shap_values"], ctx["feature_names"], top_n=TOP_N)
    table["direction"] = table["feature"].map(
        lambda f: {+1: "higher value => higher risk",
                   -1: "higher value => lower risk",
                   0: "mixed / constant"}[directions[f]]
    )
    print("\n  Top global drivers (feature contribution to predicted risk):")
    print(table.to_string(index=False))

    try:
        err = shap_reconstruction_error(ctx)
        print(f"\n  Consistency check: max |sigmoid(base + sum(SHAP)) - p| = {err:.5f}")
    except Exception:
        print("\n  Consistency check: skipped")


def print_local_section(labels, ctx):
    separator = "=" * 72
    titles = {
        "true_positive": "CORRECTLY PREDICTED HIGH-RISK / DEFAULT APPLICANT (TP)",
        "true_negative": "CORRECTLY PREDICTED LOW-RISK / NON-DEFAULT APPLICANT (TN)",
        "false_positive": "FALSE POSITIVE APPLICANT (FP) - labeled default, actually repaid",
        "false_negative": "FALSE NEGATIVE APPLICANT (FN) - labeled repaid, actually defaulted",
    }
    for key, title in titles.items():
        position = labels[key]
        if position is None:
            print(f"\n  {key.replace('_', ' ')}: no example in the sample")
            continue
        info = example_summary(position, ctx)
        print("\n" + separator)
        print(f"LOCAL EXPLANATION - {title}")
        print(separator)
        print(f"  Original row id        : {info['row_id']}")
        print(f"  Predicted default risk : {info['probability']:.4f}")
        print(f"  Locked threshold       : {info['threshold']:.3f}")
        print(f"  Predicted class        : {info['predicted_class']} "
              f"(1 = default / high risk at threshold {info['threshold']:.3f})")
        print(f"  Actual class           : {info['actual_class']}")
        print("\n  Top features increasing the model's risk score (+SHAP):")
        for name, value in info["top_increasing"]:
            print(f"      + {name:34s} {value:+.4f}")
        print("\n  Top features decreasing the model's risk score (-SHAP):")
        for name, value in info["top_decreasing"]:
            print(f"      - {name:34s} {value:+.4f}")


def print_interpretation(table, directions):
    separator = "=" * 72
    print("\n" + separator)
    print("INTERPRETATION (feature contribution to the model's prediction)")
    print(separator)
    print("  Positive SHAP value  -> feature pushes the model toward HIGHER default risk")
    print("  Negative SHAP value  -> feature pushes the model toward LOWER default risk")
    print("  These describe the model's prediction behavior, NOT causation.")
    print("\n  TOP GLOBAL RISK DRIVERS (highest mean |SHAP|):")
    for _, row in table.iterrows():
        feature = row["feature"]
        direction = {+1: "higher value => higher risk",
                     -1: "higher value => lower risk",
                     0: "mixed / constant"}[directions[feature]]
        print(f"      #{int(row['rank']):2d} {feature:34s} [{direction}]")
    print("\n  LIMITATIONS OF SHAP:")
    print("      - SHAP explains the model's prediction behavior. It does NOT")
    print("        prove that a feature causes default (correlation != causation).")
    print("      - Values are local to each prediction; importances can change")
    print("        across applicants and datasets.")
    print("      - Missing-indicator and one-hot columns are artifacts of")
    print("        preprocessing; interpret them alongside their base features.")
    print("      - SHAP assumptions (additivity, local linearity) are")
    print("        approximations of the model's true behavior.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    print("Phase 12 - SHAP explainability for the locked final XGBoost model.")
    print("Loading data and applying the same deterministic cleaning + split...")
    X_train, X_test, y_train, y_test = prepare_data()
    print(f"  X_train / y_train : {len(X_train):,} rows (fitted here only)")
    print(f"  X_test  / y_test  : {len(X_test):,} rows (frozen hold-out)")

    verify_application_only_features()
    print("\n  Verified: 'loan_grade' and 'loan_int_rate' are excluded "
          "(APPLICATION-ONLY).")

    print("\nFitting the locked Phase 11 pipeline on X_train / y_train ONLY...")
    pipeline = build_final_pipeline()
    pipeline.fit(X_train, y_train)

    print(f"\nComputing SHAP on hold-out observations (TreeExplainer, "
          f"class-1 default risk)...")
    ctx = explain_holdout(pipeline, X_test, y_test,
                          threshold=LOCKED_THRESHOLD,
                          max_rows=MAX_SHAP_ROWS, seed=SHAP_SAMPLE_SEED)
    if ctx["n_rows"] < len(X_test):
        print(f"  NOTE: deterministic chance-set sample (random_state=42) used: "
              f"{ctx['n_rows']} of {len(X_test):,} rows.")
    print(f"  Transformed features : {ctx['n_features']} (names verified vs columns)")
    print(f"  SHAP matrix shape    : {ctx['shap_values'].shape}")

    table = importance_table(ctx["shap_values"], ctx["feature_names"], top_n=TOP_N)
    directions = feature_directions(ctx["Xt"], ctx["shap_values"], ctx["feature_names"])
    print_global_section(ctx)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    bar_path = FIGURES_DIR / "shap_global_bar.png"
    summary_path = FIGURES_DIR / "shap_summary.png"
    save_shap_bar(ctx, bar_path, top_n=TOP_N)
    save_shap_summary(ctx, summary_path, top_n=TOP_N)
    print(f"\n  Saved: {bar_path}")
    print(f"  Saved: {summary_path}")

    save_shap_summary_json(table, directions)
    print(f"  Saved: {SHAP_SUMMARY_PATH}")

    labels = select_examples(ctx["y_true"], ctx["y_proba"], ctx["y_pred"])
    print_local_section(labels, ctx)

    waterfall_targets = {
        ("true_positive", "shap_high_risk_example.png"): True,
        ("true_negative", "shap_low_risk_example.png"): True,
        ("false_positive", "shap_false_positive_example.png"): True,
        ("false_negative", "shap_false_negative_example.png"): True,
    }
    for (key, filename), _ in waterfall_targets.items():
        position = labels[key]
        if position is None:
            continue
        path = FIGURES_DIR / filename
        save_waterfall(ctx, position, path)
        print(f"  Saved: {path}")

    print_interpretation(table, directions)

    print("\nPhase 12 complete. No tuning, no threshold change, no calibration,")
    print("no risk bands and no production artifacts were created. The model")
    print("remains frozen at threshold 0.325 for downstream phases.")


if __name__ == "__main__":
    main()