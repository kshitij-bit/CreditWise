"""
Phase 14 - CreditWise risk score, risk bands and decision engine.

Convert the locked model's RAW default-risk probability estimate into an
understandable financial decision-support output. All predictive-model work
is complete and FROZEN:

    Model      : Tuned XGBoost (application-only features)
    Threshold  : 0.325   (locked classification threshold)

This module is a PURE, framework-free decision layer. It never fits models,
never touches preprocessing, and never changes the locked configuration.
It maps one probability value to:

    - CreditWise Risk Score         (0-100, higher = greater predicted risk)
    - Risk Category                 (LOW / MODERATE / HIGH / VERY_HIGH)
    - Decision recommendation       (APPROVE / REVIEW / REJECT)
    - Human-readable explanation fields

Important
---------
    - The risk score is an INTERNAL CreditWise risk score. It is NOT a CIBIL
      score and NOT an external credit-bureau score.
    - The risk categories are OPERATIONAL/BUSINESS categories anchored to the
      locked 0.325 boundary. They are not statistically guaranteed borrower
      grades and were not derived by optimizing the holdout set.
    - This is decision SUPPORT for humans. It does not replace human
      underwriting and carries no legal or regulatory lending authority.

Run with:
    python ml/risk_engine.py     (prints a boundary + demo table)
"""

import math

# ---------------------------------------------------------------------------
# Locked configuration constants (single source of truth for the engine)
# ---------------------------------------------------------------------------
CLASSIFICATION_THRESHOLD = 0.325
MODEL_VERSION = "creditwise-xgb-v1"

# Risk band boundaries (LOW_RISK_MAX <= p < MODERATE_RISK_MAX <= p < ...).
LOW_RISK_MAX = 0.15
MODERATE_RISK_MAX = 0.325
HIGH_RISK_MAX = 0.60

RISK_CATEGORY_NAMES = ("LOW", "MODERATE", "HIGH", "VERY_HIGH")

# CreditWise risk score disclaimer - the score is internal, not a bureau score.
RISK_SCORE_DISCLAIMER = (
    "This is an internal CreditWise risk score (0-100, higher = greater "
    "predicted default risk). It is NOT a CIBIL score and NOT an external "
    "credit-bureau score."
)

MODEL_BASED_NOTE = (
    "Model-based decision support only: this recommendation is produced by "
    "the locked CreditWise XGBoost model. It does not replace human "
    "underwriting and carries no legal or regulatory lending authority."
)

# One clear message per risk category.
MESSAGES = {
    "LOW": "Low predicted credit risk.",
    "MODERATE": "Moderate predicted credit risk. Manual review recommended.",
    "HIGH": "High predicted credit risk. Detailed review recommended.",
    "VERY_HIGH": "Very high predicted credit risk. Application rejection recommended.",
}

DECISIONS = {
    "LOW": "APPROVE",
    "MODERATE": "REVIEW",
    "HIGH": "REVIEW",
    "VERY_HIGH": "REJECT",
}


# ---------------------------------------------------------------------------
# Pure decision functions
# ---------------------------------------------------------------------------
def validate_probability(probability):
    """
    Validate that a probability is finite and within [0, 1].

    Raises ValueError otherwise. Returns the float unchanged so the raw
    probability is never edited.
    """
    if probability is None:
        raise ValueError("probability must not be None.")
    p = float(probability)
    if not math.isfinite(p):
        raise ValueError(f"probability must be finite, got {probability!r}.")
    if p < 0.0 or p > 1.0:
        raise ValueError(f"probability must be within [0, 1], got {p!r}.")
    return p


def probability_to_risk_score(probability):
    """
    CreditWise risk score: round(probability * 100), range 0-100.

    0 = lowest predicted default risk, 100 = highest predicted default risk.

    Uses round-half-up (floor(x + 0.5)) so ties behave intuitively:
    probability 0.325 -> 33 (plain Python round() would return 32 via
    bankers' rounding).
    """
    p = validate_probability(probability)
    score = int(math.floor(p * 100.0 + 0.5))
    return min(100, max(0, score))


def get_risk_category(probability):
    """
    Risk band for a default-risk probability estimate.

        p < 0.15                    -> LOW
        0.15 <= p < 0.325           -> MODERATE
        0.325 <= p < 0.60           -> HIGH
        p >= 0.60                   -> VERY_HIGH

    The locked classification threshold 0.325 is the boundary between
    non-default-oriented and default-oriented bands (0.325 -> HIGH).
    """
    p = validate_probability(probability)
    if p < LOW_RISK_MAX:
        return "LOW"
    if p < MODERATE_RISK_MAX:
        return "MODERATE"
    if p < HIGH_RISK_MAX:
        return "HIGH"
    return "VERY_HIGH"


def get_decision(probability):
    """
    Decision recommendation for a risk category.

        LOW       -> APPROVE
        MODERATE  -> REVIEW
        HIGH      -> REVIEW
        VERY_HIGH -> REJECT

    Decision-support recommendation only - not automated/binding lending.
    """
    p = validate_probability(probability)
    return DECISIONS[get_risk_category(p)]


def build_risk_assessment(probability, model_version=None):
    """
    Build the full reusable risk-assessment output structure.

    Returns a dictionary with:
        default_risk_probability   (round-tripped raw probability, 4dp)
        risk_score                 (CreditWise 0-100 internal score)
        risk_category              (LOW / MODERATE / HIGH / VERY_HIGH)
        decision                   (APPROVE / REVIEW / REJECT)
        classification_threshold   (locked 0.325)
        model_version
        message                    (human-readable explanation)
        note                       (model-based support, not CIBIL)
        top_risk_increasing_factors / top_risk_reducing_factors
                                   (empty slots - the API can attach SHAP
                                    factors here later)

    The raw probability is returned unchanged (precision rounded to 4dp);
    this module never modifies model predictions.
    """
    p = validate_probability(probability)
    version = model_version if model_version is not None else MODEL_VERSION
    category = get_risk_category(p)
    return {
        "default_risk_probability": float(round(p, 4)),
        "risk_score": probability_to_risk_score(p),
        "risk_category": category,
        "decision": DECISIONS[category],
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "model_version": str(version),
        "message": MESSAGES[category],
        "note": MODEL_BASED_NOTE,
        "top_risk_increasing_factors": [],
        "top_risk_reducing_factors": [],
    }


# ---------------------------------------------------------------------------
# Optional demo entry point
# ---------------------------------------------------------------------------
def _print_assessment(probability):
    assessment = build_risk_assessment(probability)
    print(f"  p={probability:6.3f} -> score={assessment['risk_score']:3d}  "
          f"band={assessment['risk_category']:8s}  "
          f"decision={assessment['decision']:7s}  "
          f"message={assessment['message']}")


def main():
    separator = "=" * 72
    print("\n" + separator)
    print("CREDITWISE RISK ENGINE - DEMO")
    print(separator)
    print("  " + RISK_SCORE_DISCLAIMER)
    print("  " + MODEL_BASED_NOTE)
    print(f"\n  Classification threshold (locked): {CLASSIFICATION_THRESHOLD}")
    print("\n  Boundary tests:")
    for p in (0.00, 0.149, 0.150, 0.324, 0.325, 0.599, 0.600, 1.00):
        _print_assessment(p)
    print("\n  Example applicants:")
    for p in (0.08, 0.274, 0.52, 0.76, 0.325):
        _print_assessment(p)
    print("\n  Single assessment object example:")
    print("  ", build_risk_assessment(0.274))
    print("\nPhase 14 complete. No model retrained, threshold unchanged "
          "(0.325),\nno artifacts saved, no backend/frontend created.")


if __name__ == "__main__":
    main()