"""
Phase 17 - REST API routes (Blueprint).

All ML prediction/risk logic lives in ml.predict (the ONLY canonical entry
point). The routes only:
    1. inspect the HTTP request shape
    2. delegate to ml.predict.predict_applicant()
    3. persist the result once
    4. shape the JSON response

No route calls training, modifies the model, changes the threshold, or
accepts loan_status / loan_grade / loan_int_rate as inputs. /api/health is
public; every other route requires a valid JWT (both ANALYST and ADMIN).
"""

import logging

from pathlib import Path

from flask import Blueprint, jsonify, request, send_file
from flask_jwt_extended import jwt_required
from sqlalchemy import func

import ml.predict as predict_service
from backend.extensions import db
from backend.models import Prediction

logger = logging.getLogger("creditwise.api")

api = Blueprint("api", __name__)

MISSING_ARTIFACT_MESSAGE = (
    "Model artifacts not found. Run first: python ml/train_production.py"
)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
ALLOWED_PLOT_FILES = frozenset({"shap_global_bar.png", "shap_summary.png"})


def _error(message, status):
    return jsonify({"error": message}), status


def _load_artifacts():
    """Load the cached production artifacts or answer with a clean 503."""
    try:
        return predict_service.load_model_artifacts()
    except FileNotFoundError:
        return None


@api.route("/health", methods=["GET"])
def health():
    """Service + model availability. Also warms the artifact cache."""
    pipeline = _load_artifacts()
    if pipeline is None:
        return _error(MISSING_ARTIFACT_MESSAGE, 503)
    return jsonify({
        "status": "ok",
        "service": "CreditWise AI",
        "model_loaded": True,
    })


@api.route("/model/info", methods=["GET"])
@jwt_required()
def model_info():
    """User-facing model facts read from the canonical model_metadata.json."""
    loaded = _load_artifacts()
    if loaded is None:
        return _error(MISSING_ARTIFACT_MESSAGE, 503)
    _, metadata = loaded

    evaluation = metadata.get("evaluation", {})
    return jsonify({
        "model_name": metadata.get("model_name"),
        "model_version": metadata.get("model_version"),
        "algorithm": metadata.get("algorithm"),
        "feature_configuration": metadata.get("feature_configuration"),
        "classification_threshold": metadata.get("classification_threshold"),
        "probability_representation": metadata.get("probability_representation"),
        "evaluation_source": evaluation.get("evaluation_source"),
        "evaluation_metrics": {
            "ranking_metrics": evaluation.get("ranking_metrics"),
            "classification_metrics": evaluation.get("classification_metrics"),
            "counts": evaluation.get("counts"),
        },
    })


@api.route("/model/explain", methods=["GET"])
@jwt_required()
def model_explain():
    """Global SHAP feature-importance explanation (precomputed, no recompute)."""
    try:
        summary = predict_service.load_shap_summary()
    except FileNotFoundError:
        return _error(MISSING_ARTIFACT_MESSAGE, 503)

    allowed_plots = {"shap_global_bar.png", "shap_summary.png"}
    plots = {}
    for filename in summary.get("plots") or []:
        if filename in allowed_plots:
            plots[filename] = f"/api/model/explain/plots/{filename}"

    return jsonify({
        "model_version": summary.get("model_version"),
        "top_features": summary.get("top_features", []),
        "plots": plots,
        "disclaimer": summary.get("disclaimer"),
    }), 200


@api.route("/model/explain/plots/<path:filename>", methods=["GET"])
@jwt_required()
def model_plot(filename):
    """Serve a single known SHAP figure from the reports directory."""
    if filename not in ALLOWED_PLOT_FILES:
        return _error("Plot not found.", 404)
    path = FIGURES_DIR / filename
    if not path.exists() or not path.is_file():
        return _error("Plot file not found. Regenerate with: python ml/explain_model.py", 503)
    return send_file(path, mimetype="image/png")


@api.route("/predict", methods=["POST"])
@jwt_required()
def predict():
    """
    1. require a JSON object body
    2. delegate to ml.predict.predict_applicant() (validation + assessment)
    3. persist applicant inputs + assessment once
    4. return the persisted prediction id plus the assessment
    """
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _error(
            "Request body must be a JSON object with the applicant fields.", 400
        )

    try:
        assessment = predict_service.predict_applicant(payload)
    except ValueError as exc:
        return _error(str(exc), 400)
    except FileNotFoundError:
        return _error(MISSING_ARTIFACT_MESSAGE, 503)

    emp_length = payload["person_emp_length"]
    record = Prediction(
        person_age=int(payload["person_age"]),
        person_income=float(payload["person_income"]),
        person_home_ownership=str(payload["person_home_ownership"]),
        person_emp_length=float(emp_length) if emp_length is not None else None,
        loan_intent=str(payload["loan_intent"]),
        loan_amnt=float(payload["loan_amnt"]),
        cb_person_default_on_file=str(payload["cb_person_default_on_file"]),
        cb_person_cred_hist_length=int(payload["cb_person_cred_hist_length"]),
        default_risk_probability=assessment["default_risk_probability"],
        risk_score=assessment["risk_score"],
        risk_category=assessment["risk_category"],
        decision=assessment["decision"],
        classification_threshold=assessment["classification_threshold"],
        model_version=assessment["model_version"],
    )
    db.session.add(record)
    db.session.commit()

    response = record.to_dict()
    response["prediction_id"] = record.id
    response["message"] = assessment["message"]
    return jsonify(response), 200


@api.route("/history", methods=["GET"])
@jwt_required()
def history():
    """Recent predictions, newest first, with simple pagination."""
    try:
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 20))
    except (TypeError, ValueError):
        return _error("'page' and 'per_page' must be integers.", 400)

    if page < 1:
        return _error("'page' must be at least 1.", 400)
    if per_page < 1 or per_page > 100:
        return _error("'per_page' must be between 1 and 100.", 400)

    pagination = (
        Prediction.query.order_by(
            Prediction.created_at.desc(), Prediction.id.desc()
        ).paginate(page=page, per_page=per_page, error_out=False)
    )
    return jsonify({
        "items": [record.to_dict() for record in pagination.items],
        "page": pagination.page,
        "per_page": pagination.per_page,
        "total": pagination.total,
        "pages": pagination.pages,
    })


@api.route("/history/<int:prediction_id>", methods=["GET"])
@jwt_required()
def prediction_by_id(prediction_id):
    """One full persisted prediction, or 404."""
    record = db.session.get(Prediction, prediction_id)
    if record is None:
        return _error("Prediction not found.", 404)
    return jsonify(record.to_dict())


@api.route("/analytics", methods=["GET"])
@jwt_required()
def analytics():
    """Simple portfolio aggregates via SQL aggregation (zero-safe)."""
    total = db.session.query(func.count(Prediction.id)).scalar() or 0
    avg_score = db.session.query(func.avg(Prediction.risk_score)).scalar()
    avg_prob = db.session.query(
        func.avg(Prediction.default_risk_probability)
    ).scalar()

    decision_counts = {"APPROVE": 0, "REVIEW": 0, "REJECT": 0}
    for decision, count in (
        db.session.query(Prediction.decision, func.count(Prediction.id))
        .group_by(Prediction.decision)
        .all()
    ):
        decision_counts[decision] = count

    category_counts = {"LOW": 0, "MODERATE": 0, "HIGH": 0, "VERY_HIGH": 0}
    for category, count in (
        db.session.query(Prediction.risk_category, func.count(Prediction.id))
        .group_by(Prediction.risk_category)
        .all()
    ):
        category_counts[category] = count

    return jsonify({
        "total_predictions": total,
        "decision_counts": decision_counts,
        "risk_category_counts": category_counts,
        "average_risk_score": round(avg_score, 2) if avg_score is not None else 0.0,
        "average_default_risk_probability": (
            round(avg_prob, 4) if avg_prob is not None else 0.0
        ),
    })