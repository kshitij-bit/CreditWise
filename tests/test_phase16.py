"""
Phase 16 tests - Flask REST API + SQLite prediction history.

Phase 17 update: the shared endpoints now require a valid JWT, so these tests
authenticate as a fresh ANALYST user (public /api/health stays anonymous).

Run with:
    python -m pytest tests/test_phase16.py -v

Covers:
    - health, model/info, predict, history, analytics endpoints
    - prediction matches ml.predict.predict_applicant() exactly
    - input validation delegated to canonical ml.predict (ValueError -> 400)
    - prediction persistence and retrieval
    - pagination
    - analytics aggregation (empty + populated)
    - frozen constants: threshold 0.325, model_version creditwise-xgb-v1
    - no model training / artifact modification during API calls
    - forbidden fields never accepted or stored
"""

import pytest

import ml.predict as pr
import ml.risk_engine as re
from backend.app import create_app
from backend.extensions import db
from backend.models import Prediction

VALID_APPLICANT = {
    "person_age": 30,
    "person_income": 75000,
    "person_home_ownership": "RENT",
    "person_emp_length": 4,
    "loan_intent": "PERSONAL",
    "loan_amnt": 12000,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 5,
}

RISKY_APPLICANT = {
    "person_age": 58,
    "person_income": 28000,
    "person_home_ownership": "RENT",
    "person_emp_length": 3,
    "loan_intent": "DEBTCONSOLIDATION",
    "loan_amnt": 50000,
    "cb_person_default_on_file": "Y",
    "cb_person_cred_hist_length": 2,
}

ANALYST_CREDENTIALS = {
    "email": "analyst@creditwise.test",
    "password": "Password123",
}


@pytest.fixture()
def app(tmp_path):
    db_path = tmp_path / "test_phase16.db"
    _app = create_app({
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{db_path}",
        "TESTING": True,
    })
    with _app.app_context():
        yield _app
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


class _AuthedClient:
    """Test client that injects a Bearer token into every request."""

    def __init__(self, client, token):
        self._client = client
        self._headers = {"Authorization": f"Bearer {token}"}

    def get(self, *args, **kwargs):
        kwargs.setdefault("headers", {}).update(self._headers)
        return self._client.get(*args, **kwargs)

    def post(self, *args, **kwargs):
        kwargs.setdefault("headers", {}).update(self._headers)
        return self._client.post(*args, **kwargs)


@pytest.fixture()
def auth_client(client):
    """A test client already authenticated as a fresh ANALYST user."""
    resp = client.post("/api/auth/register", json=ANALYST_CREDENTIALS)
    assert resp.status_code == 201
    login = client.post("/api/auth/login", json=ANALYST_CREDENTIALS)
    assert login.status_code == 200
    token = login.get_json()["access_token"]
    return _AuthedClient(client, token)


def _post_predict(client, applicant=None):
    return client.post("/api/predict", json=applicant or VALID_APPLICANT)


# -----------------------------------------------------------------------
# Health + model info
# -----------------------------------------------------------------------
def test_health_returns_200(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert body["service"] == "CreditWise AI"
    assert body["model_loaded"] is True


def test_model_info_returns_canonical_metadata(auth_client):
    resp = auth_client.get("/api/model/info")
    assert resp.status_code == 200
    body = resp.get_json()
    _, metadata = pr.load_model_artifacts()

    assert body["model_name"] == metadata["model_name"]
    assert body["model_version"] == metadata["model_version"]
    assert body["algorithm"] == metadata["algorithm"]
    assert body["feature_configuration"] == metadata["feature_configuration"]
    assert body["classification_threshold"] == metadata["classification_threshold"]
    assert body["probability_representation"] == metadata["probability_representation"]
    assert body["evaluation_source"] == metadata["evaluation"]["evaluation_source"]
    assert body["evaluation_metrics"]["ranking_metrics"] == metadata["evaluation"]["ranking_metrics"]
    assert body["evaluation_metrics"]["classification_metrics"] == metadata["evaluation"]["classification_metrics"]
    assert body["evaluation_metrics"]["counts"] == metadata["evaluation"]["counts"]


# -----------------------------------------------------------------------
# Valid predict
# -----------------------------------------------------------------------
def test_valid_predict_returns_200(auth_client):
    resp = _post_predict(auth_client)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["model_version"] == "creditwise-xgb-v1"
    assert body["classification_threshold"] == 0.325
    assert "prediction_id" in body
    assert "created_at" in body
    assert "message" in body


def test_predict_output_matches_canonical_ml_predict(auth_client):
    resp = _post_predict(auth_client)
    api_body = resp.get_json()
    assessment = pr.predict_applicant(VALID_APPLICANT)

    assert api_body["default_risk_probability"] == assessment["default_risk_probability"]
    assert api_body["risk_score"] == assessment["risk_score"]
    assert api_body["risk_category"] == assessment["risk_category"]
    assert api_body["decision"] == assessment["decision"]
    assert api_body["classification_threshold"] == assessment["classification_threshold"]
    assert api_body["model_version"] == assessment["model_version"]
    assert api_body["message"] == assessment["message"]


def test_prediction_stored_exactly_once(auth_client, app):
    resp = _post_predict(auth_client)
    assert resp.status_code == 200
    with app.app_context():
        assert Prediction.query.count() == 1

    _post_predict(auth_client)
    with app.app_context():
        assert Prediction.query.count() == 2
        ids = [p.id for p in Prediction.query.all()]
        assert len(ids) == 2
        assert ids[0] != ids[1]


def test_prediction_persists_all_fields(auth_client, app):
    _post_predict(auth_client)
    with app.app_context():
        record = Prediction.query.first()
        d = record.to_dict()
    for field in [
        "person_age", "person_income", "person_home_ownership",
        "person_emp_length", "loan_intent", "loan_amnt",
        "cb_person_default_on_file", "cb_person_cred_hist_length",
        "default_risk_probability", "risk_score", "risk_category",
        "decision", "classification_threshold", "model_version",
    ]:
        assert field in d


def test_predict_with_null_emp_length(auth_client):
    applicant = dict(VALID_APPLICANT, person_emp_length=None)
    resp = auth_client.post("/api/predict", json=applicant)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["person_emp_length"] is None


# -----------------------------------------------------------------------
# Invalid predict
# -----------------------------------------------------------------------
def test_invalid_applicant_returns_400(auth_client):
    bad = dict(VALID_APPLICANT)
    bad["person_age"] = -1
    resp = auth_client.post("/api/predict", json=bad)
    assert resp.status_code == 400
    body = resp.get_json()
    assert "error" in body


@pytest.mark.parametrize("field", [
    "person_age", "person_income", "person_emp_length", "loan_amnt",
])
def test_missing_mandatory_field_returns_400(auth_client, field):
    bad = dict(VALID_APPLICANT)
    del bad[field]
    resp = auth_client.post("/api/predict", json=bad)
    assert resp.status_code == 400


def test_unknown_categorical_returns_400(auth_client):
    bad = dict(VALID_APPLICANT, person_home_ownership="PENTHOUSE")
    resp = auth_client.post("/api/predict", json=bad)
    assert resp.status_code == 400


def test_missing_json_body_returns_400(auth_client):
    resp = auth_client.post("/api/predict")
    assert resp.status_code == 400


def test_non_object_json_body_returns_400(auth_client):
    resp = auth_client.post("/api/predict", json=[VALID_APPLICANT])
    assert resp.status_code == 400


def test_forbidden_fields_rejected(auth_client):
    bad = dict(VALID_APPLICANT, loan_status=0, loan_grade="A", loan_int_rate=11.5)
    resp = auth_client.post("/api/predict", json=bad)
    assert resp.status_code == 400


# -----------------------------------------------------------------------
# History
# -----------------------------------------------------------------------
def test_history_returns_stored_predictions(auth_client):
    _post_predict(auth_client)
    _post_predict(auth_client, dict(VALID_APPLICANT, loan_amnt=5000))
    resp = auth_client.get("/api/history")
    body = resp.get_json()
    assert resp.status_code == 200
    assert body["total"] == 2
    assert body["page"] == 1
    assert body["pages"] == 1
    assert len(body["items"]) == 2
    # newest first
    assert body["items"][0]["id"] > body["items"][1]["id"]


def test_history_pagination_works(auth_client):
    for i in range(5):
        auth_client.post("/api/predict", json=dict(VALID_APPLICANT, loan_amnt=1000 + i))
    resp = auth_client.get("/api/history?page=2&per_page=2")
    body = resp.get_json()
    assert body["total"] == 5
    assert body["pages"] == 3
    assert body["page"] == 2
    assert len(body["items"]) == 2


def test_history_invalid_params_returns_400(auth_client):
    assert auth_client.get("/api/history?page=0").status_code == 400
    assert auth_client.get("/api/history?page=-1").status_code == 400
    assert auth_client.get("/api/history?per_page=0").status_code == 400
    assert auth_client.get("/api/history?per_page=101").status_code == 400
    assert auth_client.get("/api/history?page=abc").status_code == 400


def test_history_lookup_by_id(auth_client):
    created = _post_predict(auth_client).get_json()
    resp = auth_client.get(f"/api/history/{created['prediction_id']}")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["id"] == created["prediction_id"]
    assert body["person_age"] == VALID_APPLICANT["person_age"]
    assert body["risk_category"] == created["risk_category"]


def test_history_unknown_id_returns_404(auth_client):
    resp = auth_client.get("/api/history/999999")
    assert resp.status_code == 404
    assert "error" in resp.get_json()


# -----------------------------------------------------------------------
# Analytics
# -----------------------------------------------------------------------
def test_analytics_counts_correctly(auth_client):
    _post_predict(auth_client, VALID_APPLICANT)   # low risk
    _post_predict(auth_client, RISKY_APPLICANT)  # high risk

    resp = auth_client.get("/api/analytics")
    body = resp.get_json()

    a_low = pr.predict_applicant(VALID_APPLICANT)
    a_high = pr.predict_applicant(RISKY_APPLICANT)

    expected_avg_prob = round(
        (a_low["default_risk_probability"] + a_high["default_risk_probability"]) / 2, 4
    )
    expected_avg_score = round(
        (a_low["risk_score"] + a_high["risk_score"]) / 2, 2
    )

    assert body["total_predictions"] == 2
    assert body["average_default_risk_probability"] == expected_avg_prob
    assert body["average_risk_score"] == expected_avg_score
    assert body["decision_counts"][a_low["decision"]] == 1
    assert body["decision_counts"][a_high["decision"]] == 1
    assert body["risk_category_counts"][a_low["risk_category"]] == 1
    assert body["risk_category_counts"][a_high["risk_category"]] == 1


def test_analytics_empty_db_is_zero_safe(auth_client):
    resp = auth_client.get("/api/analytics")
    body = resp.get_json()
    assert body["total_predictions"] == 0
    assert body["decision_counts"] == {"APPROVE": 0, "REVIEW": 0, "REJECT": 0}
    assert body["risk_category_counts"] == {
        "LOW": 0, "MODERATE": 0, "HIGH": 0, "VERY_HIGH": 0
    }
    assert body["average_risk_score"] == 0.0
    assert body["average_default_risk_probability"] == 0.0


# -----------------------------------------------------------------------
# Error handling
# -----------------------------------------------------------------------
def test_invalid_method_returns_405(client):
    resp = client.get("/api/predict")
    assert resp.status_code == 405
    assert "error" in resp.get_json()


def test_unknown_route_returns_404(client):
    resp = client.get("/api/doesnotexist")
    assert resp.status_code == 404


# -----------------------------------------------------------------------
# Frozen constants + safety
# -----------------------------------------------------------------------
def test_threshold_remains_0325(auth_client):
    assert re.CLASSIFICATION_THRESHOLD == 0.325
    body = _post_predict(auth_client).get_json()
    assert body["classification_threshold"] == 0.325


def test_model_version_remains_creditwise_xgb_v1(auth_client):
    assert re.MODEL_VERSION == "creditwise-xgb-v1"
    body = _post_predict(auth_client).get_json()
    assert body["model_version"] == "creditwise-xgb-v1"


def test_ml_model_is_not_retrained(auth_client):
    """Artifact file is never modified during a complete API usage cycle."""
    before_mtime = pr.MODEL_PATH.stat().st_mtime_ns
    auth_client.get("/api/health")
    auth_client.get("/api/model/info")
    _post_predict(auth_client)
    auth_client.get("/api/history")
    auth_client.get("/api/analytics")
    after_mtime = pr.MODEL_PATH.stat().st_mtime_ns
    assert before_mtime == after_mtime


def test_prediction_table_excludes_forbidden_columns(app):
    """loan_status / loan_grade / loan_int_rate are deliberately absent."""
    from sqlalchemy import inspect

    with app.app_context():
        columns = {
            c["name"] for c in inspect(db.engine).get_columns("predictions")
        }
    assert "loan_status" not in columns
    assert "loan_grade" not in columns
    assert "loan_int_rate" not in columns
    assert "loan_percent_income" not in columns