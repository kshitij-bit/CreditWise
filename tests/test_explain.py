"""
Tests for the /api/model/explain explainability endpoint.

Run with:
    python -m pytest tests/test_explain.py -v

Covers:
    - anonymous request -> 401
    - authenticated Analyst -> 200
    - authenticated Admin -> 200
    - model_version == creditwise-xgb-v1
    - feature list is non-empty
    - feature importances are finite
    - no loan_status leakage feature appears
    - response includes SHAP disclaimer
    - no model retraining occurs per HTTP request
    - no SHAP recalculation occurs per HTTP request
    - known plot served with correct MIME type
    - disallowed / path-traversal filenames rejected
"""

import json
import math

import pytest
from backend.app import create_app
from backend.extensions import db
from backend.models import User
import ml.predict as pr
import backend.routes as routes

ANALYST_EMAIL = "explain.analyst@creditwise.test"
ANALYST_PASSWORD = "ExplainPass1"
ADMIN_EMAIL = "explain.admin@creditwise.test"
ADMIN_PASSWORD = "ExplainAdmin1"

SAMPLE_FEATURES = [
    {"feature": "loan_to_income_ratio", "importance": 3.21, "direction": 1},
    {"feature": "person_income", "importance": 1.87, "direction": -1},
    {"feature": "loan_amnt", "importance": 1.50, "direction": 1},
]
DISCLAIMER = ("SHAP explains the model's prediction behavior; "
              "it does not establish causality.")
MODEL_VERSION = "creditwise-xgb-v1"


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------
@pytest.fixture()
def app(tmp_path):
    db_path = tmp_path / "test_explain.db"
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


def _register(client, email=ANALYST_EMAIL, password=ANALYST_PASSWORD):
    return client.post("/api/auth/register", json={"email": email, "password": password})


def _login(client, email=ANALYST_EMAIL, password=ANALYST_PASSWORD):
    return client.post("/api/auth/login", json={"email": email, "password": password})


def _bearer(token):
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def analyst_token(client):
    assert _register(client, ANALYST_EMAIL, ANALYST_PASSWORD).status_code == 201
    resp = _login(client, ANALYST_EMAIL, ANALYST_PASSWORD)
    assert resp.status_code == 200
    return resp.get_json()["access_token"]


@pytest.fixture()
def admin_user(app):
    with app.app_context():
        user = User(
            email=ADMIN_EMAIL,
            password_hash=User.hash_password(ADMIN_PASSWORD),
            role="ADMIN",
        )
        db.session.add(user)
        db.session.commit()
    return user


@pytest.fixture()
def admin_token(client, admin_user):
    resp = _login(client, ADMIN_EMAIL, ADMIN_PASSWORD)
    assert resp.status_code == 200
    return resp.get_json()["access_token"]


@pytest.fixture()
def shap_env(tmp_path, monkeypatch):
    """Create a temp precomputed SHAP artifact and figure dir for tests."""
    summary = {
        "model_version": MODEL_VERSION,
        "explanation_type": "global_shap",
        "top_features": SAMPLE_FEATURES,
        "plots": ["shap_global_bar.png", "shap_summary.png"],
        "disclaimer": DISCLAIMER,
    }
    summary_path = tmp_path / "shap_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    fig_dir = tmp_path / "figures"
    fig_dir.mkdir(exist_ok=True)
    (fig_dir / "shap_global_bar.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00")
    (fig_dir / "shap_summary.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00")

    monkeypatch.setattr(pr, "SHAP_SUMMARY_PATH", summary_path)
    monkeypatch.setattr(routes, "FIGURES_DIR", fig_dir)
    return summary


# -----------------------------------------------------------------------
# Auth / access control
# -----------------------------------------------------------------------
def test_anonymous_explain_rejected(client, shap_env):
    resp = client.get("/api/model/explain")
    assert resp.status_code == 401
    assert "error" in resp.get_json()


def test_analyst_can_get_explain(client, shap_env, analyst_token):
    resp = client.get("/api/model/explain", headers=_bearer(analyst_token))
    assert resp.status_code == 200


def test_admin_can_get_explain(client, shap_env, admin_token):
    resp = client.get("/api/model/explain", headers=_bearer(admin_token))
    assert resp.status_code == 200


# -----------------------------------------------------------------------
# Response content
# -----------------------------------------------------------------------
def test_explain_model_version(client, shap_env, analyst_token):
    resp = client.get("/api/model/explain", headers=_bearer(analyst_token))
    assert resp.get_json()["model_version"] == MODEL_VERSION


def test_explain_features_non_empty(client, shap_env, analyst_token):
    resp = client.get("/api/model/explain", headers=_bearer(analyst_token))
    features = resp.get_json()["top_features"]
    assert len(features) > 0


def test_explain_importances_finite(client, shap_env, analyst_token):
    resp = client.get("/api/model/explain", headers=_bearer(analyst_token))
    for f in resp.get_json()["top_features"]:
        assert math.isfinite(f["importance"])


def test_explain_no_leakage_feature(client, shap_env, analyst_token):
    resp = client.get("/api/model/explain", headers=_bearer(analyst_token))
    feature_names = {f["feature"] for f in resp.get_json()["top_features"]}
    assert "loan_status" not in feature_names
    assert "loan_grade" not in feature_names
    assert "loan_int_rate" not in feature_names


def test_explain_includes_disclaimer(client, shap_env, analyst_token):
    resp = client.get("/api/model/explain", headers=_bearer(analyst_token))
    disclaimer = resp.get_json().get("disclaimer", "")
    assert "causality" in disclaimer.lower()


# -----------------------------------------------------------------------
# No model retraining / SHAP recalculation per request
# -----------------------------------------------------------------------
def test_explain_no_retraining_or_shap_recompute(client, shap_env, analyst_token, monkeypatch):
    def _fail(*args, **kwargs):
        raise AssertionError("model artefact loaded or SHAP computed at request time")

    import shap as _shap

    monkeypatch.setattr(pr, "load_model_artifacts", _fail)
    monkeypatch.setattr(_shap, "TreeExplainer", _fail)
    resp = client.get("/api/model/explain", headers=_bearer(analyst_token))
    assert resp.status_code == 200


# -----------------------------------------------------------------------
# Plot serving
# -----------------------------------------------------------------------
def test_explain_plot_served(client, shap_env, analyst_token):
    resp = client.get(
        "/api/model/explain/plots/shap_global_bar.png",
        headers=_bearer(analyst_token),
    )
    assert resp.status_code == 200
    assert resp.content_type == "image/png"


def test_explain_disallowed_plot_rejected(client, shap_env, analyst_token):
    resp = client.get(
        "/api/model/explain/plots/passwd",
        headers=_bearer(analyst_token),
    )
    assert resp.status_code == 404


def test_explain_path_traversal_plot_rejected(client, shap_env, analyst_token):
    resp = client.get(
        "/api/model/explain/plots/..%2F..%2F..%2Fapp.py",
        headers=_bearer(analyst_token),
    )
    assert resp.status_code == 404


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
