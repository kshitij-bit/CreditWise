"""
Phase 17 tests - JWT authentication and role-based access control.

Run with:
    python -m pytest tests/test_phase17.py -v

Covers:
    - registration (success, duplicate, validation, no admin escalation)
    - password hashing (never plaintext, never returned)
    - login (success, wrong password, unknown email, inactive user)
    - /api/auth/me
    - protected routes: anonymous 401, analyst 200
    - role matrix: analyst forbidden / admin allowed on /api/admin/users
    - /api/health stays public
    - frozen ML constants and prediction parity at API level
"""

import pytest

import ml.predict as pr
import ml.risk_engine as re
from backend.app import create_app
from backend.extensions import db
from backend.models import User

ANALYST_EMAIL = "analyst@creditwise.test"
ANALYST_PASSWORD = "Password123"
ADMIN_EMAIL = "boss@creditwise.test"
ADMIN_PASSWORD = "AdminPass123"

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


@pytest.fixture()
def app(tmp_path):
    db_path = tmp_path / "test_phase17.db"
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


def _register(client, email=ANALYST_EMAIL, password=ANALYST_PASSWORD, **extra):
    payload = {"email": email, "password": password}
    payload.update(extra)
    return client.post("/api/auth/register", json=payload)


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


def _assert_no_hash_present(body):
    """Recursively assert no 'password_hash' key anywhere in a parsed body."""
    if isinstance(body, dict):
        assert "password_hash" not in body
        for value in body.values():
            _assert_no_hash_present(value)
    elif isinstance(body, list):
        for value in body:
            _assert_no_hash_present(value)


# -----------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------
def test_register_analyst_success(client):
    resp = _register(client)
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["user"]["email"] == ANALYST_EMAIL
    assert body["user"]["role"] == "ANALYST"
    assert "id" in body["user"]
    assert "created_at" in body["user"]


def test_register_normalizes_email(client):
    resp = client.post("/api/auth/register",
                       json={"email": "  Analyst@CreditWise.TEST ", "password": ANALYST_PASSWORD})
    assert resp.status_code == 201
    assert resp.get_json()["user"]["email"] == "analyst@creditwise.test"


def test_password_stored_as_hash(client, app):
    _register(client)
    with app.app_context():
        user = User.query.filter_by(email=ANALYST_EMAIL).first()
    assert user is not None
    assert user.password_hash != ANALYST_PASSWORD
    assert user.password_hash.startswith(("pbkdf2:", "scrypt:"))
    assert user.check_password(ANALYST_PASSWORD)


def test_register_response_never_contains_hash(client):
    resp = _register(client)
    assert resp.status_code == 201
    _assert_no_hash_present(resp.get_json())


def test_duplicate_email_rejected(client):
    assert _register(client).status_code == 201
    assert _register(client).status_code == 409


def test_duplicate_email_case_insensitive(client):
    assert _register(client).status_code == 201
    resp = _register(client, email="ANALYST@CREDITWISE.TEST")
    assert resp.status_code == 409


def test_register_requires_email_and_password(client):
    assert _register(client, email="", password=ANALYST_PASSWORD).status_code == 400
    assert _register(client, email=ANALYST_EMAIL, password="").status_code == 400
    assert client.post("/api/auth/register", json={}).status_code == 400
    assert client.post("/api/auth/register").status_code == 400


def test_register_short_password_rejected(client):
    resp = _register(client, password="Short1")
    assert resp.status_code == 400
    assert "at least 8" in resp.get_json()["error"]


def test_register_cannot_create_admin(client, app):
    resp = _register(client, role="ADMIN", is_active=False)
    assert resp.status_code == 201
    assert resp.get_json()["user"]["role"] == "ANALYST"
    with app.app_context():
        user = User.query.filter_by(email=ANALYST_EMAIL).first()
        assert user.role == "ANALYST"
        assert user.is_active is True


# -----------------------------------------------------------------------
# Login
# -----------------------------------------------------------------------
def test_login_success(client):
    _register(client)
    resp = _login(client)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["access_token"]
    assert body["user"]["email"] == ANALYST_EMAIL
    assert body["user"]["role"] == "ANALYST"


def test_login_incorrect_password_rejected(client):
    _register(client)
    resp = _login(client, password="WrongPassword1")
    assert resp.status_code == 401
    assert resp.get_json()["error"] == "Invalid email or password."


def test_login_unknown_email_rejected(client):
    resp = _login(client, email="nobody@creditwise.test")
    assert resp.status_code == 401
    assert resp.get_json()["error"] == "Invalid email or password."


def test_login_inactive_user_rejected(client, app):
    with app.app_context():
        user = User(
            email="inactive@creditwise.test",
            password_hash=User.hash_password("InactivePass1"),
            role="ANALYST",
            is_active=False,
        )
        db.session.add(user)
        db.session.commit()
    resp = _login(client, email="inactive@creditwise.test", password="InactivePass1")
    assert resp.status_code == 401


def test_login_response_never_contains_hash(client):
    _register(client)
    _assert_no_hash_present(_login(client).get_json())


# -----------------------------------------------------------------------
# /api/auth/me
# -----------------------------------------------------------------------
def test_me_works(client, analyst_token):
    resp = client.get("/api/auth/me", headers=_bearer(analyst_token))
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["email"] == ANALYST_EMAIL
    assert body["role"] == "ANALYST"
    assert "id" in body
    assert "created_at" in body
    assert "password_hash" not in body


def test_me_without_token_rejected(client):
    resp = client.get("/api/auth/me")
    assert resp.status_code == 401


# -----------------------------------------------------------------------
# Protected routes - anonymous rejection
# -----------------------------------------------------------------------
def test_anonymous_predict_rejected(client):
    resp = client.post("/api/predict", json=VALID_APPLICANT)
    assert resp.status_code == 401
    assert "error" in resp.get_json()


def test_anonymous_protected_routes_rejected(client):
    assert client.get("/api/model/info").status_code == 401
    assert client.get("/api/history").status_code == 401
    assert client.get("/api/analytics").status_code == 401
    assert client.get("/api/admin/users").status_code == 401


def test_invalid_token_rejected(client):
    resp = client.get("/api/history", headers=_bearer("not.a.real.token"))
    assert resp.status_code == 401
    assert "error" in resp.get_json()


def test_modulated_authorization_header_rejected(client):
    resp = client.get("/api/history", headers={"Authorization": "token"})
    assert resp.status_code == 401


# -----------------------------------------------------------------------
# Admin CLI
# -----------------------------------------------------------------------
def test_create_admin_cli_core_creates_admin(app):
    from backend.create_admin import create_admin_user

    admin, message = create_admin_user(app, "cliadmin@creditwise.test", "CliAdminPass1")
    assert admin is not None
    assert admin.role == "ADMIN"
    assert "ADMIN user created" in message
    assert admin.check_password("CliAdminPass1")


def test_create_admin_cli_core_rejects_existing_email(app):
    from backend.create_admin import create_admin_user

    first = create_admin_user(app, "cliadmin@creditwise.test", "CliAdminPass1")
    assert first[0] is not None

    second, message = create_admin_user(app, "CLIADMIN@creditwise.test", "AnotherPass1")
    assert second is None
    assert "already registered" in message

    with app.app_context():
        assert User.query.filter_by(email="cliadmin@creditwise.test").count() == 1


# -----------------------------------------------------------------------
# Analyst access
# -----------------------------------------------------------------------
def test_analyst_can_call_predict(client, analyst_token):
    resp = client.post("/api/predict", json=VALID_APPLICANT, headers=_bearer(analyst_token))
    assert resp.status_code == 200
    assert resp.get_json()["model_version"] == "creditwise-xgb-v1"


def test_analyst_can_call_history(client, analyst_token):
    client.post("/api/predict", json=VALID_APPLICANT, headers=_bearer(analyst_token))
    resp = client.get("/api/history", headers=_bearer(analyst_token))
    assert resp.status_code == 200
    assert resp.get_json()["total"] == 1


def test_analyst_can_call_analytics(client, analyst_token):
    resp = client.get("/api/analytics", headers=_bearer(analyst_token))
    assert resp.status_code == 200
    assert resp.get_json()["total_predictions"] == 0


def test_analyst_can_call_model_info(client, analyst_token):
    resp = client.get("/api/model/info", headers=_bearer(analyst_token))
    assert resp.status_code == 200
    assert resp.get_json()["model_version"] == "creditwise-xgb-v1"


def test_analyst_forbidden_admin_users(client, analyst_token):
    resp = client.get("/api/admin/users", headers=_bearer(analyst_token))
    assert resp.status_code == 403


# -----------------------------------------------------------------------
# Admin access
# -----------------------------------------------------------------------
def test_admin_allowed_admin_users(client, analyst_token, admin_token):
    resp = client.get("/api/admin/users", headers=_bearer(admin_token))
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["total"] >= 1
    emails = [u["email"] for u in body["users"]]
    assert ANALYST_EMAIL in emails
    for user in body["users"]:
        assert "password_hash" not in user


def test_admin_users_never_expose_hash(client, admin_token):
    _assert_no_hash_present(client.get("/api/admin/users", headers=_bearer(admin_token)).get_json())


# -----------------------------------------------------------------------
# Public health
# -----------------------------------------------------------------------
def test_health_remains_public(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


# -----------------------------------------------------------------------
# Frozen ML constants + prediction parity (through the authenticated API)
# -----------------------------------------------------------------------
def test_threshold_remains_0325(client, analyst_token):
    assert re.CLASSIFICATION_THRESHOLD == 0.325
    resp = client.post("/api/predict", json=VALID_APPLICANT, headers=_bearer(analyst_token))
    assert resp.get_json()["classification_threshold"] == 0.325


def test_model_version_remains_creditwise_xgb_v1(client, analyst_token):
    assert re.MODEL_VERSION == "creditwise-xgb-v1"
    resp = client.post("/api/predict", json=VALID_APPLICANT, headers=_bearer(analyst_token))
    assert resp.get_json()["model_version"] == "creditwise-xgb-v1"


def test_api_prediction_matches_canonical_ml_predict(client, analyst_token):
    resp = client.post("/api/predict", json=VALID_APPLICANT, headers=_bearer(analyst_token))
    api_body = resp.get_json()
    assessment = pr.predict_applicant(VALID_APPLICANT)
    assert api_body["default_risk_probability"] == assessment["default_risk_probability"]
    assert api_body["risk_score"] == assessment["risk_score"]
    assert api_body["risk_category"] == assessment["risk_category"]
    assert api_body["decision"] == assessment["decision"]
    assert api_body["classification_threshold"] == assessment["classification_threshold"]
    assert api_body["model_version"] == assessment["model_version"]