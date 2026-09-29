"""
Debug reproduction - CreditWise login "400" investigation.

Uses the real Flask test client (create_app + temp SQLite) to prove the
backend contract for POST /api/auth/login:

    case A  valid email + valid password      -> 200
    case B  valid email + wrong password      -> 401 (generic)
    case C  missing email                     -> 401 (generic)
    case D  missing password                  -> 401 (generic)
    case E  JSON body is NOT an object (bare string / the bug) -> 400

The frontend bug that produced the 400: LoginPage called
    login(form.email, form.password)          # two string args
while AuthContext.login(credentials) / api/auth.login(credentials) expect a
single object.  A bare string was sent as the axios JSON body, so
request.get_json() yielded a str (not a dict) and the route returned 400
"Request body must be a JSON object.".

Run with:
    python -m pytest tests/test_debug_login_400.py -v
"""

import pytest

import ml.predict as pr
import ml.risk_engine as re
from backend.app import create_app
from backend.extensions import db
from backend.models import User

EMAIL = "debug@creditwise.test"
PASSWORD = "Password123"


def _register(client):
    return client.post(
        "/api/auth/register",
        json={"email": EMAIL, "password": PASSWORD},
    )


def _login(client, **payload):
    return client.post("/api/auth/login", json=payload)


@pytest.fixture()
def client(tmp_path):
    app = create_app({
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'debug.db'}",
        "TESTING": True,
    })
    with app.app_context():
        db.create_all()
        _register(app.test_client())
        _app = app
    return _app.test_client()


def test_case_a_valid_credentials_ok(client):
    resp = _login(client, email=EMAIL, password=PASSWORD)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["access_token"]
    assert body["user"]["email"] == EMAIL


def test_case_b_wrong_password_401(client):
    resp = _login(client, email=EMAIL, password="WrongPassword1")
    assert resp.status_code == 401
    assert resp.get_json()["error"] == "Invalid email or password."


def test_case_c_missing_email_401(client):
    resp = _login(client, password=PASSWORD)
    assert resp.status_code == 401
    assert resp.get_json()["error"] == "Invalid email or password."


def test_case_d_missing_password_401(client):
    resp = _login(client, email=EMAIL)
    assert resp.status_code == 401
    assert resp.get_json()["error"] == "Invalid email or password."


def test_case_e_non_object_body_400(client):
    # Reproduces the original frontend bug: bare string sent as JSON body.
    resp = client.post("/api/auth/login", data="any@example.com", content_type="application/json")
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Request body must be a JSON object."
