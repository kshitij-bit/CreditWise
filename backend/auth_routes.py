"""
Phase 17 - Authentication and role-based routes.

    POST  /api/auth/register   public   -> always creates an ANALYST
    POST  /api/auth/login      public   -> access_token + safe user info
    GET   /api/auth/me         JWT      -> current user's safe info
    GET   /api/admin/users     ADMIN    -> safe user list

Passwords are Werkzeug-hashed on the User model and are never serialized.
The registration payload can never escalate a role: the payload's "role"
key (if present) is simply ignored - users are always created as ANALYST.
"""

from flask import Blueprint, jsonify, request
from flask_jwt_extended import create_access_token, jwt_required

from backend.config import DEFAULT_ROLE, MIN_PASSWORD_LENGTH
from backend.extensions import db
from backend.models import User
from backend.security import get_current_user, role_required

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")
admin_bp = Blueprint("admin", __name__, url_prefix="/api/admin")

GENERIC_LOGIN_ERROR = "Invalid email or password."


def _error(message, status):
    return jsonify({"error": message}), status


@auth_bp.route("/register", methods=["POST"])
def register():
    """Public self-registration. Always assigns role=ANALYST."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _error("Request body must be a JSON object.", 400)

    email = payload.get("email")
    password = payload.get("password")

    if not email or not isinstance(email, str) or not email.strip():
        return _error("'email' is required.", 400)
    if not password or not isinstance(password, str):
        return _error("'password' is required.", 400)

    normalized = User.normalize_email(email)
    if len(password) < MIN_PASSWORD_LENGTH:
        return _error(
            f"'password' must be at least {MIN_PASSWORD_LENGTH} characters.", 400
        )

    existing = User.query.filter_by(email=normalized).first()
    if existing is not None:
        return _error("Email is already registered.", 409)

    user = User(
        email=normalized,
        password_hash=User.hash_password(password),
        role=DEFAULT_ROLE,
    )
    db.session.add(user)
    db.session.commit()
    return jsonify({"user": user.to_dict()}), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    """Validates credentials and returns a JWT plus safe user info."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _error("Request body must be a JSON object.", 400)

    email = payload.get("email")
    password = payload.get("password")
    if not email or not password:
        return _error(GENERIC_LOGIN_ERROR, 401)

    user = User.query.filter_by(email=User.normalize_email(email)).first()
    # One generic message for unknown email / wrong password / inactive user.
    if user is None or not user.is_active or not user.check_password(password):
        return _error(GENERIC_LOGIN_ERROR, 401)

    token = create_access_token(
        identity=user.id, additional_claims={"role": user.role}
    )
    return jsonify({
        "access_token": token,
        "user": user.to_dict(),
    }), 200


@auth_bp.route("/me", methods=["GET"])
@jwt_required()
def me():
    """Current authenticated user's safe information."""
    user = get_current_user()
    if user is None:
        return _error("User not found.", 404)
    return jsonify(user.to_dict()), 200


@admin_bp.route("/users", methods=["GET"])
@role_required("ADMIN")
def admin_users():
    """ADMIN-only view of all registered users (no password hashes)."""
    users = User.query.order_by(User.created_at.desc(), User.id.desc()).all()
    return jsonify({
        "total": len(users),
        "users": [user.to_dict() for user in users],
    }), 200