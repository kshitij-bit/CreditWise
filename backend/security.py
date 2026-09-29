"""
Phase 17 - Reusable authorization helpers.

There is exactly ONE role-check implementation in the project. Routes either
use @jwt_required() (any authenticated user) or @role_required("ADMIN") (which
also requires valid JWTs). Never duplicate role logic in individual routes.
"""

from functools import wraps

from flask import jsonify
from flask_jwt_extended import get_jwt, get_jwt_identity, verify_jwt_in_request

from backend.config import VALID_ROLES
from backend.extensions import db
from backend.models import User

FORBIDDEN_MESSAGE = "You do not have permission to perform this action."


def role_required(*roles):
    """
    Require a valid JWT whose 'role' claim is one of the given roles.

    Unauthenticated requests are rejected by verify_jwt_in_request() (central
    JWT handlers answer 401). Authenticated users without a matching role
    receive 403. The role travels inside the token (chosen simple approach:
    role changes take effect on the next login).
    """
    for role in roles:
        if role not in VALID_ROLES:
            raise ValueError(f"Unknown role: {role}")

    def decorator(view):
        @wraps(view)
        def wrapper(*args, **kwargs):
            verify_jwt_in_request()
            if get_jwt().get("role") not in roles:
                return jsonify({"error": FORBIDDEN_MESSAGE}), 403
            return view(*args, **kwargs)
        return wrapper

    return decorator


def get_current_user():
    """Load the authing user from the DB using the JWT identity (or None)."""
    try:
        user_id = int(get_jwt_identity())
    except (TypeError, ValueError):
        return None
    return db.session.get(User, user_id) if user_id is not None else None