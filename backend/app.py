"""
Phase 17 - CreditWise AI Flask application factory.

Run the development server with:
    python backend/app.py        -> http://127.0.0.1:5000

The ML model is loaded (cached) from ml/artifacts on demand - never trained
on startup. The SQLite database is created once via create_all() (idempotent:
it never deletes or recreates existing data).
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

from backend.auth_routes import admin_bp, auth_bp
from backend.config import DATA_DIR, Config
from backend.extensions import db, jwt
from backend.routes import api

logger = logging.getLogger("creditwise.api")

JSON_404 = {"error": "Not found."}


def create_app(config_override=None):
    """Application factory. config_override lets tests swap the database."""
    app = Flask(__name__)
    app.config.from_object(Config)
    if config_override:
        app.config.update(config_override)

    db.init_app(app)
    jwt.init_app(app)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    app.register_blueprint(api, url_prefix="/api")
    app.register_blueprint(auth_bp)
    app.register_blueprint(admin_bp)

    _register_jwt_handlers(app)
    _register_error_handlers(app)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with app.app_context():
        db.create_all()

    return app


def _register_jwt_handlers(app):
    """Central JSON answers for invalid/missing/expired JWTs (no 500s)."""

    @jwt.unauthorized_loader
    def missing_token(reason):
        return jsonify({"error": "Missing or invalid Authorization header."}), 401

    @jwt.invalid_token_loader
    def invalid_token(reason):
        return jsonify({"error": "Invalid token."}), 401

    @jwt.expired_token_loader
    def expired_token(header, payload):
        return jsonify({"error": "Token has expired."}), 401

    @jwt.user_identity_loader
    def user_identity(identity):
        # The JWT "sub"/identity must be JSON-serializable; store a string.
        return str(identity)


def _register_error_handlers(app):
    def reply(data, status):
        return jsonify(data), status

    @app.errorhandler(404)
    def not_found(error):
        return reply(JSON_404, 404)

    @app.errorhandler(405)
    def method_not_allowed(error):
        return reply({"error": "Method not allowed."}, 405)

    @app.errorhandler(HTTPException)
    def http_exception(error):
        return reply({"error": error.description}, error.code or 500)

    @app.errorhandler(Exception)
    def unexpected(error):
        logger.exception("Unhandled error: %s", error)
        return reply({"error": "Internal server error."}, 500)


def main():
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    print("\nCreditWise AI API running at http://127.0.0.1:5000\n")
    app.run(host="127.0.0.1", port=5000)


if __name__ == "__main__":
    main()