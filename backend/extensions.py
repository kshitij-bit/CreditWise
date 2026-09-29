"""
Phase 17 - Shared Flask extension instances.

Created once here, bound to the app inside backend.app.create_app().
"""

from flask_jwt_extended import JWTManager
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
jwt = JWTManager()