"""
Phase 17 - Backend configuration for CreditWise AI.

The development database lives under data/ and is created automatically on
startup (data/ is committed to the repo; the actual .db file is git-ignored).

JWT secret: read from the JWT_SECRET_KEY environment variable. A clearly
labelled DEVELOPMENT-ONLY fallback is used when the variable is unset so the
server can start on a student machine; it must never be used in production.
"""

import os
from datetime import timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_PATH = DATA_DIR / "creditwise.db"

# Roles allowed for User records.
VALID_ROLES = ("ANALYST", "ADMIN")
DEFAULT_ROLE = "ANALYST"
MIN_PASSWORD_LENGTH = 8

# DEV-ONLY FALLBACK: set JWT_SECRET_KEY in the environment for anything real.
DEV_JWT_SECRET_KEY = "dev-only-insecure-jwt-secret-do-not-use-in-production"


class Config:
    """Local development configuration (SQLite + JWT auth)."""

    SQLALCHEMY_DATABASE_URI = f"sqlite:///{DATABASE_PATH}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JSON_SORT_KEYS = False

    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", DEV_JWT_SECRET_KEY)
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(minutes=60)
    JWT_TOKEN_LOCATION = ["headers"]