"""
Phase 17 - Admin bootstrap CLI.

Creates an ADMIN user (or prints a clean message if the email already exists).

Usage:
    python backend/create_admin.py --email admin@example.com

Password is prompted securely with getpass (never typed on the command line,
never printed or logged). Emails are normalized the same way as the register
endpoint so uniqueness checks stay consistent.
"""

import argparse
import getpass
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app import create_app  # noqa: E402  (sys.path bootstrap first)
from backend.config import MIN_PASSWORD_LENGTH  # noqa: E402
from backend.extensions import db  # noqa: E402
from backend.models import User  # noqa: E402


def create_admin_user(app, email, password):
    """
    Create an ADMIN user in the given app's database (or report cleanly if
    the email already exists). Returns (user_or_None, message).
    """
    with app.app_context():
        normalized = User.normalize_email(email)
        existing = User.query.filter_by(email=normalized).first()
        if existing is not None:
            return None, (
                f"Email '{normalized}' is already registered as role="
                f"'{existing.role}' (admin not created, existing account untouched)."
            )
        if len(password) < MIN_PASSWORD_LENGTH:
            return None, (
                f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
            )

        admin = User(
            email=normalized,
            password_hash=User.hash_password(password),
            role="ADMIN",
        )
        db.session.add(admin)
        db.session.commit()
        return admin, f"ADMIN user created: {admin.email} (id={admin.id})."


def prompt_password():
    """Prompt and confirm a password with getpass; never echo it."""
    while True:
        password = getpass.getpass("Password (hidden): ")
        if len(password) < MIN_PASSWORD_LENGTH:
            print(
                f"Password must be at least {MIN_PASSWORD_LENGTH} characters.",
                file=sys.stderr,
            )
            continue
        confirmation = getpass.getpass("Confirm password (hidden): ")
        if password != confirmation:
            print("Passwords do not match.", file=sys.stderr)
            continue
        return password


def main():
    parser = argparse.ArgumentParser(description="Create a CreditWise ADMIN user.")
    parser.add_argument("--email", required=True, help="Admin email address")
    args = parser.parse_args()

    app = create_app()
    password = prompt_password()
    admin, message = create_admin_user(app, args.email, password)
    print(message, file=sys.stderr if admin is None else sys.stdout)
    if admin is None:
        sys.exit(1)


if __name__ == "__main__":
    main()