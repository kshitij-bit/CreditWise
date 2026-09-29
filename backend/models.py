"""
Phase 17 - SQLAlchemy models.

Prediction: stores the RAW applicant inputs plus the complete CreditWise
assessment for every /api/predict call. The applicant inputs are exactly the
canonical application-only fields from ml.predict; loan_status / loan_grade /
loan_int_rate are deliberately NOT part of the schema and can never be stored.

User: local authentication record for the Flask API. Passwords are stored as
Werkzeug hashes only - never in plaintext - and the hash is never serialized.
"""

from datetime import datetime, timezone

from werkzeug.security import check_password_hash, generate_password_hash

from backend.config import DEFAULT_ROLE, VALID_ROLES
from backend.extensions import db


class Prediction(db.Model):
    """One persisted prediction, created by the /api/predict route."""

    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Applicant inputs (canonical ml.predict raw fields).
    person_age = db.Column(db.Integer, nullable=False)
    person_income = db.Column(db.Float, nullable=False)
    person_home_ownership = db.Column(db.String(16), nullable=False)
    person_emp_length = db.Column(db.Float, nullable=True)
    loan_intent = db.Column(db.String(32), nullable=False)
    loan_amnt = db.Column(db.Float, nullable=False)
    cb_person_default_on_file = db.Column(db.String(2), nullable=False)
    cb_person_cred_hist_length = db.Column(db.Integer, nullable=False)

    # Prediction outputs (produced by ml.predict.predict_applicant()).
    default_risk_probability = db.Column(db.Float, nullable=False)
    risk_score = db.Column(db.Integer, nullable=False)
    risk_category = db.Column(db.String(16), nullable=False)
    decision = db.Column(db.String(16), nullable=False)
    classification_threshold = db.Column(db.Float, nullable=False)
    model_version = db.Column(db.String(32), nullable=False)

    def to_dict(self):
        """JSON-safe representation of the full persisted prediction."""
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "person_age": self.person_age,
            "person_income": self.person_income,
            "person_home_ownership": self.person_home_ownership,
            "person_emp_length": self.person_emp_length,
            "loan_intent": self.loan_intent,
            "loan_amnt": self.loan_amnt,
            "cb_person_default_on_file": self.cb_person_default_on_file,
            "cb_person_cred_hist_length": self.cb_person_cred_hist_length,
            "default_risk_probability": self.default_risk_probability,
            "risk_score": self.risk_score,
            "risk_category": self.risk_category,
            "decision": self.decision,
            "classification_threshold": self.classification_threshold,
            "model_version": self.model_version,
        }


class User(db.Model):
    """Local API user. Role is one of ANALYST / ADMIN."""

    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, index=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(16), nullable=False, default=DEFAULT_ROLE)
    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    is_active = db.Column(db.Boolean, nullable=False, default=True)

    @staticmethod
    def normalize_email(email):
        """One consistent normalization for unique email matching."""
        return str(email).strip().lower()

    @staticmethod
    def hash_password(password):
        """Werkzeug-hash a plaintext password (never store it directly)."""
        return generate_password_hash(password)

    def check_password(self, password):
        """Verify a plaintext password against the stored hash."""
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        """Safe serialization - intentionally never includes password_hash."""
        return {
            "id": self.id,
            "email": self.email,
            "role": self.role,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "is_active": self.is_active,
        }

    def __repr__(self):
        return f"<User {self.id} {self.email} {self.role}>"