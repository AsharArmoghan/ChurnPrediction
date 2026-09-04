"""
models.py

SQLAlchemy ORM definition for the Churn Prediction System.

The ``Customer`` table mirrors the exact feature schema used by the ML
pipeline so rows can be queried at runtime and fed directly into the
trained churn prediction model.

Schema
------
    customer_id               : string primary key  (e.g. "CUST123")
    name                     : friendly display name (UI only)
    tenure_months            : numeric feature
    monthly_charges          : numeric feature
    total_charges            : numeric feature
    contract_type            : categorical feature ("month-to-month" / ...)
    payment_method           : categorical feature ("credit_card" / ...)
    support_tickets_last_3mo : numeric feature
    avg_session_duration_min : numeric feature
    login_frequency_weekly   : numeric feature
    nps_score                : numeric feature
    churn                    : label  -> Churn (0/1)
"""
from sqlalchemy import Column, Float, Integer, String

from database import Base


class Customer(Base):
    """A single customer record stored in the SQLite ``customers`` table."""

    __tablename__ = "customers"

    # --- Primary key (string customer id from the source JSON) ---
    customer_id = Column(String(32), primary_key=True, index=True)
    name = Column(String(120), nullable=True)

    # --- Numeric model input features ---
    tenure_months = Column(Integer, nullable=False)
    monthly_charges = Column(Float, nullable=False)
    total_charges = Column(Float, nullable=False)
    support_tickets_last_3mo = Column(Integer, nullable=False)
    avg_session_duration_min = Column(Float, nullable=False)
    login_frequency_weekly = Column(Integer, nullable=False)
    nps_score = Column(Integer, nullable=False)

    # --- Categorical model input features ---
    contract_type = Column(String(32), nullable=False)
    payment_method = Column(String(32), nullable=False)

    # --- Target / ground-truth label (not used as model input) ---
    churn = Column(Integer, nullable=False, default=0)  # 0 or 1

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Customer customer_id={self.customer_id!r}>"
