"""
seed.py

Populate the SQLite database with 100 sample customers using the feature
schema required for churn prediction. The row generator uses realistic
logic to create synthetic customer data with churn labels.

Run with::  python seed.py
"""
from __future__ import annotations

import numpy as np

from database import SessionLocal, init_db
from models import Customer

CONTRACT_TYPES = ["month-to-month", "one-year", "two-year"]
PAYMENT_METHODS = ["credit_card", "bank_transfer", "paypal", "mailed_check"]


def build_sample_customers(n: int = 100, random_state: int = 7) -> list[Customer]:
    """Return ``n`` :class:`Customer` ORM objects with synthetic attributes."""
    rng = np.random.default_rng(random_state)

    customers: list[Customer] = []
    for i in range(1, n + 1):
        tenure = int(rng.integers(1, 73))
        monthly_charges = round(float(np.clip(rng.normal(70, 25), 15, 150)), 2)
        total_charges = round(float(np.clip(monthly_charges * tenure * rng.normal(1.0, 0.03), 0, None)), 2)
        support_tickets = int(np.clip(rng.poisson(1.6), 0, 12))
        avg_session = round(float(np.clip(rng.normal(15, 6), 1, 60)), 2)
        login_freq = int(rng.integers(0, 8))
        nps = int(rng.integers(0, 11))
        contract_type = str(rng.choice(CONTRACT_TYPES, p=[0.55, 0.30, 0.15]))
        payment_method = str(rng.choice(PAYMENT_METHODS, p=[0.45, 0.30, 0.15, 0.10]))

        # Generate realistic churn signal based on customer behavior
        churn_logit = (
            -0.9 * (contract_type == "two-year")
            - 0.6 * (contract_type == "one-year")
            + 0.5 * (payment_method == "mailed_check")
            - 0.04 * tenure
            + 0.03 * (monthly_charges - 70)
            + 0.55 * support_tickets
            - 0.08 * login_freq
            - 0.10 * nps
            + 0.03 * avg_session
            + rng.normal(0, 0.5)
        )
        churn = 1 if rng.uniform() < 1.0 / (1.0 + np.exp(-churn_logit)) else 0

        customers.append(
            Customer(
                customer_id=f"CUST{i:04d}",
                name=f"Customer {i:03d}",
                tenure_months=tenure,
                monthly_charges=monthly_charges,
                total_charges=total_charges,
                contract_type=contract_type,
                payment_method=payment_method,
                support_tickets_last_3mo=support_tickets,
                avg_session_duration_min=avg_session,
                login_frequency_weekly=login_freq,
                nps_score=nps,
                churn=churn,
            )
        )
    return customers


def seed() -> None:
    """Create tables if needed and (re)populate the customers table."""
    init_db()
    with SessionLocal() as db:
        db.query(Customer).delete()  # idempotent reseed
        db.bulk_save_objects(build_sample_customers())
        db.commit()
    print("✓ Seeded 100 sample customers into the database.")


if __name__ == "__main__":
    seed()
