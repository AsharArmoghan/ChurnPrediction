"""
main.py

FastAPI backend for the Churn Prediction System.

On startup it loads the simplified churn prediction model from
``simple_churn_model.py`` into memory for low-latency predictions.

Endpoints
---------
GET /                              -> health / service metadata
GET /api/customers                 -> list all customers
GET /api/predict/churn/{cust_id}  -> churn probability (%) for one customer
GET /api/dashboard-summary        -> aggregate dashboard metrics

Run with::

    python -m uvicorn main:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import os
import warnings
from typing import List

# Silence noisy scikit-learn warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
import pandas as pd

from database import get_db, init_db
from models import Customer
from run_churn_model import SimpleChurnPredictor

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Churn Prediction System",
    version="2.0.0",
    description="Predict customer churn probability via REST API.",
)

# Allow the static dashboard to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# In-memory model holder (populated in startup_event)
PREDICTOR: SimpleChurnPredictor = None

# Threshold (0-1) above which a customer is considered "high-risk" churn
HIGH_RISK_THRESHOLD = 0.50


# ---------------------------------------------------------------------------
# Pydantic response schemas
# ---------------------------------------------------------------------------
class CustomerOut(BaseModel):
    customer_id: str
    name: str | None
    tenure_months: int
    monthly_charges: float
    total_charges: float
    contract_type: str
    payment_method: str
    support_tickets_last_3mo: int
    avg_session_duration_min: float
    login_frequency_weekly: int
    nps_score: int
    churn: int

    class Config:
        from_attributes = True


class ChurnPrediction(BaseModel):
    customer_id: str
    churn_probability: float      # percentage 0-100 (2 dp)
    risk_level: str               # "High" / "Medium" / "Low"


class DashboardSummary(BaseModel):
    total_customers: int
    high_risk_churn_count: int
    medium_risk_churn_count: int
    low_risk_churn_count: int
    average_churn_risk_pct: float


# ---------------------------------------------------------------------------
# Startup: load model
# ---------------------------------------------------------------------------
@app.on_event("startup")
def startup_event() -> None:
    """Load the trained churn prediction model and ensure DB schema exists."""
    global PREDICTOR

    init_db()

    model_path = os.path.join(MODELS_DIR, "simple_churn_model.pkl")

    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Missing model artifact: {model_path}. "
            "Run `python simple_churn_model.py` first to train the model."
        )

    PREDICTOR = SimpleChurnPredictor.load_model(model_path)
    print("✓ Churn prediction model loaded successfully.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_customer_or_404(customer_id: str, db: Session) -> Customer:
    """Return a customer row or raise a clean 404."""
    customer = db.get(Customer, customer_id)
    if customer is None:
        raise HTTPException(
            status_code=404,
            detail=f"Customer {customer_id} not found."
        )
    return customer


def _build_feature_dataframe(customer: Customer) -> pd.DataFrame:
    """Map a Customer ORM row to a DataFrame for model prediction."""
    return pd.DataFrame([{
        "tenure_months": int(customer.tenure_months),
        "monthly_charges": float(customer.monthly_charges),
        "total_charges": float(customer.total_charges),
        "support_tickets_last_3mo": int(customer.support_tickets_last_3mo),
        "avg_session_duration_min": float(customer.avg_session_duration_min),
        "login_frequency_weekly": int(customer.login_frequency_weekly),
        "nps_score": int(customer.nps_score),
        "contract_type": str(customer.contract_type),
        "payment_method": str(customer.payment_method),
    }])


def _get_risk_level(probability: float) -> str:
    """Determine risk level based on churn probability."""
    if probability >= 0.70:
        return "High"
    elif probability >= 0.40:
        return "Medium"
    else:
        return "Low"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", tags=["meta"])
def root():
    """Health check and service information."""
    return {
        "service": "Churn Prediction System",
        "version": "2.0.0",
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/api/customers", response_model=List[CustomerOut], tags=["customers"])
def list_customers(db: Session = Depends(get_db)):
    """Return all customers stored in the database."""
    return db.query(Customer).order_by(Customer.customer_id).all()


@app.get("/api/predict/churn/{customer_id}", response_model=ChurnPrediction, tags=["predict"])
def predict_churn(customer_id: str, db: Session = Depends(get_db)):
    """Predict churn probability for a specific customer."""
    customer = _get_customer_or_404(customer_id, db)

    # Build feature DataFrame
    X = _build_feature_dataframe(customer)

    # Get prediction probability
    proba = PREDICTOR.predict_proba(X)[0]
    pct = round(float(proba) * 100, 2)
    level = _get_risk_level(proba)

    return ChurnPrediction(
        customer_id=customer_id,
        churn_probability=pct,
        risk_level=level
    )


@app.get("/api/dashboard-summary", response_model=DashboardSummary, tags=["summary"])
def dashboard_summary(db: Session = Depends(get_db)):
    """Aggregate portfolio-level churn metrics across all customers."""
    customers = db.query(Customer).all()

    if not customers:
        return DashboardSummary(
            total_customers=0,
            high_risk_churn_count=0,
            medium_risk_churn_count=0,
            low_risk_churn_count=0,
            average_churn_risk_pct=0.0,
        )

    total_customers = len(customers)

    # Build features for all customers
    rows = [_build_feature_dataframe(c).iloc[0].to_dict() for c in customers]
    X = pd.DataFrame(rows)

    # Get predictions for all customers
    churn_probs = PREDICTOR.predict_proba(X)

    # Calculate statistics
    high_risk = int((churn_probs >= 0.70).sum())
    medium_risk = int(((churn_probs >= 0.40) & (churn_probs < 0.70)).sum())
    low_risk = int((churn_probs < 0.40).sum())
    average_risk = float(churn_probs.mean()) * 100

    return DashboardSummary(
        total_customers=total_customers,
        high_risk_churn_count=high_risk,
        medium_risk_churn_count=medium_risk,
        low_risk_churn_count=low_risk,
        average_churn_risk_pct=round(average_risk, 2),
    )
