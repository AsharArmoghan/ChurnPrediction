"""
database.py

Database bootstrap for the Churn Prediction System.

Uses SQLAlchemy 2.0 style declarations against a local SQLite file
(``database.db``) to simulate a small production store. The same engine is
shared by the FastAPI backend and the seeding script.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# ---------------------------------------------------------------------------
# Engine / session configuration
# ---------------------------------------------------------------------------
# SQLite lives next to this file. ``check_same_thread=False`` is required so
# the asynchronous FastAPI event loop can share the connection pool.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'database.db')}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    future=True,
)

# All ORM models inherit from this declarative base.
Base = declarative_base()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def init_db() -> None:
    """Create all tables if they do not yet exist.

    Importing ``models`` here (lazily) ensures the mapped tables are
    registered on ``Base.metadata`` before ``create_all`` runs, while
    avoiding a circular import at module load time.
    """
    import models  # noqa: F401  (registers Customer on Base.metadata)

    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency that yields a scoped DB session.

    The session is guaranteed to be closed when the request finishes.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
