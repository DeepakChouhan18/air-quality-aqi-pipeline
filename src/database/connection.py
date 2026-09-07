"""
Single place that creates the SQLAlchemy engine used everywhere else.
Centralizing this means connection settings only change in one place
(src/config.py) and every script/test can reuse the same engine logic.
"""

from sqlalchemy import create_engine

from src.config import DATABASE_URL

_engine = None


def get_engine():
    """Return a shared SQLAlchemy engine (created once, reused)."""
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    return _engine
