"""
tests/test_db_client.py
-----------------------
Connection-URL normalisation (no database needed).
"""

import pytest

from database.db_client import _normalise_url


@pytest.mark.parametrize("url, expected", [
    ("postgresql://u:p@h/db", "postgresql+psycopg2://u:p@h/db"),
    ("postgres://u:p@h/db?sslmode=require", "postgresql+psycopg2://u:p@h/db?sslmode=require"),
    ("postgresql+asyncpg://u:p@h/db", "postgresql+psycopg2://u:p@h/db"),
    ("postgresql+psycopg2://u:p@h/db", "postgresql+psycopg2://u:p@h/db"),
])
def test_driver_pinned_to_psycopg2(url, expected):
    # SQLAlchemy 2.1 defaults bare postgresql:// to psycopg 3; writes need psycopg2
    assert _normalise_url(url) == expected
