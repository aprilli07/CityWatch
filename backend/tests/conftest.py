import os

import psycopg
import pytest

TEST_DB_URL = os.environ.get("TEST_DATABASE_URL", "postgresql://localhost:5432/citywatch_test")

# Configure the app for tests *before* it is imported anywhere.
os.environ["DATABASE_URL"] = TEST_DB_URL
os.environ["STAFF_API_TOKEN"] = "test-staff-token"
os.environ["SUPABASE_URL"] = ""          # store photos on local disk
os.environ["SUPABASE_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""
os.environ["RISK_MODEL_PATH"] = "/nonexistent/risk.joblib"  # don't trigger background rescoring


def _db_available() -> bool:
    admin = TEST_DB_URL.rsplit("/", 1)[0] + "/postgres"
    try:
        with psycopg.connect(admin, autocommit=True, connect_timeout=3) as conn:
            name = TEST_DB_URL.rsplit("/", 1)[1]
            if not conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone():
                conn.execute(f'CREATE DATABASE "{name}"')
        return True
    except psycopg.OperationalError:
        return False


@pytest.fixture(scope="session")
def db():
    if not _db_available():
        pytest.skip("Postgres not available for integration tests")
    from db.migrate import migrate

    with psycopg.connect(TEST_DB_URL, autocommit=True) as conn:
        conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    migrate(TEST_DB_URL)
    with psycopg.connect(TEST_DB_URL, autocommit=True) as conn:
        # Two assets in Waterloo: a road segment and a stop sign
        conn.execute("""
            INSERT INTO assets (id, asset_type, subtype, name, length_m, geom, centroid, h3_cell) VALUES
            (1, 'road', 'residential', 'King St', 120,
             ST_GeomFromText('LINESTRING(-80.5230 43.4640, -80.5215 43.4645)', 4326),
             ST_GeomFromText('POINT(-80.52225 43.46425)', 4326), '882b9b5a1dfffff'),
            (2, 'sign', 'stop', NULL, NULL,
             ST_GeomFromText('POINT(-80.5300 43.4700)', 4326),
             ST_GeomFromText('POINT(-80.5300 43.4700)', 4326), '882b9b5a1dfffff')""")
    yield TEST_DB_URL


@pytest.fixture(scope="session")
def client(db):
    from fastapi.testclient import TestClient

    from app.config import get_settings

    get_settings.cache_clear()
    from app.main import app

    with TestClient(app) as c:
        yield c
