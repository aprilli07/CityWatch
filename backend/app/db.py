from collections.abc import Iterator
from contextlib import contextmanager

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.config import get_settings

_pool: ConnectionPool | None = None


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            get_settings().database_url,
            min_size=1,
            max_size=10,
            kwargs={"row_factory": dict_row},
            open=True,
        )
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


@contextmanager
def connection() -> Iterator:
    with get_pool().connection() as conn:
        yield conn


def fetch_all(sql: str, params: dict | tuple | None = None) -> list[dict]:
    with connection() as conn:
        return conn.execute(sql, params).fetchall()


def fetch_one(sql: str, params: dict | tuple | None = None) -> dict | None:
    with connection() as conn:
        return conn.execute(sql, params).fetchone()


def query_df(conn, sql: str, params: dict | tuple | None = None):
    """Run a query on a psycopg connection and return a pandas DataFrame."""
    import pandas as pd

    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [c.name for c in cur.description]
        rows = [tuple(r.values()) if isinstance(r, dict) else r for r in cur.fetchall()]
    return pd.DataFrame(rows, columns=cols)
