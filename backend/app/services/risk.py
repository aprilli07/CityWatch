"""Live risk scoring: features as of now -> asset_risk + region_risk tables.

Runs after every new report (debounced, in a background thread) so the heatmap
reflects new data within seconds, and on demand via POST /api/admin/risk/refresh.
"""

from __future__ import annotations

import logging
import threading
import time

import joblib
import numpy as np
import pandas as pd
import psycopg

from app.config import get_settings
from app.db import query_df
from ml.risk.data import load_context

log = logging.getLogger(__name__)

REGION_PRIOR_K = 10
_bundle = None
_lock = threading.Lock()
_running = threading.Lock()
_dirty = threading.Event()
last_run: dict = {}


def _load_bundle():
    global _bundle
    with _lock:
        if _bundle is None:
            path = get_settings().risk_model_path
            if not path.exists():
                return None
            _bundle = joblib.load(path)
    return _bundle


def score_all(as_of: pd.Timestamp | None = None) -> dict:
    bundle = _load_bundle()
    if bundle is None:
        raise RuntimeError("risk model not trained; run `python -m ml.risk.train`")
    t0 = time.perf_counter()
    as_of = as_of or pd.Timestamp.now(tz="UTC")
    with psycopg.connect(get_settings().database_url) as conn:
        ctx = load_context(conn)
        X = ctx.build(as_of.tz_localize(None) if as_of.tzinfo else as_of)[bundle["features"]]
        probs = {h: m.predict_proba(X)[:, 1] for h, m in bundle["models"].items()}

        assets = ctx.assets[["id", "h3_cell", "lon", "lat"]].copy()
        for h, p in probs.items():
            assets[f"risk_{h}d"] = p
        open_stats = query_df(conn, """
            SELECT h3_cell, COUNT(*) AS open_issues, AVG(severity) AS avg_severity
            FROM reports WHERE status IN ('open', 'in_progress') GROUP BY h3_cell""")
        regions = assets.groupby("h3_cell").agg(
            asset_count=("id", "size"), lon=("lon", "mean"), lat=("lat", "mean"),
            risk_30d=("risk_30d", "mean"), risk_90d=("risk_90d", "mean"), risk_365d=("risk_365d", "mean"),
        ).reset_index().merge(open_stats, on="h3_cell", how="left")
        # Shrink each region's mean toward the city mean (empirical Bayes, k pseudo-assets) so a
        # region with 1-2 assets can't top the ranking on a single noisy probability.
        for h in probs:
            col, city = f"risk_{h}d", float(np.mean(probs[h]))
            regions[col] = (regions[col] * regions["asset_count"] + REGION_PRIOR_K * city) / (
                regions["asset_count"] + REGION_PRIOR_K)
        regions["open_issues"] = regions["open_issues"].fillna(0).astype(int)
        regions["avg_severity"] = regions["avg_severity"].astype(float).fillna(0.0)

        day = as_of.date()
        version = bundle["version"]
        with conn.cursor() as cur:
            cur.execute("TRUNCATE asset_risk, region_risk")
            with cur.copy("COPY asset_risk (asset_id, risk_30d, risk_90d, risk_365d, as_of, model_version)"
                          " FROM STDIN") as cp:
                for r in assets.itertuples():
                    cp.write_row((r.id, float(r.risk_30d), float(r.risk_90d), float(r.risk_365d), day, version))
            with cur.copy("COPY region_risk (h3_cell, centroid, asset_count, open_issues, avg_severity,"
                          " risk_30d, risk_90d, risk_365d, as_of, model_version) FROM STDIN") as cp:
                for r in regions.itertuples():
                    cp.write_row((r.h3_cell, f"SRID=4326;POINT({r.lon} {r.lat})", int(r.asset_count),
                                  int(r.open_issues), float(r.avg_severity), float(r.risk_30d),
                                  float(r.risk_90d), float(r.risk_365d), day, version))
        conn.commit()
    result = {
        "as_of": str(day), "model_version": version, "assets": len(assets), "regions": len(regions),
        "seconds": round(time.perf_counter() - t0, 2),
        "mean_risk": {h: float(np.mean(p)) for h, p in probs.items()},
    }
    last_run.clear()
    last_run.update(result)
    return result


def request_refresh() -> None:
    """Debounced background rescore: concurrent requests coalesce into one rerun."""
    _dirty.set()
    if not _running.acquire(blocking=False):
        return

    def worker():
        try:
            while _dirty.is_set():
                _dirty.clear()
                try:
                    r = score_all()
                    from app.routers.insights import invalidate_cache  # avoid import cycle

                    invalidate_cache()
                    log.info("Risk refreshed: %s", r)
                except Exception:
                    log.exception("Risk refresh failed")
        finally:
            _running.release()
            if _dirty.is_set():  # a request arrived after the loop exited
                request_refresh()

    threading.Thread(target=worker, daemon=True).start()
