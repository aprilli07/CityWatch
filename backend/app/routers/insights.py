"""Read-side endpoints: stats, heatmap, regions, leaderboard."""

import json
import time
from typing import Literal

import h3
from fastapi import APIRouter, HTTPException, Query

from app.config import ARTIFACTS_DIR, get_settings
from app.db import fetch_all, fetch_one

router = APIRouter(prefix="/api", tags=["insights"])

Period = Literal["current", "30d", "90d", "365d"]
Horizon = Literal["30d", "90d", "365d"]
_cache: dict[str, tuple[float, object]] = {}


def _cached(key: str, fn):
    ttl = get_settings().heatmap_cache_seconds
    hit = _cache.get(key)
    if hit and time.monotonic() - hit[0] < ttl:
        return hit[1]
    value = fn()
    _cache[key] = (time.monotonic(), value)
    return value


def invalidate_cache() -> None:
    _cache.clear()


@router.get("/stats")
def stats():
    return {"success": True, "data": _cached("stats", _stats)}


def _stats() -> dict:
    row = fetch_one("""
        SELECT
          COUNT(*) FILTER (WHERE status IN ('open', 'in_progress'))                       AS active_count,
          COUNT(*) FILTER (WHERE status = 'resolved' AND resolved_at >= now() - interval '7 days')
                                                                                          AS resolved_this_week,
          COUNT(*) FILTER (WHERE created_at >= now() - interval '7 days')                 AS reported_this_week,
          COUNT(*)                                                                        AS total_count,
          percentile_cont(0.5) WITHIN GROUP (
              ORDER BY EXTRACT(EPOCH FROM resolved_at - created_at) / 86400.0)
              FILTER (WHERE status = 'resolved' AND resolved_at >= now() - interval '90 days')
                                                                                          AS median_days_to_resolve
        FROM reports""")
    types = fetch_all("""
        SELECT issue_type, COUNT(*) AS n FROM reports
        WHERE issue_type IS NOT NULL AND status <> 'rejected' GROUP BY issue_type ORDER BY n DESC""")
    row["issue_type_counts"] = {t["issue_type"]: t["n"] for t in types}
    if row["median_days_to_resolve"] is not None:
        row["median_days_to_resolve"] = round(float(row["median_days_to_resolve"]), 1)
    return row


def _heatmap(period: str) -> list[dict]:
    rows = fetch_all("""
        SELECT h3_cell, ST_Y(centroid) AS latitude, ST_X(centroid) AS longitude, asset_count,
               open_issues AS issue_count, avg_severity, risk_30d, risk_90d, risk_365d, as_of
        FROM region_risk""")
    if not rows:
        return []
    col = "avg_severity" if period == "current" else f"risk_{period}"
    top = max((r[col] for r in rows), default=0) or 1.0
    for r in rows:
        r["risk"] = r[col]
        # Heatmap intensity: severity is already 0-1; risk is scaled relative to the riskiest region.
        r["value"] = r[col] if period == "current" else r[col] / top
        r["predict_30d"], r["predict_90d"], r["predict_365d"] = r["risk_30d"], r["risk_90d"], r["risk_365d"]
    rows.sort(key=lambda r: r["value"], reverse=True)
    return rows


@router.get("/heatmap")
def heatmap(period: Period = "current"):
    return {"success": True, "data": _cached(f"heatmap:{period}", lambda: _heatmap(period))}


@router.get("/regions")
def regions_geojson(horizon: Horizon = "30d"):
    """H3 hexagons as GeoJSON polygons, for a choropleth layer."""
    def build():
        feats = []
        for r in _heatmap(horizon):
            ring = [[lng, lat] for lat, lng in h3.cell_to_boundary(r["h3_cell"])]
            ring.append(ring[0])
            feats.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {k: r[k] for k in ("h3_cell", "asset_count", "issue_count", "avg_severity",
                                                  "risk", "value")},
            })
        return {"type": "FeatureCollection", "features": feats}

    return _cached(f"regions:{horizon}", build)


@router.get("/regions/top")
def top_regions(horizon: Horizon = "30d", limit: int = Query(10, ge=1, le=100)):
    return {"success": True, "data": _heatmap(horizon)[:limit]}


@router.get("/regions/{h3_cell}")
def region_detail(h3_cell: str):
    region = fetch_one("""
        SELECT h3_cell, ST_Y(centroid) AS latitude, ST_X(centroid) AS longitude, asset_count, open_issues,
               avg_severity, risk_30d, risk_90d, risk_365d, as_of, model_version
        FROM region_risk WHERE h3_cell = %s""", (h3_cell,))
    if not region:
        raise HTTPException(404, "region not found")
    region["riskiest_assets"] = fetch_all("""
        SELECT a.id, a.asset_type, a.subtype, a.name, ar.risk_30d, ar.risk_90d, ar.risk_365d,
               ST_Y(a.centroid) AS latitude, ST_X(a.centroid) AS longitude
        FROM asset_risk ar JOIN assets a ON a.id = ar.asset_id
        WHERE a.h3_cell = %s ORDER BY ar.risk_30d DESC LIMIT 10""", (h3_cell,))
    region["open_reports"] = fetch_all("""
        SELECT id, issue_type, severity, status, created_at FROM reports
        WHERE h3_cell = %s AND status IN ('open', 'in_progress') ORDER BY severity DESC NULLS LAST LIMIT 20""",
        (h3_cell,))
    return {"success": True, "data": region}


@router.get("/leaderboard")
def leaderboard(window: Literal["all", "30d", "365d"] = "all", limit: int = Query(20, ge=1, le=100)):
    since = {"all": None, "30d": "30 days", "365d": "365 days"}[window]
    rows = fetch_all("""
        SELECT u.id, COALESCE(u.display_name, split_part(u.email, '@', 1), 'anonymous') AS name,
               COUNT(r.id) AS report_count,
               COUNT(r.id) FILTER (WHERE r.status = 'resolved') AS resolved_count,
               RANK() OVER (ORDER BY COUNT(r.id) DESC) AS rank
        FROM users u JOIN reports r ON r.user_id = u.id
        WHERE r.status <> 'rejected'
          AND (%(since)s::interval IS NULL OR r.created_at >= now() - %(since)s::interval)
        GROUP BY u.id ORDER BY report_count DESC, u.id LIMIT %(limit)s""",
        {"since": since, "limit": limit})
    return {"success": True, "data": rows}


@router.get("/model")
def model_info():
    """Evaluation metrics for the deployed models (written by the training scripts)."""
    out = {}
    for name in ("risk_metrics", "cv_metrics"):
        p = ARTIFACTS_DIR / f"{name}.json"
        if p.exists():
            out[name] = json.loads(p.read_text())
    return {"success": True, "data": out}
