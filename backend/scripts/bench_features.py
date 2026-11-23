"""Benchmark: original per-asset buffer scan vs. KD-tree neighbour features.

The hackathon version computed each spatial feature by looping over assets and
testing every incident against a buffer polygon (O(assets x incidents)), once
per feature. This reproduces that approach with Shapely 2 (already vectorised
over incidents, so it is a *generous* re-implementation) for the 4 spatial
features it had, and compares with FeatureContext.build, which computes ~60
spatial/temporal features from one KD-tree pass.

Usage: python -m scripts.bench_features [--assets 2000]
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import psycopg
import shapely

from app.config import ARTIFACTS_DIR, get_settings
from ml.risk.data import load_context


def legacy_features(ctx, snap_day: float, n_assets: int) -> np.ndarray:
    inc_pts = shapely.points(ctx.xy[ctx.inc_asset])
    past = ctx.inc_t < snap_day
    rep_recent = (ctx.rep_t < snap_day) & (ctx.rep_t >= snap_day - 30)
    # Reports are only available to FeatureContext as pairs; rebuild their points for the legacy loop.
    rep_pts = shapely.points(ctx._rep_xy[rep_recent])
    out = np.zeros((n_assets, 4))
    for i in range(n_assets):
        p = shapely.points(ctx.xy[i])
        b100, b150 = shapely.buffer(p, 100), shapely.buffer(p, 150)
        in100 = shapely.within(inc_pts, b100) & past                 # incidents_within_100m
        in150 = shapely.within(inc_pts, b150) & past                 # avg_severity_within_150m
        r100 = shapely.within(rep_pts, b100)                         # reports_last_30d / report severity
        out[i] = [in100.sum(), ctx.inc_sev[in150].mean() if in150.any() else 0, r100.sum(), r100.sum()]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", type=int, default=2000, help="assets to time the legacy loop on (then extrapolate)")
    args = ap.parse_args()
    with psycopg.connect(get_settings().database_url) as conn:
        ctx = load_context(conn)
        rep = conn.execute("SELECT ST_X(loc), ST_Y(loc) FROM reports WHERE status <> 'rejected'").fetchall()
    from ml.risk.features import _to_utm

    rx, ry = _to_utm.transform(np.array([r[0] for r in rep]), np.array([r[1] for r in rep]))
    ctx._rep_xy = np.column_stack([rx, ry])

    snap = pd.Timestamp("2024-06-01")
    snap_day = (snap.tz_localize("UTC") - pd.Timestamp("1970-01-01", tz="UTC")).total_seconds() / 86400

    t = time.perf_counter()
    legacy_features(ctx, snap_day, args.assets)
    legacy_s = (time.perf_counter() - t) * ctx.n / args.assets

    runs = []
    for _ in range(3):
        t = time.perf_counter()
        X = ctx.build(snap)
        runs.append(time.perf_counter() - t)
    new_s = min(runs)

    result = {
        "assets": ctx.n, "incidents": int(len(ctx.inc_t)), "reports": int(len(ctx.rep_t)),
        "legacy_4_features_s_per_snapshot": round(legacy_s, 2),
        "new_all_features_s_per_snapshot": round(new_s, 3),
        "new_feature_count": int(X.shape[1]),
        "speedup": round(legacy_s / new_s, 1),
    }
    (ARTIFACTS_DIR / "feature_benchmark.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
