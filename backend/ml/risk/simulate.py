"""Simulate a realistic incident + citizen-report history over real OSM assets.

There is no public dataset of per-asset infrastructure failures for Waterloo,
so the history is *simulated* (and flagged `is_simulated = true` in the DB).
The generator is a daily hazard process designed so that the signals a real
city would see are present and learnable:

* asset fragility: road class, segment length, random per-asset latent factor
* spatial hotspots: clusters of fragile infrastructure (e.g. old neighbourhoods)
* real weather: freeze-thaw cycles and precipitation from the Open-Meteo
  archive drive pothole formation with a ~2-week lag
* ageing: hazard drifts upward over time
* self-excitation: a failed asset is more likely to fail again soon (Hawkes)
* citizen reports: severe incidents get more reports; some incidents are never
  reported; ~12% of reports are noise not tied to any failure

Usage: python -m ml.risk.simulate [--incidents 11000] [--seed 7]
"""

from __future__ import annotations

import argparse
import uuid
from datetime import date, timedelta

import h3
import numpy as np
import pandas as pd
import psycopg

from app.config import get_settings
from app.db import query_df

CLASS_FACTOR = {
    "motorway": 1.5, "trunk": 1.5, "primary": 1.6, "secondary": 1.4, "tertiary": 1.2,
    "residential": 1.0, "unclassified": 0.9, "service": 0.7, "living_street": 0.6,
}
SIGN_FACTOR = {"traffic_signals": 0.7, "stop": 0.5, "give_way": 0.5, "crossing": 0.45}
ROAD_TYPES = ["potholes", "damaged_roads", "damaged_concrete"]
SIGN_TYPES = ["broken_road_signs", "graffiti"]
NOISE_TYPES = ["garbage", "graffiti", "illegal_parking", "fallen_trees", "dead_animals", "damaged_electric_poles"]
SEV_BETA = {
    "potholes": (5, 3), "damaged_roads": (3, 4), "damaged_concrete": (4, 3),
    "broken_road_signs": (4, 3), "graffiti": (2, 6),
}
N_USERS = 2500
M_PER_DEG_LAT = 111_320.0


def load_inputs(conn) -> tuple[pd.DataFrame, pd.DataFrame]:
    assets = query_df(conn, 
        "SELECT id, asset_type, subtype, length_m, ST_X(centroid) lon, ST_Y(centroid) lat FROM assets ORDER BY id",
    )
    weather = query_df(conn, "SELECT * FROM weather_daily ORDER BY day")
    weather["day"] = pd.to_datetime(weather["day"])
    return assets, weather


def fragility(assets: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    n = len(assets)
    is_road = (assets["asset_type"] == "road").to_numpy()
    sub = assets["subtype"].fillna("").str.replace("_link", "", regex=False).to_numpy()
    cls = np.array([
        CLASS_FACTOR.get(s, 0.9) if r else SIGN_FACTOR.get(s, 0.4) for s, r in zip(sub, is_road, strict=False)
    ])
    length = assets["length_m"].fillna(0).to_numpy()
    len_f = np.where(is_road, np.clip((length / 100.0) ** 0.6, 0.3, 4.0), 1.0)
    latent = rng.lognormal(0.0, 0.5, n)

    # Hotspots: Gaussian bumps of extra fragility around random assets
    lat0 = assets["lat"].mean()
    x = (assets["lon"].to_numpy() - assets["lon"].mean()) * M_PER_DEG_LAT * np.cos(np.radians(lat0))
    y = (assets["lat"].to_numpy() - lat0) * M_PER_DEG_LAT
    bump = np.zeros(n)
    for c in rng.choice(n, 15, replace=False):
        sigma, amp = rng.uniform(250, 700), rng.uniform(1.0, 3.0)
        bump += amp * np.exp(-((x - x[c]) ** 2 + (y - y[c]) ** 2) / (2 * sigma**2))
    return cls * len_f * latent * (1.0 + bump)


def weather_drivers(weather: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ft = ((weather["temp_min"] < 0) & (weather["temp_max"] > 0)).astype(float)
    ft_roll = ft.rolling(14, min_periods=1).mean().to_numpy()       # lagged freeze-thaw damage
    precip_roll = weather["precip_mm"].fillna(0).rolling(7, min_periods=1).mean().to_numpy()
    snow = (weather["snow_cm"].fillna(0) > 0).astype(float).to_numpy()
    g_road = 1.0 + 3.0 * ft_roll + 0.08 * precip_roll + 0.3 * snow
    return g_road, ft_roll


def run_hazard(base: float, frag: np.ndarray, is_road: np.ndarray, g_road: np.ndarray,
               rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n, days = len(frag), len(g_road)
    excite = np.zeros(n)
    decay = np.exp(-1 / 60)
    hit_asset, hit_day = [], []
    for d in range(days):
        age = 1.0 + 0.08 * d / 365.0
        g = np.where(is_road, g_road[d], 1.0 + 0.2 * (g_road[d] - 1.0))
        lam = base * frag * age * g * (1.0 + 1.5 * excite)
        hits = np.flatnonzero(rng.random(n) < -np.expm1(-lam))
        excite *= decay
        excite[hits] += 1.0
        hit_asset.append(hits)
        hit_day.append(np.full(len(hits), d))
    return np.concatenate(hit_asset), np.concatenate(hit_day)


def simulate(assets: pd.DataFrame, weather: pd.DataFrame, target_incidents: int, seed: int):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2020-01-01")
    weather = weather[weather["day"] >= start].reset_index(drop=True)
    days = weather["day"]
    is_road = (assets["asset_type"] == "road").to_numpy()
    frag = fragility(assets, rng)
    g_road, ft_roll = weather_drivers(weather)

    # Calibrate the base hazard to hit the target incident count
    base = target_incidents / (frag.sum() * len(days) * 2.5)
    for _ in range(3):
        a_idx, d_idx = run_hazard(base, frag, is_road, g_road, np.random.default_rng(seed))
        base *= target_incidents / max(len(a_idx), 1)
    a_idx, d_idx = run_hazard(base, frag, is_road, g_road, rng)

    # ---- incidents ----
    n_inc = len(a_idx)
    road_hit = is_road[a_idx]
    ftr = ft_roll[d_idx]
    types = np.empty(n_inc, dtype=object)
    for i in range(n_inc):
        if road_hit[i]:
            p = np.array([0.45 + 0.25 * ftr[i], 0.35 - 0.15 * ftr[i], 0.2 - 0.1 * ftr[i]])
            types[i] = ROAD_TYPES[rng.choice(3, p=p / p.sum())]
        else:
            types[i] = SIGN_TYPES[rng.choice(2, p=[0.7, 0.3])]
    sev = np.array([rng.beta(*SEV_BETA[t]) for t in types])
    created = days.to_numpy()[d_idx] + pd.to_timedelta(rng.uniform(6, 22, n_inc), unit="h").to_numpy()
    repair_days = rng.gamma(2.0, 10 + 20 * (1 - sev))
    resolved = created + pd.to_timedelta(repair_days, unit="D").to_numpy()
    end = days.iloc[-1] + pd.Timedelta(days=1)
    resolved = np.where(resolved <= end.to_datetime64(), resolved, np.datetime64("NaT"))
    incidents = pd.DataFrame({
        "id": np.arange(1, n_inc + 1),
        "asset_id": assets["id"].to_numpy()[a_idx],
        "issue_type": types, "severity": sev.round(3),
        "created_at": created, "resolved_at": resolved,
    })

    # ---- users (Zipf-distributed activity) ----
    user_ids = [f"sim|{i:04d}" for i in range(N_USERS)]
    weights = 1.0 / np.arange(1, N_USERS + 1) ** 0.75  # a few power users, long tail
    weights /= weights.sum()

    # ---- reports from incidents ----
    lon = assets["lon"].to_numpy()[a_idx]
    lat = assets["lat"].to_numpy()[a_idx]
    k = rng.poisson(1.2 + 1.5 * sev)
    src = np.repeat(np.arange(n_inc), k)
    n_rep = len(src)
    jitter_m = rng.normal(0, 15, (n_rep, 2))
    r_lat = lat[src] + jitter_m[:, 1] / M_PER_DEG_LAT
    r_lon = lon[src] + jitter_m[:, 0] / (M_PER_DEG_LAT * np.cos(np.radians(lat[src])))
    r_time = created[src] + pd.to_timedelta(rng.exponential(3.0, n_rep), unit="D").to_numpy()
    r_type = types[src].copy()
    confuse = rng.random(n_rep) < 0.15
    r_type[confuse] = rng.choice(ROAD_TYPES + SIGN_TYPES, confuse.sum())
    r_sev = np.clip(sev[src] + rng.normal(0, 0.1, n_rep), 0, 1)
    inc_res = resolved[src]
    r_status = np.where(~pd.isna(inc_res), "resolved", np.where(rng.random(n_rep) < 0.5, "open", "in_progress"))
    reports = pd.DataFrame({
        "lon": r_lon, "lat": r_lat, "created_at": r_time, "issue_type": r_type,
        "severity": r_sev.round(3), "status": r_status, "resolved_at": inc_res,
        "incident_id": incidents["id"].to_numpy()[src], "asset_id": incidents["asset_id"].to_numpy()[src],
    })

    # ---- noise reports (not tied to any failure) ----
    n_noise = int(0.12 * n_rep)
    pick = rng.integers(0, len(assets), n_noise)
    span = (end - start).total_seconds()
    noise = pd.DataFrame({
        "lon": assets["lon"].to_numpy()[pick] + rng.normal(0, 0.0003, n_noise),
        "lat": assets["lat"].to_numpy()[pick] + rng.normal(0, 0.0002, n_noise),
        "created_at": (start + pd.to_timedelta(rng.uniform(0, span, n_noise), unit="s")).to_numpy(),
        "issue_type": rng.choice(NOISE_TYPES, n_noise),
        "severity": rng.beta(2, 5, n_noise).round(3),
        "status": np.where(rng.random(n_noise) < 0.6, "rejected", "resolved"),
        "incident_id": pd.array([pd.NA] * n_noise, dtype="Int64"),
        "asset_id": pd.array([pd.NA] * n_noise, dtype="Int64"),
    })
    noise["resolved_at"] = noise["created_at"] + pd.to_timedelta(rng.gamma(2, 7, n_noise), unit="D")
    reports = pd.concat([reports, noise], ignore_index=True)
    reports = reports[reports["created_at"] < end.to_datetime64()].reset_index(drop=True)
    reports["user_id"] = rng.choice(user_ids, len(reports), p=weights)
    reports["id"] = [str(uuid.uuid4()) for _ in range(len(reports))]
    res = get_settings().h3_resolution
    reports["h3_cell"] = [h3.latlng_to_cell(a, o, res) for a, o in zip(reports["lat"], reports["lon"], strict=False)]

    # weather snapshot for each report (city-level daily)
    wx = weather.set_index("day")
    ft_daily = ((weather["temp_min"] < 0) & (weather["temp_max"] > 0)).astype(int)
    ft7 = pd.Series(ft_daily.rolling(7, min_periods=1).sum().shift(1).fillna(0).to_numpy(), index=wx.index)
    rd = pd.to_datetime(reports["created_at"]).dt.normalize()
    reports["temp"] = rd.map(wx["temp_mean"]).to_numpy()
    reports["humidity"] = rd.map(wx["humidity"]).to_numpy()
    reports["wind_speed"] = rd.map(wx["wind_max"]).to_numpy()
    reports["rainfall"] = rd.map(wx["precip_mm"]).to_numpy()
    reports["snowfall"] = rd.map(wx["snow_cm"]).to_numpy()
    reports["freeze_thaw_cycles"] = rd.map(ft7).fillna(0).astype(int).to_numpy()
    return incidents, reports, user_ids


def _none(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA or pd.isna(v) else v


def write(conn, incidents: pd.DataFrame, reports: pd.DataFrame, user_ids: list[str]) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM reports WHERE is_simulated")
        cur.execute("DELETE FROM incidents WHERE is_simulated")
        cur.execute("DELETE FROM users WHERE id LIKE 'sim|%%'")
        with cur.copy("COPY users (id, display_name) FROM STDIN") as cp:
            for u in user_ids:
                cp.write_row((u, f"resident-{u[4:]}"))
        offset = cur.execute("SELECT COALESCE(MAX(id), 0) FROM incidents").fetchone()[0]
        with cur.copy(
            "COPY incidents (id, asset_id, issue_type, severity, created_at, resolved_at, is_simulated) FROM STDIN"
        ) as cp:
            for r in incidents.itertuples():
                cp.write_row((r.id + offset, r.asset_id, r.issue_type, r.severity,
                              pd.Timestamp(r.created_at).tz_localize("UTC"),
                              _none(pd.Timestamp(r.resolved_at).tz_localize("UTC")
                                    if not pd.isna(r.resolved_at) else None), True))
        cur.execute("SELECT setval('incidents_id_seq', (SELECT MAX(id) FROM incidents))")
        cols = ("id, user_id, loc, h3_cell, issue_type, severity, status, created_at, resolved_at, asset_id,"
                " incident_id, temp, humidity, wind_speed, rainfall, snowfall, freeze_thaw_cycles, is_simulated")
        with cur.copy(f"COPY reports ({cols}) FROM STDIN") as cp:
            for r in reports.itertuples():
                cp.write_row((
                    r.id, r.user_id, f"SRID=4326;POINT({r.lon} {r.lat})", r.h3_cell, r.issue_type, r.severity,
                    r.status, pd.Timestamp(r.created_at).tz_localize("UTC"),
                    pd.Timestamp(r.resolved_at).tz_localize("UTC") if not pd.isna(r.resolved_at) else None,
                    _none(r.asset_id), (int(r.incident_id) + offset) if not pd.isna(r.incident_id) else None,
                    _none(r.temp), _none(r.humidity), _none(r.wind_speed), _none(r.rainfall), _none(r.snowfall),
                    int(r.freeze_thaw_cycles), True,
                ))
    conn.commit()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--incidents", type=int, default=11_000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    with psycopg.connect(get_settings().database_url) as conn:
        assets, weather = load_inputs(conn)
        weather = weather[weather["day"] <= pd.Timestamp(date.today() - timedelta(days=7))]
        incidents, reports, users = simulate(assets, weather, args.incidents, args.seed)
        write(conn, incidents, reports, users)
    print(f"Simulated {len(incidents):,} incidents + {len(reports):,} reports "
          f"= {len(incidents) + len(reports):,} events from {len(users)} users")


if __name__ == "__main__":
    main()
