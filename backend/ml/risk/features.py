"""Point-in-time feature engineering for asset failure-risk forecasting.

Design goals
------------
* **No leakage**: every feature for snapshot date S uses only events strictly
  before S. Labels use only events in (S, S + horizon].
* **Fast**: spatial neighbours are found once with a KD-tree (O(n log n)), then
  each snapshot is a handful of vectorised numpy masks + bincounts. The old
  implementation looped `GeoSeries.within(buffer)` per asset (O(assets x events)).
* **Shared**: training (ml/risk/train.py) and live scoring (app/services/risk.py)
  call the same `FeatureContext.build`, so train/serve features cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import h3
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

HORIZONS = (30, 90, 365)

OWN_WINDOWS = (30, 90, 180, 365, 730)
NBR_INC_RADII = (50, 100, 250, 500)
NBR_INC_WINDOWS = (30, 90, 365)
NBR_SEV_RADII = (100, 250, 500)
NBR_SEV_WINDOWS = (90, 365)
TYPE_RADIUS, TYPE_WINDOW = 250, 365
ISSUE_TYPES = ("potholes", "damaged_roads", "damaged_concrete", "broken_road_signs", "graffiti")
REP_RADII = (50, 100, 250)
REP_WINDOWS = (30, 90, 365)
REP_SEV_RADII = (100, 250)
REP_SEV_WINDOWS = (90, 365)
WEATHER_WINDOWS = (7, 30, 90, 180)
MAX_RADIUS = max(NBR_INC_RADII + REP_RADII)

ROAD_CLASSES = (
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "residential", "unclassified", "service", "living_street", "link",
)
SIGN_KINDS = ("stop", "traffic_signals", "give_way", "crossing", "other")
NO_EVENT_DAYS = 3650.0

_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32617", always_xy=True)  # UTM 17N (Ontario)


def _days(ts: pd.Series) -> np.ndarray:
    """Datetime series -> float days since epoch (NaN for missing)."""
    ts = pd.to_datetime(ts, utc=True)
    out = (ts - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy() / 86400.0
    return out.astype(float)


def _day(ts) -> float:
    return (pd.Timestamp(ts, tz="UTC") - pd.Timestamp("1970-01-01", tz="UTC")).total_seconds() / 86400.0


def normalise_subtype(asset_type: str, subtype: str | None) -> str:
    s = (subtype or "").lower()
    if asset_type == "road":
        if s.endswith("_link"):
            return "link"
        return s if s in ROAD_CLASSES else "unclassified"
    return s if s in SIGN_KINDS else "other"


def _pairs(tree_pts: np.ndarray, query_pts: np.ndarray, radius: float):
    """All (query_idx, point_idx, distance) pairs within radius."""
    if len(tree_pts) == 0:
        return np.empty(0, int), np.empty(0, int), np.empty(0, float)
    tree = cKDTree(tree_pts)
    hits = tree.query_ball_point(query_pts, r=radius)
    lens = np.fromiter((len(h) for h in hits), int, len(hits))
    qi = np.repeat(np.arange(len(query_pts)), lens)
    pj = np.fromiter((j for h in hits for j in h), int, lens.sum())
    d = np.linalg.norm(query_pts[qi] - tree_pts[pj], axis=1)
    return qi, pj, d


@dataclass
class FeatureContext:
    assets: pd.DataFrame
    incidents: pd.DataFrame
    reports: pd.DataFrame
    weather: pd.DataFrame

    def __post_init__(self) -> None:
        a = self.assets.reset_index(drop=True)
        self.assets = a
        self.n = len(a)
        self.asset_ids = a["id"].to_numpy()
        x, y = _to_utm.transform(a["lon"].to_numpy(), a["lat"].to_numpy())
        self.xy = np.column_stack([x, y])
        cx, cy = _to_utm.transform(a["lon"].mean(), a["lat"].mean())
        self.centre = np.array([cx, cy])
        id_to_idx = pd.Series(np.arange(self.n), index=self.asset_ids)

        # --- incidents: located at their asset ---
        inc = self.incidents
        inc = inc[inc["asset_id"].isin(id_to_idx.index)].reset_index(drop=True)
        self.inc_asset = id_to_idx.loc[inc["asset_id"]].to_numpy()
        self.inc_t = _days(inc["created_at"])
        self.inc_resolved = _days(inc["resolved_at"]) if "resolved_at" in inc else np.full(len(inc), np.nan)
        self.inc_sev = inc["severity"].astype(float).fillna(0.5).to_numpy()
        self.inc_type = inc["issue_type"].fillna("").to_numpy()
        inc_xy = self.xy[self.inc_asset] if len(inc) else np.empty((0, 2))
        self.ip_a, self.ip_j, self.ip_d = _pairs(inc_xy, self.xy, MAX_RADIUS)

        # --- reports: located at their GPS point ---
        rep = self.reports.reset_index(drop=True)
        rx, ry = _to_utm.transform(rep["lon"].to_numpy(), rep["lat"].to_numpy())
        rep_xy = np.column_stack([rx, ry]) if len(rep) else np.empty((0, 2))
        self.rep_t = _days(rep["created_at"])
        self.rep_sev = rep["severity"].astype(float).fillna(0.5).to_numpy()
        self.rp_a, self.rp_j, self.rp_d = _pairs(rep_xy, self.xy, MAX_RADIUS)

        # --- regions (H3) ---
        cells = a["h3_cell"].to_numpy()
        self.region_codes, self.region_of = np.unique(cells, return_inverse=True)
        code_index = {c: i for i, c in enumerate(self.region_codes)}
        self.ring = [
            np.array([code_index[n] for n in h3.grid_disk(c, 1) if n in code_index]) for c in self.region_codes
        ]
        self.assets_per_region = np.bincount(self.region_of, minlength=len(self.region_codes))

        # --- weather (daily, city level) ---
        w = self.weather.copy()
        w["d"] = np.floor(_days(pd.to_datetime(w["day"])))
        self.w = w.sort_values("d").reset_index(drop=True)

        self._static = self._static_features()

    # ------------------------------------------------------------------ static
    def _static_features(self) -> dict[str, np.ndarray]:
        a = self.assets
        f: dict[str, np.ndarray] = {}
        is_road = (a["asset_type"] == "road").to_numpy()
        f["is_road"] = is_road.astype(float)
        sub = [normalise_subtype(t, s) for t, s in zip(a["asset_type"], a["subtype"], strict=False)]
        sub = np.array(sub)
        for c in ROAD_CLASSES:
            f[f"road_{c}"] = (is_road & (sub == c)).astype(float)
        for c in SIGN_KINDS:
            f[f"sign_{c}"] = (~is_road & (sub == c)).astype(float)
        length = a["length_m"].astype(float).fillna(0).to_numpy()
        f["length_m"] = length
        f["log_length"] = np.log1p(length)
        f["lanes"] = a["lanes"].astype(float).fillna(1).to_numpy()
        f["maxspeed"] = a["maxspeed"].astype(float).fillna(50).to_numpy()
        rel = (self.xy - self.centre) / 1000.0
        f["x_km"] = rel[:, 0]
        f["y_km"] = rel[:, 1]
        f["dist_centre_km"] = np.hypot(rel[:, 0], rel[:, 1])
        f["region_asset_count"] = self.assets_per_region[self.region_of].astype(float)
        return f

    # ---------------------------------------------------------------- dynamic
    def _bincount(self, idx: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        return np.bincount(idx, weights=weights, minlength=self.n).astype(float)

    def build(self, snapshot) -> pd.DataFrame:
        """Feature matrix (one row per asset) as of `snapshot` (exclusive)."""
        s = _day(snapshot)
        f: dict[str, np.ndarray] = dict(self._static)

        # Own incident history
        age = s - self.inc_t
        past = age > 0
        for w in OWN_WINDOWS:
            f[f"own_inc_{w}d"] = self._bincount(self.inc_asset[past & (age <= w)])
        f["own_inc_total"] = self._bincount(self.inc_asset[past])
        last = np.full(self.n, -np.inf)
        np.maximum.at(last, self.inc_asset[past], self.inc_t[past])
        f["own_days_since_inc"] = np.where(np.isfinite(last), s - last, NO_EVENT_DAYS)
        m365 = past & (age <= 365)
        cnt365 = f["own_inc_365d"]
        f["own_sev_mean_365d"] = np.divide(
            self._bincount(self.inc_asset[m365], self.inc_sev[m365]), cnt365,
            out=np.zeros(self.n), where=cnt365 > 0,
        )
        mx = np.zeros(self.n)
        np.maximum.at(mx, self.inc_asset[m365], self.inc_sev[m365])
        f["own_sev_max_365d"] = mx
        open_now = past & ~(self.inc_resolved <= s)  # NaN resolved -> still open
        f["own_open_inc"] = self._bincount(self.inc_asset[open_now])
        f["own_trend_90_vs_365"] = f["own_inc_90d"] - f["own_inc_365d"] / 4.0

        # Neighbourhood incidents (KD-tree pairs, masked by time + distance)
        page = s - self.inc_t[self.ip_j]
        ppast = page > 0
        for w in NBR_INC_WINDOWS:
            mw = ppast & (page <= w)
            for r in NBR_INC_RADII:
                f[f"nbr_inc_{r}m_{w}d"] = self._bincount(self.ip_a[mw & (self.ip_d <= r)])
        for w in NBR_SEV_WINDOWS:
            mw = ppast & (page <= w)
            for r in NBR_SEV_RADII:
                m = mw & (self.ip_d <= r)
                c = self._bincount(self.ip_a[m])
                sv = self._bincount(self.ip_a[m], self.inc_sev[self.ip_j[m]])
                f[f"nbr_sev_{r}m_{w}d"] = np.divide(sv, c, out=np.zeros(self.n), where=c > 0)
        mt = ppast & (page <= TYPE_WINDOW) & (self.ip_d <= TYPE_RADIUS)
        types = self.inc_type[self.ip_j]
        for t in ISSUE_TYPES:
            f[f"nbr_{t}_{TYPE_RADIUS}m_{TYPE_WINDOW}d"] = self._bincount(self.ip_a[mt & (types == t)])

        # Citizen reports nearby
        rage = s - self.rep_t[self.rp_j]
        rpast = rage > 0
        for w in REP_WINDOWS:
            mw = rpast & (rage <= w)
            for r in REP_RADII:
                f[f"rep_{r}m_{w}d"] = self._bincount(self.rp_a[mw & (self.rp_d <= r)])
        for w in REP_SEV_WINDOWS:
            mw = rpast & (rage <= w)
            for r in REP_SEV_RADII:
                m = mw & (self.rp_d <= r)
                c = self._bincount(self.rp_a[m])
                sv = self._bincount(self.rp_a[m], self.rep_sev[self.rp_j[m]])
                f[f"rep_sev_{r}m_{w}d"] = np.divide(sv, c, out=np.zeros(self.n), where=c > 0)
        m = rpast & (self.rp_d <= 100)
        lastr = np.full(self.n, -np.inf)
        np.maximum.at(lastr, self.rp_a[m], self.rep_t[self.rp_j[m]])
        f["days_since_rep_100m"] = np.where(np.isfinite(lastr), s - lastr, NO_EVENT_DAYS)

        # Region (H3) aggregates
        nreg = len(self.region_codes)
        for w in (90, 365):
            reg = np.bincount(self.region_of, weights=f[f"own_inc_{w}d"], minlength=nreg)
            f[f"region_inc_{w}d"] = reg[self.region_of]
        reg365 = np.bincount(self.region_of, weights=f["own_inc_365d"], minlength=nreg)
        f["region_inc_per_asset_365d"] = (reg365 / np.maximum(self.assets_per_region, 1))[self.region_of]
        ring_sum = np.array([reg365[r].sum() for r in self.ring])
        f["region_ring_inc_365d"] = ring_sum[self.region_of]
        reg_rep = np.bincount(self.region_of, weights=f["rep_50m_90d"], minlength=nreg)
        f["region_rep_90d"] = reg_rep[self.region_of]

        # Weather (city level): rolling windows ending the day before the snapshot
        wd = self.w
        for w in WEATHER_WINDOWS:
            sl = wd[(wd["d"] >= np.floor(s) - w) & (wd["d"] < np.floor(s))]
            ft = float(((sl["temp_min"] < 0) & (sl["temp_max"] > 0)).sum())
            vals = {
                "precip": float(sl["precip_mm"].sum()),
                "snow": float(sl["snow_cm"].sum()),
                "freeze_thaw": ft,
                "temp_mean": float(sl["temp_mean"].mean()) if len(sl) else 0.0,
                "temp_min": float(sl["temp_min"].min()) if len(sl) else 0.0,
            }
            for k, v in vals.items():
                f[f"wx_{k}_{w}d"] = np.full(self.n, v if np.isfinite(v) else 0.0)
        for w in (30, 90):
            sl = wd[(wd["d"] >= np.floor(s) - w) & (wd["d"] < np.floor(s))]
            f[f"wx_days_below_zero_{w}d"] = np.full(self.n, float((sl["temp_mean"] < 0).sum()))
        month = pd.Timestamp(snapshot).month
        f["season_sin"] = np.full(self.n, np.sin(2 * np.pi * month / 12))
        f["season_cos"] = np.full(self.n, np.cos(2 * np.pi * month / 12))

        # Interactions the trees would otherwise need many splits to find
        f["ft90_x_own365"] = f["wx_freeze_thaw_90d"] * f["own_inc_365d"]
        f["ft90_x_road"] = f["wx_freeze_thaw_90d"] * f["is_road"]
        f["precip30_x_nbr100_365"] = f["wx_precip_30d"] * f["nbr_inc_100m_365d"]

        return pd.DataFrame(f)

    def labels(self, snapshot) -> pd.DataFrame:
        """fail_in_{h}d = asset has >= 1 incident in (snapshot, snapshot + h]."""
        s = _day(snapshot)
        ahead = self.inc_t - s
        out = {}
        for h in HORIZONS:
            m = (ahead > 0) & (ahead <= h)
            out[f"fail_in_{h}d"] = (self._bincount(self.inc_asset[m]) > 0).astype(int)
        return pd.DataFrame(out)


def feature_names(ctx: FeatureContext) -> list[str]:
    return list(ctx.build(pd.Timestamp("2000-01-01")).columns)
