"""Unit tests for the risk feature pipeline, especially the no-leakage guarantee."""

import numpy as np
import pandas as pd
import pytest

from ml.risk.features import HORIZONS, FeatureContext

SNAP = pd.Timestamp("2024-06-01")


def make_ctx(extra_incidents: list[tuple] = (), extra_reports: list[tuple] = ()) -> FeatureContext:
    assets = pd.DataFrame({
        "id": [1, 2, 3],
        "asset_type": ["road", "road", "sign"],
        "subtype": ["primary", "residential_link", "stop"],
        "length_m": [200.0, 80.0, None],
        "lanes": [2, None, None],
        "maxspeed": [60, None, None],
        "h3_cell": ["882b9b5a1dfffff", "882b9b5a1dfffff", "882b9b5a1bfffff"],
        "lon": [-80.5200, -80.5205, -80.5400],
        "lat": [43.4640, 43.4642, 43.4700],
    })
    inc = [
        (1, "potholes", 0.8, "2024-05-20", None),
        (1, "potholes", 0.6, "2023-12-01", "2023-12-20"),
        (2, "damaged_roads", 0.4, "2024-03-01", "2024-03-10"),
        (1, "potholes", 0.7, "2024-06-15", None),   # after SNAP: label only
        *extra_incidents,
    ]
    incidents = pd.DataFrame(inc, columns=["asset_id", "issue_type", "severity", "created_at", "resolved_at"])
    for c in ("created_at", "resolved_at"):
        incidents[c] = pd.to_datetime(incidents[c], utc=True)
    rep = [("2024-05-25", 0.7, "potholes", -80.52001, 43.46401), *extra_reports]
    reports = pd.DataFrame(rep, columns=["created_at", "severity", "issue_type", "lon", "lat"])
    reports["created_at"] = pd.to_datetime(reports["created_at"], utc=True)
    days = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    weather = pd.DataFrame({
        "day": days,
        "temp_min": np.where(days.month <= 3, -5.0, 10.0),
        "temp_max": np.where(days.month <= 3, 3.0, 20.0),
        "temp_mean": 5.0, "precip_mm": 1.0, "snow_cm": 0.0,
    })
    return FeatureContext(assets, incidents, reports, weather)


def test_has_100_plus_features_and_no_nans():
    X = make_ctx().build(SNAP)
    assert X.shape == (3, X.shape[1])
    assert X.shape[1] >= 100
    assert not X.isna().any().any()


def test_own_history_counts():
    X = make_ctx().build(SNAP)
    assert X.loc[0, "own_inc_30d"] == 1      # 2024-05-20
    assert X.loc[0, "own_inc_365d"] == 2     # + 2023-12-01
    assert X.loc[0, "own_open_inc"] == 1     # 2024-05-20 unresolved at SNAP
    assert X.loc[0, "own_days_since_inc"] == pytest.approx(12, abs=1)
    assert X.loc[2, "own_inc_total"] == 0


def test_neighbourhood_counts_respect_radius():
    X = make_ctx().build(SNAP)
    # asset 2 is ~45 m from asset 1: within 50 m and 100 m buffers
    assert X.loc[1, "nbr_inc_100m_365d"] == 3
    # the sign is ~2 km away: sees nothing
    assert X.loc[2, "nbr_inc_500m_365d"] == 0
    assert X.loc[0, "rep_50m_30d"] == 1


def test_no_future_leakage():
    """Adding events after the snapshot must not change any feature."""
    base = make_ctx().build(SNAP)
    future = make_ctx(
        extra_incidents=[(2, "potholes", 0.9, "2024-06-02", None), (3, "graffiti", 0.2, "2024-09-01", None)],
        extra_reports=[("2024-06-03", 0.9, "potholes", -80.5205, 43.4642)],
    ).build(SNAP)
    pd.testing.assert_frame_equal(base, future)


def test_labels():
    y = make_ctx().labels(SNAP)
    assert list(y.columns) == [f"fail_in_{h}d" for h in HORIZONS]
    assert y.loc[0, "fail_in_30d"] == 1   # 2024-06-15
    assert y.loc[1, "fail_in_365d"] == 0
    assert y.loc[2, "fail_in_365d"] == 0


def test_subtype_one_hot():
    X = make_ctx().build(SNAP)
    assert X.loc[0, "road_primary"] == 1
    assert X.loc[1, "road_link"] == 1      # *_link roads are grouped
    assert X.loc[2, "sign_stop"] == 1
    assert X.filter(like="road_").loc[2].sum() == 0
