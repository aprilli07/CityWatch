"""Load the frames FeatureContext needs from Postgres."""

import pandas as pd

from app.db import query_df
from ml.risk.features import FeatureContext


def load_context(conn) -> FeatureContext:
    assets = query_df(conn, """
        SELECT id, asset_type, subtype, length_m, lanes, maxspeed, h3_cell,
               ST_X(centroid) AS lon, ST_Y(centroid) AS lat
        FROM assets ORDER BY id""")
    incidents = query_df(conn, "SELECT asset_id, issue_type, severity, created_at, resolved_at FROM incidents")
    reports = query_df(conn, """
        SELECT created_at, severity, issue_type, ST_X(loc) AS lon, ST_Y(loc) AS lat
        FROM reports WHERE status <> 'rejected'""")
    weather = query_df(conn, "SELECT day, temp_min, temp_max, temp_mean, precip_mm, snow_cm FROM weather_daily")
    for df, cols in ((incidents, ["created_at", "resolved_at"]), (reports, ["created_at"])):
        for c in cols:
            df[c] = pd.to_datetime(df[c], utc=True)
    return FeatureContext(assets, incidents, reports, weather)
