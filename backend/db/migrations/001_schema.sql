-- CityWatch core schema (PostgreSQL 15+ with PostGIS 3+).
-- Works on a local Postgres or on Supabase (both ship PostGIS).

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- gen_random_uuid()

-- ---------------------------------------------------------------------------
-- Users (id is the Auth0 `sub`)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    email         TEXT,
    display_name  TEXT,
    role          TEXT NOT NULL DEFAULT 'resident' CHECK (role IN ('resident', 'staff')),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Assets: road segments and traffic signs (from OpenStreetMap)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS assets (
    id          BIGINT PRIMARY KEY,
    asset_type  TEXT NOT NULL CHECK (asset_type IN ('road', 'sign')),
    subtype     TEXT,                 -- road: highway class; sign: sign kind
    name        TEXT,
    osm_id      TEXT,
    length_m    REAL,                 -- roads only
    lanes       SMALLINT,
    maxspeed    SMALLINT,
    geom        geometry(Geometry, 4326) NOT NULL,
    centroid    geometry(Point, 4326) NOT NULL,
    h3_cell     TEXT NOT NULL         -- H3 resolution-8 region id
);
CREATE INDEX IF NOT EXISTS assets_geom_gix     ON assets USING GIST (geom);
CREATE INDEX IF NOT EXISTS assets_geog_gix     ON assets USING GIST ((geom::geography));
CREATE INDEX IF NOT EXISTS assets_h3_idx       ON assets (h3_cell);

-- ---------------------------------------------------------------------------
-- Incidents: confirmed failures on an asset (one per failure event)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS incidents (
    id           BIGSERIAL PRIMARY KEY,
    asset_id     BIGINT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    issue_type   TEXT,
    severity     REAL CHECK (severity BETWEEN 0 AND 1),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at  TIMESTAMPTZ,
    is_simulated BOOLEAN NOT NULL DEFAULT false
);
CREATE INDEX IF NOT EXISTS incidents_asset_time_idx ON incidents (asset_id, created_at);
CREATE INDEX IF NOT EXISTS incidents_open_idx ON incidents (asset_id) WHERE resolved_at IS NULL;

-- ---------------------------------------------------------------------------
-- Reports: citizen photo reports
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS reports (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             TEXT REFERENCES users(id) ON DELETE SET NULL,
    image_url           TEXT,
    loc                 geometry(Point, 4326) NOT NULL,
    h3_cell             TEXT NOT NULL,
    issue_type          TEXT,
    cv_confidence       REAL,
    severity            REAL CHECK (severity BETWEEN 0 AND 1),
    status              TEXT NOT NULL DEFAULT 'open'
                        CHECK (status IN ('open', 'in_progress', 'resolved', 'rejected')),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at         TIMESTAMPTZ,
    asset_id            BIGINT REFERENCES assets(id) ON DELETE SET NULL,
    incident_id         BIGINT REFERENCES incidents(id) ON DELETE SET NULL,
    temp                REAL,
    humidity            REAL,
    wind_speed          REAL,
    rainfall            REAL,
    snowfall            REAL,
    freeze_thaw_cycles  SMALLINT,
    weather_conditions  TEXT,
    is_simulated        BOOLEAN NOT NULL DEFAULT false
);
CREATE INDEX IF NOT EXISTS reports_loc_gix      ON reports USING GIST ((loc::geography));
CREATE INDEX IF NOT EXISTS reports_created_idx  ON reports (created_at DESC);
CREATE INDEX IF NOT EXISTS reports_user_idx     ON reports (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS reports_status_idx   ON reports (status);
CREATE INDEX IF NOT EXISTS reports_h3_idx       ON reports (h3_cell);

-- ---------------------------------------------------------------------------
-- City-level daily weather (Open-Meteo archive)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS weather_daily (
    day         DATE PRIMARY KEY,
    temp_min    REAL,
    temp_max    REAL,
    temp_mean   REAL,
    precip_mm   REAL,
    snow_cm     REAL,
    humidity    REAL,
    wind_max    REAL
);

-- ---------------------------------------------------------------------------
-- Model outputs
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS asset_risk (
    asset_id    BIGINT PRIMARY KEY REFERENCES assets(id) ON DELETE CASCADE,
    risk_30d    REAL NOT NULL,
    risk_90d    REAL NOT NULL,
    risk_365d   REAL NOT NULL,
    as_of       DATE NOT NULL,
    model_version TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS region_risk (
    h3_cell       TEXT PRIMARY KEY,
    centroid      geometry(Point, 4326) NOT NULL,
    asset_count   INTEGER NOT NULL,
    open_issues   INTEGER NOT NULL,
    avg_severity  REAL NOT NULL,
    risk_30d      REAL NOT NULL,
    risk_90d      REAL NOT NULL,
    risk_365d     REAL NOT NULL,
    as_of         DATE NOT NULL,
    model_version TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- Nearest asset within radius (uses the geography GIST index + KNN ordering)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION find_nearest_asset(lon_in DOUBLE PRECISION, lat_in DOUBLE PRECISION, radius_m DOUBLE PRECISION)
RETURNS TABLE (id BIGINT, asset_type TEXT, distance_m DOUBLE PRECISION)
LANGUAGE sql STABLE AS $$
    SELECT a.id, a.asset_type,
           ST_Distance(a.geom::geography, ST_SetSRID(ST_MakePoint(lon_in, lat_in), 4326)::geography) AS distance_m
    FROM assets a
    WHERE ST_DWithin(a.geom::geography, ST_SetSRID(ST_MakePoint(lon_in, lat_in), 4326)::geography, radius_m)
    ORDER BY a.geom <-> ST_SetSRID(ST_MakePoint(lon_in, lat_in), 4326)
    LIMIT 1;
$$;
