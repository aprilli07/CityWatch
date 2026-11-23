"""Load real road segments and traffic-control signs from OpenStreetMap into `assets`.

Usage: python -m scripts.ingest_osm ["Waterloo, Ontario, Canada" ...]
"""

import json
import sys
import time

import h3
import osmnx as ox
import psycopg
from pyproj import Geod

from app.config import get_settings

DEFAULT_PLACES = ["Waterloo, Ontario, Canada"]
SIGN_TAGS = {"highway": ["stop", "give_way", "traffic_signals", "crossing"], "traffic_sign": True}
_geod = Geod(ellps="WGS84")


def _first(v):
    return v[0] if isinstance(v, list) else v


def _int_or_none(v):
    v = _first(v)
    try:
        return int(str(v).split()[0])
    except (TypeError, ValueError):
        return None


OVERPASS_MIRRORS = [
    "https://overpass-api.de/api",
    "https://overpass.kumi.systems/api",
    "https://maps.mail.ru/osm/tools/overpass/api",
    "https://overpass.private.coffee/api",
]


def with_mirrors(fn, *args, **kwargs):
    """Retry an osmnx call across public Overpass mirrors (they are often at capacity)."""
    last = None
    for attempt in range(2):
        for url in OVERPASS_MIRRORS:
            ox.settings.overpass_url = url
            try:
                return fn(*args, **kwargs)
            except Exception as e:  # noqa: BLE001 - any mirror failure -> try the next
                last = e
                print(f"  {url} failed ({type(e).__name__}); trying next mirror")
                time.sleep(2 * (attempt + 1))
    raise RuntimeError("all Overpass mirrors failed") from last


def load_assets(places: list[str]) -> list[tuple]:
    res = get_settings().h3_resolution
    G = with_mirrors(ox.graph_from_place, places, network_type="drive")
    G = ox.convert.to_undirected(G)  # one row per physical segment, not per direction
    edges = ox.graph_to_gdfs(G, nodes=False).reset_index()

    rows: list[tuple] = []
    for i, e in enumerate(edges.itertuples()):
        geom = e.geometry
        mid = geom.interpolate(0.5, normalized=True)
        length = abs(_geod.geometry_length(geom))
        rows.append((
            1_000_000 + i, "road", _first(getattr(e, "highway", None)), _first(getattr(e, "name", None)),
            str(_first(getattr(e, "osmid", ""))), length, _int_or_none(getattr(e, "lanes", None)),
            _int_or_none(getattr(e, "maxspeed", None)), json.dumps(geom.__geo_interface__),
            mid.x, mid.y, h3.latlng_to_cell(mid.y, mid.x, res),
        ))

    signs = with_mirrors(ox.features_from_place, places, tags=SIGN_TAGS)
    signs = signs[signs.geometry.geom_type == "Point"].reset_index()
    for i, s in enumerate(signs.itertuples()):
        kind = getattr(s, "highway", None)
        if not isinstance(kind, str):
            kind = "other"
        p = s.geometry
        rows.append((
            2_000_000 + i, "sign", kind, None, str(getattr(s, "id", "")), None, None, None,
            json.dumps(p.__geo_interface__), p.x, p.y, h3.latlng_to_cell(p.y, p.x, res),
        ))
    return rows


def write_assets(rows: list[tuple]) -> None:
    with psycopg.connect(get_settings().database_url) as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE assets CASCADE")
        with cur.copy(
            "COPY assets (id, asset_type, subtype, name, osm_id, length_m, lanes, maxspeed, geom, centroid, h3_cell)"
            " FROM STDIN"
        ) as cp:
            for r in rows:
                (aid, typ, sub, name, osm, length, lanes, speed, gj, lon, lat, cell) = r
                cp.write_row((aid, typ, sub, name, osm, length, lanes, speed,
                              f"SRID=4326;{_geojson_to_wkt(gj)}", f"SRID=4326;POINT({lon} {lat})", cell))


def _geojson_to_wkt(gj: str) -> str:
    from shapely.geometry import shape

    return shape(json.loads(gj)).wkt


if __name__ == "__main__":
    places = sys.argv[1:] or DEFAULT_PLACES
    rows = load_assets(places)
    write_assets(rows)
    roads = sum(1 for r in rows if r[1] == "road")
    cells = len({r[-1] for r in rows})
    print(f"Loaded {roads} road segments, {len(rows) - roads} signs across {cells} H3 regions")
