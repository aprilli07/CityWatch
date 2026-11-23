"""Backfill city-level daily weather from the Open-Meteo archive (free, no key).

Usage: python -m scripts.backfill_weather [start=2019-06-01] [end=today-6d]
"""

import sys
from datetime import date, timedelta

import httpx
import psycopg

from app.config import get_settings

URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY = (
    "temperature_2m_min,temperature_2m_max,temperature_2m_mean,precipitation_sum,"
    "snowfall_sum,relative_humidity_2m_mean,wind_speed_10m_max"
)


def fetch(start: str, end: str) -> list[tuple]:
    s = get_settings()
    r = httpx.get(URL, params={
        "latitude": s.city_lat, "longitude": s.city_lon, "start_date": start, "end_date": end,
        "daily": DAILY, "timezone": "America/Toronto",
    }, timeout=60)
    r.raise_for_status()
    d = r.json()["daily"]
    return list(zip(
        d["time"], d["temperature_2m_min"], d["temperature_2m_max"], d["temperature_2m_mean"],
        d["precipitation_sum"], d["snowfall_sum"], d["relative_humidity_2m_mean"],
        d["wind_speed_10m_max"], strict=True,
    ))


def main(start: str, end: str) -> int:
    rows = [r for r in fetch(start, end) if r[1] is not None]
    with psycopg.connect(get_settings().database_url) as conn, conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO weather_daily (day, temp_min, temp_max, temp_mean, precip_mm, snow_cm, humidity, wind_max)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (day) DO UPDATE SET"
            " temp_min=EXCLUDED.temp_min, temp_max=EXCLUDED.temp_max, temp_mean=EXCLUDED.temp_mean,"
            " precip_mm=EXCLUDED.precip_mm, snow_cm=EXCLUDED.snow_cm, humidity=EXCLUDED.humidity,"
            " wind_max=EXCLUDED.wind_max",
            rows,
        )
    return len(rows)


if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else "2019-06-01"
    end = sys.argv[2] if len(sys.argv) > 2 else (date.today() - timedelta(days=6)).isoformat()
    print(f"Stored {main(start, end)} days of weather ({start} .. {end})")
