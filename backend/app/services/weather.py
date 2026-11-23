"""Weather at report time via Open-Meteo (free, no API key).

One HTTP call returns the report day plus the 7 days before it, which is all we
need for the point-in-time conditions and the freeze-thaw cycle count. The old
OpenWeather integration needed 9 paid calls per report.
"""

import logging
from datetime import UTC, date, datetime, timedelta

import httpx

from app.config import get_settings

log = logging.getLogger(__name__)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
ARCHIVE_LAG_DAYS = 5  # the archive trails real time by a few days

EMPTY = {
    "temp": None,
    "humidity": None,
    "wind_speed": None,
    "rainfall": None,
    "snowfall": None,
    "freeze_thaw_cycles": None,
    "weather_conditions": None,
}

# WMO weather codes -> short description
WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "rime fog", 51: "light drizzle", 53: "drizzle", 55: "dense drizzle",
    56: "freezing drizzle", 57: "freezing drizzle", 61: "light rain", 63: "rain",
    65: "heavy rain", 66: "freezing rain", 67: "freezing rain", 71: "light snow",
    73: "snow", 75: "heavy snow", 77: "snow grains", 80: "rain showers", 81: "rain showers",
    82: "violent rain showers", 85: "snow showers", 86: "snow showers", 95: "thunderstorm",
    96: "thunderstorm with hail", 99: "thunderstorm with hail",
}

_cache: dict[tuple, dict] = {}


def count_freeze_thaw(mins: list, maxs: list) -> int:
    return sum(1 for lo, hi in zip(mins, maxs, strict=False) if lo is not None and hi is not None and lo < 0 < hi)


def parse_response(data: dict, when: datetime) -> dict:
    daily = data["daily"]
    hourly = data["hourly"]
    target = when.date().isoformat()
    day_idx = daily["time"].index(target)
    # freeze-thaw over the 7 days *before* the report day
    ft = count_freeze_thaw(daily["temperature_2m_min"][:day_idx], daily["temperature_2m_max"][:day_idx])
    hour_key = when.strftime("%Y-%m-%dT%H:00")
    h = hourly["time"].index(hour_key) if hour_key in hourly["time"] else None

    def hv(name):
        return hourly[name][h] if h is not None else None

    code = hv("weather_code")
    return {
        "temp": hv("temperature_2m"),
        "humidity": hv("relative_humidity_2m"),
        "wind_speed": hv("wind_speed_10m"),
        "rainfall": daily["precipitation_sum"][day_idx] or 0.0,
        "snowfall": daily["snowfall_sum"][day_idx] or 0.0,
        "freeze_thaw_cycles": ft,
        "weather_conditions": WMO.get(code) if code is not None else None,
    }


async def get_weather(lat: float, lon: float, when: datetime, client: httpx.AsyncClient | None = None) -> dict:
    when = when.astimezone(UTC)
    key = (round(lat, 2), round(lon, 2), when.strftime("%Y-%m-%dT%H"))
    if key in _cache:
        return _cache[key]

    start = (when.date() - timedelta(days=7)).isoformat()
    end = when.date().isoformat()
    use_archive = when.date() < date.today() - timedelta(days=ARCHIVE_LAG_DAYS)
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "timezone": "UTC",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,snowfall_sum",
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
    }
    try:
        owns = client is None
        client = client or httpx.AsyncClient(timeout=get_settings().external_timeout_s)
        try:
            r = await client.get(ARCHIVE_URL if use_archive else FORECAST_URL, params=params)
            r.raise_for_status()
            result = parse_response(r.json(), when)
        finally:
            if owns:
                await client.aclose()
    except Exception as e:  # weather is enrichment; never fail a report over it
        log.warning("Weather lookup failed: %s", e)
        return dict(EMPTY)

    if len(_cache) > 2048:
        _cache.clear()
    _cache[key] = result
    return result
