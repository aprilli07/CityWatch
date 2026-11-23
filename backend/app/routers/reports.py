import asyncio
from datetime import UTC, datetime
from typing import Literal

import h3
import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app.config import get_settings
from app.db import connection, fetch_all, fetch_one
from app.deps import require_staff
from app.routers.insights import invalidate_cache
from app.services import risk, storage
from app.services.classifier import classifier
from app.services.severity import score_severity
from app.services.weather import get_weather

router = APIRouter(prefix="/api/reports", tags=["reports"])

# Report types that represent failure of a physical asset (and so open an incident).
ASSET_FAILURE_TYPES = {
    "potholes", "damaged_roads", "damaged_concrete",
    "broken_road_signs", "graffiti", "damaged_electric_poles",
}
Status = Literal["open", "in_progress", "resolved", "rejected"]
TRANSITIONS: dict[str, set[str]] = {
    "open": {"in_progress", "resolved", "rejected"},
    "in_progress": {"resolved", "rejected", "open"},
    "resolved": {"open"},
    "rejected": {"open"},
}
REPORT_COLUMNS = """
    r.id, r.user_id, r.image_url, r.issue_type, r.cv_confidence, r.severity, r.status,
    r.created_at, r.resolved_at, r.asset_id, r.incident_id, r.h3_cell,
    ST_Y(r.loc) AS latitude, ST_X(r.loc) AS longitude
"""


def _parse_ts(value: str | None) -> datetime:
    if not value:
        return datetime.now(UTC)
    try:
        ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as e:
        raise HTTPException(422, f"invalid photo_timestamp: {value}") from e
    ts = ts if ts.tzinfo else ts.replace(tzinfo=UTC)
    return min(ts, datetime.now(UTC))


@router.post("")
async def create_report(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    latitude: float = Form(..., ge=-90, le=90),
    longitude: float = Form(..., ge=-180, le=180),
    photo_timestamp: str | None = Form(None),
    user_id: str | None = Form(None),
):
    s = get_settings()
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(415, "file must be an image")
    img = await file.read()
    if not img:
        raise HTTPException(422, "empty file")
    if len(img) > s.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"image larger than {s.max_upload_mb} MB")
    when = _parse_ts(photo_timestamp)

    # Classification (CPU, thread pool), weather and upload run concurrently.
    async with httpx.AsyncClient(timeout=s.external_timeout_s) as client:
        cv, weather, image_url = await asyncio.gather(
            run_in_threadpool(classifier.predict, img),
            get_weather(latitude, longitude, when, client),
            storage.save_image(img, user_id, file.content_type or "image/jpeg"),
        )
    severity, severity_source = await score_severity(img, cv["prediction"], cv["confidence"])
    cell = h3.latlng_to_cell(latitude, longitude, s.h3_resolution)

    with connection() as conn, conn.transaction():
        if user_id:
            conn.execute("INSERT INTO users (id) VALUES (%s) ON CONFLICT DO NOTHING", (user_id,))
        asset = conn.execute(
            "SELECT id, asset_type, distance_m FROM find_nearest_asset(%s, %s, %s)",
            (longitude, latitude, s.asset_match_radius_m),
        ).fetchone()
        incident_id = None
        if asset and cv["prediction"] in ASSET_FAILURE_TYPES:
            open_inc = conn.execute(
                "SELECT id FROM incidents WHERE asset_id = %s AND resolved_at IS NULL"
                " ORDER BY created_at DESC LIMIT 1",
                (asset["id"],),
            ).fetchone()
            if open_inc:
                incident_id = open_inc["id"]
            else:
                incident_id = conn.execute(
                    "INSERT INTO incidents (asset_id, issue_type, severity, created_at)"
                    " VALUES (%s, %s, %s, %s) RETURNING id",
                    (asset["id"], cv["prediction"], severity, when),
                ).fetchone()["id"]
        report = conn.execute(
            f"""INSERT INTO reports AS r (user_id, image_url, loc, h3_cell, issue_type, cv_confidence, severity,
                    created_at, asset_id, incident_id, temp, humidity, wind_speed, rainfall, snowfall,
                    freeze_thaw_cycles, weather_conditions)
                VALUES (%(user_id)s, %(image_url)s, ST_SetSRID(ST_MakePoint(%(lon)s, %(lat)s), 4326), %(cell)s,
                    %(issue_type)s, %(conf)s, %(severity)s, %(when)s, %(asset_id)s, %(incident_id)s, %(temp)s,
                    %(humidity)s, %(wind_speed)s, %(rainfall)s, %(snowfall)s, %(freeze_thaw_cycles)s,
                    %(weather_conditions)s)
                RETURNING {REPORT_COLUMNS}""",
            {
                "user_id": user_id, "image_url": image_url, "lon": longitude, "lat": latitude, "cell": cell,
                "issue_type": cv["prediction"], "conf": cv["confidence"], "severity": severity, "when": when,
                "asset_id": asset["id"] if asset else None, "incident_id": incident_id, **weather,
            },
        ).fetchone()

    invalidate_cache()
    if s.risk_model_path.exists():
        background.add_task(risk.request_refresh)
    return {
        "success": True,
        "report": report,
        "incident_id": incident_id,
        "classification": cv,
        "severity_source": severity_source,
        "matched_asset": asset,
        "weather": weather,
    }


@router.get("")
def list_reports(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    user_id: str | None = None,
    status: Status | None = None,
    h3_cell: str | None = None,
):
    where, params = [], {"limit": limit, "offset": offset}
    for col, val in (("user_id", user_id), ("status", status), ("h3_cell", h3_cell)):
        if val is not None:
            where.append(f"r.{col} = %({col})s")
            params[col] = val
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    rows = fetch_all(
        f"SELECT {REPORT_COLUMNS} FROM reports r {clause} ORDER BY r.created_at DESC"
        " LIMIT %(limit)s OFFSET %(offset)s",
        params,
    )
    return {"success": True, "data": rows}


@router.get("/{report_id}")
def get_report(report_id: str):
    try:
        row = fetch_one(
            f"SELECT {REPORT_COLUMNS}, r.temp, r.humidity, r.wind_speed, r.rainfall, r.snowfall,"
            " r.freeze_thaw_cycles, r.weather_conditions FROM reports r WHERE r.id = %s::uuid",
            (report_id,),
        )
    except Exception as e:
        raise HTTPException(404, "report not found") from e
    if not row:
        raise HTTPException(404, "report not found")
    return {"success": True, "data": row}


class StatusUpdate(BaseModel):
    status: Status


@router.patch("/{report_id}/status", dependencies=[Depends(require_staff)])
def update_status(report_id: str, body: StatusUpdate, background: BackgroundTasks):
    with connection() as conn, conn.transaction():
        row = conn.execute(
            "SELECT status, incident_id FROM reports WHERE id = %s::uuid FOR UPDATE", (report_id,)
        ).fetchone()
        if not row:
            raise HTTPException(404, "report not found")
        if body.status != row["status"] and body.status not in TRANSITIONS[row["status"]]:
            raise HTTPException(409, f"cannot move report from {row['status']} to {body.status}")
        resolved = body.status == "resolved"
        conn.execute(
            "UPDATE reports SET status = %s, resolved_at = CASE WHEN %s THEN now() ELSE NULL END"
            " WHERE id = %s::uuid",
            (body.status, resolved, report_id),
        )
        if row["incident_id"]:
            # An incident is resolved once none of its reports are still open.
            conn.execute(
                """UPDATE incidents SET resolved_at = CASE
                       WHEN EXISTS (SELECT 1 FROM reports WHERE incident_id = %(i)s
                                    AND status IN ('open', 'in_progress')) THEN NULL
                       ELSE COALESCE(resolved_at, now()) END
                   WHERE id = %(i)s""",
                {"i": row["incident_id"]},
            )
    invalidate_cache()
    if get_settings().risk_model_path.exists():
        background.add_task(risk.request_refresh)
    return {"success": True, "id": report_id, "status": body.status}
