import cv2
import numpy as np
import pytest

from app.routers import reports as reports_router
from app.services import classifier as classifier_mod

STAFF = {"X-Staff-Token": "test-staff-token"}


def _jpeg() -> bytes:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No network or TF model in tests: stub weather and the classifier."""
    async def fake_weather(lat, lon, when, client=None):
        return {"temp": -2.0, "humidity": 80.0, "wind_speed": 12.0, "rainfall": 1.5, "snowfall": 0.0,
                "freeze_thaw_cycles": 3, "weather_conditions": "light snow"}

    monkeypatch.setattr(reports_router, "get_weather", fake_weather)
    monkeypatch.setattr(classifier_mod.classifier, "predict",
                        lambda b: {"prediction": "potholes", "confidence": 0.9, "top3": []})


def _submit(client, lat=43.46425, lon=-80.52225, user="auth0|alice"):
    return client.post(
        "/api/reports",
        data={"latitude": lat, "longitude": lon, "user_id": user, "photo_timestamp": "2026-03-01T15:00:00Z"},
        files={"file": ("p.jpg", _jpeg(), "image/jpeg")},
    )


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["db"] is True


def test_report_matches_asset_and_opens_incident(client):
    r = _submit(client)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["matched_asset"]["id"] == 1
    assert body["incident_id"] is not None
    rep = body["report"]
    assert rep["issue_type"] == "potholes"
    assert 0 <= rep["severity"] <= 1
    assert body["severity_source"] == "heuristic"
    assert rep["status"] == "open"

    # A second report on the same asset joins the open incident instead of creating a new one
    r2 = _submit(client, user="auth0|bob")
    assert r2.json()["incident_id"] == body["incident_id"]


def test_report_far_from_assets_has_no_incident(client):
    r = _submit(client, lat=43.40, lon=-80.40)
    assert r.status_code == 200
    assert r.json()["matched_asset"] is None
    assert r.json()["incident_id"] is None


def test_rejects_non_image_and_bad_coords(client):
    r = client.post("/api/reports", data={"latitude": 43.4, "longitude": -80.5},
                    files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 415
    r = client.post("/api/reports", data={"latitude": 123, "longitude": -80.5},
                    files={"file": ("p.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 422


def test_list_and_get_reports(client):
    rows = client.get("/api/reports", params={"user_id": "auth0|alice"}).json()["data"]
    assert rows and all(r["latitude"] and r["longitude"] for r in rows)
    one = client.get(f"/api/reports/{rows[0]['id']}").json()["data"]
    assert one["freeze_thaw_cycles"] == 3
    assert client.get("/api/reports/not-a-uuid").status_code == 404


def test_status_workflow_requires_staff_and_valid_transition(client):
    rid = client.get("/api/reports", params={"user_id": "auth0|alice"}).json()["data"][0]["id"]
    assert client.patch(f"/api/reports/{rid}/status", json={"status": "resolved"}).status_code == 401
    r = client.patch(f"/api/reports/{rid}/status", json={"status": "in_progress"}, headers=STAFF)
    assert r.status_code == 200
    r = client.patch(f"/api/reports/{rid}/status", json={"status": "resolved"}, headers=STAFF)
    assert r.status_code == 200
    # resolved -> in_progress is not allowed (must reopen first)
    r = client.patch(f"/api/reports/{rid}/status", json={"status": "in_progress"}, headers=STAFF)
    assert r.status_code == 409


def test_stats_and_leaderboard(client):
    s = client.get("/api/stats").json()["data"]
    assert s["total_count"] >= 3
    assert s["resolved_this_week"] >= 1
    assert s["issue_type_counts"]["potholes"] >= 1
    lb = client.get("/api/leaderboard").json()["data"]
    assert lb[0]["rank"] == 1
    assert lb[0]["report_count"] >= lb[-1]["report_count"]


def test_heatmap_empty_before_scoring(client):
    r = client.get("/api/heatmap", params={"period": "30d"})
    assert r.status_code == 200
    assert r.json()["data"] == []
    assert client.get("/api/heatmap", params={"period": "bogus"}).status_code == 422
