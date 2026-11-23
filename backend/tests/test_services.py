from datetime import UTC, datetime

import cv2
import numpy as np
import pytest

from app.services.severity import heuristic_severity, parse_llm_severity
from app.services.weather import count_freeze_thaw, parse_response
from ml.cv.preprocess import IMG_SIZE, ImageRejected, average_hash, is_near_duplicate, load_and_prepare


def test_freeze_thaw_count():
    assert count_freeze_thaw([-3, 1, -1, None], [2, 5, -0.5, 3]) == 1


def test_parse_open_meteo_response():
    days = [f"2026-02-{d:02d}" for d in range(1, 9)]
    data = {
        "daily": {
            "time": days,
            "temperature_2m_min": [-5, -4, 1, -2, -6, -1, -3, -2],
            "temperature_2m_max": [2, 3, 6, 1, -1, 4, 2, 5],
            "precipitation_sum": [0, 0, 0, 0, 0, 0, 0, 4.2],
            "snowfall_sum": [0, 0, 0, 0, 0, 0, 0, 1.1],
        },
        "hourly": {
            "time": ["2026-02-08T14:00", "2026-02-08T15:00"],
            "temperature_2m": [1.0, 1.5],
            "relative_humidity_2m": [70, 72],
            "wind_speed_10m": [10, 11],
            "weather_code": [71, 73],
        },
    }
    w = parse_response(data, datetime(2026, 2, 8, 15, 20, tzinfo=UTC))
    assert w["temp"] == 1.5
    assert w["rainfall"] == 4.2 and w["snowfall"] == 1.1
    assert w["freeze_thaw_cycles"] == 5     # days 1,2,4,6,7 (excludes report day)
    assert w["weather_conditions"] == "snow"


@pytest.mark.parametrize("text,expected", [
    ('{"severity": 0.7}', 0.7),
    ('```json\n{"severity": 0.25}\n```', 0.25),
    ('{"severity": 1.4}', None),
    ("no json here", None),
])
def test_parse_llm_severity(text, expected):
    assert parse_llm_severity(text) == expected


def test_heuristic_severity():
    assert heuristic_severity(None, 0.9) is None
    hi = heuristic_severity("potholes", 0.95)
    lo = heuristic_severity("potholes", 0.2)
    assert 0.5 < lo < hi <= 1.0  # confident predictions move further from the midpoint
    assert heuristic_severity("graffiti", 0.9) < 0.5


def test_preprocess_shape_and_rejection():
    img = np.zeros((480, 640, 3), np.uint8)
    img[100:200, 100:300] = 255
    ok, buf = cv2.imencode(".jpg", img)
    out = load_and_prepare(buf.tobytes())
    assert out.shape == (IMG_SIZE, IMG_SIZE, 3) and out.dtype == np.uint8
    with pytest.raises(ImageRejected):
        load_and_prepare(b"not an image")
    ok, tiny = cv2.imencode(".jpg", np.zeros((20, 20, 3), np.uint8))
    with pytest.raises(ImageRejected):
        load_and_prepare(tiny.tobytes())


def _scene(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((224, 224, 3), 90, np.uint8)
    for _ in range(12):
        x, y = rng.integers(0, 200, 2)
        cv2.rectangle(img, (int(x), int(y)), (int(x) + 40, int(y) + 30), rng.integers(0, 255, 3).tolist(), -1)
    return img


def test_near_duplicate_detection():
    img = _scene(1)
    # re-encoded at low quality and resized: the kind of copy augmented datasets contain
    ok, buf = cv2.imencode(".jpg", cv2.resize(img, (180, 180)), [cv2.IMWRITE_JPEG_QUALITY, 50])
    copy = cv2.resize(cv2.imdecode(buf, cv2.IMREAD_COLOR), (224, 224))
    kept = np.array([average_hash(img)], dtype=np.uint64)
    assert is_near_duplicate(average_hash(copy), kept)
    assert not is_near_duplicate(average_hash(_scene(2)), kept)
