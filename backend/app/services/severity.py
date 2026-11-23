"""Severity scoring (0-1).

If GEMINI_API_KEY is set, a vision LLM rates the photo (bounded by a timeout,
never blocking the report for minutes). Otherwise a transparent heuristic
combines a per-class base severity with the classifier's confidence.
"""

import base64
import json
import logging
import re

import httpx

from app.config import get_settings

log = logging.getLogger(__name__)

# Base severity per defect class: how urgent the issue is if confirmed.
BASE_SEVERITY = {
    "potholes": 0.75,
    "damaged_roads": 0.6,
    "damaged_concrete": 0.55,
    "broken_road_signs": 0.65,
    "damaged_electric_poles": 0.85,
    "fallen_trees": 0.7,
    "dead_animals": 0.4,
    "garbage": 0.3,
    "graffiti": 0.2,
    "illegal_parking": 0.25,
}

PROMPT = (
    "You are a municipal infrastructure inspector. Rate the severity of the damage or "
    "hazard in this photo from 0.0 (no issue) to 1.0 (dangerous, needs immediate repair). "
    'Consider size, depth, extent and safety risk. Reply with JSON only: {"severity": 0.7}'
)


def heuristic_severity(issue_type: str | None, confidence: float | None) -> float | None:
    if issue_type is None:
        return None
    base = BASE_SEVERITY.get(issue_type, 0.5)
    conf = confidence if confidence is not None else 0.5
    # Low-confidence predictions are pulled toward the middle of the scale.
    return round(0.5 + (base - 0.5) * conf, 3)


def parse_llm_severity(text: str) -> float | None:
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return None
    try:
        v = float(json.loads(m.group(0))["severity"])
    except (ValueError, KeyError, TypeError):
        return None
    return v if 0.0 <= v <= 1.0 else None


async def llm_severity(img_bytes: bytes, mime: str = "image/jpeg") -> float | None:
    s = get_settings()
    if not s.gemini_api_key:
        return None
    body = {
        "contents": [{"parts": [
            {"text": PROMPT},
            {"inline_data": {"mime_type": mime, "data": base64.b64encode(img_bytes).decode()}},
        ]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 64, "responseMimeType": "application/json"},
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{s.gemini_model}:generateContent"
    try:
        async with httpx.AsyncClient(timeout=s.external_timeout_s) as client:
            r = await client.post(url, params={"key": s.gemini_api_key}, json=body)
            r.raise_for_status()
            text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            return parse_llm_severity(text)
    except Exception as e:
        log.warning("LLM severity failed, falling back to heuristic: %s", e)
        return None


async def score_severity(
    img_bytes: bytes, issue_type: str | None, confidence: float | None
) -> tuple[float | None, str]:
    v = await llm_severity(img_bytes)
    if v is not None:
        return v, "llm"
    return heuristic_severity(issue_type, confidence), "heuristic"
