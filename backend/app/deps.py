import secrets

from fastapi import Header, HTTPException

from app.config import get_settings


def require_staff(x_staff_token: str | None = Header(None)) -> None:
    """Staff-only endpoints need the shared STAFF_API_TOKEN in the X-Staff-Token header."""
    expected = get_settings().staff_api_token
    if not expected:
        raise HTTPException(503, "staff endpoints disabled: set STAFF_API_TOKEN")
    if not x_staff_token or not secrets.compare_digest(x_staff_token, expected):
        raise HTTPException(401, "invalid staff token")
