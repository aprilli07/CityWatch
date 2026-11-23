"""Photo storage: Supabase Storage when configured, local ./uploads otherwise.

Uploads go through the API (with the server-side key), so the mobile app no
longer needs any Supabase credentials.
"""

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

import httpx

from app.config import BACKEND_DIR, get_settings

log = logging.getLogger(__name__)

UPLOAD_DIR = BACKEND_DIR / "uploads"
_bucket_ready = False


def _object_name(user_id: str | None) -> str:
    safe_user = "".join(c if c.isalnum() or c in "-_" else "-" for c in (user_id or "anonymous"))
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"{safe_user}/{stamp}-{uuid.uuid4().hex[:8]}.jpg"


async def _ensure_bucket(client: httpx.AsyncClient, base: str, headers: dict, bucket: str) -> None:
    global _bucket_ready
    if _bucket_ready:
        return
    r = await client.get(f"{base}/storage/v1/bucket/{bucket}", headers=headers)
    if r.status_code != 200:
        r = await client.post(
            f"{base}/storage/v1/bucket", headers=headers, json={"id": bucket, "name": bucket, "public": True}
        )
        if r.status_code not in (200, 201, 409):
            raise RuntimeError(f"could not create bucket: {r.status_code} {r.text}")
    _bucket_ready = True


async def save_image(img_bytes: bytes, user_id: str | None, content_type: str = "image/jpeg") -> str:
    """Store the image and return a public URL."""
    s = get_settings()
    name = _object_name(user_id)
    if s.supabase_url and s.supabase_key:
        headers = {"Authorization": f"Bearer {s.supabase_key}", "apikey": s.supabase_key}
        async with httpx.AsyncClient(timeout=30) as client:
            await _ensure_bucket(client, s.supabase_url, headers, s.storage_bucket)
            r = await client.post(
                f"{s.supabase_url}/storage/v1/object/{s.storage_bucket}/{name}",
                headers={**headers, "Content-Type": content_type},
                content=img_bytes,
            )
            r.raise_for_status()
        return f"{s.supabase_url}/storage/v1/object/public/{s.storage_bucket}/{name}"

    path = UPLOAD_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(img_bytes)
    return f"{s.public_base_url}/uploads/{name}"
