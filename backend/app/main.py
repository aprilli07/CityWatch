"""CityWatch API.

Run: uvicorn app.main:app --reload --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.db import close_pool, fetch_one, get_pool
from app.deps import require_staff
from app.routers import insights, reports
from app.services import risk
from app.services.classifier import classifier
from app.services.storage import UPLOAD_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_pool()
    yield
    close_pool()


app = FastAPI(title="CityWatch API", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.include_router(reports.router)
app.include_router(insights.router)


@app.get("/health")
def health():
    db_ok = fetch_one("SELECT 1 AS ok") is not None
    return {"status": "ok" if db_ok else "degraded", "db": db_ok, "cv_model": classifier.available,
            "last_risk_refresh": risk.last_run or None}


class UserIn(BaseModel):
    id: str
    email: str | None = None
    display_name: str | None = None


@app.post("/api/users", tags=["users"])
def upsert_user(user: UserIn):
    row = fetch_one(
        """INSERT INTO users (id, email, display_name) VALUES (%s, %s, %s)
           ON CONFLICT (id) DO UPDATE SET email = COALESCE(EXCLUDED.email, users.email),
                display_name = COALESCE(EXCLUDED.display_name, users.display_name)
           RETURNING id, email, display_name, role, created_at""",
        (user.id, user.email, user.display_name),
    )
    return {"success": True, "data": row}


@app.post("/api/admin/risk/refresh", tags=["admin"], dependencies=[Depends(require_staff)])
def refresh_risk(background: BackgroundTasks, wait: bool = False):
    if wait:
        result = risk.score_all()
        insights.invalidate_cache()
        return {"success": True, "data": result}
    background.add_task(risk.request_refresh)
    return {"success": True, "queued": True}
