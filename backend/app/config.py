from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BACKEND_DIR / "ml" / "artifacts"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    database_url: str = "postgresql://localhost:5432/citywatch"

    # Supabase Storage for report photos. If unset, photos are saved to ./uploads.
    supabase_url: str | None = None
    supabase_key: str | None = None
    storage_bucket: str = "reports"
    public_base_url: str = "http://localhost:8000"

    # Optional LLM severity scoring. Without it, severity comes from CV confidence.
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"

    # City used for the weather archive and default map centre (Kitchener-Waterloo).
    city_lat: float = 43.4643
    city_lon: float = -80.5204

    # Shared secret for staff-only endpoints (status changes, risk refresh).
    staff_api_token: str | None = None

    h3_resolution: int = 8
    asset_match_radius_m: float = 25.0
    heatmap_cache_seconds: int = 60
    external_timeout_s: float = 8.0
    max_upload_mb: int = 10

    cv_model_path: Path = ARTIFACTS_DIR / "cv_model.keras"
    cv_labels_path: Path = ARTIFACTS_DIR / "cv_labels.json"
    risk_model_path: Path = ARTIFACTS_DIR / "risk_models.joblib"


@lru_cache
def get_settings() -> Settings:
    return Settings()
