from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    # Project root (repo root in local dev, /app in Docker)
    project_root: Path = Path(__file__).resolve().parents[2]

    # Database URL:
    # - Default for local dev: sqlite file in the project root (sms_ai.db)
    # - Override in Docker / production using the DATABASE_URL env var
    database_url: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{(Path(__file__).resolve().parents[2] / 'sms_ai.db')}",
    )

    # Optional glossary CSV path (can be overridden by env var)
    # If None, we default to data/glossary.csv under project root.
    glossary_csv: Path | None = None

    # --- Twilio settings for async outbound SMS ---
    twilio_account_sid: str | None = os.getenv("TWILIO_ACCOUNT_SID")
    twilio_auth_token: str | None = os.getenv("TWILIO_AUTH_TOKEN")
    twilio_from_number: str | None = os.getenv("TWILIO_FROM_NUMBER")

    def model_post_init(self, __context: object) -> None:  # type: ignore[override]
        env_path = os.getenv("GLOSSARY_CSV_PATH")
        if env_path:
            object.__setattr__(self, "glossary_csv", Path(env_path))
        elif self.glossary_csv is None:
            object.__setattr__(self, "glossary_csv", self.project_root / "data" / "glossary.csv")


@lru_cache
def get_settings() -> Settings:
    return Settings()
