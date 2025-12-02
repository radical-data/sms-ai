from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    # SQLite file in project root by default
    database_url: str = f"sqlite:///{(Path(__file__).resolve().parents[2] / 'sms_ai.db')}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
