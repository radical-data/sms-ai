from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    # Project root
    project_root: Path = Path(__file__).resolve().parents[2]

    # SQLite file in project root by default
    database_url: str = f"sqlite:///{(Path(__file__).resolve().parents[2] / 'sms_ai.db')}"

    # Optional glossary CSV path (can be overridden by env var)
    # If None, we default to data/glossary.csv under project root.
    glossary_csv: Path | None = None

    def model_post_init(self, __context: object) -> None:  # type: ignore[override]
        env_path = os.getenv("GLOSSARY_CSV_PATH")
        if env_path:
            object.__setattr__(self, "glossary_csv", Path(env_path))
        elif self.glossary_csv is None:
            object.__setattr__(self, "glossary_csv", self.project_root / "data" / "glossary.csv")


@lru_cache
def get_settings() -> Settings:
    return Settings()
