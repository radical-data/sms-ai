from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    DateTime,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .config import get_settings


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    phone: Mapped[str] = mapped_column(String, nullable=False)
    direction: Mapped[str] = mapped_column(String(3), nullable=False)  # "in" / "out"
    text: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class Turn(Base):
    __tablename__ = "turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    phone: Mapped[str] = mapped_column(String, nullable=False)
    incoming_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    outgoing_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    lang_detected: Mapped[str | None] = mapped_column(String(8), nullable=True)
    question_tsn_raw: Mapped[str | None] = mapped_column(String, nullable=True)
    question_en: Mapped[str | None] = mapped_column(String, nullable=True)
    answer_en: Mapped[str | None] = mapped_column(String, nullable=True)
    answer_tsn: Mapped[str | None] = mapped_column(String, nullable=True)
    llm_model: Mapped[str | None] = mapped_column(String, nullable=True)
    translation_backend: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


# --- Engine & Session factory ---

settings = get_settings()

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """Create tables if they don't exist."""
    Base.metadata.create_all(bind=engine)
