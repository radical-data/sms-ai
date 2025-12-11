from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Form, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session

from .db import SessionLocal, Turn, init_db
from .pipeline import handle_message
from .sms import InboundSms


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: runs once before the app starts serving requests
    init_db()
    yield
    # Shutdown: runs once when the app is shutting down (nothing to do yet)


app = FastAPI(title="sms.ai", version="0.1.0", lifespan=lifespan)

BASE_DIR = Path(__file__).resolve().parents[2]
DEMO_HTML_PATH = BASE_DIR / "static" / "demo.html"

# --- Admin protection ---

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
ALLOWED_ADMIN_IPS = {"127.0.0.1", "::1"}


def verify_admin(request: Request) -> None:
    """
    Simple protection for /admin endpoints:
    - only allow requests from ALLOWED_ADMIN_IPS
    - require X-Admin-Token header that matches ADMIN_TOKEN env var
    """
    client_host = request.client.host if request.client else None

    if client_host not in ALLOWED_ADMIN_IPS:
        raise HTTPException(status_code=403, detail="Forbidden")

    if not ADMIN_TOKEN:
        # Misconfiguration; safer to refuse access than to expose data.
        raise HTTPException(status_code=500, detail="ADMIN_TOKEN not configured")

    header_token = request.headers.get("X-Admin-Token")
    if header_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid admin token")


# --- DB dependency ---


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- Routes ---


@app.get("/")
def demo_page() -> FileResponse:
    """
    Web demo page that mimics the SMS UI.

    Notes:
    - Uses a fake phone number stored in localStorage.
    - Sends JSON requests to /test/inbound.
    - Does NOT use Twilio or /sms/inbound.
    """
    return FileResponse(DEMO_HTML_PATH)


@app.post("/test/inbound")
def test_inbound(payload: InboundSms, db: Session = Depends(get_db)) -> JSONResponse:
    """
    Test endpoint for Milestone 1.

    Accepts JSON:

      { "phone": "+27123456789", "text": "Dumela" }

    Stores the message and returns an echo.
    """
    result = handle_message(db=db, phone=payload.phone, text=payload.text)
    return JSONResponse(
        {
            "status": "ok",
            "message_id": result.message_id,
            "echo": result.echo_text,
        }
    )


@app.post("/sms/inbound")
def sms_inbound(
    From_: str = Form(..., alias="From"),
    Body: str = Form(..., alias="Body"),
    db: Session = Depends(get_db),
) -> Response:
    """
    Twilio-style SMS webhook endpoint.

    For now we test this locally by mimicking Twilio:
      - Content-Type: application/x-www-form-urlencoded
      - Fields:
          From: the sender's phone number
          Body: the SMS text

    We:
      - read those values (parsed by FastAPI from the form body)
      - pass them to handle_message(...)
      - return a TwiML XML response with the echo text
    """
    from_number = From_
    body = Body

    result = handle_message(db=db, phone=from_number, text=body)

    # TwiML XML response. Later Twilio will send this back as an SMS.
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Message>{result.echo_text}</Message>
</Response>"""

    return Response(content=twiml, media_type="application/xml")


@app.get("/admin/turns")
def admin_turns(
    limit: int = 50,
    db: Session = Depends(get_db),
    _: None = Depends(verify_admin),
) -> JSONResponse:
    """
    Very small admin endpoint to inspect recent turns.

    Example:
      GET /admin/turns
      GET /admin/turns?limit=10
    """
    # Clamp limit to a reasonable range
    safe_limit = max(1, min(limit, 200))
    turns = db.query(Turn).order_by(Turn.created_at.desc()).limit(safe_limit).all()

    payload = [
        {
            "id": t.id,
            "phone": t.phone,
            "created_at": t.created_at,
            "lang_detected": t.lang_detected,
            "question_tsn_raw": t.question_tsn_raw,
            "question_en": t.question_en,
            "answer_en": t.answer_en,
            "answer_tsn": t.answer_tsn,
            "llm_model": t.llm_model,
            "translation_backend": t.translation_backend,
            "reasoning_summary": t.reasoning_summary,
            "safety_flags_json": t.safety_flags_json,
        }
        for t in turns
    ]
    return JSONResponse(payload)
