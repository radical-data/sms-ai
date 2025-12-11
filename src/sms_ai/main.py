from __future__ import annotations

from collections.abc import Generator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
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

# --- DB dependency ---


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- Routes ---


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
async def sms_inbound(request: Request, db: Session = Depends(get_db)) -> Response:
    """
    Twilio-style SMS webhook endpoint.

    For now we test this locally by mimicking Twilio:
      - Content-Type: application/x-www-form-urlencoded
      - Fields:
          From: the sender's phone number
          Body: the SMS text

    We:
      - read those values
      - pass them to handle_message(...)
      - return a TwiML XML response with the echo text
    """
    form = await request.form()
    from_number = form.get("From")
    body = form.get("Body")

    if not from_number or not body:
        # This should not happen under normal Twilio usage
        raise HTTPException(
            status_code=400,
            detail="Missing 'From' or 'Body' in webhook payload",
        )

    # After the None check, we know these are strings (not UploadFile)
    assert isinstance(from_number, str)
    assert isinstance(body, str)

    result = handle_message(db=db, phone=from_number, text=body)

    # TwiML XML response. Later Twilio will send this back as an SMS.
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Message>{result.echo_text}</Message>
</Response>"""

    return Response(content=twiml, media_type="application/xml")


@app.get("/admin/turns")
def admin_turns(limit: int = 50, db: Session = Depends(get_db)) -> JSONResponse:
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
