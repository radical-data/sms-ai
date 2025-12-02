from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from .db import SessionLocal, init_db
from .pipeline import handle_message
from .sms import InboundSms

app = FastAPI(title="sms.ai MVP", version="0.1.0")

# --- DB dependency ---

def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
def on_startup() -> None:
    init_db()


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

