from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from .db import Message


@dataclass
class PipelineResult:
    echo_text: str
    message_id: int


def handle_message(db: Session, phone: str, text: str) -> PipelineResult:
    """
    Core business logic for Milestone 1:
    - stores the incoming message
    - returns a simple echo response string
    """
    incoming = Message(phone=phone, direction="in", text=text)
    db.add(incoming)
    db.commit()
    db.refresh(incoming)

    echo = f"We got: {text}"

    # For Milestone 1 we do NOT store the outgoing echo as a message yet.
    # We'll add 'out' messages and Turns in later milestones.

    return PipelineResult(echo_text=echo, message_id=incoming.id)
