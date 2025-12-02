from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from .db import Message, Turn
from .llm import ask_llm


@dataclass
class PipelineResult:
    echo_text: str
    message_id: int


def handle_message(db: Session, phone: str, text: str) -> PipelineResult:
    """
    Core business logic for Milestone 3A:
    - stores the incoming message
    - calls the LLM (English in → English out)
    - stores a basic Turn
    - returns the LLM answer as echo_text
    """
    # 1. Save incoming message
    incoming = Message(phone=phone, direction="in", text=text)
    db.add(incoming)
    db.commit()
    db.refresh(incoming)

    # 2. For now we assume the user wrote in English
    question_en = text

    # 3. Call LLM via LangChain
    answer_en = ask_llm(question_en)

    # 4. Save outgoing message
    outgoing = Message(phone=phone, direction="out", text=answer_en)
    db.add(outgoing)
    db.commit()
    db.refresh(outgoing)

    # 5. Save Turn
    turn = Turn(
        phone=phone,
        incoming_id=incoming.id,
        outgoing_id=outgoing.id,
        lang_detected="en",  # hard-coded for now
        question_tsn_raw=None,
        question_en=question_en,
        answer_en=answer_en,
        answer_tsn=None,
        llm_model="openai:gpt-4o-mini",
        translation_backend=None,
    )
    db.add(turn)
    db.commit()

    # 6. Return LLM answer for endpoints to send back
    return PipelineResult(echo_text=answer_en, message_id=incoming.id)
