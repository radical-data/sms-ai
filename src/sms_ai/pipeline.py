from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from .db import Message, Turn
from .language import translate
from .llm import ask_llm


@dataclass
class PipelineResult:
    echo_text: str
    message_id: int


def handle_message(db: Session, phone: str, text: str) -> PipelineResult:
    """
    Core business logic for Milestone 3B (Setswana-only):
    - stores the incoming message (assumed Setswana)
    - translates Tswana -> English
    - calls the LLM in English
    - translates English -> Tswana
    - stores a Turn with all intermediate steps
    - returns the Setswana answer as echo_text
    """
    # 1. Save incoming message (assumed Setswana)
    incoming = Message(phone=phone, direction="in", text=text)
    db.add(incoming)
    db.commit()
    db.refresh(incoming)

    # 2. Treat raw text as Setswana
    question_tsn_raw = text

    # 3. Tswana -> English
    question_en = translate(question_tsn_raw, source="tsn", target="en")

    # 4. LLM in English
    answer_en = ask_llm(question_en)

    # 5. English -> Tswana
    answer_tsn = translate(answer_en, source="en", target="tsn")

    # This is what we send back to the user
    answer_for_user = answer_tsn

    # 6. Save outgoing message (Setswana answer)
    outgoing = Message(phone=phone, direction="out", text=answer_for_user)
    db.add(outgoing)
    db.commit()
    db.refresh(outgoing)

    # 7. Save Turn with full sandwich
    turn = Turn(
        phone=phone,
        incoming_id=incoming.id,
        outgoing_id=outgoing.id,
        lang_detected="tsn",  # hard-coded, we assume all input is Setswana
        question_tsn_raw=question_tsn_raw,
        question_en=question_en,
        answer_en=answer_en,
        answer_tsn=answer_tsn,
        llm_model="openai:gpt-4o-mini",
        translation_backend="openai:gpt-4o-mini:chat-translation",
    )
    db.add(turn)
    db.commit()

    # 8. Return the Setswana answer for endpoints to return/send via SMS
    return PipelineResult(echo_text=answer_for_user, message_id=incoming.id)
