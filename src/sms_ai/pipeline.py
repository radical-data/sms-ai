from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from sqlalchemy.orm import Session

from .db import Message, Turn
from .language import translate
from .llm import ask_llm

MAX_SMS_CHARS: Final[int] = 320


def clamp_sms(text: str) -> str:
    """
    Ensure the SMS body is not excessively long.

    We aim for roughly <= 2 GSM-7 SMS segments (~160 chars each).

    If it's longer, we truncate and add a short continuation hint in Setswana.
    """
    if len(text) <= MAX_SMS_CHARS:
        return text

    tail = " Go na le tshedimosetso e nngwe, kopa tswelela o botse gape."

    # If the tail itself is longer than the budget, hard-cut the text.
    if len(tail) >= MAX_SMS_CHARS:
        return text[:MAX_SMS_CHARS].rstrip()

    allowed = MAX_SMS_CHARS - len(tail)
    truncated = text[:allowed].rstrip()
    return truncated + tail


def maybe_add_warning(answer_en: str, answer_tsn: str) -> str:
    """
    Hook for future safety tweaks.

    For now this is a no-op: we just return the Setswana answer unchanged.

    Later we can inspect answer_en for risky content and conditionally
    append a very short warning in Setswana.
    """
    return answer_tsn


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

    # 6. Apply safety/UX hooks:
    #    - (later) maybe_add_warning can append a short warning for risky content
    #    - clamp_sms ensures we don't send excessively long replies over SMS
    answer_for_user = maybe_add_warning(answer_en=answer_en, answer_tsn=answer_tsn)
    answer_for_user = clamp_sms(answer_for_user)

    # 7. Save outgoing message (Setswana answer)
    outgoing = Message(phone=phone, direction="out", text=answer_for_user)
    db.add(outgoing)
    db.commit()
    db.refresh(outgoing)

    # 8. Save Turn with full sandwich
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

    # 9. Return the Setswana answer for endpoints to return/send via SMS
    return PipelineResult(echo_text=answer_for_user, message_id=incoming.id)
