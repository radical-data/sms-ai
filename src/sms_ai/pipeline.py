from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from sqlalchemy.orm import Session

from .agent import run_agent
from .db import Message, Turn

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


def maybe_add_warning(answer_en: str, answer_tsn: str, safety_flags: Mapping[str, Any]) -> str:
    """
    Append very short warnings in Setswana based on safety flags.
    Currently:
      - if needs_human_review: add a standard warning.
    """
    # Base answer
    out = answer_tsn

    needs_human_review = bool(safety_flags.get("needs_human_review"))
    if needs_human_review:
        warning = (
            " Tlhokomeliso: maemo a a ka nna a le masisi. "
            "Bua le ofisiri ya temo kgotsa molemi yo o nang le maitemogelo."
        )
        out = out.rstrip()
        if not out.endswith("."):
            out += "."
        out += warning

    return out


@dataclass
class PipelineResult:
    echo_text: str
    message_id: int


def handle_message(db: Session, phone: str, text: str) -> PipelineResult:
    """
    New core business logic:
    - stores the incoming message
    - calls the agent (single LLM call)
    - uses the agent's structured output to save a Turn
    - returns the answer in the user's language as echo_text
    """
    # 1. Save incoming message (raw text)
    incoming = Message(phone=phone, direction="in", text=text)
    db.add(incoming)
    db.commit()
    db.refresh(incoming)

    # 2. Call the agent
    agent_result = run_agent(text)

    detected_language = agent_result["detected_language"]
    english_translation = agent_result["english_translation"]
    answer_en = agent_result["answer_english"]
    final_answer = agent_result["final_answer_user_language"]
    safety_flags = agent_result.get("safety_flags", {})
    reasoning_summary = agent_result.get("reasoning_summary", "")

    # 3. Decide what to store as question_tsn_raw / question_en / answer_tsn
    if detected_language in ("tsn", "mixed"):
        question_tsn_raw = text
        question_en = english_translation
        answer_tsn = final_answer
    elif detected_language == "en":
        question_tsn_raw = None
        question_en = text  # original is already English
        answer_tsn = None  # final answer is in English
    else:  # "other"
        question_tsn_raw = None
        question_en = english_translation or text
        answer_tsn = None

    # 4. Apply safety/UX hooks
    if detected_language in ("tsn", "mixed"):
        answer_for_user = maybe_add_warning(
            answer_en=answer_en,
            answer_tsn=final_answer,
            safety_flags=safety_flags,
        )
    else:
        # For English or "other", we currently just send the final answer as-is.
        answer_for_user = final_answer

    answer_for_user = clamp_sms(answer_for_user)

    # 5. Save outgoing message
    outgoing = Message(phone=phone, direction="out", text=answer_for_user)
    db.add(outgoing)
    db.commit()
    db.refresh(outgoing)

    # 6. Save Turn with full sandwich as best we can
    turn = Turn(
        phone=phone,
        incoming_id=incoming.id,
        outgoing_id=outgoing.id,
        lang_detected=detected_language,
        question_tsn_raw=question_tsn_raw,
        question_en=question_en,
        answer_en=answer_en,
        answer_tsn=answer_tsn,
        llm_model="gemini-3-pro-preview",
        translation_backend="gemini3:tavily-single-call",
        reasoning_summary=reasoning_summary,
        safety_flags_json=json.dumps(safety_flags),
    )
    db.add(turn)
    db.commit()

    # 7. Return answer for endpoints to send via SMS/HTTP
    return PipelineResult(echo_text=answer_for_user, message_id=incoming.id)
