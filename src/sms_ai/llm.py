from __future__ import annotations

from typing import Final, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

SYSTEM_PROMPT: Final[str] = (
    "You are an agricultural assistant helping smallholder farmers near "
    "Johannesburg, South Africa. Farmers will send you short questions about "
    "crops and livestock. Give short, clear, practical answers in simple English. "
    "If you are not sure, say you are not sure. Do NOT give exact pesticide "
    "dosages unless you are extremely sure."
)

_model: ChatOpenAI | None = None


def get_model() -> ChatOpenAI:
    """
    Lazily create and cache a ChatOpenAI model.

    Relies on the OPENAI_API_KEY environment variable.
    """
    global _model
    if _model is None:
        _model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=256,  # short, focused answers  # type: ignore[call-arg]
        )
    return _model


def ask_llm(question_en: str) -> str:
    """
    Call the LLM with the given English question and return a short English answer.
    """
    model = get_model()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question_en),
    ]
    response = model.invoke(messages)
    # For ChatOpenAI, response.content is always a string
    content = cast(str, response.content)  # type: ignore[reportUnknownMemberType]
    return content.strip()
