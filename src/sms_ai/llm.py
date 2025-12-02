from __future__ import annotations

from typing import Final, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

SYSTEM_PROMPT: Final[str] = (
    "You are an agricultural assistant helping smallholder farmers near "
    "Johannesburg, South Africa. Farmers send you brief questions by SMS "
    "about crops and livestock. Reply in simple English with short, clear, "
    "practical advice: ideally 2–4 short sentences at most. Focus on low-cost, "
    "low-risk actions the farmer can take. If you are not sure, or the problem "
    "sounds serious or life-threatening for people or animals, say that you are "
    "not sure and recommend talking to a local agricultural extension officer "
    "or an experienced farmer. Do NOT give exact chemical or medicine dosages, "
    "spray recipes, or injection instructions. Do NOT pretend to be completely "
    "certain when you are not."
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
