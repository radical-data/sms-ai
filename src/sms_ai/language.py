from __future__ import annotations

from typing import Final, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

TRANSLATION_SYSTEM_PROMPT: Final[str] = (
    "You are a professional translator specialising in agricultural communication "
    "between English and Setswana (South African Tswana). "
    "Translate the user's message accurately and faithfully, without adding, removing, "
    "or interpreting information. Maintain the user's tone and level of formality. "
    "Use natural, rural Setswana phrasing where appropriate. "
    "If a term is ambiguous, choose the most practical farming-related meaning based on context. "
    "If you are uncertain, provide your best direct translation without explanation."
)

_translation_model: ChatOpenAI | None = None


def get_translation_model() -> ChatOpenAI:
    """
    Lazily create and cache a ChatOpenAI model for translation.

    Relies on the OPENAI_API_KEY environment variable, loaded via dotenv in __init__.
    """
    global _translation_model
    if _translation_model is None:
        _translation_model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=256,  # short translations  # type: ignore[call-arg]
        )
    return _translation_model


def translate(text: str, source: str, target: str) -> str:
    """
    Translate between English ('en') and Setswana ('tsn') using the LLM.

    Supported directions:
      - tsn -> en
      - en -> tsn

    If source == target, returns the text unchanged.
    """
    if source == target:
        return text

    if {source, target} != {"en", "tsn"}:
        raise ValueError(f"Unsupported translation direction: {source} -> {target}")

    if source == "tsn" and target == "en":
        direction_instruction = (
            "Translate from Setswana (South African Tswana) into English. "
            "Do not add explanations, just translate."
        )
    elif source == "en" and target == "tsn":
        direction_instruction = (
            "Translate from English into Setswana (South African Tswana). "
            "Keep it natural and easy for rural farmers to understand. "
            "Do not add explanations, just translate."
        )
    else:
        # Should be unreachable given the set check above
        raise ValueError(f"Unsupported translation direction: {source} -> {target}")

    model = get_translation_model()

    messages = [
        SystemMessage(content=TRANSLATION_SYSTEM_PROMPT + " " + direction_instruction),
        HumanMessage(content=text),
    ]

    response = model.invoke(messages)
    content = cast(str, response.content)  # type: ignore[reportUnknownMemberType]
    return content.strip()
