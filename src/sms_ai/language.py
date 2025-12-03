from __future__ import annotations

from typing import Final, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .glossary import GlossaryEntry, find_terms_for_en, find_terms_for_tsn

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


def _build_glossary_prompt(
    entries: list[GlossaryEntry],
    source: str,
    target: str,
) -> str:
    """
    Build a short, structured glossary instruction for the LLM.

    - For tsn -> en: show Setswana term(s) and corresponding English term.

    - For en -> tsn: show English term and preferred Setswana term.
    """
    if not entries:
        return ""

    lines: list[str] = []
    lines.append(
        "When translating, you MUST use the following domain glossary "
        "whenever the term clearly matches. Always use the exact target-language "
        "term given here, and do not invent alternative technical terms."
    )

    if source == "tsn" and target == "en":
        lines.append("")
        lines.append("Setswana → English glossary:")
        for e in entries:
            variants = ", ".join(e.setswana_variants) if e.setswana_variants else ""
            variant_part = f" (variants: {variants})" if variants else ""
            lines.append(f"- {e.setswana_preferred}{variant_part} → {e.english_label}")
    elif source == "en" and target == "tsn":
        lines.append("")
        lines.append("English → Setswana glossary:")
        for e in entries:
            lines.append(f"- {e.english_label} → {e.setswana_preferred}")
    else:
        # We only support en <-> tsn
        return ""

    return "\n".join(lines)


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
        glossary_entries = find_terms_for_tsn(text)
    elif source == "en" and target == "tsn":
        direction_instruction = (
            "Translate from English into Setswana (South African Tswana). "
            "Keep it natural and easy for rural farmers to understand. "
            "Do not add explanations, just translate."
        )
        glossary_entries = find_terms_for_en(text)
    else:
        # Should be unreachable due to set check above
        raise ValueError(f"Unsupported translation direction: {source} -> {target}")

    glossary_prompt = _build_glossary_prompt(glossary_entries, source=source, target=target)

    model = get_translation_model()

    system_content = TRANSLATION_SYSTEM_PROMPT + " " + direction_instruction
    if glossary_prompt:
        system_content += "\n\n" + glossary_prompt

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=text),
    ]

    response = model.invoke(messages)
    content = cast(str, response.content)  # type: ignore[reportUnknownMemberType]
    return content.strip()
