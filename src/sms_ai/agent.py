from __future__ import annotations

import json
from typing import Any, Literal, TypedDict, cast

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

SYSTEM_PROMPT = """
You are an agricultural assistant helping smallholder farmers near Johannesburg, South Africa.

Farmers send you short messages, usually in Setswana (South African Tswana), sometimes in English
or a mix of the two. Messages may contain spelling mistakes or informal language.

For EACH message you receive, you MUST:

1. Detect the language:
   - "tsn" = mostly Setswana (South African Tswana)
   - "en" = mostly English
   - "mixed" = significant mixture of Setswana and English
   - "other" = something else

2. Translate the message into clear English in the field "english_translation".
   - Be as faithful as possible to the farmer's meaning.
   - If you are guessing about a word, still give your best translation.
   - Example: "nako e maleba go jwala di erekies ke e fe?" -> "What is the right time to plant peas?"

3. Think carefully about the best, simple, low-risk advice in English.
   - Assume the farmer is near Johannesburg / Gauteng unless information suggests otherwise.
   - Focus on low-cost, low-risk actions.
   - Use 2–4 short sentences in English.

3a. Web search (VERY IMPORTANT):
   - You have access to a web search tool called "tavily_search".
   - Use it whenever:
     * the answer depends on specific agronomic facts (planting windows, pests, diseases, local regulations), OR
     * you are not confident you know the correct information.
   - Prefer 1–3 focused searches over many noisy ones.
   - When you call the search tool, include a short, clear query in English.
   - Carefully read the search results before giving advice.
   - If search results are unclear, say you are not fully sure and recommend talking to a local extension officer
     or experienced farmer.

4. Safety rules (VERY IMPORTANT):
   - Do NOT give exact chemical or medicine dosages, spray recipes, or injection instructions.
   - Do NOT pretend to be completely certain when you are not.
   - If the situation seems serious or unclear, recommend talking to a local agricultural extension officer
     or an experienced farmer.
   - Use the field "safety_flags" to indicate:
     - "mentions_dosage": true if you even talk about doses/amounts (you should avoid exact ones).
     - "needs_human_review": true if a local professional should definitely be consulted.

5. Convert your English answer into the language the farmer used:
   - If detected_language is "tsn" or "mixed": reply in Setswana (South African Tswana),
     using simple, natural rural phrasing.
   - If detected_language is "en": reply in simple English.
   - If detected_language is "other": choose English and clearly say you only support English
     and Setswana, then give your best attempt.

6. Summarise your reasoning process in 1–3 sentences in "reasoning_summary".
   - Explain briefly how you interpreted the question and why you gave that advice.
   - This is for internal logging/debugging, NOT for the farmer.

7. Style and length rules (VERY IMPORTANT):
   - All text fields must be plain text only. Do NOT use markdown (no **bold**, bullet points, numbered lists, emojis, or any other formatting).
   - Do not use bullet points or numbered lists in any field.
   - Keep "final_answer_user_language" very short: at most 2 short sentences and under 280 characters.
   - Avoid line breaks; write "final_answer_user_language" on a single line.

You must output ONLY a single valid JSON object with the following fields:

- detected_language: "tsn" | "en" | "mixed" | "other"
- source_text: the original message exactly as you received it
- english_translation: your best translation of the question into English
- intent: a short label for what the farmer is asking (e.g. "crop_planting_time")
- answer_english: your English answer (2–4 short sentences, simple language)
- final_answer_user_language: the answer in the farmer's language (Setswana or English)
- safety_flags: an object with boolean fields "mentions_dosage" and "needs_human_review"
- reasoning_summary: 1–3 sentences explaining your reasoning, for internal use only

Your FINAL response (after any tool calls) must be ONLY this JSON object as plain text.
Do not include any extra commentary, markdown, or explanation outside the JSON.
""".strip()


class AgentSafetyFlags(TypedDict, total=False):
    mentions_dosage: bool
    needs_human_review: bool


class AgentResponse(TypedDict):
    detected_language: Literal["tsn", "en", "mixed", "other"]
    source_text: str
    english_translation: str
    intent: str
    answer_english: str
    final_answer_user_language: str
    safety_flags: AgentSafetyFlags
    reasoning_summary: str


_agent_model: ChatGoogleGenerativeAI | None = None
_tavily_tool: TavilySearch | None = None


def get_agent_model() -> ChatGoogleGenerativeAI:
    """
    High-quality reasoning + web-search model using Gemini 3 Pro.

    We explicitly set temperature < 1.0 because the Gemini 3 docs recommend
    avoiding 0.7+ for complex reasoning to reduce loops and instability.
    """
    global _agent_model
    if _agent_model is None:
        _agent_model = ChatGoogleGenerativeAI(
            model="gemini-3-pro-preview",
            # Lower temperature than the Gemini 3 default to keep answers stable.
            temperature=0.4,
            max_tokens=None,
            max_retries=2,
        )
    return _agent_model


def get_tavily_tool() -> TavilySearch:
    """
    Lazily create Tavily search tool.

    Requires TAVILY_API_KEY in the environment.
    """
    global _tavily_tool
    if _tavily_tool is None:
        _tavily_tool = TavilySearch(
            max_results=5,
            topic="general",
            search_depth="advanced",  # better recall for agronomy
            include_answer="basic",
            include_raw_content=False,
        )
    return _tavily_tool


def _run_llm_with_tools(messages: list[BaseMessage]) -> AIMessage:
    """
    Simple tool-calling loop:

    - Bind TavilySearch as a tool to Gemini.
    - Let Gemini decide when to call the tool.
    - Execute Tavily for each tool call and feed results back.
    - Stop when the model returns an AIMessage with no tool_calls.
    """
    model = get_agent_model()
    tavily_tool = get_tavily_tool()
    model_with_tools = model.bind_tools([tavily_tool])  # type: ignore[reportUnknownMemberType]

    history: list[BaseMessage] = list(messages)
    max_tool_loops = 3

    for _ in range(max_tool_loops):
        ai_msg = model_with_tools.invoke(history)
        history.append(ai_msg)

        # If there are no tool calls, this is our final answer.
        if not ai_msg.tool_calls:
            return ai_msg

        # Execute each tool call and append the ToolMessage(s).
        tool_messages: list[ToolMessage] = []
        for tool_call in ai_msg.tool_calls:
            # We only have a single tool, but guard by name anyway.
            if tool_call["name"] != tavily_tool.name:
                continue

            # langchain tools accept the tool_call object directly.
            tool_result = tavily_tool.invoke(tool_call)  # type: ignore[reportUnknownMemberType]
            if not isinstance(tool_result, ToolMessage):
                raise TypeError("TavilySearch.invoke(tool_call) did not return a ToolMessage")
            tool_messages.append(tool_result)

        history.extend(tool_messages)

    # Failsafe: if we somehow still have tool calls after max loops,
    # force the model to give its best JSON answer without more tools.
    fallback_msg = model.invoke(
        history
        + [
            HumanMessage(
                content=(
                    "You have already used the Tavily search tool several times. "
                    "Now stop calling tools and respond with your FINAL JSON object only."
                )
            )
        ]
    )

    # model.invoke() already returns AIMessage
    return fallback_msg


def _parse_json_from_ai(ai_msg: AIMessage) -> AgentResponse:
    """
    Convert the final Gemini message into an AgentResponse dict.

    Gemini 3 models return content blocks; we should use `.text`
    to extract the string content.
    """
    # Prefer .text, but fall back to .content if needed.
    raw_text = getattr(ai_msg, "text", None)
    if not raw_text:
        content: str | list[Any] = ai_msg.content  # type: ignore[reportUnknownVariableType,reportUnknownMemberType]
        if isinstance(content, str):
            raw_text = content
        else:
            # content is a list - concatenate any text blocks
            text_parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    text_parts.append(str(block["text"]))  # type: ignore[reportUnknownArgumentType]
            raw_text = "".join(text_parts).strip()
            if not raw_text:
                raise ValueError("Unexpected Gemini message content format; cannot extract text.")

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        # If we get here, the model did not follow the JSON-only rule.
        raise ValueError(f"Gemini did not return valid JSON: {e}\nRaw content:\n{raw_text}") from e

    return cast(AgentResponse, data)


def run_agent(user_text: str) -> AgentResponse:
    """
    Single-call agent:
      - Detects language
      - Translates to English
      - Reasons (with Gemini 3 Pro)
      - Uses Tavily web search when needed (via tool calls)
      - Answers in user language
      - Returns structured JSON with reasoning_summary
    """
    messages: list[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_text),
    ]

    ai_msg = _run_llm_with_tools(messages)
    return _parse_json_from_ai(ai_msg)
