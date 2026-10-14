from __future__ import annotations

import json
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from sms_ai import agent, main
from sms_ai.db import Message, Turn


@pytest.fixture
def model_response(monkeypatch: pytest.MonkeyPatch) -> Mock:
    # Mock only the provider boundary: exercise real JSON parsing and persistence.
    response = Mock(
        return_value=AIMessage(
            content=json.dumps(
                {
                    "detected_language": "en",
                    "source_text": "Hello",
                    "english_translation": "Hello",
                    "intent": "greeting",
                    "answer_english": "How can I help?",
                    "final_answer_user_language": "**Hello!**\n- How can I help?",
                    "safety_flags": {},
                    "reasoning_summary": "The user greeted us.",
                }
            )
        )
    )
    monkeypatch.setattr(agent, "_run_llm_with_tools", response)
    return response


def test_web_message_is_parsed_and_persisted(
    database: sessionmaker[Session], model_response: Mock
) -> None:
    with TestClient(main.app) as client:
        response = client.post("/test/inbound", json={"phone": "+27123456789", "text": "Hello"})
    assert response.status_code == 200
    assert response.json()["echo"] == "Hello! How can I help?"
    model_response.assert_called_once()
    with database() as db:
        messages = db.scalars(select(Message).order_by(Message.id)).all()
        assert [(m.direction, m.text) for m in messages] == [
            ("in", "Hello"),
            ("out", "Hello! How can I help?"),
        ]
        turn = db.scalars(select(Turn)).one()
        assert turn.incoming_id == response.json()["message_id"] == messages[0].id
        assert turn.outgoing_id == messages[1].id
        assert turn.lang_detected == "en"
        assert turn.question_en == "Hello"


def test_invalid_web_message_never_calls_provider(
    database: sessionmaker[Session], model_response: Mock
) -> None:
    with TestClient(main.app) as client:
        response = client.post("/test/inbound", json={"phone": "+27123456789"})
    assert response.status_code == 422
    model_response.assert_not_called()
    with database() as db:
        assert db.scalars(select(Message)).all() == []


def test_sms_background_task_sends_persisted_answer(
    database: sessionmaker[Session], model_response: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    send_sms = Mock()
    monkeypatch.setattr(main, "send_sms", send_sms)
    with TestClient(main.app) as client:
        response = client.post("/sms/inbound", data={"From": "+27123456789", "Body": "Hello"})
    assert response.status_code == 200
    assert "<Response></Response>" in response.text
    send_sms.assert_called_once_with(to="+27123456789", body="Hello! How can I help?")
    model_response.assert_called_once()
    with database() as db:
        assert len(db.scalars(select(Message)).all()) == 2
        assert db.scalars(select(Turn)).one().lang_detected == "en"
