from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from sms_ai.main import app

# Live provider calls are opt-in, even when a local .env contains API keys.
if os.environ.get("RUN_LIVE_TESTS") != "1":
    pytest.skip("Set RUN_LIVE_TESTS=1 to run live provider tests", allow_module_level=True)

# Both providers must be configured.
if not all(os.environ.get(key) for key in ("GOOGLE_API_KEY", "TAVILY_API_KEY")):
    pytest.skip("Provider keys not set, skipping LLM smoke test", allow_module_level=True)


def test_health_inbound_test_endpoint(database: sessionmaker[Session]) -> None:
    with TestClient(app) as client:
        resp = client.post(
            "/test/inbound",
            json={"phone": "+27123456789", "text": "Hello"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "message_id" in data
    # echo is now an LLM answer, not a fixed string
    assert isinstance(data["echo"], str)
    assert data["echo"].strip() != ""
