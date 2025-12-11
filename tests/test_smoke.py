from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from sms_ai.main import app

# Skip this module if no API key is configured (avoids flaky CI without secrets)
if not os.environ.get("GOOGLE_API_KEY"):
    pytest.skip("GOOGLE_API_KEY not set, skipping LLM smoke test", allow_module_level=True)


def test_health_inbound_test_endpoint() -> None:
    client = TestClient(app)
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
