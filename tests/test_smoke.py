from __future__ import annotations

from fastapi.testclient import TestClient

from sms_ai.main import app


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
    assert data["echo"] == "We got: Hello"
