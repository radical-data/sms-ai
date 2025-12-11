from __future__ import annotations

from twilio.rest import Client

from .config import get_settings


def get_twilio_client() -> Client:
    settings = get_settings()

    if not settings.twilio_account_sid or not settings.twilio_auth_token:
        raise RuntimeError(
            "Twilio credentials are not configured (TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN)"
        )

    return Client(settings.twilio_account_sid, settings.twilio_auth_token)


def send_sms(to: str, body: str) -> None:
    """
    Send an SMS using the configured Twilio account.

    This is used by the async background worker after the LLM has finished.
    """
    settings = get_settings()
    if not settings.twilio_from_number:
        raise RuntimeError("TWILIO_FROM_NUMBER is not configured")

    client = get_twilio_client()
    client.messages.create(
        to=to,
        from_=settings.twilio_from_number,
        body=body,
    )
