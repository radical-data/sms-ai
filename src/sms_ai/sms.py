from __future__ import annotations

from pydantic import BaseModel


class InboundSms(BaseModel):
    phone: str
    text: str


# Later we'll add provider-specific parsing and send_sms() functions here.

