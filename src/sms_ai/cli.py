from __future__ import annotations

from .db import SessionLocal
from .pipeline import handle_message

CLI_TS_PHONE = "+999000000_tswana_cli"


def chat_tsn() -> None:
    """
    Interactive CLI chat assuming Tswana input.

    Uses the full pipeline (translate tsn->en, LLM, translate en->tsn)
    and logs to the database with a fixed pseudo-phone number.
    """
    db = SessionLocal()
    print("Tswana CLI mode (via full pipeline). Type /quit to exit.\n")
    try:
        while True:
            try:
                user_input = input("tsn> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_input:
                continue
            if user_input.lower() in {"/q", "/quit", "/exit"}:
                break
            result = handle_message(db=db, phone=CLI_TS_PHONE, text=user_input)
            print(f"bot> {result.echo_text}\n")
    finally:
        db.close()


def main() -> None:
    chat_tsn()


if __name__ == "__main__":
    main()
