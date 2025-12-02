from __future__ import annotations

import argparse

from .db import SessionLocal
from .llm import ask_llm
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


def chat_en() -> None:
    """
    Interactive CLI chat in English, directly against the LLM.

    This bypasses translation and does NOT log to the database.
    """
    print("English CLI mode (direct LLM, no DB). Type /quit to exit.\n")
    while True:
        try:
            user_input = input("en> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in {"/q", "/quit", "/exit"}:
            break
        answer = ask_llm(user_input)
        print(f"bot> {answer}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="sms-ai CLI tools (chat in Tswana or English).")
    parser.add_argument(
        "mode",
        choices=["tsn", "en"],
        help="Chat mode: 'tsn' (Tswana pipeline) or 'en' (direct English LLM).",
    )
    args = parser.parse_args()

    if args.mode == "tsn":
        chat_tsn()
    else:
        chat_en()


if __name__ == "__main__":
    main()
