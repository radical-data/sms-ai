from __future__ import annotations

import argparse
from collections.abc import Iterable

from .db import SessionLocal, Turn


def _format_str(value: str | None) -> str:
    """Normalise None/whitespace for display."""
    if value is None:
        return ""
    return value.strip()


def iter_recent_turns(limit: int) -> Iterable[Turn]:
    """Yield recent turns ordered by newest first."""
    db = SessionLocal()
    try:
        turns = db.query(Turn).order_by(Turn.created_at.desc()).limit(limit).all()
        # detach results from session before closing
        yield from turns
    finally:
        db.close()


def print_recent_turns(limit: int) -> None:
    """Print recent turns in a human-readable form."""
    for turn in iter_recent_turns(limit):
        print("-" * 80)
        print(
            f"Turn #{turn.id} | phone={turn.phone} | "
            f"lang={turn.lang_detected or '-'} | at={turn.created_at}"
        )
        print()
        print(f"Q_TS: {_format_str(turn.question_tsn_raw)}")
        print(f"Q_EN: {_format_str(turn.question_en)}")
        print()
        print(f"A_EN: {_format_str(turn.answer_en)}")
        print(f"A_TS: {_format_str(turn.answer_tsn)}")
        print()


def export_recent_turns_csv(limit: int, csv_path: str) -> None:
    """
    Export recent turns to a CSV file.

    Includes a blank 'tag' column so you can mark rows manually
    as 'ok', 'weird', 'wrong', 'unsafe', etc.
    """
    import csv

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "created_at",
                "phone",
                "lang_detected",
                "question_tsn_raw",
                "question_en",
                "answer_en",
                "answer_tsn",
                "llm_model",
                "translation_backend",
                "tag",  # for manual review
            ]
        )
        for turn in iter_recent_turns(limit):
            writer.writerow(
                [
                    turn.id,
                    turn.created_at.isoformat() if turn.created_at else "",
                    turn.phone,
                    turn.lang_detected or "",
                    _format_str(turn.question_tsn_raw),
                    _format_str(turn.question_en),
                    _format_str(turn.answer_en),
                    _format_str(turn.answer_tsn),
                    turn.llm_model or "",
                    turn.translation_backend or "",
                    "",  # tag left blank for you to fill in
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect recent turns stored in the sms-ai SQLite database."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of most recent turns to show/export (default: 20).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="",
        help="Optional path to export turns as CSV. If omitted, only prints to stdout.",
    )
    args = parser.parse_args()

    if args.csv:
        export_recent_turns_csv(limit=args.limit, csv_path=args.csv)
        print(f"Exported {args.limit} turns to {args.csv}")
    else:
        print_recent_turns(limit=args.limit)


if __name__ == "__main__":
    main()
