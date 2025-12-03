from __future__ import annotations

import argparse
from typing import Any, cast

from sms_ai.glossary import preview_matches_for_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str)
    parser.add_argument("--source", choices=["tsn", "en"], default="tsn")
    args = parser.parse_args()

    # Type checker can't infer that args.source is a valid LangCode literal,
    # but argparse ensures it's one of the choices ("tsn" or "en"), so the cast is safe.
    matches = preview_matches_for_text(args.text, source=args.source)  # type: ignore[arg-type]

    if not matches:
        print("No glossary matches.")
        return

    for m in matches:
        token = cast(str, m["token"])
        print(f"token: {token}")
        entries = cast(list[dict[str, Any]], m["entries"])
        for e in entries:
            variants_list = cast(list[str], e["setswana_variants"])
            variants = ", ".join(variants_list) or ""
            variant_part = f" (variants: {variants})" if variants else ""
            setswana_preferred = cast(str, e["setswana_preferred"])
            english_label = cast(str, e["english_label"])
            print(f"  {setswana_preferred}  <->  {english_label}{variant_part}")
        print()


if __name__ == "__main__":
    main()
