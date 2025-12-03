from __future__ import annotations

import csv
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from .config import get_settings

LangCode = Literal["en", "tsn"]

try:
    from rapidfuzz import fuzz
except ImportError:
    # Type checker complains because fuzz is expected to be a module, but we set it to None
    # when RapidFuzz is not available. This is intentional for graceful degradation.
    fuzz = None  # type: ignore[assignment]


WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñʼ'\-]+", re.UNICODE)


@dataclass(frozen=True)
class GlossaryEntry:
    english_label: str
    english_pos: str | None
    setswana_preferred: str
    setswana_variants: tuple[str, ...]
    setswana_pos: str | None

    @property
    def all_setswana_forms(self) -> tuple[str, ...]:
        return (self.setswana_preferred, *self.setswana_variants)

    @property
    def all_english_forms(self) -> tuple[str, ...]:
        # Future: add more English variants here if needed.
        return (self.english_label,)


@dataclass(frozen=True)
class GlossaryIndex:
    entries: tuple[GlossaryEntry, ...]
    # normalised form -> entries that contain this form
    tsn_index: dict[str, list[GlossaryEntry]]
    en_index: dict[str, list[GlossaryEntry]]
    # for fuzzy matching
    tsn_forms: tuple[str, ...]
    en_forms: tuple[str, ...]


def _normalise(text: str) -> str:
    """Lowercase, strip accents, trim spaces."""
    text = text.strip().lower()
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _tokenise(text: str) -> list[str]:
    return [_normalise(m.group(0)) for m in WORD_RE.finditer(text)]


def _build_index(entries: list[GlossaryEntry]) -> GlossaryIndex:
    tsn_index: dict[str, list[GlossaryEntry]] = {}
    en_index: dict[str, list[GlossaryEntry]] = {}

    for entry in entries:
        for form in entry.all_setswana_forms:
            key = _normalise(form)
            if not key:
                continue
            tsn_index.setdefault(key, []).append(entry)

        for form in entry.all_english_forms:
            key = _normalise(form)
            if not key:
                continue
            en_index.setdefault(key, []).append(entry)

    return GlossaryIndex(
        entries=tuple(entries),
        tsn_index=tsn_index,
        en_index=en_index,
        tsn_forms=tuple(tsn_index.keys()),
        en_forms=tuple(en_index.keys()),
    )


@lru_cache
def get_glossary_index() -> GlossaryIndex:
    settings = get_settings()
    csv_path = settings.glossary_csv

    # If the file does not exist, return an empty index.
    if not csv_path or not Path(csv_path).is_file():
        return _build_index([])

    entries: list[GlossaryEntry] = []

    with Path(csv_path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            english_label = (row.get("english_label") or "").strip()
            if not english_label:
                continue

            english_pos = (row.get("english_pos") or "").strip() or None
            setswana_preferred = (row.get("setswana_preferred") or "").strip()
            if not setswana_preferred:
                continue

            setswana_pos = (row.get("setswana_pos") or "").strip() or None

            variants_raw = (row.get("setswana_variants") or "").strip()
            if variants_raw:
                variants = tuple(v.strip() for v in variants_raw.split("|") if v.strip())
            else:
                variants = ()

            entries.append(
                GlossaryEntry(
                    english_label=english_label,
                    english_pos=english_pos,
                    setswana_preferred=setswana_preferred,
                    setswana_variants=variants,
                    setswana_pos=setswana_pos,
                )
            )

    return _build_index(entries)


def _unique(entries: Iterable[GlossaryEntry]) -> list[GlossaryEntry]:
    seen: set[tuple[str, str]] = set()
    out: list[GlossaryEntry] = []
    for e in entries:
        key = (e.english_label, e.setswana_preferred)
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


def _score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if fuzz is not None:
        # RapidFuzz ratio is 0–100
        return float(fuzz.ratio(a, b))
    # Fallback: simple equality as 100, else 0
    return 100.0 if a == b else 0.0


def _match_tokens(
    tokens: list[str],
    index_map: dict[str, list[GlossaryEntry]],
    all_forms: tuple[str, ...],
    *,
    max_terms: int = 30,
    min_score: float = 80.0,
) -> list[GlossaryEntry]:
    """
    Match tokens against indexed forms, with exact then fuzzy matching.

    - Exact matches are included directly.
    - For non-exact tokens, we fuzzy match against all_forms.
    """
    if not tokens:
        return []

    matches: list[GlossaryEntry] = []

    # Exact matches
    for token in tokens:
        if token in index_map:
            matches.extend(index_map[token])

    # Fuzzy matches (only if RapidFuzz available OR you want difflib fallback)
    remaining_tokens = [t for t in tokens if t not in index_map]
    if remaining_tokens and all_forms:
        # If RapidFuzz is available, score against all_forms
        for token in remaining_tokens:
            best_score = 0.0
            best_forms: list[str] = []

            for form in all_forms:
                s = _score(token, form)
                if s >= min_score:
                    if s > best_score:
                        best_score = s
                        best_forms = [form]
                    elif s == best_score:
                        best_forms.append(form)

            if best_forms and best_score >= min_score:
                for form in best_forms:
                    matches.extend(index_map.get(form, []))

    # Deduplicate and limit
    return _unique(matches)[:max_terms]


def _entries_for_token(
    token: str,
    *,
    source: LangCode,
) -> list[GlossaryEntry]:
    """
    Return glossary entries that match a single *normalised* token
    for the given source language, using the same logic as _match_tokens.
    """
    idx = get_glossary_index()

    if source == "tsn":
        index_map = idx.tsn_index
        all_forms = idx.tsn_forms
    elif source == "en":
        index_map = idx.en_index
        all_forms = idx.en_forms
    else:
        raise ValueError(f"Unsupported source language: {source}")

    # Reuse _match_tokens, but with only this one token
    # Use a very high max_terms so we effectively don't limit per token
    return _match_tokens(
        tokens=[token],
        index_map=index_map,
        all_forms=all_forms,
        max_terms=999,
    )


def find_terms_for_tsn(text: str, max_terms: int = 30) -> list[GlossaryEntry]:
    """Find relevant glossary entries for Setswana source text."""
    idx = get_glossary_index()
    tokens = _tokenise(text)
    return _match_tokens(tokens, idx.tsn_index, idx.tsn_forms, max_terms=max_terms)


def find_terms_for_en(text: str, max_terms: int = 30) -> list[GlossaryEntry]:
    """Find relevant glossary entries for English source text."""
    idx = get_glossary_index()
    tokens = _tokenise(text)
    return _match_tokens(tokens, idx.en_index, idx.en_forms, max_terms=max_terms)


def preview_matches_for_text(
    text: str,
    source: LangCode,
) -> list[dict[str, object]]:
    """
    Debug helper: for a given text and source language, return a list of
    per-token matches:

      {
        "token": original_token_from_text,
        "normalised_token": normalised_form,
        "entries": [
          {
            "english_label": ...,
            "english_pos": ...,
            "setswana_preferred": ...,
            "setswana_variants": [...],
            "setswana_pos": ...,
          },
          ...
        ],
      }

    Only tokens that have at least one glossary match are included.
    """
    results: list[dict[str, object]] = []

    # Use WORD_RE directly so we can keep the original surface form,
    # then normalise it for lookup.
    for match in WORD_RE.finditer(text):
        raw_token = match.group(0)
        normalised = _normalise(raw_token)

        entries = _entries_for_token(normalised, source=source)
        if not entries:
            continue

        results.append(
            {
                "token": raw_token,
                "normalised_token": normalised,
                "entries": [
                    {
                        "english_label": e.english_label,
                        "english_pos": e.english_pos,
                        "setswana_preferred": e.setswana_preferred,
                        "setswana_variants": list(e.setswana_variants),
                        "setswana_pos": e.setswana_pos,
                    }
                    for e in entries
                ],
            }
        )

    return results
