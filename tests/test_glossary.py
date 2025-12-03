from __future__ import annotations

from pathlib import Path

import pytest

from sms_ai.glossary import (
    find_terms_for_en,
    find_terms_for_tsn,
    get_glossary_index,
)


def _normalise(text: str) -> str:
    """Helper to match the normalization used in glossary.py."""
    import unicodedata

    text = text.strip().lower()
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


@pytest.fixture
def sample_glossary_csv(tmp_path: Path) -> Path:
    """Create a temporary CSV file with sample glossary entries."""
    csv_path = tmp_path / "glossary.csv"
    content = """english_label,english_pos,setswana_preferred,setswana_variants,setswana_pos
abdomen,noun,mpa,,noun
absorb,verb,gapa,gabisa|gapa godimo,verb
absorption,noun,monyelo,,noun
acacia,noun,sika lootlharemmitlwa,,noun
accident,noun,kotsi,,noun
account,noun,letlotlo,,noun"""
    csv_path.write_text(content, encoding="utf-8")
    return csv_path


def test_load_glossary_index(sample_glossary_csv: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_glossary_index loads entries correctly."""
    # Clear cache first
    get_glossary_index.cache_clear()

    # Override the glossary path
    monkeypatch.setenv("GLOSSARY_CSV_PATH", str(sample_glossary_csv))

    # Reload settings to pick up the env var
    from sms_ai.config import get_settings

    get_settings.cache_clear()
    idx = get_glossary_index()

    assert len(idx.entries) == 6

    # Check that we have the expected entries
    entry_dict = {e.english_label: e for e in idx.entries}
    assert "abdomen" in entry_dict
    assert entry_dict["abdomen"].setswana_preferred == "mpa"
    assert entry_dict["absorb"].setswana_variants == ("gabisa", "gapa godimo")

    # Check indices are populated
    assert len(idx.tsn_index) > 0
    assert len(idx.en_index) > 0
    assert "mpa" in idx.tsn_index or _normalise("mpa") in idx.tsn_index
    assert "abdomen" in idx.en_index or _normalise("abdomen") in idx.en_index


def test_find_terms_for_tsn_exact_match(
    sample_glossary_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that find_terms_for_tsn finds exact matches."""
    get_glossary_index.cache_clear()
    monkeypatch.setenv("GLOSSARY_CSV_PATH", str(sample_glossary_csv))

    from sms_ai.config import get_settings

    get_settings.cache_clear()

    # Test exact match
    results = find_terms_for_tsn("mpa")
    assert len(results) > 0
    assert any(e.english_label == "abdomen" and e.setswana_preferred == "mpa" for e in results)


def test_find_terms_for_en_exact_match(
    sample_glossary_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that find_terms_for_en finds exact matches."""
    get_glossary_index.cache_clear()
    monkeypatch.setenv("GLOSSARY_CSV_PATH", str(sample_glossary_csv))

    from sms_ai.config import get_settings

    get_settings.cache_clear()

    # Test exact match
    results = find_terms_for_en("abdomen")
    assert len(results) > 0
    assert any(e.english_label == "abdomen" and e.setswana_preferred == "mpa" for e in results)


def test_find_terms_for_tsn_fuzzy_match(
    sample_glossary_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that find_terms_for_tsn finds fuzzy matches for misspelled terms."""
    get_glossary_index.cache_clear()
    monkeypatch.setenv("GLOSSARY_CSV_PATH", str(sample_glossary_csv))

    from sms_ai.config import get_settings

    get_settings.cache_clear()

    # Test fuzzy match with slight misspelling
    # "mpaa" should match "mpa" if RapidFuzz is available
    results = find_terms_for_tsn("mpaa")
    # If RapidFuzz is available, we should get a match
    # If not, we might not get a match (fallback only does exact matches)
    # So we check if we got results OR if RapidFuzz is not available
    import importlib.util

    if importlib.util.find_spec("rapidfuzz") is not None:
        # RapidFuzz is available, so we should get fuzzy matches
        assert len(results) > 0
        assert any(e.english_label == "abdomen" for e in results)
    # If RapidFuzz not available, fuzzy matching won't work
    # This is acceptable - the fallback only does exact matches


def test_find_terms_for_tsn_variants(
    sample_glossary_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that variants are also matched."""
    get_glossary_index.cache_clear()
    monkeypatch.setenv("GLOSSARY_CSV_PATH", str(sample_glossary_csv))

    from sms_ai.config import get_settings

    get_settings.cache_clear()

    # Test that variant "gabisa" matches the "absorb" entry
    results = find_terms_for_tsn("gabisa")
    assert len(results) > 0
    assert any(e.english_label == "absorb" for e in results)


def test_empty_glossary_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that missing glossary file returns empty index."""
    get_glossary_index.cache_clear()
    non_existent = tmp_path / "nonexistent.csv"
    monkeypatch.setenv("GLOSSARY_CSV_PATH", str(non_existent))

    from sms_ai.config import get_settings

    get_settings.cache_clear()

    idx = get_glossary_index()
    assert len(idx.entries) == 0
    assert len(idx.tsn_index) == 0
    assert len(idx.en_index) == 0
