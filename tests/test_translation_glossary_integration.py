from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from sms_ai.language import translate


class FakeModel:
    """Fake model that records messages and returns a dummy response."""

    def __init__(self) -> None:
        self.called_messages: list[list[Any]] = []

    def invoke(self, messages: list[Any]) -> Any:
        """Record messages and return a fake response."""
        self.called_messages.append(messages)
        return type("Response", (), {"content": "dummy translation"})()


@pytest.fixture
def sample_glossary_csv(tmp_path: Path) -> Path:
    """Create a temporary CSV file with sample glossary entries."""
    csv_path = tmp_path / "glossary.csv"
    content = """english_label,english_pos,setswana_preferred,setswana_variants,setswana_pos
abdomen,noun,mpa,,noun
absorb,verb,gapa,gabisa|gapa godimo,verb
absorption,noun,monyelo,,noun"""
    csv_path.write_text(content, encoding="utf-8")
    return csv_path


def test_translate_tsn_to_en_includes_glossary(
    sample_glossary_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that translate includes glossary in system prompt for tsn->en."""
    # Clear caches
    from sms_ai.config import get_settings
    from sms_ai.glossary import get_glossary_index

    get_settings.cache_clear()
    get_glossary_index.cache_clear()

    # Override glossary path
    monkeypatch.setenv("GLOSSARY_CSV_PATH", str(sample_glossary_csv))

    # Create fake model
    fake_model = FakeModel()

    # Monkeypatch get_translation_model
    def get_fake_model() -> FakeModel:
        return fake_model

    monkeypatch.setattr("sms_ai.language.get_translation_model", get_fake_model)

    # Call translate with a term that should match
    result = translate("mpa", source="tsn", target="en")

    # Verify the model was called
    assert len(fake_model.called_messages) == 1
    messages = fake_model.called_messages[0]

    # Verify message structure
    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)

    # Verify system message contains glossary
    # Type checker can't infer the exact type of SystemMessage.content (it's a union type),
    # but we know from the context that it's a string here. The cast is safe.
    system_content = cast(str, messages[0].content)  # type: ignore[reportUnknownMemberType]
    assert "Setswana → English glossary" in system_content
    assert "mpa" in system_content
    assert "abdomen" in system_content

    # Verify result
    assert result == "dummy translation"


def test_translate_en_to_tsn_includes_glossary(
    sample_glossary_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that translate includes glossary in system prompt for en->tsn."""
    # Clear caches
    from sms_ai.config import get_settings
    from sms_ai.glossary import get_glossary_index

    get_settings.cache_clear()
    get_glossary_index.cache_clear()

    # Override glossary path
    monkeypatch.setenv("GLOSSARY_CSV_PATH", str(sample_glossary_csv))

    # Create fake model
    fake_model = FakeModel()

    # Monkeypatch get_translation_model
    def get_fake_model() -> FakeModel:
        return fake_model

    monkeypatch.setattr("sms_ai.language.get_translation_model", get_fake_model)

    # Call translate with a term that should match
    result = translate("abdomen", source="en", target="tsn")

    # Verify the model was called
    assert len(fake_model.called_messages) == 1
    messages = fake_model.called_messages[0]

    # Verify message structure
    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)

    # Verify system message contains glossary
    # Type checker can't infer the exact type of SystemMessage.content (it's a union type),
    # but we know from the context that it's a string here. The cast is safe.
    system_content = cast(str, messages[0].content)  # type: ignore[reportUnknownMemberType]
    assert "English → Setswana glossary" in system_content
    assert "abdomen" in system_content
    assert "mpa" in system_content

    # Verify result
    assert result == "dummy translation"


def test_translate_no_glossary_match(
    sample_glossary_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that translate works even when no glossary terms match."""
    # Clear caches
    from sms_ai.config import get_settings
    from sms_ai.glossary import get_glossary_index

    get_settings.cache_clear()
    get_glossary_index.cache_clear()

    # Override glossary path
    monkeypatch.setenv("GLOSSARY_CSV_PATH", str(sample_glossary_csv))

    # Create fake model
    fake_model = FakeModel()

    # Monkeypatch get_translation_model
    def get_fake_model() -> FakeModel:
        return fake_model

    monkeypatch.setattr("sms_ai.language.get_translation_model", get_fake_model)

    # Call translate with text that doesn't match any glossary terms
    result = translate("hello world", source="en", target="tsn")

    # Verify the model was called
    assert len(fake_model.called_messages) == 1
    messages = fake_model.called_messages[0]

    # Verify message structure
    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)

    # Verify system message does NOT contain glossary (no matches)
    # Type checker can't infer the exact type of SystemMessage.content (it's a union type),
    # but we know from the context that it's a string here. The cast is safe.
    system_content = cast(str, messages[0].content)  # type: ignore[reportUnknownMemberType]
    # If no glossary terms match, the glossary prompt should be empty
    # So "English → Setswana glossary" should not appear
    assert "English → Setswana glossary" not in system_content

    # Verify result
    assert result == "dummy translation"


def test_translate_missing_glossary_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that translate works even when glossary file is missing."""
    # Clear caches
    from sms_ai.config import get_settings
    from sms_ai.glossary import get_glossary_index

    get_settings.cache_clear()
    get_glossary_index.cache_clear()

    # Set non-existent glossary path
    non_existent = tmp_path / "nonexistent.csv"
    monkeypatch.setenv("GLOSSARY_CSV_PATH", str(non_existent))

    # Create fake model
    fake_model = FakeModel()

    # Monkeypatch get_translation_model
    def get_fake_model() -> FakeModel:
        return fake_model

    monkeypatch.setattr("sms_ai.language.get_translation_model", get_fake_model)

    # Call translate - should work without errors
    result = translate("hello", source="en", target="tsn")

    # Verify the model was called
    assert len(fake_model.called_messages) == 1
    assert result == "dummy translation"
