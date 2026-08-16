"""
Comprehensive tests for src/foundry_client.py.

Tests GenerationSettings validation, message validation helpers,
stream text extraction, and the FoundryClient class interface.
No real Foundry Local runtime is invoked — the SDK manager is mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.foundry_client import (
    GenerationSettings,
    FoundryClientStatus,
    validate_messages,
    extract_stream_text,
    normalize_text,
    FoundryClient,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_MAX_TOKENS,
)


# ---------------------------------------------------------------------------
# Module-level availability
# ---------------------------------------------------------------------------

class TestModuleAvailability:
    def test_foundry_client_class_importable(self) -> None:
        assert FoundryClient is not None

    def test_generation_settings_importable(self) -> None:
        assert GenerationSettings is not None

    def test_validate_messages_importable(self) -> None:
        assert callable(validate_messages)

    def test_extract_stream_text_importable(self) -> None:
        assert callable(extract_stream_text)


# ---------------------------------------------------------------------------
# GenerationSettings
# ---------------------------------------------------------------------------

class TestGenerationSettings:
    def test_defaults_are_correct(self) -> None:
        s = GenerationSettings()
        assert s.temperature == DEFAULT_TEMPERATURE
        assert s.top_p == DEFAULT_TOP_P
        assert s.max_tokens == DEFAULT_MAX_TOKENS

    def test_valid_custom_settings(self) -> None:
        s = GenerationSettings(temperature=0.5, top_p=0.9, max_tokens=256)
        assert s.temperature == 0.5
        assert s.top_p == 0.9
        assert s.max_tokens == 256

    def test_temperature_zero_is_valid(self) -> None:
        s = GenerationSettings(temperature=0.0)
        assert s.temperature == 0.0

    def test_temperature_two_is_valid(self) -> None:
        s = GenerationSettings(temperature=2.0)
        assert s.temperature == 2.0

    def test_temperature_negative_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            GenerationSettings(temperature=-0.1)

    def test_temperature_above_two_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            GenerationSettings(temperature=2.1)

    def test_temperature_bool_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            GenerationSettings(temperature=True)

    def test_top_p_zero_is_valid(self) -> None:
        s = GenerationSettings(top_p=0.0)
        assert s.top_p == 0.0

    def test_top_p_one_is_valid(self) -> None:
        s = GenerationSettings(top_p=1.0)
        assert s.top_p == 1.0

    def test_top_p_above_one_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            GenerationSettings(top_p=1.1)

    def test_max_tokens_one_is_valid(self) -> None:
        s = GenerationSettings(max_tokens=1)
        assert s.max_tokens == 1

    def test_max_tokens_zero_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            GenerationSettings(max_tokens=0)

    def test_max_tokens_float_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            GenerationSettings(max_tokens=100.5)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# validate_messages
# ---------------------------------------------------------------------------

class TestValidateMessages:
    def test_valid_single_user_message(self) -> None:
        msgs = [{"role": "user", "content": "Enflasyon nedir?"}]
        result = validate_messages(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_valid_system_plus_user(self) -> None:
        msgs = [
            {"role": "system", "content": "Sen bir finansal asistansin."},
            {"role": "user", "content": "Bilesik faiz nedir?"},
        ]
        result = validate_messages(msgs)
        assert len(result) == 2

    def test_empty_messages_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            validate_messages([])

    def test_non_sequence_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            validate_messages("Bu bir mesaj")  # type: ignore[arg-type]

    def test_missing_user_role_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_messages([{"role": "system", "content": "Sistem mesaji."}])

    def test_invalid_role_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_messages([{"role": "gecelik", "content": "Bilinmeyen rol."}])

    def test_empty_content_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            validate_messages([{"role": "user", "content": ""}])

    def test_whitespace_content_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            validate_messages([{"role": "user", "content": "   "}])

    def test_roles_normalized_to_lowercase(self) -> None:
        msgs = [{"role": "USER", "content": "Sorum var."}]
        result = validate_messages(msgs)
        assert result[0]["role"] == "user"

    def test_assistant_role_accepted_with_user(self) -> None:
        msgs = [
            {"role": "user", "content": "Sorum bir."},
            {"role": "assistant", "content": "Cevabim."},
            {"role": "user", "content": "Sorum iki."},
        ]
        result = validate_messages(msgs)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# normalize_text (foundry_client internal)
# ---------------------------------------------------------------------------

class TestFoundryNormalizeText:
    def test_strips_whitespace(self) -> None:
        result = normalize_text("  Metin  ", field_name="test")
        assert result == result.strip()

    def test_removes_null_bytes(self) -> None:
        result = normalize_text("Faiz\x00Orani", field_name="test")
        assert "\x00" not in result

    def test_empty_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            normalize_text("", field_name="test")

    def test_non_string_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            normalize_text(123, field_name="test")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# extract_stream_text
# ---------------------------------------------------------------------------

class TestExtractStreamText:
    def _make_chunk(self, content: str | None) -> MagicMock:
        chunk = MagicMock()
        if content is None:
            chunk.choices = []
        else:
            delta = MagicMock()
            delta.content = content
            choice = MagicMock()
            choice.delta = delta
            chunk.choices = [choice]
        return chunk

    def test_none_chunk_returns_empty(self) -> None:
        assert extract_stream_text(None) == ""

    def test_empty_choices_returns_empty(self) -> None:
        chunk = self._make_chunk(None)
        assert extract_stream_text(chunk) == ""

    def test_valid_content_returned(self) -> None:
        chunk = self._make_chunk("Bilesik ")
        result = extract_stream_text(chunk)
        assert "Bilesik" in result


# ---------------------------------------------------------------------------
# FoundryClientStatus
# ---------------------------------------------------------------------------

class TestFoundryClientStatus:
    def test_all_fields_stored(self) -> None:
        status = FoundryClientStatus(
            model_alias="qwen3.5-2b-text",
            manager_initialized=True,
            model_resolved=True,
            model_loaded=True,
            client_ready=True,
            closed=False,
        )
        assert status.model_alias == "qwen3.5-2b-text"
        assert status.client_ready is True
        assert status.closed is False
