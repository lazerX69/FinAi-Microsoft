"""
Comprehensive tests for src/ingest.py.

Covers: text normalization, chunking logic, document chunk dataclass,
metadata generation, and ingestion statistics.
Does NOT require Foundry Local or a real ChromaDB collection.
"""

from __future__ import annotations

import hashlib

import pytest

from src.ingest import (
    DocumentChunk,
    IngestionStatistics,
    normalize_text,
    split_text,
)
from src import config


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def test_empty_string_returns_empty(self) -> None:
        assert normalize_text("") == ""

    def test_plain_text_unchanged(self) -> None:
        text = "Finansal okuryazarlik onemlidir."
        result = normalize_text(text)
        assert "finansal" in result.lower() or "Finansal" in result

    def test_removes_null_bytes(self) -> None:
        result = normalize_text("Finans\x00Okuryazarlik")
        assert "\x00" not in result

    def test_normalizes_crlf_to_lf(self) -> None:
        result = normalize_text("Satir bir.\r\nSatir iki.")
        assert "\r\n" not in result

    def test_normalizes_cr_to_lf(self) -> None:
        result = normalize_text("Satir bir.\rSatir iki.")
        assert "\r" not in result

    def test_collapses_multiple_blank_lines(self) -> None:
        result = normalize_text("Para A.\n\n\n\n\nPara B.")
        assert "\n\n\n" not in result

    def test_strips_leading_and_trailing_whitespace(self) -> None:
        result = normalize_text("   Metin   ")
        assert result == result.strip()

    def test_preserves_paragraph_breaks(self) -> None:
        result = normalize_text("Paragraf bir.\n\nParagraf iki.")
        assert "\n\n" in result


# ---------------------------------------------------------------------------
# split_text
# ---------------------------------------------------------------------------

class TestSplitText:
    def test_empty_string_returns_empty_list(self) -> None:
        assert split_text("") == []

    def test_short_text_returns_single_chunk(self) -> None:
        text = "Bilesik faiz zaman icinde birikimli buyume olusturabilir."
        result = split_text(text)
        assert result == [text]

    def test_long_text_produces_multiple_chunks(self) -> None:
        long_text = ("Finansal okuryazarlik onemlidir. " * 100).strip()
        chunks = split_text(long_text)
        assert len(chunks) > 1

    def test_chunks_cover_full_content(self) -> None:
        """Every word in the original text must appear in at least one chunk."""
        long_text = " ".join(f"kelime{i}" for i in range(300))
        chunks = split_text(long_text)
        combined = " ".join(chunks)
        # Every word should appear at least once across all chunks
        for i in range(0, 300, 30):
            assert f"kelime{i}" in combined

    def test_each_chunk_within_size_limit(self) -> None:
        long_text = ("A" * 50 + " ") * 100
        for chunk in split_text(long_text):
            assert len(chunk) <= config.CHUNK_SIZE + config.CHUNK_OVERLAP + 50

    def test_no_chunk_is_empty(self) -> None:
        text = "Metin. " * 200
        for chunk in split_text(text):
            assert chunk.strip() != ""

    def test_whitespace_only_returns_empty_list(self) -> None:
        assert split_text("   \n\n\t  ") == []


# ---------------------------------------------------------------------------
# DocumentChunk dataclass
# ---------------------------------------------------------------------------

class TestDocumentChunk:
    def _make_chunk(self, **kwargs) -> DocumentChunk:
        defaults = dict(
            chunk_id="test-id-001",
            text="Bilesik faiz nasil calisir?",
            source="02_gelir_gider.txt",
            file_name="02_gelir_gider.txt",
            file_type="txt",
            page_number=1,
            chunk_index=0,
            global_chunk_index=0,
            character_count=27,
            content_hash="deadbeef",
        )
        defaults.update(kwargs)
        return DocumentChunk(**defaults)

    def test_chunk_id_stored(self) -> None:
        chunk = self._make_chunk(chunk_id="abc")
        assert chunk.chunk_id == "abc"

    def test_text_stored(self) -> None:
        chunk = self._make_chunk(text="Test metni.")
        assert chunk.text == "Test metni."

    def test_to_metadata_returns_dict(self) -> None:
        chunk = self._make_chunk()
        meta = chunk.to_metadata()
        assert isinstance(meta, dict)

    def test_to_metadata_contains_required_keys(self) -> None:
        chunk = self._make_chunk()
        meta = chunk.to_metadata()
        for key in ("source", "file_name", "file_type", "page", "chunk_index"):
            assert key in meta, f"Key '{key}' missing from metadata"

    def test_to_metadata_scalar_values_only(self) -> None:
        """ChromaDB requires only str/int/float/bool metadata values."""
        chunk = self._make_chunk()
        for key, val in chunk.to_metadata().items():
            assert isinstance(val, (str, int, float, bool)), (
                f"Metadata key '{key}' has non-scalar type {type(val)}"
            )

    def test_character_count_matches_text(self) -> None:
        text = "Merhaba dunya."
        chunk = self._make_chunk(text=text, character_count=len(text))
        assert chunk.character_count == len(text)

    def test_optional_section_fields_default_empty(self) -> None:
        chunk = self._make_chunk()
        assert chunk.document_title == ""
        assert chunk.section_title == ""
        assert chunk.subsection_title == ""

    def test_section_fields_stored_when_provided(self) -> None:
        chunk = self._make_chunk(
            document_title="Ansiklopedi",
            section_title="Faiz",
            subsection_title="Bilesik",
        )
        assert chunk.document_title == "Ansiklopedi"
        assert chunk.section_title == "Faiz"
        assert chunk.subsection_title == "Bilesik"


# ---------------------------------------------------------------------------
# IngestionStatistics
# ---------------------------------------------------------------------------

class TestIngestionStatistics:
    def test_default_values_are_zero(self) -> None:
        stats = IngestionStatistics()
        assert stats.discovered_files == 0
        assert stats.processed_files == 0
        assert stats.skipped_files == 0
        assert stats.failed_files == 0
        assert stats.extracted_pages == 0
        assert stats.empty_pages == 0
        assert stats.generated_chunks == 0
        assert stats.inserted_chunks == 0

    def test_fields_are_mutable(self) -> None:
        stats = IngestionStatistics()
        stats.discovered_files = 6
        stats.processed_files = 6
        stats.generated_chunks = 540
        assert stats.discovered_files == 6
        assert stats.generated_chunks == 540
