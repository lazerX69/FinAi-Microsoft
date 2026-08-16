"""
Comprehensive tests for src/retriever.py.

Tests query normalization, RetrievalResult property accessors,
RetrievalResponse helpers, and the Retriever class using a mocked
VectorStore and mocked embedding function so no real models are needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import src.retriever as retriever_module
from src.retriever import (
    RetrievalResult,
    RetrievalResponse,
    Retriever,
    normalize_query,
    DEFAULT_MIN_SCORE,
    DEFAULT_MAX_CONTEXT_CHARACTERS,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestRetrieverConstants:
    def test_default_min_score_in_valid_range(self) -> None:
        assert 0.0 <= DEFAULT_MIN_SCORE <= 1.0

    def test_default_max_context_characters_positive(self) -> None:
        assert DEFAULT_MAX_CONTEXT_CHARACTERS > 0

    def test_min_query_length_positive(self) -> None:
        assert retriever_module.MIN_QUERY_LENGTH > 0

    def test_max_query_length_greater_than_min(self) -> None:
        assert retriever_module.MAX_QUERY_LENGTH > retriever_module.MIN_QUERY_LENGTH


# ---------------------------------------------------------------------------
# normalize_query
# ---------------------------------------------------------------------------

class TestNormalizeQuery:
    def test_strips_whitespace(self) -> None:
        result = normalize_query("  Enflasyon nedir?  ")
        assert result == result.strip()

    def test_too_short_raises(self) -> None:
        with pytest.raises((ValueError, RuntimeError)):
            normalize_query("a")

    def test_too_long_raises(self) -> None:
        with pytest.raises((ValueError, RuntimeError)):
            normalize_query("A" * 5_000)

    def test_valid_query_passes_through(self) -> None:
        result = normalize_query("Bilesik faiz nasil calisir?")
        assert len(result) > 0


# ---------------------------------------------------------------------------
# RetrievalResult properties
# ---------------------------------------------------------------------------

def _make_result(**overrides) -> RetrievalResult:
    defaults = dict(
        document="Bilesik faiz anaparanin faize tabi tutulmasidir.",
        metadata={
            "source": "03_tasarruf.txt",
            "file_name": "03_tasarruf.txt",
            "file_type": "txt",
            "page": 1,
            "chunk_index": 2,
            "global_chunk_index": 7,
            "character_count": 50,
            "content_hash": "abc",
            "document_title": "",
            "section_title": "Tasarruf",
            "subsection_title": "",
        },
        distance=0.05,
        score=0.95,
        rank=1,
        result_id="chunk-abc",
    )
    defaults.update(overrides)
    return RetrievalResult(**defaults)


class TestRetrievalResultProperties:
    def test_source_from_metadata(self) -> None:
        r = _make_result()
        assert r.source == "03_tasarruf.txt"

    def test_file_name_from_metadata(self) -> None:
        r = _make_result()
        assert r.file_name == "03_tasarruf.txt"

    def test_page_number_integer(self) -> None:
        r = _make_result()
        assert r.page_number == 1

    def test_page_number_none_when_missing(self) -> None:
        r = _make_result(metadata={"source": "x.txt"})
        assert r.page_number is None

    def test_chunk_index_integer(self) -> None:
        r = _make_result()
        assert r.chunk_index == 2

    def test_chunk_index_none_when_missing(self) -> None:
        r = _make_result(metadata={"source": "x.txt"})
        assert r.chunk_index is None

    def test_citation_includes_page(self) -> None:
        r = _make_result()
        citation = r.citation
        assert "03_tasarruf.txt" in citation
        assert "1" in citation  # page number

    def test_citation_without_page(self) -> None:
        r = _make_result(metadata={"source": "no_page.txt"})
        assert "no_page.txt" in r.citation

    def test_to_dict_has_required_keys(self) -> None:
        r = _make_result()
        d = r.to_dict()
        for key in ("rank", "document", "distance", "score", "source", "citation"):
            assert key in d

    def test_content_hash_returned(self) -> None:
        r = _make_result()
        assert r.content_hash == "abc"

    def test_score_stored_correctly(self) -> None:
        r = _make_result(score=0.87)
        assert abs(r.score - 0.87) < 1e-9


# ---------------------------------------------------------------------------
# RetrievalResponse
# ---------------------------------------------------------------------------

class TestRetrievalResponse:
    def _make_response(self, n_results: int = 2) -> RetrievalResponse:
        results = tuple(
            _make_result(rank=i + 1, score=0.9 - i * 0.1) for i in range(n_results)
        )
        return RetrievalResponse(
            query="Bilesik faiz nedir?",
            results=results,
            requested_top_k=4,
            searched_top_k=16,
            collection_count=540,
            min_score=0.55,
        )

    def test_has_query(self) -> None:
        resp = self._make_response()
        assert resp.query == "Bilesik faiz nedir?"

    def test_has_results(self) -> None:
        resp = self._make_response(n_results=3)
        assert len(resp.results) == 3

    def test_has_context_via_property(self) -> None:
        """RetrievalResponse exposes result_count, not a context string."""
        resp = self._make_response(n_results=2)
        assert resp.result_count == 2

    def test_empty_results_allowed(self) -> None:
        resp = RetrievalResponse(
            query="q",
            results=(),
            requested_top_k=4,
            searched_top_k=16,
            collection_count=0,
            min_score=0.55,
        )
        assert resp.results == ()

    def test_has_results_property_false_when_empty(self) -> None:
        resp = RetrievalResponse(
            query="q",
            results=(),
            requested_top_k=4,
            searched_top_k=16,
            collection_count=0,
            min_score=0.55,
        )
        assert resp.has_results is False

    def test_best_score_none_when_no_results(self) -> None:
        resp = RetrievalResponse(
            query="q",
            results=(),
            requested_top_k=4,
            searched_top_k=16,
            collection_count=0,
            min_score=0.55,
        )
        assert resp.best_score is None

    def test_best_score_when_results_present(self) -> None:
        resp = self._make_response(n_results=2)
        # first result has score 0.9
        assert resp.best_score is not None
        assert resp.best_score > 0.0



# ---------------------------------------------------------------------------
# Retriever class (mocked dependencies)
# ---------------------------------------------------------------------------

class TestRetriever:
    def _make_retriever(self, mock_store: MagicMock, fake_emb: list[float]) -> Retriever:
        with (
            patch("src.retriever.VectorStore", return_value=mock_store),
            patch("src.retriever.embed_texts", return_value=[fake_emb]),
        ):
            return Retriever()

    def test_document_count_delegates_to_store(
        self, mock_vector_store: MagicMock, fake_embedding: list[float]
    ) -> None:
        r = self._make_retriever(mock_vector_store, fake_embedding)
        assert r.document_count == 8

    def test_retrieve_returns_retrieval_response(
        self, mock_vector_store: MagicMock, fake_embedding: list[float]
    ) -> None:
        r = self._make_retriever(mock_vector_store, fake_embedding)
        with patch("src.retriever.embed_texts", return_value=[fake_embedding]):
            resp = r.retrieve("Bilesik faiz nedir?")
        assert isinstance(resp, RetrievalResponse)

    def test_retrieve_has_query_field(
        self, mock_vector_store: MagicMock, fake_embedding: list[float]
    ) -> None:
        r = self._make_retriever(mock_vector_store, fake_embedding)
        with patch("src.retriever.embed_texts", return_value=[fake_embedding]):
            resp = r.retrieve("Enflasyon nedir?")
        assert len(resp.query) > 0

    def test_retrieve_too_short_query_raises(
        self, mock_vector_store: MagicMock, fake_embedding: list[float]
    ) -> None:
        r = self._make_retriever(mock_vector_store, fake_embedding)
        with pytest.raises((ValueError, RuntimeError)):
            r.retrieve("a")
