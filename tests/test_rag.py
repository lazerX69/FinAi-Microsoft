"""
Comprehensive tests for src/rag.py.

Tests: RAGSource, RAGResponse, StreamEvent dataclasses, and the pure
helper functions (normalize_answer, tokenize_words, repeated_ngram_ratio,
contains_source_reference, answer_grounding_ratio, etc.).

The FinGuideRAG class (which requires live models) is tested with mocked
dependencies only.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import src.rag as rag_module
from src.rag import (
    RAGSource,
    RAGResponse,
    StreamEvent,
    normalize_answer,
    tokenize_words,
    repeated_ngram_ratio,
    contains_source_reference,
    answer_grounding_ratio,
    repeated_sentence_ratio,
    contains_unreliable_language,
    NO_CONTEXT_ANSWER,
    LOW_CONFIDENCE_ANSWER,
    DEFAULT_SYSTEM_PROMPT,
)


# ---------------------------------------------------------------------------
# Module-level type availability
# ---------------------------------------------------------------------------

class TestModuleTypes:
    def test_rag_response_importable(self) -> None:
        assert rag_module.RAGResponse is not None

    def test_rag_source_importable(self) -> None:
        assert rag_module.RAGSource is not None

    def test_stream_event_importable(self) -> None:
        assert rag_module.StreamEvent is not None

    def test_finguide_rag_importable(self) -> None:
        assert hasattr(rag_module, "FinGuideRAG")


# ---------------------------------------------------------------------------
# RAGSource
# ---------------------------------------------------------------------------

def _make_source(**overrides) -> RAGSource:
    defaults = dict(
        index=1,
        source="03_tasarruf.txt",
        page=1,
        chunk_index=0,
        score=0.88,
        text="Bilesik faiz anaparanin faize tabi tutulmasidir.",
    )
    defaults.update(overrides)
    return RAGSource(**defaults)


class TestRAGSource:
    def test_citation_with_page(self) -> None:
        s = _make_source(source="test.txt", page=3)
        assert "test.txt" in s.citation
        assert "3" in s.citation

    def test_citation_without_page(self) -> None:
        s = _make_source(source="test.txt", page=None)
        assert "test.txt" in s.citation
        assert "sayfa" not in s.citation

    def test_to_dict_has_required_keys(self) -> None:
        d = _make_source().to_dict()
        for key in ("index", "source", "page", "chunk_index", "score", "text", "citation"):
            assert key in d

    def test_score_stored(self) -> None:
        s = _make_source(score=0.77)
        assert abs(s.score - 0.77) < 1e-9

    def test_text_stored(self) -> None:
        s = _make_source(text="Test metni.")
        assert s.text == "Test metni."


# ---------------------------------------------------------------------------
# RAGResponse
# ---------------------------------------------------------------------------

def _make_rag_response(**overrides) -> RAGResponse:
    from src.retriever import RetrievalResponse
    retrieval = MagicMock(spec=RetrievalResponse)
    retrieval.to_dict.return_value = {}
    defaults = dict(
        question="Bilesik faiz nedir?",
        answer="Bilesik faiz anaparanin faize tabi tutulmasidir. [Kaynak 1]",
        sources=(_make_source(),),
        retrieval=retrieval,
        context="[Kaynak 1] Bilesik faiz metni.",
        used_fallback=False,
        generation_seconds=0.75,
    )
    defaults.update(overrides)
    return RAGResponse(**defaults)


class TestRAGResponse:
    def test_has_sources_true(self) -> None:
        resp = _make_rag_response()
        assert resp.has_sources is True

    def test_has_sources_false_when_empty(self) -> None:
        resp = _make_rag_response(sources=())
        assert resp.has_sources is False

    def test_source_count(self) -> None:
        resp = _make_rag_response(sources=(_make_source(), _make_source(index=2)))
        assert resp.source_count == 2

    def test_used_fallback_stored(self) -> None:
        resp = _make_rag_response(used_fallback=True)
        assert resp.used_fallback is True

    def test_generation_seconds_stored(self) -> None:
        resp = _make_rag_response(generation_seconds=1.23)
        assert abs(resp.generation_seconds - 1.23) < 1e-9

    def test_to_dict_has_required_keys(self) -> None:
        resp = _make_rag_response()
        d = resp.to_dict()
        for key in ("question", "answer", "used_fallback", "generation_seconds",
                    "source_count", "sources"):
            assert key in d

    def test_to_dict_sources_are_list(self) -> None:
        resp = _make_rag_response()
        assert isinstance(resp.to_dict()["sources"], list)


# ---------------------------------------------------------------------------
# StreamEvent
# ---------------------------------------------------------------------------

class TestStreamEvent:
    def test_event_stored(self) -> None:
        e = StreamEvent(event="token", text="Bilesik ")
        assert e.event == "token"
        assert e.text == "Bilesik "

    def test_response_defaults_to_none(self) -> None:
        e = StreamEvent(event="retrieval_started")
        assert e.response is None


# ---------------------------------------------------------------------------
# normalize_answer
# ---------------------------------------------------------------------------

class TestNormalizeAnswer:
    def test_strips_whitespace(self) -> None:
        result = normalize_answer("  Cevap.  ")
        assert result == result.strip()

    def test_removes_null_bytes(self) -> None:
        result = normalize_answer("Cevap\x00Metni")
        assert "\x00" not in result

    def test_collapses_triple_newlines(self) -> None:
        result = normalize_answer("A.\n\n\n\nB.")
        assert "\n\n\n" not in result

    def test_non_string_returns_empty(self) -> None:
        result = normalize_answer(None)  # type: ignore[arg-type]
        assert result == ""

    def test_removes_duplicate_consecutive_lines(self) -> None:
        text = "Bilesik faiz onemlidir.\nBilesik faiz onemlidir."
        result = normalize_answer(text)
        assert result.count("Bilesik faiz onemlidir") == 1


# ---------------------------------------------------------------------------
# tokenize_words
# ---------------------------------------------------------------------------

class TestTokenizeWords:
    def test_returns_list(self) -> None:
        assert isinstance(tokenize_words("Merhaba dunya."), list)

    def test_splits_on_punctuation(self) -> None:
        words = tokenize_words("Faiz, risk ve getiri.")
        assert "faiz" in words
        assert "risk" in words
        assert "getiri" in words

    def test_empty_string_returns_empty(self) -> None:
        assert tokenize_words("") == []

    def test_all_lowercase(self) -> None:
        words = tokenize_words("Enflasyon ARTIS")
        for w in words:
            assert w == w.lower() or w == w.casefold()


# ---------------------------------------------------------------------------
# repeated_ngram_ratio
# ---------------------------------------------------------------------------

class TestRepeatedNgramRatio:
    def test_no_repetition_returns_low_value(self) -> None:
        words = "a b c d e f g h i j k l m".split()
        ratio = repeated_ngram_ratio(words)
        assert ratio < 0.3

    def test_all_repetition_returns_high_value(self) -> None:
        words = ("bilesik faiz " * 20).split()
        ratio = repeated_ngram_ratio(words)
        assert ratio > 0.5

    def test_short_text_returns_zero(self) -> None:
        ratio = repeated_ngram_ratio(["a", "b"])
        assert ratio == 0.0

    def test_empty_returns_zero(self) -> None:
        assert repeated_ngram_ratio([]) == 0.0


# ---------------------------------------------------------------------------
# contains_source_reference
# ---------------------------------------------------------------------------

class TestContainsSourceReference:
    def test_detects_kaynak_tag(self) -> None:
        assert contains_source_reference("Cevap buradadir. [Kaynak 1]")

    def test_detects_case_insensitive(self) -> None:
        assert contains_source_reference("[kaynak 2]")

    def test_no_reference_returns_false(self) -> None:
        assert not contains_source_reference("Hicbir kaynak belirtilmedi.")

    def test_empty_returns_false(self) -> None:
        assert not contains_source_reference("")


# ---------------------------------------------------------------------------
# answer_grounding_ratio
# ---------------------------------------------------------------------------

class TestAnswerGroundingRatio:
    def test_fully_grounded_answer(self) -> None:
        answer = "bilesik faiz anaparanin faize tabi tutulmasidir"
        source = "bilesik faiz anaparanin faize tabi tutulmasidir"
        ratio = answer_grounding_ratio(answer, [source])
        assert ratio > 0.7

    def test_ungrounded_answer_low_ratio(self) -> None:
        answer = "mars uzayda bir gezegendir rotasyon hizli"
        source = "bilesik faiz anaparanin faize tabi tutulmasidir"
        ratio = answer_grounding_ratio(answer, [source])
        assert ratio < 0.4

    def test_empty_answer_returns_zero(self) -> None:
        assert answer_grounding_ratio("", ["kaynak metni"]) == 0.0

    def test_empty_sources_returns_zero(self) -> None:
        assert answer_grounding_ratio("cevap metni", []) == 0.0


# ---------------------------------------------------------------------------
# repeated_sentence_ratio
# ---------------------------------------------------------------------------

class TestRepeatedSentenceRatio:
    def test_no_repetition_returns_zero(self) -> None:
        text = "Faiz artti. Enflasyon dusuk kaldi. Tasarruf onemlidir."
        ratio = repeated_sentence_ratio(text)
        assert ratio < 0.3

    def test_full_repetition_returns_high(self) -> None:
        sent = "Bilesik faiz onemlidir."
        text = " ".join([sent] * 5)
        ratio = repeated_sentence_ratio(text)
        assert ratio > 0.5

    def test_single_sentence_returns_zero(self) -> None:
        ratio = repeated_sentence_ratio("Tek cumle.")
        assert ratio == 0.0


# ---------------------------------------------------------------------------
# contains_unreliable_language
# ---------------------------------------------------------------------------

class TestContainsUnreliableLanguage:
    def test_ai_self_reference_detected(self) -> None:
        assert contains_unreliable_language("bir yapay zeka modeli olarak cevap veremem")

    def test_english_ai_disclaimer_detected(self) -> None:
        assert contains_unreliable_language("as an AI I cannot provide advice")

    def test_clean_answer_not_flagged(self) -> None:
        assert not contains_unreliable_language(
            "Bilesik faiz anaparanin faize tabi tutulmasidir."
        )

    def test_empty_string_not_flagged(self) -> None:
        assert not contains_unreliable_language("")


# ---------------------------------------------------------------------------
# Constant values sanity checks
# ---------------------------------------------------------------------------

class TestConstants:
    def test_no_context_answer_non_empty(self) -> None:
        assert len(NO_CONTEXT_ANSWER) > 10

    def test_low_confidence_answer_non_empty(self) -> None:
        assert len(LOW_CONFIDENCE_ANSWER) > 10

    def test_default_system_prompt_has_rules(self) -> None:
        # System prompt should contain at least one numbered rule
        assert "1." in DEFAULT_SYSTEM_PROMPT
