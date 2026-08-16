"""
Integration Q&A Test Suite for FinGuide AI.

PURPOSE
-------
This file documents and tests the expected behavior of the full RAG pipeline
against a curated set of real questions. It serves two roles:

  1. Living documentation: Every entry in QA_CASES is an explicit contract
     about what the system should and should not answer.

  2. Automated regression test: Running this suite after any change to models,
     prompts, chunk settings, or documents verifies no regression occurred.

HOW TO RUN
----------
  # All integration tests (requires populated ChromaDB collection):
  python -m pytest tests/integration/ -v

  # Skip integration tests (fast CI without a populated DB):
  python -m pytest -m "not integration" -v

  # Run only integration tests:
  python -m pytest -m "integration" -v

PREREQUISITE
------------
Before running these tests you must have indexed the documents:

  python -m src.ingest

MARKERS
-------
All tests here are marked ``integration``. They are deliberately excluded
from fast CI runs so that unit tests remain quick and offline.

Q&A TEST CASE CATALOG
----------------------
Below is the complete, human-readable catalog of test cases. Each entry
documents the expected system behavior so reviewers can understand intent
without running the code.

+-----+-----------------------------------------------------------+-----------+-------------------------------------------+
| ID  | Question                                                  | Answerable| Expected behavior / key terms             |
+-----+-----------------------------------------------------------+-----------+-------------------------------------------+
| Q01 | Bilesik faiz nedir?                                       | YES       | answer includes "ana para" or "faiz"      |
| Q02 | Enflasyon satın alma gücünü nasıl etkiler?                | YES       | answer includes "satın alma" or "deger"   |
| Q03 | Acil durum fonu neden önemlidir?                          | YES       | answer includes "beklenmedik" or "guvenli"|
| Q04 | Yatırım fonu nedir?                                       | YES       | answer includes "fon" or "yatirim"        |
| Q05 | Likidite ile ne kastedilir?                               | YES       | answer includes "nakde" or "likidite"     |
| Q06 | Risk ve getiri arasındaki ilişki nedir?                   | YES       | answer includes "risk" and "getiri"       |
| Q07 | Net değer nasıl hesaplanır?                               | YES       | answer includes "varlik" or "yukumluluk"  |
| Q08 | Bütçe planlaması neden yapılmalıdır?                      | YES       | answer includes "butce" or "gider"        |
| Q09 | Mars'ta tarım yapmak için hangi gübre kullanılmalıdır?    | NO        | system must refuse / say it does not know |
| Q10 | Uzayda yıldızlar nasıl oluşur?                            | NO        | system must refuse / say it does not know |
| Q11 | Futbolda ofsayt kuralı nedir?                             | NO        | system must refuse / say it does not know |
| Q12 | (empty string)                                            | ERROR     | should raise or return validation error   |
+-----+-----------------------------------------------------------+-----------+-------------------------------------------+
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Mark all tests in this module as integration tests.
# Run with:  pytest -m integration
# Skip with: pytest -m "not integration"
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rag_service():
    """
    Load a real FinGuideRAG instance once for the whole module.

    This fixture is SLOW (loads embedding model and locates Foundry service).
    It is only used in integration tests, never in unit tests.
    """
    from src.rag import FinGuideRAG
    service = FinGuideRAG()
    yield service


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _ask(rag_service, question: str, use_llm: bool = False):
    """Send a question and return the RAGResponse."""
    return rag_service.answer(question, use_llm=use_llm)


def _ask_llm(rag_service, question: str):
    """Send a question using LLM generation."""
    return rag_service.answer(question, use_llm=True)


# ---------------------------------------------------------------------------
# Q01 – Q08: Answerable questions (topic is covered in the knowledge base)
# ---------------------------------------------------------------------------

class TestAnswerableQuestions:
    """
    The system MUST return a meaningful, source-grounded answer for these
    questions. All of them are covered in the 6 indexed document files.
    """

    def test_q01_bilesik_faiz(self, rag_service) -> None:
        """Q01: Bilesik faiz nedir?"""
        resp = _ask(rag_service, "Bilesik faiz nedir?")
        assert resp.has_sources, "Q01: Expected at least one source to be retrieved"
        lower = resp.answer.lower()
        assert any(term in lower for term in ("faiz", "ana para", "anapara", "birikimli")), (
            f"Q01: Answer does not contain expected terms. Got: {resp.answer[:200]}"
        )

    def test_q02_enflasyon_etkisi(self, rag_service) -> None:
        """Q02: Enflasyon satın alma gücünü nasıl etkiler?"""
        resp = _ask(rag_service, "Enflasyon satin alma gucunu nasil etkiler?")
        assert resp.has_sources, "Q02: Expected sources"
        lower = resp.answer.lower()
        assert any(term in lower for term in ("enflasyon", "satin alma", "guc", "artis")), (
            f"Q02: Unexpected answer: {resp.answer[:200]}"
        )

    def test_q03_acil_durum_fonu(self, rag_service) -> None:
        """Q03: Acil durum fonu neden önemlidir?"""
        resp = _ask(rag_service, "Acil durum fonu neden onemlidir?")
        assert resp.has_sources, "Q03: Expected sources"
        lower = resp.answer.lower()
        assert any(term in lower for term in ("acil", "beklenmedik", "fon", "guvenlik")), (
            f"Q03: Unexpected answer: {resp.answer[:200]}"
        )

    def test_q04_yatirim_fonu(self, rag_service) -> None:
        """Q04: Yatırım fonu nedir?"""
        resp = _ask(rag_service, "Yatirim fonu nedir?")
        assert resp.has_sources, "Q04: Expected sources"
        lower = resp.answer.lower()
        assert any(term in lower for term in ("fon", "yatirim", "kolektif", "yatirimci")), (
            f"Q04: Unexpected answer: {resp.answer[:200]}"
        )

    def test_q05_likidite(self, rag_service) -> None:
        """Q05: Likidite ile ne kastedilir?"""
        resp = _ask(rag_service, "Likidite ile ne kastedilir?")
        assert resp.has_sources, "Q05: Expected sources"
        lower = resp.answer.lower()
        assert any(term in lower for term in ("likidite", "nakde", "varlik", "hizlica")), (
            f"Q05: Unexpected answer: {resp.answer[:200]}"
        )

    def test_q06_risk_getiri_iliskisi(self, rag_service) -> None:
        """Q06: Risk ve getiri arasındaki ilişki nedir?"""
        resp = _ask(rag_service, "Risk ve getiri arasindaki iliski nedir?")
        assert resp.has_sources, "Q06: Expected sources"
        lower = resp.answer.lower()
        assert "risk" in lower, f"Q06: 'risk' missing from answer: {resp.answer[:200]}"
        assert any(term in lower for term in ("getiri", "iliski", "yuksek")), (
            f"Q06: Expected 'getiri' or 'iliski': {resp.answer[:200]}"
        )

    def test_q07_net_deger(self, rag_service) -> None:
        """Q07: Net değer nasıl hesaplanır?"""
        resp = _ask(rag_service, "Net deger nasil hesaplanir?")
        assert resp.has_sources, "Q07: Expected sources"
        lower = resp.answer.lower()
        assert any(term in lower for term in ("net", "varlik", "yukumluluk", "deger")), (
            f"Q07: Unexpected answer: {resp.answer[:200]}"
        )

    def test_q08_butce_planlamasi(self, rag_service) -> None:
        """Q08: Bütçe planlaması neden yapılmalıdır?"""
        resp = _ask(rag_service, "Butce planlamasi neden yapilmalidir?")
        assert resp.has_sources, "Q08: Expected sources"
        lower = resp.answer.lower()
        assert any(term in lower for term in ("butce", "gider", "gelir", "plan")), (
            f"Q08: Unexpected answer: {resp.answer[:200]}"
        )


# ---------------------------------------------------------------------------
# Q09 – Q11: Unanswerable / off-topic questions
# ---------------------------------------------------------------------------

class TestUnanswerableQuestions:
    """
    The system MUST refuse to answer these questions because they are
    completely outside the financial literacy domain. The retriever should
    return no (or low-confidence) results, and the answer should be a safe
    fallback message — NOT a hallucinated response.
    """

    REFUSAL_INDICATORS = (
        "cevaplayamiyorum",
        "bulunamadi",
        "yeterli",
        "bilgi",
        "kaynak",
        "ekleyin",
        "ilgili",
        "bilinmiyor",
    )

    def _assert_refusal(self, resp, question_id: str) -> None:
        lower = resp.answer.lower()
        # Either: no sources retrieved, OR answer contains refusal language
        has_refusal_language = any(
            indicator in lower for indicator in self.REFUSAL_INDICATORS
        )
        no_high_confidence_sources = (
            not resp.has_sources
            or all(s.score < 0.60 for s in resp.sources)
        )
        assert has_refusal_language or no_high_confidence_sources, (
            f"{question_id}: System should refuse or express uncertainty. "
            f"Got: {resp.answer[:300]}"
        )

    def test_q09_mars_tarim(self, rag_service) -> None:
        """Q09: Mars'ta tarım — completely off-topic."""
        resp = _ask(rag_service, "Mars'ta tarim yapmak icin hangi gubre kullanilmalidir?")
        self._assert_refusal(resp, "Q09")

    def test_q10_yildizlar(self, rag_service) -> None:
        """Q10: Star formation — astronomy, not finance."""
        resp = _ask(rag_service, "Uzayda yildizlar nasil olusur?")
        self._assert_refusal(resp, "Q10")

    def test_q11_ofsayt(self, rag_service) -> None:
        """Q11: Football offside rule — sports, not finance."""
        resp = _ask(rag_service, "Futbolda ofsayt kurali nedir?")
        self._assert_refusal(resp, "Q11")


# ---------------------------------------------------------------------------
# Q12: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_q12_empty_question_raises_or_returns_error(self, rag_service) -> None:
        """Q12: Empty input must be rejected before reaching the retriever."""
        with pytest.raises((ValueError, RuntimeError, TypeError)):
            rag_service.answer("", use_llm=False)

    def test_whitespace_only_question_raises(self, rag_service) -> None:
        """Whitespace-only input must also be rejected."""
        with pytest.raises((ValueError, RuntimeError, TypeError)):
            rag_service.answer("   ", use_llm=False)

    def test_very_long_question_handled_gracefully(self, rag_service) -> None:
        """Very long questions should raise a validation error, not crash."""
        with pytest.raises((ValueError, RuntimeError)):
            rag_service.answer("Sorum: " + "finansal " * 1000, use_llm=False)


# ---------------------------------------------------------------------------
# Source metadata integrity
# ---------------------------------------------------------------------------

class TestSourceMetadata:
    """
    Verify that retrieved sources always contain valid metadata.
    This catches regressions in the ingestion pipeline.
    """

    def test_sources_have_non_empty_source_field(self, rag_service) -> None:
        resp = _ask(rag_service, "Enflasyon nedir?")
        if resp.has_sources:
            for source in resp.sources:
                assert source.source, "Source field must be non-empty"

    def test_sources_have_valid_scores(self, rag_service) -> None:
        resp = _ask(rag_service, "Likidite nedir?")
        if resp.has_sources:
            for source in resp.sources:
                assert 0.0 <= source.score <= 1.0, (
                    f"Score out of range: {source.score}"
                )

    def test_sources_have_text(self, rag_service) -> None:
        resp = _ask(rag_service, "Bilesik faiz nedir?")
        if resp.has_sources:
            for source in resp.sources:
                assert len(source.text.strip()) > 0, "Source text must not be empty"

    def test_response_generation_seconds_positive(self, rag_service) -> None:
        resp = _ask(rag_service, "Bilesik faiz nedir?")
        assert resp.generation_seconds >= 0.0


# ---------------------------------------------------------------------------
# Multi-document retrieval verification
# ---------------------------------------------------------------------------

class TestMultiDocumentRetrieval:
    """
    After splitting the encyclopedia into 6 files, the retriever must be
    able to pull relevant chunks from different source documents.
    """

    EXPECTED_SOURCE_FILES = {
        "01_finansal_okuryazarlik_temelleri.txt",
        "02_gelir_gider_varlik_nakit_akisi.txt",
        "03_tasarruf_yatirim_likidite_risk_getiri.txt",
        "04_finansal_hedefler_ve_karar_verme.txt",
        "05_sozluk_yanlis_inanislar_finansal_saglik.txt",
        "06_bankacilik_belgeler_mini_ansiklopedi.txt",
    }

    def test_at_least_two_source_files_in_index(self, rag_service) -> None:
        """
        After ingestion of all 6 files, querying different topics should
        retrieve chunks from more than one file.
        """
        seen_sources: set[str] = set()

        questions = [
            "Finansal okuryazarlik nedir?",
            "Acil durum fonu nasil kurulur?",
            "Bilesik faiz nedir?",
        ]

        for q in questions:
            resp = _ask(rag_service, q)
            for s in resp.sources:
                seen_sources.add(s.source.split("/")[-1].split("\\")[-1])

        assert len(seen_sources) >= 2, (
            f"Expected results from at least 2 different files, "
            f"got: {seen_sources}"
        )
