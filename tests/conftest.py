"""
Shared pytest fixtures for the FinGuide AI test suite.

This conftest provides lightweight, in-memory fixtures that do NOT require
Foundry Local or a real ChromaDB collection, so unit tests can run fast
and offline.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Temporary directory fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Return a temporary directory that is cleaned up after the test."""
    return tmp_path


# ---------------------------------------------------------------------------
# Sample text fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_financial_texts() -> list[str]:
    """A small list of realistic Turkish financial literacy sentences."""
    return [
        "Bilesik faiz, ana para uzerine biriken faizin de faize tabi tutulmasidir.",
        "Enflasyon, genel fiyat duzeyinin zaman icinde artmasiyla satin alma gucunun azalmasidir.",
        "Acil durum fonu, beklenmedik harcamalar icin kolay erisilebilir finansal birikimdir.",
        "Yatirim fonu, bircok yatirimcinin parasini bir araya getirerek yonetilen kolektif bir yatirim aracidir.",
        "Likidite, bir varligin deger kaybi olmadan hizlica nakde cevrilebilme ozellgidir.",
        "Risk ve getiri arasinda genellikle pozitif bir iliski vardir.",
        "Butce, belirli bir donemde gelir ve giderlerin planlanmasidir.",
        "Net deger, toplam varliklardan toplam yukumlulukler cikararak elde edilen finansal gostergedir.",
    ]


@pytest.fixture
def sample_short_text() -> str:
    return "Bilesik faiz nedir?"


@pytest.fixture
def sample_long_text() -> str:
    return (
        "Bilesik faiz, ana para uzerine kazanilan faizin de bir sonraki donemde "
        "faize dahil edilmesi ilkesine dayanir. Bu sayede zaman gectikce birikim "
        "katlanarak buyur. Ornegin yillik yuzde on faiz oraniy1a bin lira "
        "yatirildiginda birinci yil sonunda bin yuz lira olur."
    )


# ---------------------------------------------------------------------------
# Fake embedding fixture (1024-dim zeros - fast, no model needed)
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_embedding() -> list[float]:
    """1024-dimensional zero vector, usable as a dummy embedding."""
    return [0.0] * 1024


@pytest.fixture
def fake_embeddings(sample_financial_texts: list[str]) -> list[list[float]]:
    """One fake embedding per sample text."""
    return [[float(i) / 1000.0] * 1024 for i in range(len(sample_financial_texts))]


# ---------------------------------------------------------------------------
# Mocked VectorStore fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Return a MagicMock that mimics the VectorStore interface."""
    store = MagicMock()
    store.count.return_value = 8
    store.is_empty.return_value = False
    store.get_collection_name.return_value = "financial_literacy"
    store.search.return_value = {
        "ids": [["chunk-001", "chunk-002"]],
        "documents": [
            [
                "Bilesik faiz, ana para uzerine biriken faizin de faize tabi tutulmasidir.",
                "Enflasyon, genel fiyat duzeyinin artmasiyla satin alma gucunun azalmasidir.",
            ]
        ],
        "metadatas": [
            [
                {
                    "source": "03_tasarruf_yatirim_likidite_risk_getiri.txt",
                    "file_name": "03_tasarruf_yatirim_likidite_risk_getiri.txt",
                    "file_type": "txt",
                    "page": 1,
                    "chunk_index": 0,
                    "global_chunk_index": 5,
                    "character_count": 72,
                    "content_hash": "abc123",
                    "document_title": "Finansal Kavramlar Ansiklopedisi",
                    "section_title": "Tasarruf",
                    "subsection_title": "",
                },
                {
                    "source": "05_sozluk_yanlis_inanislar_finansal_saglik.txt",
                    "file_name": "05_sozluk_yanlis_inanislar_finansal_saglik.txt",
                    "file_type": "txt",
                    "page": 1,
                    "chunk_index": 1,
                    "global_chunk_index": 6,
                    "character_count": 80,
                    "content_hash": "def456",
                    "document_title": "Finansal Kavramlar Ansiklopedisi",
                    "section_title": "Enflasyon",
                    "subsection_title": "",
                },
            ]
        ],
        "distances": [[0.05, 0.18]],
    }
    return store


# ---------------------------------------------------------------------------
# Mocked FoundryClient fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_foundry_client() -> MagicMock:
    """Return a MagicMock that mimics the FoundryClient interface."""
    client = MagicMock()
    client.is_ready = True
    client.complete.return_value = (
        "Bilesik faiz, ana para ve uzerinde biriken faizin tekrar faize "
        "tabi tutulmasidir. [Kaynak 1]"
    )
    client.stream.return_value = iter(
        [
            "Bilesik faiz, ",
            "ana para ve biriken faizin ",
            "tekrar faize tabi tutulmasidir.",
        ]
    )
    client.close.return_value = None
    return client
