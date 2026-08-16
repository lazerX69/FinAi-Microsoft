"""
Comprehensive tests for src/vector_store.py.

Uses a temporary ChromaDB directory so tests never touch the real
production database. Each test class gets a fresh, isolated store.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from src.vector_store import VectorStore
from src import config


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_store(tmp_path: Path) -> VectorStore:
    """
    Return a VectorStore backed by a fresh, isolated ChromaDB directory.

    Each call creates a unique directory AND a unique collection name, so
    tests cannot share or pollute each other's data.
    """
    import chromadb
    from chromadb.config import Settings
    import uuid

    unique_id = uuid.uuid4().hex[:12]
    chroma_path = tmp_path / f"chroma_{unique_id}"
    chroma_path.mkdir()

    # Build an isolated client directly instead of going through config
    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False),
    )

    collection_name = f"test_{unique_id}"
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Create a store shell and swap in our isolated client + collection
    store = object.__new__(VectorStore)
    store.client = client
    store.collection = collection
    return store


def _random_embedding(dim: int = 32) -> list[float]:
    import random
    rng = random.Random()
    return [rng.gauss(0, 1) for _ in range(dim)]


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------

class TestVectorStoreAPI:
    def test_has_add_documents(self) -> None:
        assert callable(getattr(VectorStore, "add_documents", None))

    def test_has_search(self) -> None:
        assert callable(getattr(VectorStore, "search", None))

    def test_has_count(self) -> None:
        assert callable(getattr(VectorStore, "count", None))

    def test_has_reset(self) -> None:
        assert callable(getattr(VectorStore, "reset", None))

    def test_has_is_empty(self) -> None:
        assert callable(getattr(VectorStore, "is_empty", None))

    def test_has_peek(self) -> None:
        assert callable(getattr(VectorStore, "peek", None))

    def test_has_get_collection_name(self) -> None:
        assert callable(getattr(VectorStore, "get_collection_name", None))


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestVectorStoreInit:
    def test_new_store_is_empty(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        assert store.count() == 0

    def test_new_store_is_empty_flag(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        assert store.is_empty() is True

    def test_collection_name_is_string(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        assert isinstance(store.get_collection_name(), str)


# ---------------------------------------------------------------------------
# add_documents / count
# ---------------------------------------------------------------------------

class TestAddDocuments:
    def test_add_single_document_increments_count(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        store.add_documents(
            ids=["doc-1"],
            documents=["Bilesik faiz anaparanin buyumesidir."],
            embeddings=[_random_embedding()],
            metadatas=[{"source": "test.txt"}],
        )
        assert store.count() == 1

    def test_add_multiple_documents(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        n = 5
        store.add_documents(
            ids=[f"doc-{i}" for i in range(n)],
            documents=[f"Metin {i}" for i in range(n)],
            embeddings=[_random_embedding() for _ in range(n)],
            metadatas=[{"source": f"file{i}.txt"} for i in range(n)],
        )
        assert store.count() == n

    def test_add_empty_list_does_nothing(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        initial = store.count()
        store.add_documents(ids=[], documents=[], embeddings=[], metadatas=[])
        assert store.count() == initial

    def test_is_empty_false_after_insert(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        store.add_documents(
            ids=["x"],
            documents=["deneme"],
            embeddings=[_random_embedding()],
            metadatas=[{"source": "s.txt"}],
        )
        assert store.is_empty() is False


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_all_documents(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        store.add_documents(
            ids=["a", "b"],
            documents=["Metin A", "Metin B"],
            embeddings=[_random_embedding(), _random_embedding()],
            metadatas=[{"source": "a.txt"}, {"source": "b.txt"}],
        )
        count_before_reset = store.count()
        assert count_before_reset >= 2  # at least the 2 we just added
        store.reset()
        assert store.count() == 0

    def test_reset_makes_store_empty(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        store.add_documents(
            ids=["z"],
            documents=["Deneme"],
            embeddings=[_random_embedding()],
            metadatas=[{"source": "z.txt"}],
        )
        store.reset()
        assert store.is_empty() is True

    def test_can_add_after_reset(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        store.add_documents(
            ids=["old"],
            documents=["Eski belge"],
            embeddings=[_random_embedding()],
            metadatas=[{"source": "old.txt"}],
        )
        store.reset()
        store.add_documents(
            ids=["new"],
            documents=["Yeni belge"],
            embeddings=[_random_embedding()],
            metadatas=[{"source": "new.txt"}],
        )
        assert store.count() == 1


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

class TestSearch:
    def test_search_returns_dict(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        emb = _random_embedding()
        store.add_documents(
            ids=["s1"],
            documents=["Enflasyon fiyatlarin artisidir."],
            embeddings=[emb],
            metadatas=[{"source": "test.txt"}],
        )
        result = store.search(embedding=emb, top_k=1)
        assert isinstance(result, dict)

    def test_search_result_has_documents_key(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        emb = _random_embedding()
        store.add_documents(
            ids=["s2"],
            documents=["Likidite onemlidir."],
            embeddings=[emb],
            metadatas=[{"source": "test.txt"}],
        )
        result = store.search(embedding=emb, top_k=1)
        assert "documents" in result

    def test_search_respects_top_k(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        for i in range(5):
            store.add_documents(
                ids=[f"t{i}"],
                documents=[f"Belge {i}"],
                embeddings=[_random_embedding()],
                metadatas=[{"source": "t.txt"}],
            )
        result = store.search(embedding=_random_embedding(), top_k=3)
        docs = result.get("documents", [[]])[0]
        assert len(docs) <= 3


# ---------------------------------------------------------------------------
# peek
# ---------------------------------------------------------------------------

class TestPeek:
    def test_peek_returns_result_on_empty_store(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        result = store.peek(limit=5)
        assert result is not None

    def test_peek_limit_respected(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        for i in range(10):
            store.add_documents(
                ids=[f"p{i}"],
                documents=[f"Peek belge {i}"],
                embeddings=[_random_embedding()],
                metadatas=[{"source": "p.txt"}],
            )
        result = store.peek(limit=3)
        # ChromaDB peek returns a dict with 'ids', 'documents', etc.
        ids = result.get("ids", [])
        assert len(ids) <= 3
