"""
Comprehensive tests for src/embedding.py.

Tests text normalization, cosine similarity math, and the public API.
The actual SentenceTransformer model is NOT loaded — embed_texts and
embed_query are patched so tests run fast and offline.
"""

from __future__ import annotations

import math
from unittest.mock import patch, MagicMock

import pytest

from src.embedding import (
    normalize_embedding_text,
    cosine_similarity,
    embed_texts,
    embed_query,
    clear_embedding_model_cache,
)


# ---------------------------------------------------------------------------
# normalize_embedding_text
# ---------------------------------------------------------------------------

class TestNormalizeEmbeddingText:
    def test_plain_text_returned_unchanged(self) -> None:
        text = "Bilesik faiz nedir?"
        result = normalize_embedding_text(text)
        assert "Bilesik" in result or "bilesik" in result.lower()

    def test_removes_null_bytes(self) -> None:
        result = normalize_embedding_text("Faiz\x00orani")
        assert "\x00" not in result

    def test_strips_leading_trailing_whitespace(self) -> None:
        result = normalize_embedding_text("   Metin   ")
        assert result == result.strip()

    def test_collapses_internal_whitespace(self) -> None:
        result = normalize_embedding_text("cok   bosluk   var")
        assert "  " not in result

    def test_empty_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            normalize_embedding_text("")

    def test_whitespace_only_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            normalize_embedding_text("   \n\t  ")

    def test_non_string_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            normalize_embedding_text(42)  # type: ignore[arg-type]

    def test_text_exceeding_max_length_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            normalize_embedding_text("A" * 60_000)

    def test_normalizes_crlf(self) -> None:
        result = normalize_embedding_text("Satir bir.\r\nSatir iki.")
        assert "\r" not in result


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors_return_one(self) -> None:
        v = [1.0, 0.0, 0.0]
        score = cosine_similarity(v, v)
        assert abs(score - 1.0) < 1e-6

    def test_orthogonal_vectors_return_zero(self) -> None:
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        assert abs(cosine_similarity(v1, v2)) < 1e-6

    def test_opposite_vectors_return_minus_one(self) -> None:
        v1 = [1.0, 0.0]
        v2 = [-1.0, 0.0]
        score = cosine_similarity(v1, v2)
        assert abs(score - (-1.0)) < 1e-6

    def test_result_in_range_minus_one_to_one(self) -> None:
        import random
        rng = random.Random(42)
        for _ in range(20):
            v1 = [rng.gauss(0, 1) for _ in range(128)]
            v2 = [rng.gauss(0, 1) for _ in range(128)]
            score = cosine_similarity(v1, v2)
            assert -1.0 <= score <= 1.0

    def test_zero_vector_returns_zero(self) -> None:
        v1 = [0.0, 0.0, 0.0]
        v2 = [1.0, 2.0, 3.0]
        assert cosine_similarity(v1, v2) == 0.0

    def test_mismatched_dimensions_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_symmetry(self) -> None:
        v1 = [0.5, 0.3, 0.2]
        v2 = [0.1, 0.9, 0.0]
        assert abs(cosine_similarity(v1, v2) - cosine_similarity(v2, v1)) < 1e-6

    def test_normalized_vectors(self) -> None:
        """Normalized vectors: cosine similarity == dot product."""
        import math
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.6, 0.8, 0.0]  # already unit length
        expected = sum(a * b for a, b in zip(v1, v2))
        assert abs(cosine_similarity(v1, v2) - expected) < 1e-5


# ---------------------------------------------------------------------------
# embed_texts (mocked model)
# ---------------------------------------------------------------------------

class TestEmbedTexts:
    def _make_mock_model(self, dim: int = 8) -> MagicMock:
        import numpy as np
        mock = MagicMock()

        def fake_encode(texts, **kwargs):
            return np.ones((len(texts), dim), dtype="float32")

        mock.encode.side_effect = fake_encode
        return mock

    def test_empty_list_returns_empty(self) -> None:
        result = embed_texts([])
        assert result == []

    def test_returns_list_of_lists(self) -> None:
        mock_model = self._make_mock_model(dim=8)
        with patch("src.embedding.get_embedding_model", return_value=mock_model):
            result = embed_texts(["Metin bir", "Metin iki"])
        assert isinstance(result, list)
        assert all(isinstance(e, list) for e in result)

    def test_one_embedding_per_text(self) -> None:
        mock_model = self._make_mock_model(dim=8)
        texts = ["A", "B", "C"]
        with patch("src.embedding.get_embedding_model", return_value=mock_model):
            result = embed_texts(texts)
        assert len(result) == len(texts)

    def test_non_sequence_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            embed_texts("metin")  # type: ignore[arg-type]

    def test_string_in_sequence_normalized(self) -> None:
        mock_model = self._make_mock_model(dim=4)
        with patch("src.embedding.get_embedding_model", return_value=mock_model):
            result = embed_texts(["  Bilesik faiz  "])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# embed_query (mocked model)
# ---------------------------------------------------------------------------

class TestEmbedQuery:
    def _make_mock_model(self, dim: int = 8) -> MagicMock:
        import numpy as np
        mock = MagicMock()

        def fake_encode(texts, **kwargs):
            return np.ones((len(texts), dim), dtype="float32")

        mock.encode.side_effect = fake_encode
        return mock

    def test_returns_single_list(self) -> None:
        mock_model = self._make_mock_model(dim=8)
        with patch("src.embedding.get_embedding_model", return_value=mock_model):
            result = embed_query("Bilesik faiz nedir?")
        assert isinstance(result, list)
        assert len(result) == 8

    def test_empty_query_raises(self) -> None:
        with pytest.raises((ValueError, RuntimeError)):
            embed_query("")


# ---------------------------------------------------------------------------
# clear_embedding_model_cache
# ---------------------------------------------------------------------------

class TestClearEmbeddingModelCache:
    def test_clear_does_not_raise(self) -> None:
        """Clearing the LRU cache should be safe to call at any time."""
        clear_embedding_model_cache()
