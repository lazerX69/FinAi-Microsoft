"""
Comprehensive tests for src/config.py.

Covers: path resolution, value types, constraint relationships,
and environment-level sanity checks.
"""

from __future__ import annotations

import src.config as config


# ---------------------------------------------------------------------------
# Path tests
# ---------------------------------------------------------------------------

class TestPaths:
    def test_base_dir_exists(self) -> None:
        assert config.BASE_DIR.exists(), "BASE_DIR must point to a real directory"

    def test_documents_dir_is_subdir_of_base(self) -> None:
        assert config.DOCUMENTS_DIR.parent == config.BASE_DIR

    def test_data_dir_is_subdir_of_base(self) -> None:
        assert config.DATA_DIR.parent == config.BASE_DIR

    def test_chroma_dir_is_subdir_of_data(self) -> None:
        assert config.CHROMA_DIR.parent == config.DATA_DIR

    def test_model_cache_dir_is_subdir_of_data(self) -> None:
        assert config.MODEL_CACHE_DIR.parent == config.DATA_DIR


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

class TestModelConfig:
    def test_model_name_is_string(self) -> None:
        assert isinstance(config.MODEL_NAME, str)
        assert len(config.MODEL_NAME) > 0

    def test_model_alias_equals_model_name(self) -> None:
        assert config.MODEL_ALIAS == config.MODEL_NAME

    def test_collection_name_is_string(self) -> None:
        assert isinstance(config.COLLECTION_NAME, str)
        assert len(config.COLLECTION_NAME) > 0


# ---------------------------------------------------------------------------
# Embedding configuration
# ---------------------------------------------------------------------------

class TestEmbeddingConfig:
    def test_embedding_model_is_string(self) -> None:
        assert isinstance(config.EMBEDDING_MODEL, str)
        assert "/" in config.EMBEDDING_MODEL, "Should be in HuggingFace org/model format"

    def test_embedding_device_is_valid(self) -> None:
        assert config.EMBEDDING_DEVICE in ("cpu", "cuda", "mps")

    def test_embedding_batch_size_positive(self) -> None:
        assert config.EMBEDDING_BATCH_SIZE > 0

    def test_embedding_normalize_is_bool(self) -> None:
        assert isinstance(config.EMBEDDING_NORMALIZE, bool)


# ---------------------------------------------------------------------------
# Chunk configuration
# ---------------------------------------------------------------------------

class TestChunkConfig:
    def test_chunk_size_positive(self) -> None:
        assert config.CHUNK_SIZE > 0

    def test_chunk_overlap_non_negative(self) -> None:
        assert config.CHUNK_OVERLAP >= 0

    def test_chunk_overlap_less_than_chunk_size(self) -> None:
        assert config.CHUNK_OVERLAP < config.CHUNK_SIZE, (
            "CHUNK_OVERLAP must be strictly less than CHUNK_SIZE"
        )

    def test_min_chunk_characters_positive(self) -> None:
        assert config.MIN_CHUNK_CHARACTERS > 0

    def test_min_chunk_characters_less_than_chunk_size(self) -> None:
        assert config.MIN_CHUNK_CHARACTERS < config.CHUNK_SIZE


# ---------------------------------------------------------------------------
# Retrieval configuration
# ---------------------------------------------------------------------------

class TestRetrievalConfig:
    def test_top_k_positive(self) -> None:
        assert config.TOP_K > 0

    def test_retrieval_candidate_multiplier_at_least_one(self) -> None:
        assert config.RETRIEVAL_CANDIDATE_MULTIPLIER >= 1

    def test_default_min_score_in_range(self) -> None:
        assert 0.0 <= config.DEFAULT_MIN_SCORE <= 1.0

    def test_max_context_characters_positive(self) -> None:
        assert config.MAX_CONTEXT_CHARACTERS > 0

    def test_min_answer_confidence_in_range(self) -> None:
        assert 0.0 <= config.MIN_ANSWER_CONFIDENCE <= 1.0


# ---------------------------------------------------------------------------
# Answer strategy
# ---------------------------------------------------------------------------

class TestAnswerStrategyConfig:
    def test_prefer_deterministic_answers_is_bool(self) -> None:
        assert isinstance(config.PREFER_DETERMINISTIC_ANSWERS, bool)

    def test_enable_llm_generation_is_bool(self) -> None:
        assert isinstance(config.ENABLE_LLM_GENERATION, bool)

    def test_max_answer_sentences_positive(self) -> None:
        assert config.MAX_ANSWER_SENTENCES > 0

    def test_max_answer_characters_positive(self) -> None:
        assert config.MAX_ANSWER_CHARACTERS > 0
