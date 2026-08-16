"""
FinAi embedding katmanı.

Bu modül:

- Çok dilli SentenceTransformer modelini process başına bir kez yükler.
- Belge ve sorgu embeddinglerini normalize edilmiş biçimde üretir.
- Yerel cache kullanımını destekler.
- Hugging Face ve Transformers log gürültüsünü azaltır.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Sequence


os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL,
    EMBEDDING_NORMALIZE,
    MODEL_CACHE_DIR,
)


LOGGER = logging.getLogger("finai.embedding")

MAX_EMBEDDING_TEXT_LENGTH = 50_000


def configure_dependency_logging() -> None:
    """Embedding bağımlılıklarının gereksiz HTTP loglarını azaltır."""

    for logger_name in (
        "httpx",
        "httpcore",
        "huggingface_hub",
        "transformers",
        "sentence_transformers",
        "filelock",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def normalize_embedding_text(
    text: str,
    *,
    field_name: str = "text",
) -> str:
    """Embedding öncesinde tek bir metni doğrular ve normalize eder."""

    if not isinstance(text, str):
        raise TypeError(f"{field_name} string olmalıdır.")

    normalized = text.replace("\x00", " ")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = " ".join(normalized.split()).strip()

    if not normalized:
        raise ValueError(f"{field_name} boş olamaz.")

    if len(normalized) > MAX_EMBEDDING_TEXT_LENGTH:
        raise ValueError(
            f"{field_name} en fazla "
            f"{MAX_EMBEDDING_TEXT_LENGTH} karakter olabilir."
        )

    return normalized


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Embedding modelini process boyunca yalnızca bir kez yükler.

    İlk çevrim içi çalıştırmada model Hugging Face üzerinden indirilir.
    Sonraki çalıştırmalarda yerel cache kullanılabilir.
    """

    configure_dependency_logging()
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault(
        "SENTENCE_TRANSFORMERS_HOME",
        str(MODEL_CACHE_DIR.resolve()),
    )
    os.environ.setdefault(
        "HF_HUB_DISABLE_SYMLINKS_WARNING",
        "1",
    )

    LOGGER.info(
        "Çok dilli embedding modeli yükleniyor: %s",
        EMBEDDING_MODEL,
    )

    try:
        model = SentenceTransformer(
            EMBEDDING_MODEL,
            device=EMBEDDING_DEVICE,
            cache_folder=str(MODEL_CACHE_DIR.resolve()),
        )
    except Exception as exc:
        raise RuntimeError(
            "Embedding modeli yüklenemedi. İlk kullanımda internet "
            "bağlantısını, disk alanını ve model cache izinlerini kontrol edin. "
            f"Model: {EMBEDDING_MODEL}"
        ) from exc

    dimension_getter = getattr(
        model,
        "get_embedding_dimension",
        None,
    )

    if callable(dimension_getter):
        embedding_dimension = int(dimension_getter())
    else:
        legacy_dimension_getter = getattr(
            model,
            "get_sentence_embedding_dimension",
            None,
        )

        if not callable(legacy_dimension_getter):
            raise RuntimeError(
                "Embedding modelinin vektör boyutu belirlenemedi."
            )

        embedding_dimension = int(legacy_dimension_getter())

    LOGGER.info(
        "Embedding modeli hazır | model=%s | boyut=%d",
        EMBEDDING_MODEL,
        embedding_dimension,
    )

    return model


def embed_texts(
    texts: Sequence[str],
) -> list[list[float]]:
    """Birden fazla metin için normalize edilmiş embedding üretir."""

    if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
        raise TypeError("texts bir metin dizisi olmalıdır.")

    if not texts:
        return []

    normalized_texts = [
        normalize_embedding_text(
            text,
            field_name=f"texts[{index}]",
        )
        for index, text in enumerate(texts)
    ]

    model = get_embedding_model()

    try:
        embeddings = model.encode(
            normalized_texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=EMBEDDING_NORMALIZE,
            show_progress_bar=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "Belge embeddingleri üretilemedi."
        ) from exc

    array = np.asarray(embeddings, dtype=np.float32)

    if array.ndim != 2:
        raise RuntimeError(
            "Embedding modeli iki boyutlu bir sonuç döndürmedi."
        )

    if array.shape[0] != len(normalized_texts):
        raise RuntimeError(
            "Embedding sayısı ile metin sayısı eşleşmiyor."
        )

    if not np.isfinite(array).all():
        raise RuntimeError(
            "Embedding çıktısında geçersiz sayısal değer bulundu."
        )

    return array.tolist()


def embed_query(query: str) -> list[float]:
    """Tek bir kullanıcı sorgusu için embedding üretir."""

    normalized_query = normalize_embedding_text(
        query,
        field_name="query",
    )

    embeddings = embed_texts([normalized_query])

    if not embeddings:
        raise RuntimeError(
            "Sorgu için embedding üretilemedi."
        )

    return embeddings[0]


def cosine_similarity(
    vector1: Iterable[float],
    vector2: Iterable[float],
) -> float:
    """İki sayısal vektör arasındaki cosine similarity değerini hesaplar."""

    v1 = np.asarray(list(vector1), dtype=np.float32)
    v2 = np.asarray(list(vector2), dtype=np.float32)

    if v1.ndim != 1 or v2.ndim != 1:
        raise ValueError(
            "Cosine similarity yalnızca tek boyutlu vektörlerle hesaplanabilir."
        )

    if v1.shape != v2.shape:
        raise ValueError(
            "Cosine similarity için vektör boyutları eşit olmalıdır."
        )

    if not np.isfinite(v1).all() or not np.isfinite(v2).all():
        raise ValueError(
            "Vektörler geçersiz sayısal değer içeremez."
        )

    denominator = float(
        np.linalg.norm(v1) * np.linalg.norm(v2)
    )

    if denominator == 0.0:
        return 0.0

    similarity = float(np.dot(v1, v2) / denominator)

    return max(-1.0, min(1.0, similarity))


def clear_embedding_model_cache() -> None:
    """
    Testlerde veya kontrollü yeniden yüklemelerde model nesnesi cache'ini temizler.

    İndirilen model dosyalarını diskten silmez.
    """

    get_embedding_model.cache_clear()
