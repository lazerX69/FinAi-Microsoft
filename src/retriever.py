"""
FinAi retrieval katmanı.

Bu modül, kullanıcı sorularını embedding vektörüne dönüştürür ve
ChromaDB içinde en alakalı belge parçalarını bulur.

Temel sorumlulukları:

- Kullanıcı sorgusunu doğrulamak ve normalize etmek
- Sorgu embedding'i üretmek
- ChromaDB üzerinde cosine similarity araması yapmak
- ChromaDB sonuçlarını güvenli biçimde ayrıştırmak
- Distance değerlerini similarity skoruna dönüştürmek
- Düşük kaliteli sonuçları filtrelemek
- Yinelenen içerikleri kaldırmak
- LLM'e gönderilecek kaynaklı context metnini oluşturmak
- Komut satırından retrieval testi yapılmasını sağlamak

Çalıştırma:

    python -m src.retriever "Enflasyon nedir?"

Özel sonuç sayısı:

    python -m src.retriever "Bütçe nasıl hazırlanır?" --top-k 5

Minimum benzerlik eşiği:

    python -m src.retriever "Faiz nedir?" --min-score 0.25
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from src.config import (
    RETRIEVAL_CANDIDATE_MULTIPLIER,
    TOP_K,
)
from src.embedding import embed_texts
from src.vector_store import VectorStore


LOGGER = logging.getLogger("finai.retriever")

DEFAULT_MIN_SCORE = 0.55
DEFAULT_MAX_CONTEXT_CHARACTERS = 6_000
DEFAULT_CONTEXT_SEPARATOR = "\n\n---\n\n"

MIN_QUERY_LENGTH = 2
MAX_QUERY_LENGTH = 2_000


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """
    Tek bir retrieval sonucunu temsil eder.

    Attributes:
        document:
            ChromaDB'den getirilen kaynak metin parçası.
        metadata:
            Kaynak dosya, sayfa ve chunk gibi bilgiler.
        distance:
            ChromaDB tarafından döndürülen cosine distance değeri.
        score:
            Distance değerinden hesaplanan 0-1 aralığındaki benzerlik skoru.
        rank:
            Sonucun sıralamadaki bir tabanlı konumu.
        result_id:
            ChromaDB kaydının kimliği. Mevcutsa doldurulur.
    """

    document: str
    metadata: Mapping[str, Any]
    distance: float
    score: float
    rank: int
    result_id: str | None = None

    @property
    def source(self) -> str:
        """Kaynak dosya yolunu döndürür."""

        value = self.metadata.get("source")

        if isinstance(value, str) and value.strip():
            return value.strip()

        file_name = self.metadata.get("file_name")

        if isinstance(file_name, str) and file_name.strip():
            return file_name.strip()

        return "Bilinmeyen kaynak"

    @property
    def file_name(self) -> str:
        """Kaynağın yalnızca dosya adını döndürür."""

        value = self.metadata.get("file_name")

        if isinstance(value, str) and value.strip():
            return value.strip()

        return self.source.rsplit("/", maxsplit=1)[-1].rsplit("\\", maxsplit=1)[-1]

    @property
    def page_number(self) -> int | None:
        """Sayfa numarasını güvenli biçimde döndürür."""

        value = self.metadata.get("page")

        if value is None:
            value = self.metadata.get("page_number")

        if isinstance(value, bool):
            return None

        if isinstance(value, int):
            return value if value > 0 else None

        if isinstance(value, float) and value.is_integer():
            page = int(value)
            return page if page > 0 else None

        if isinstance(value, str):
            try:
                page = int(value.strip())
                return page if page > 0 else None
            except ValueError:
                return None

        return None

    @property
    def chunk_index(self) -> int | None:
        """Chunk indeksini güvenli biçimde döndürür."""

        value = self.metadata.get("chunk_index")

        if isinstance(value, bool):
            return None

        if isinstance(value, int):
            return value if value >= 0 else None

        if isinstance(value, float) and value.is_integer():
            chunk_index = int(value)
            return chunk_index if chunk_index >= 0 else None

        if isinstance(value, str):
            try:
                chunk_index = int(value.strip())
                return chunk_index if chunk_index >= 0 else None
            except ValueError:
                return None

        return None

    @property
    def content_hash(self) -> str | None:
        """İçerik hash değerini döndürür."""

        value = self.metadata.get("content_hash")

        if isinstance(value, str) and value.strip():
            return value.strip()

        return None

    @property
    def citation(self) -> str:
        """
        İnsan tarafından okunabilir kaynak etiketi üretir.

        Örnek:

            finans.pdf, sayfa 3
        """

        if self.page_number is not None:
            return f"{self.source}, sayfa {self.page_number}"

        return self.source

    def to_dict(self) -> dict[str, Any]:
        """Sonucu JSON uyumlu sözlüğe dönüştürür."""

        return {
            "id": self.result_id,
            "rank": self.rank,
            "document": self.document,
            "metadata": dict(self.metadata),
            "distance": self.distance,
            "score": self.score,
            "source": self.source,
            "file_name": self.file_name,
            "page": self.page_number,
            "chunk_index": self.chunk_index,
            "citation": self.citation,
        }


@dataclass(frozen=True, slots=True)
class RetrievalResponse:
    """
    Bir retrieval işleminin bütün çıktısını temsil eder.

    Attributes:
        query:
            Normalize edilmiş kullanıcı sorgusu.
        results:
            Filtrelenmiş ve sıralanmış retrieval sonuçları.
        requested_top_k:
            Kullanıcının istediği sonuç sayısı.
        searched_top_k:
            ChromaDB'den gerçekte istenen aday sayısı.
        collection_count:
            Arama anında koleksiyondaki toplam kayıt sayısı.
        min_score:
            Uygulanan minimum benzerlik eşiği.
    """

    query: str
    results: tuple[RetrievalResult, ...]
    requested_top_k: int
    searched_top_k: int
    collection_count: int
    min_score: float

    @property
    def has_results(self) -> bool:
        """En az bir retrieval sonucu olup olmadığını döndürür."""

        return bool(self.results)

    @property
    def result_count(self) -> int:
        """Filtrelenmiş sonuç sayısını döndürür."""

        return len(self.results)

    @property
    def best_score(self) -> float | None:
        """En yüksek similarity skorunu döndürür."""

        if not self.results:
            return None

        return self.results[0].score

    def to_dict(self) -> dict[str, Any]:
        """Yanıtı JSON uyumlu sözlüğe dönüştürür."""

        return {
            "query": self.query,
            "requested_top_k": self.requested_top_k,
            "searched_top_k": self.searched_top_k,
            "collection_count": self.collection_count,
            "min_score": self.min_score,
            "result_count": self.result_count,
            "best_score": self.best_score,
            "results": [result.to_dict() for result in self.results],
        }


@dataclass(frozen=True, slots=True)
class ContextDocument:
    """
    LLM context'i içinde kullanılacak belge parçası.

    Bu sınıf retrieval sonucunu, prompt üretiminde ihtiyaç duyulan
    minimum bilgilerle temsil eder.
    """

    index: int
    text: str
    source: str
    page_number: int | None
    score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def citation_label(self) -> str:
        """Prompt içinde kullanılacak kaynak başlığını üretir."""

        if self.page_number is not None:
            return (
                f"Kaynak {self.index}: {self.source}, "
                f"sayfa {self.page_number}"
            )

        return f"Kaynak {self.index}: {self.source}"


def configure_logging(verbose: bool = False) -> None:
    """
    Komut satırı çalıştırmaları için logging ayarlarını yapılandırır.

    Args:
        verbose:
            True olduğunda DEBUG seviyesinde log üretir.
    """

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


def normalize_query(query: str) -> str:
    """
    Kullanıcı sorgusunu retrieval işlemi için normalize eder.

    - Baş ve sondaki boşlukları kaldırır.
    - Satır sonlarını boşluğa dönüştürür.
    - Ardışık boşlukları tek boşluğa indirir.
    - Null karakterlerini kaldırır.

    Args:
        query:
            Ham kullanıcı sorgusu.

    Returns:
        Normalize edilmiş sorgu.

    Raises:
        TypeError:
            Sorgu string değilse.
        ValueError:
            Sorgu boşsa, çok kısaysa veya izin verilen uzunluğu aşıyorsa.
    """

    if not isinstance(query, str):
        raise TypeError(
            "Retrieval sorgusu string olmalıdır. "
            f"Alınan tür: {type(query).__name__}"
        )

    normalized = query.replace("\x00", " ")
    normalized = normalized.replace("\r\n", " ").replace("\r", " ")
    normalized = normalized.replace("\n", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if not normalized:
        raise ValueError("Retrieval sorgusu boş olamaz.")

    if len(normalized) < MIN_QUERY_LENGTH:
        raise ValueError(
            f"Retrieval sorgusu en az {MIN_QUERY_LENGTH} karakter olmalıdır."
        )

    if len(normalized) > MAX_QUERY_LENGTH:
        raise ValueError(
            "Retrieval sorgusu izin verilen maksimum uzunluğu aşıyor. "
            f"Maksimum: {MAX_QUERY_LENGTH}, alınan: {len(normalized)}"
        )

    return normalized


def validate_top_k(top_k: int) -> int:
    """
    top_k değerini doğrular.

    Returns:
        Doğrulanmış top_k değeri.
    """

    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k bir tam sayı olmalıdır.")

    if top_k <= 0:
        raise ValueError("top_k sıfırdan büyük olmalıdır.")

    return top_k


def validate_min_score(min_score: float) -> float:
    """
    Minimum similarity skorunu doğrular.

    Skor 0 ile 1 arasında olmalıdır.
    """

    if isinstance(min_score, bool) or not isinstance(
        min_score,
        (int, float),
    ):
        raise TypeError("min_score sayısal bir değer olmalıdır.")

    normalized_score = float(min_score)

    if not math.isfinite(normalized_score):
        raise ValueError("min_score sonlu bir sayı olmalıdır.")

    if not 0.0 <= normalized_score <= 1.0:
        raise ValueError("min_score 0.0 ile 1.0 arasında olmalıdır.")

    return normalized_score


def validate_max_context_characters(max_characters: int) -> int:
    """Context karakter limitini doğrular."""

    if isinstance(max_characters, bool) or not isinstance(
        max_characters,
        int,
    ):
        raise TypeError("max_characters bir tam sayı olmalıdır.")

    if max_characters <= 0:
        raise ValueError("max_characters sıfırdan büyük olmalıdır.")

    return max_characters


def cosine_distance_to_score(distance: float) -> float:
    """
    ChromaDB cosine distance değerini 0-1 similarity skoruna dönüştürür.

    ChromaDB cosine space kullanıldığında genel ilişki:

        similarity = 1 - distance

    Teorik olarak cosine similarity -1 ile 1 arasında olabilir.
    Uygulama ve kullanıcı arayüzünde kolay yorumlanabilmesi için sonuç
    0 ile 1 aralığına sınırlandırılır.

    Args:
        distance:
            ChromaDB cosine distance değeri.

    Returns:
        0 ile 1 arasında similarity skoru.
    """

    if isinstance(distance, bool) or not isinstance(
        distance,
        (int, float),
    ):
        return 0.0

    numeric_distance = float(distance)

    if not math.isfinite(numeric_distance):
        return 0.0

    similarity = 1.0 - numeric_distance
    return max(0.0, min(1.0, similarity))


def _first_nested_list(
    raw_value: Any,
) -> list[Any]:
    """
    ChromaDB query sonucundaki ilk sorguya ait listeyi döndürür.

    ChromaDB tek bir query embedding gönderildiğinde alanları genellikle
    aşağıdaki biçimde döndürür:

        [["sonuç 1", "sonuç 2"]]

    Eksik veya beklenmeyen sonuçlarda boş liste döndürülür.
    """

    if raw_value is None:
        return []

    if not isinstance(raw_value, Sequence) or isinstance(
        raw_value,
        (str, bytes),
    ):
        return []

    if len(raw_value) == 0:
        return []

    first_value = raw_value[0]

    if not isinstance(first_value, Sequence) or isinstance(
        first_value,
        (str, bytes),
    ):
        return []

    return list(first_value)


def parse_search_results(
    raw_results: Mapping[str, Any],
) -> list[RetrievalResult]:
    """
    ChromaDB query çıktısını RetrievalResult listesine dönüştürür.

    Eksik document, metadata veya distance alanları güvenli biçimde
    ele alınır. Boş belgeler sonuçlara dahil edilmez.

    Args:
        raw_results:
            VectorStore.search() çıktısı.

    Returns:
        Distance değerine göre sıralanmış sonuç listesi.
    """

    if not isinstance(raw_results, Mapping):
        raise TypeError(
            "VectorStore.search() sonucu mapping biçiminde olmalıdır."
        )

    documents = _first_nested_list(raw_results.get("documents"))
    metadatas = _first_nested_list(raw_results.get("metadatas"))
    distances = _first_nested_list(raw_results.get("distances"))
    ids = _first_nested_list(raw_results.get("ids"))

    parsed_results: list[RetrievalResult] = []

    for index, raw_document in enumerate(documents):
        if not isinstance(raw_document, str):
            LOGGER.debug(
                "String olmayan retrieval belgesi atlandı. İndeks: %d",
                index,
            )
            continue

        document = raw_document.strip()

        if not document:
            LOGGER.debug(
                "Boş retrieval belgesi atlandı. İndeks: %d",
                index,
            )
            continue

        raw_metadata: Any = (
            metadatas[index]
            if index < len(metadatas)
            else {}
        )

        metadata: Mapping[str, Any]

        if isinstance(raw_metadata, Mapping):
            metadata = dict(raw_metadata)
        else:
            metadata = {}

        raw_distance: Any = (
            distances[index]
            if index < len(distances)
            else 1.0
        )

        try:
            distance = float(raw_distance)
        except (TypeError, ValueError):
            distance = 1.0

        if not math.isfinite(distance):
            distance = 1.0

        raw_id: Any = ids[index] if index < len(ids) else None
        result_id = str(raw_id) if raw_id is not None else None

        parsed_results.append(
            RetrievalResult(
                document=document,
                metadata=metadata,
                distance=distance,
                score=cosine_distance_to_score(distance),
                rank=0,
                result_id=result_id,
            )
        )

    parsed_results.sort(
        key=lambda result: (
            -result.score,
            result.distance,
            result.source.lower(),
        )
    )

    return [
        RetrievalResult(
            document=result.document,
            metadata=result.metadata,
            distance=result.distance,
            score=result.score,
            rank=rank,
            result_id=result.result_id,
        )
        for rank, result in enumerate(parsed_results, start=1)
    ]


def deduplicate_results(
    results: Sequence[RetrievalResult],
) -> list[RetrievalResult]:
    """
    Yinelenen retrieval sonuçlarını kaldırır.

    Öncelikle ingest sırasında oluşturulan content_hash kullanılır.
    Hash bulunmuyorsa normalize edilmiş belge metni üzerinden kontrol
    yapılır. En yüksek skorlu ilk sonuç korunur.
    """

    unique_results: list[RetrievalResult] = []
    seen_keys: set[str] = set()

    for result in results:
        if result.content_hash:
            deduplication_key = f"hash:{result.content_hash}"
        else:
            normalized_document = re.sub(
                r"\s+",
                " ",
                result.document,
            ).strip().casefold()

            deduplication_key = f"text:{normalized_document}"

        if deduplication_key in seen_keys:
            LOGGER.debug(
                "Yinelenen retrieval sonucu atlandı: %s",
                result.citation,
            )
            continue

        seen_keys.add(deduplication_key)
        unique_results.append(result)

    return [
        RetrievalResult(
            document=result.document,
            metadata=result.metadata,
            distance=result.distance,
            score=result.score,
            rank=rank,
            result_id=result.result_id,
        )
        for rank, result in enumerate(unique_results, start=1)
    ]


def filter_results(
    results: Sequence[RetrievalResult],
    *,
    min_score: float,
    limit: int,
) -> list[RetrievalResult]:
    """
    Sonuçları similarity eşiğine göre filtreler ve sınırlar.

    Args:
        results:
            Ayrıştırılmış retrieval sonuçları.
        min_score:
            Minimum kabul edilen similarity skoru.
        limit:
            Döndürülecek maksimum sonuç sayısı.
    """

    validated_score = validate_min_score(min_score)
    validated_limit = validate_top_k(limit)

    filtered_results = [
        result
        for result in results
        if result.score >= validated_score
    ][:validated_limit]

    return [
        RetrievalResult(
            document=result.document,
            metadata=result.metadata,
            distance=result.distance,
            score=result.score,
            rank=rank,
            result_id=result.result_id,
        )
        for rank, result in enumerate(filtered_results, start=1)
    ]


TURKISH_STOP_WORDS = frozenset(
    {
        "acaba",
        "ama",
        "ancak",
        "arasındaki",
        "arasında",
        "bana",
        "benim",
        "bir",
        "biraz",
        "bu",
        "bunun",
        "da",
        "daha",
        "de",
        "diye",
        "en",
        "fark",
        "farkı",
        "gibi",
        "hakkında",
        "hangi",
        "için",
        "ile",
        "ise",
        "mı",
        "mi",
        "mu",
        "mü",
        "nasıl",
        "ne",
        "neden",
        "nedir",
        "nelerdir",
        "olan",
        "olarak",
        "olur",
        "ve",
        "veya",
    }
)

LEXICAL_WEIGHT = 0.38
SEMANTIC_WEIGHT = 0.62
EXACT_PHRASE_BONUS = 0.12
ALL_TERMS_BONUS = 0.10
TITLE_MATCH_BONUS = 0.08


def lexical_tokens(text: str) -> list[str]:
    """Hybrid retrieval için Türkçe uyumlu kelime listesi üretir."""

    if not isinstance(text, str):
        return []

    return re.findall(
        r"[0-9a-zçğıöşü]+",
        text.casefold(),
    )


def meaningful_query_terms(query: str) -> tuple[str, ...]:
    """Sorgudaki ayırt edici kelimeleri deterministik sırada döndürür."""

    terms: list[str] = []
    seen: set[str] = set()

    for token in lexical_tokens(query):
        if len(token) < 2:
            continue

        if token in TURKISH_STOP_WORDS:
            continue

        if token in seen:
            continue

        seen.add(token)
        terms.append(token)

    return tuple(terms)


def normalize_lexical_text(text: str) -> str:
    """Tam ifade araması için metni sadeleştirir."""

    return " ".join(lexical_tokens(text))


def build_query_phrases(
    query: str,
    terms: Sequence[str],
) -> tuple[str, ...]:
    """
    Sorgudan tam veya kısmi kavram ifadeleri üretir.

    Örneğin:
        "Net değer ile nakit akışı arasındaki fark nedir?"

    sorgusu için:
        "net değer"
        "nakit akışı"

    gibi ikili ifadeler oluşturulur.
    """

    normalized_query = normalize_lexical_text(query)
    phrases: list[str] = []

    if normalized_query:
        phrases.append(normalized_query)

    for index in range(len(terms) - 1):
        phrase = f"{terms[index]} {terms[index + 1]}"

        if phrase not in phrases:
            phrases.append(phrase)

    return tuple(phrases)


def metadata_search_text(
    result: RetrievalResult,
) -> str:
    """Başlık ve dosya metadata alanlarını lexical arama metnine ekler."""

    values: list[str] = [
        result.source,
        result.file_name,
    ]

    for key in (
        "document_title",
        "section_title",
        "subsection_title",
        "title",
        "heading",
    ):
        value = result.metadata.get(key)

        if isinstance(value, str) and value.strip():
            values.append(value.strip())

    return " ".join(values)


def lexical_relevance_score(
    *,
    query: str,
    result: RetrievalResult,
) -> float:
    """Bir retrieval sonucunun sorguyla lexical uygunluğunu hesaplar."""

    query_terms = meaningful_query_terms(query)

    if not query_terms:
        return 0.0

    document_text = normalize_lexical_text(result.document)
    metadata_text = normalize_lexical_text(
        metadata_search_text(result)
    )

    document_tokens = set(document_text.split())
    metadata_tokens = set(metadata_text.split())

    matched_document_terms = {
        term
        for term in query_terms
        if term in document_tokens
    }

    matched_metadata_terms = {
        term
        for term in query_terms
        if term in metadata_tokens
    }

    document_overlap = (
        len(matched_document_terms) / len(query_terms)
    )

    metadata_overlap = (
        len(matched_metadata_terms) / len(query_terms)
    )

    score = document_overlap
    score += metadata_overlap * TITLE_MATCH_BONUS

    if len(matched_document_terms) == len(query_terms):
        score += ALL_TERMS_BONUS

    query_phrases = build_query_phrases(
        query,
        query_terms,
    )

    matched_phrase_count = sum(
        1
        for phrase in query_phrases
        if len(phrase.split()) >= 2
        and phrase in document_text
    )

    if matched_phrase_count:
        score += min(
            EXACT_PHRASE_BONUS * matched_phrase_count,
            EXACT_PHRASE_BONUS * 2,
        )

    return max(0.0, min(1.0, score))


def rerank_hybrid_results(
    query: str,
    results: Sequence[RetrievalResult],
) -> list[RetrievalResult]:
    """
    Semantic sonuçları lexical eşleşmeyle yeniden sıralar.

    RetrievalResult.score alanı bu aşamadan sonra nihai hybrid skoru temsil eder.
    Distance alanı ChromaDB'nin orijinal semantic distance değerini korur.
    """

    reranked: list[
        tuple[
            float,
            float,
            float,
            RetrievalResult,
        ]
    ] = []

    for result in results:
        lexical_score = lexical_relevance_score(
            query=query,
            result=result,
        )

        hybrid_score = (
            result.score * SEMANTIC_WEIGHT
            + lexical_score * LEXICAL_WEIGHT
        )

        reranked.append(
            (
                hybrid_score,
                lexical_score,
                result.score,
                result,
            )
        )

    reranked.sort(
        key=lambda item: (
            -item[0],
            -item[1],
            -item[2],
            item[3].source.casefold(),
        )
    )

    return [
        replace(
            result,
            score=max(0.0, min(1.0, hybrid_score)),
            rank=rank,
        )
        for rank, (
            hybrid_score,
            _,
            _,
            result,
        ) in enumerate(reranked, start=1)
    ]


class Retriever:
    """
    FinAi retrieval servisi.

    Sınıf bir VectorStore örneğini kullanır ve kullanıcı sorularından
    LLM context'ine kadar gerekli retrieval sürecini yönetir.

    Args:
        vector_store:
            Test veya dependency injection amacıyla dışarıdan verilebilen
            VectorStore örneği. Verilmezse yeni örnek oluşturulur.
        default_top_k:
            retrieve() çağrısında top_k verilmediğinde kullanılacak değer.
        default_min_score:
            retrieve() çağrısında eşik verilmediğinde kullanılacak skor.
        candidate_multiplier:
            Filtreleme ve duplicate temizliği öncesinde ChromaDB'den daha
            fazla aday çekmek için kullanılan çarpan.
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        *,
        default_top_k: int = TOP_K,
        default_min_score: float = DEFAULT_MIN_SCORE,
        candidate_multiplier: int = RETRIEVAL_CANDIDATE_MULTIPLIER,
    ) -> None:
        self.vector_store = vector_store or VectorStore()
        self.default_top_k = validate_top_k(default_top_k)
        self.default_min_score = validate_min_score(default_min_score)

        if isinstance(candidate_multiplier, bool) or not isinstance(
            candidate_multiplier,
            int,
        ):
            raise TypeError("candidate_multiplier bir tam sayı olmalıdır.")

        if candidate_multiplier <= 0:
            raise ValueError(
                "candidate_multiplier sıfırdan büyük olmalıdır."
            )

        self.candidate_multiplier = candidate_multiplier

    @property
    def document_count(self) -> int:
        """ChromaDB koleksiyonundaki toplam kayıt sayısını döndürür."""

        return self.vector_store.count()

    @property
    def is_ready(self) -> bool:
        """Retriever'ın sorgulanabilir durumda olup olmadığını döndürür."""

        return self.document_count > 0

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> RetrievalResponse:
        """
        Kullanıcı sorgusu için en alakalı belge parçalarını getirir.

        Args:
            query:
                Kullanıcı sorusu.
            top_k:
                Döndürülecek maksimum sonuç sayısı.
            min_score:
                Minimum similarity skoru.

        Returns:
            RetrievalResponse.

        Raises:
            RuntimeError:
                Vektör koleksiyonu boşsa veya embedding/search işlemi
                başarısız olursa.
        """

        normalized_query = normalize_query(query)

        requested_top_k = validate_top_k(
            top_k if top_k is not None else self.default_top_k
        )

        effective_min_score = validate_min_score(
            min_score
            if min_score is not None
            else self.default_min_score
        )

        collection_count = self.document_count

        if collection_count <= 0:
            raise RuntimeError(
                "ChromaDB koleksiyonu boş. Önce belgeleri indeksleyin: "
                "python -m src.ingest"
            )

        searched_top_k = min(
            collection_count,
            max(
                requested_top_k,
                requested_top_k * self.candidate_multiplier,
            ),
        )

        LOGGER.info(
            "Retrieval başlatıldı | top_k=%d | aday=%d | eşik=%.3f",
            requested_top_k,
            searched_top_k,
            effective_min_score,
        )

        try:
            query_embeddings = embed_texts([normalized_query])
        except Exception as exc:
            raise RuntimeError(
                "Kullanıcı sorgusu için embedding üretilemedi."
            ) from exc

        if not query_embeddings:
            raise RuntimeError(
                "Embedding modeli kullanıcı sorgusu için sonuç üretmedi."
            )

        query_embedding = query_embeddings[0]

        if not isinstance(query_embedding, Sequence) or isinstance(
            query_embedding,
            (str, bytes),
        ):
            raise RuntimeError(
                "Embedding modeli geçersiz sorgu vektörü döndürdü."
            )

        if len(query_embedding) == 0:
            raise RuntimeError(
                "Embedding modeli boş sorgu vektörü döndürdü."
            )

        try:
            raw_results = self.vector_store.search(
                embedding=list(query_embedding),
                top_k=searched_top_k,
            )
        except Exception as exc:
            raise RuntimeError(
                "ChromaDB similarity araması başarısız oldu."
            ) from exc

        parsed_results = parse_search_results(raw_results)
        unique_results = deduplicate_results(parsed_results)

        reranked_results = rerank_hybrid_results(
            normalized_query,
            unique_results,
        )

        filtered_results = filter_results(
            reranked_results,
            min_score=effective_min_score,
            limit=requested_top_k,
        )

        if filtered_results:
            LOGGER.info(
                "Hybrid retrieval tamamlandı | sonuç=%d | "
                "en iyi skor=%.4f",
                len(filtered_results),
                filtered_results[0].score,
            )
        else:
            LOGGER.info(
                "Retrieval tamamlandı ancak eşik üzerinde sonuç bulunamadı."
            )

        return RetrievalResponse(
            query=normalized_query,
            results=tuple(filtered_results),
            requested_top_k=requested_top_k,
            searched_top_k=searched_top_k,
            collection_count=collection_count,
            min_score=effective_min_score,
        )

    def build_context_documents(
        self,
        response: RetrievalResponse,
    ) -> tuple[ContextDocument, ...]:
        """
        RetrievalResponse içeriğini prompt dostu belgelere dönüştürür.
        """

        return tuple(
            ContextDocument(
                index=index,
                text=result.document,
                source=result.source,
                page_number=result.page_number,
                score=result.score,
                metadata=result.metadata,
            )
            for index, result in enumerate(response.results, start=1)
        )

    def build_context(
        self,
        response: RetrievalResponse,
        *,
        max_characters: int = DEFAULT_MAX_CONTEXT_CHARACTERS,
        separator: str = DEFAULT_CONTEXT_SEPARATOR,
        include_scores: bool = False,
    ) -> str:
        """
        Retrieval sonuçlarından LLM'e gönderilecek context metnini üretir.

        Her kaynak açık bir başlıkla ayrılır. Context karakter limiti
        aşılmadan mümkün olan en fazla kaynak eklenir.

        Args:
            response:
                retrieve() tarafından üretilen yanıt.
            max_characters:
                Context'in maksimum toplam karakter sayısı.
            separator:
                Kaynak blokları arasında kullanılacak ayraç.
            include_scores:
                True olduğunda kaynak başlığında similarity skoru bulunur.

        Returns:
            Kaynaklandırılmış context metni. Sonuç yoksa boş string.
        """

        validated_limit = validate_max_context_characters(max_characters)

        if not isinstance(separator, str):
            raise TypeError("separator string olmalıdır.")

        if not response.results:
            return ""

        context_parts: list[str] = []
        current_length = 0

        for context_document in self.build_context_documents(response):
            heading = context_document.citation_label

            if include_scores:
                heading += f" | Benzerlik: {context_document.score:.4f}"

            block = (
                f"[{heading}]\n"
                f"{context_document.text.strip()}"
            )

            separator_length = len(separator) if context_parts else 0
            remaining_characters = (
                validated_limit
                - current_length
                - separator_length
            )

            if remaining_characters <= 0:
                break

            if len(block) <= remaining_characters:
                context_parts.append(block)
                current_length += separator_length + len(block)
                continue

            # İlk kaynak limite sığmıyorsa tamamen boş context dönmek yerine
            # belgeyi güvenli biçimde kırpar.
            if not context_parts:
                truncation_marker = "\n[Metin karakter limiti nedeniyle kısaltıldı.]"
                available_text_length = (
                    remaining_characters
                    - len(heading)
                    - len(truncation_marker)
                    - 3
                )

                if available_text_length > 0:
                    truncated_text = context_document.text[
                        :available_text_length
                    ].rstrip()

                    context_parts.append(
                        f"[{heading}]\n"
                        f"{truncated_text}"
                        f"{truncation_marker}"
                    )

            break

        return separator.join(context_parts)

    def retrieve_context(
        self,
        query: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        max_characters: int = DEFAULT_MAX_CONTEXT_CHARACTERS,
        include_scores: bool = False,
    ) -> tuple[RetrievalResponse, str]:
        """
        Retrieval ve context üretimini tek çağrıda gerçekleştirir.

        Returns:
            ``(retrieval_response, context)`` ikilisi.
        """

        response = self.retrieve(
            query,
            top_k=top_k,
            min_score=min_score,
        )

        context = self.build_context(
            response,
            max_characters=max_characters,
            include_scores=include_scores,
        )

        return response, context


def format_result_for_terminal(result: RetrievalResult) -> str:
    """
    Retrieval sonucunu terminalde okunabilir biçimde formatlar.
    """

    preview = re.sub(
        r"\s+",
        " ",
        result.document,
    ).strip()

    if len(preview) > 500:
        preview = preview[:497].rstrip() + "..."

    lines = [
        f"Sonuç {result.rank}",
        f"Kaynak     : {result.citation}",
        f"Benzerlik  : {result.score:.4f}",
        f"Distance   : {result.distance:.4f}",
    ]

    if result.chunk_index is not None:
        lines.append(f"Chunk       : {result.chunk_index}")

    lines.extend(
        [
            "",
            preview,
        ]
    )

    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    """Komut satırı argümanlarını tanımlar."""

    parser = argparse.ArgumentParser(
        description=(
            "FinAi ChromaDB koleksiyonunda semantic search yapar."
        )
    )

    parser.add_argument(
        "query",
        type=str,
        help="Belgelerde aranacak kullanıcı sorusu.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=(
            "Döndürülecek maksimum sonuç sayısı. "
            f"Varsayılan: {TOP_K}"
        ),
    )

    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help=(
            "Minimum cosine similarity skoru. "
            f"Varsayılan: {DEFAULT_MIN_SCORE}"
        ),
    )

    parser.add_argument(
        "--show-context",
        action="store_true",
        help="LLM'e gönderilecek birleşik context metnini gösterir.",
    )

    parser.add_argument(
        "--max-context-characters",
        type=int,
        default=DEFAULT_MAX_CONTEXT_CHARACTERS,
        help=(
            "Birleşik context için maksimum karakter sayısı. "
            f"Varsayılan: {DEFAULT_MAX_CONTEXT_CHARACTERS}"
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Ayrıntılı DEBUG loglarını gösterir.",
    )

    return parser


def main() -> int:
    """
    Komut satırı giriş noktası.

    Returns:
        Başarı halinde 0, hata halinde 1.
    """

    parser = build_argument_parser()
    args = parser.parse_args()

    configure_logging(verbose=args.verbose)

    try:
        retriever = Retriever()

        response, context = retriever.retrieve_context(
            args.query,
            top_k=args.top_k,
            min_score=args.min_score,
            max_characters=args.max_context_characters,
            include_scores=True,
        )
    except KeyboardInterrupt:
        LOGGER.warning("Retrieval işlemi kullanıcı tarafından durduruldu.")
        return 130
    except Exception:
        LOGGER.exception("Retrieval işlemi başarısız oldu.")
        return 1

    print()
    print("=" * 72)
    print("FinAi Retrieval Sonuçları")
    print("=" * 72)
    print(f"Sorgu             : {response.query}")
    print(f"Koleksiyon kaydı  : {response.collection_count}")
    print(f"İstenen sonuç     : {response.requested_top_k}")
    print(f"Aranan aday       : {response.searched_top_k}")
    print(f"Minimum skor      : {response.min_score:.4f}")
    print(f"Bulunan sonuç     : {response.result_count}")
    print("=" * 72)

    if not response.results:
        print()
        print(
            "Belirlenen benzerlik eşiğinin üzerinde sonuç bulunamadı."
        )
        print(
            "Daha genel bir soru deneyebilir veya --min-score değerini "
            "düşürebilirsin."
        )
        return 0

    for result in response.results:
        print()
        print(format_result_for_terminal(result))
        print()
        print("-" * 72)

    if args.show_context:
        print()
        print("=" * 72)
        print("LLM Context")
        print("=" * 72)
        print(context or "Context oluşturulamadı.")
        print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())