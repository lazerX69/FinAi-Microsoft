"""
FinAi document ingestion pipeline.

Bu modül:

- documents/ klasöründeki TXT ve PDF belgelerini bulur.
- Metinleri temizler ve anlamlı parçalara ayırır.
- Her parçaya deterministik ve benzersiz bir kimlik üretir.
- Sentence Transformers ile embedding oluşturur.
- Verileri toplu şekilde ChromaDB koleksiyonuna kaydeder.
- Kaynak dosya, sayfa ve parça bilgilerini metadata olarak saklar.

Çalıştırma:

    python -m src.ingest

Varsayılan davranış mevcut ChromaDB koleksiyonunu sıfırlar ve bütün belgeleri
yeniden indeksler.

Koleksiyonu sıfırlamadan çalıştırmak için:

    python -m src.ingest --no-reset

Özel batch boyutu kullanmak için:

    python -m src.ingest --batch-size 32
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENTS_DIR,
)
from src.embedding import embed_texts
from src.vector_store import VectorStore


LOGGER = logging.getLogger("finai.ingest")

SUPPORTED_EXTENSIONS = frozenset({".txt", ".pdf"})
DEFAULT_BATCH_SIZE = 64
MIN_CHUNK_LENGTH = 40

# TXT belgelerinde denenebilecek güvenli kodlamalar.
TEXT_ENCODINGS = (
    "utf-8-sig",
    "utf-8",
    "cp1254",
    "windows-1252",
    "latin-1",
)


@dataclass(frozen=True, slots=True)
class HeadingSection:
    """
    TXT belgesindeki başlık hiyerarşisine bağlı anlamlı metin bölümü.
    """

    document_title: str
    section_title: str
    subsection_title: str
    text: str


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """
    Vektör veritabanına kaydedilecek tek bir metin parçası.

    Attributes:
        chunk_id:
            ChromaDB içinde kullanılacak deterministik benzersiz kimlik.
        text:
            Embedding üretilecek ve ChromaDB'ye kaydedilecek metin.
        source:
            Kaynak dosyanın documents/ klasörüne göre göreli yolu.
        file_name:
            Kaynak dosyanın yalnızca dosya adı.
        file_type:
            Dosya uzantısı. Örneğin "pdf" veya "txt".
        page_number:
            PDF sayfa numarası. TXT belgelerinde 1 olarak kullanılır.
        chunk_index:
            İlgili sayfa veya metin içindeki sıfır tabanlı parça indeksi.
        global_chunk_index:
            Dosya genelindeki sıfır tabanlı parça indeksi.
        character_count:
            Temizlenmiş metnin karakter sayısı.
        content_hash:
            Metin içeriğinin SHA-256 özeti.
    """

    chunk_id: str
    text: str
    source: str
    file_name: str
    file_type: str
    page_number: int
    chunk_index: int
    global_chunk_index: int
    character_count: int
    content_hash: str
    document_title: str = ""
    section_title: str = ""
    subsection_title: str = ""

    def to_metadata(self) -> dict[str, str | int]:
        """
        ChromaDB ile uyumlu, yalnızca skaler değerler içeren metadata döndürür.
        """

        return {
            "source": self.source,
            "file_name": self.file_name,
            "file_type": self.file_type,
            "page": self.page_number,
            "chunk_index": self.chunk_index,
            "global_chunk_index": self.global_chunk_index,
            "character_count": self.character_count,
            "content_hash": self.content_hash,
            "document_title": self.document_title,
            "section_title": self.section_title,
            "subsection_title": self.subsection_title,
        }


@dataclass(slots=True)
class IngestionStatistics:
    """İndeksleme işlemi boyunca toplanan özet bilgiler."""

    discovered_files: int = 0
    processed_files: int = 0
    skipped_files: int = 0
    failed_files: int = 0
    extracted_pages: int = 0
    empty_pages: int = 0
    generated_chunks: int = 0
    inserted_chunks: int = 0


def configure_logging(verbose: bool = False) -> None:
    """
    Komut satırı çalıştırmaları için logging ayarlarını yapılandırır.

    Args:
        verbose:
            True olduğunda DEBUG seviyesinde ayrıntılı çıktı üretir.
    """

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


def validate_settings(batch_size: int) -> None:
    """
    İndeksleme ayarlarını işlem başlamadan doğrular.

    Raises:
        ValueError:
            Chunk veya batch ayarları geçersiz olduğunda.
    """

    if CHUNK_SIZE <= 0:
        raise ValueError("CHUNK_SIZE sıfırdan büyük olmalıdır.")

    if CHUNK_OVERLAP < 0:
        raise ValueError("CHUNK_OVERLAP negatif olamaz.")

    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError(
            "CHUNK_OVERLAP, CHUNK_SIZE değerinden küçük olmalıdır. "
            f"Mevcut değerler: CHUNK_SIZE={CHUNK_SIZE}, "
            f"CHUNK_OVERLAP={CHUNK_OVERLAP}"
        )

    if batch_size <= 0:
        raise ValueError("Batch boyutu sıfırdan büyük olmalıdır.")


def normalize_text(text: str) -> str:
    """
    Belgeden çıkarılan metni RAG kullanımı için temizler.

    Temizlik işlemi:

    - Null karakterlerini kaldırır.
    - Windows ve eski Mac satır sonlarını normalize eder.
    - Satır sonlarında bölünmüş kelimeleri birleştirir.
    - Gereksiz boşlukları azaltır.
    - Paragraf sınırlarını korur.

    Args:
        text:
            Ham belge metni.

    Returns:
        Normalize edilmiş metin.
    """

    if not text:
        return ""

    normalized = text.replace("\x00", " ")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    # PDF satır sonlarında bölünmüş kelimeleri birleştirir:
    # "finan-\nsal" -> "finansal"
    normalized = re.sub(
        r"(?<=\w)-\s*\n\s*(?=\w)",
        "",
        normalized,
    )

    # Tek satır sonlarını boşluğa dönüştürürken paragraf ayrımlarını korur.
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n[ \t]+", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r"(?<!\n)\n(?!\n)", " ", normalized)

    # Bir paragraf içindeki fazla boşlukları temizler.
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)

    paragraphs = [
        paragraph.strip()
        for paragraph in normalized.split("\n\n")
        if paragraph.strip()
    ]

    return "\n\n".join(paragraphs).strip()


def find_preferred_split_position(
    text: str,
    start: int,
    maximum_end: int,
) -> int:
    """
    Chunk bitişi için mümkün olan en doğal sınırı bulur.

    Öncelik sırası:

    1. Paragraf sonu
    2. Cümle sonu
    3. Noktalama işareti
    4. Kelime sonu
    5. Zorunlu karakter sınırı

    Chunk'ın aşırı küçülmesini önlemek için arama, hedef alanın son
    yaklaşık yüzde 40'lık bölümünde yapılır.
    """

    if maximum_end >= len(text):
        return len(text)

    minimum_end = start + max(
        int(CHUNK_SIZE * 0.60),
        MIN_CHUNK_LENGTH,
    )
    minimum_end = min(minimum_end, maximum_end)

    search_area = text[minimum_end:maximum_end]

    separators = (
        "\n\n",
        ". ",
        "? ",
        "! ",
        "; ",
        ": ",
        ", ",
        " ",
    )

    for separator in separators:
        relative_position = search_area.rfind(separator)

        if relative_position != -1:
            split_position = minimum_end + relative_position + len(separator)
            if split_position > start:
                return split_position

    return maximum_end


def split_text(text: str) -> list[str]:
    """
    Metni karakter tabanlı overlap ile anlamlı parçalara ayırır.

    Bu fonksiyon mevcut config.py içindeki CHUNK_SIZE ve CHUNK_OVERLAP
    değerlerini kullanır. Bölme sırasında mümkün olduğunca paragraf,
    cümle ve kelime sınırları tercih edilir.

    Args:
        text:
            Normalize edilmiş belge veya sayfa metni.

    Returns:
        Boş olmayan metin parçaları.
    """

    cleaned_text = normalize_text(text)

    if not cleaned_text:
        return []

    if len(cleaned_text) <= CHUNK_SIZE:
        return [cleaned_text]

    chunks: list[str] = []
    start = 0
    text_length = len(cleaned_text)

    while start < text_length:
        maximum_end = min(start + CHUNK_SIZE, text_length)
        end = find_preferred_split_position(
            text=cleaned_text,
            start=start,
            maximum_end=maximum_end,
        )

        if end <= start:
            end = maximum_end

        chunk = cleaned_text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = end - CHUNK_OVERLAP

        # Overlap başlangıcını mümkünse kelime sınırına taşır.
        if next_start > 0:
            nearby_space = cleaned_text.find(
                " ",
                next_start,
                min(end, next_start + 40),
            )
            if nearby_space != -1:
                next_start = nearby_space + 1

        # Her iterasyonda mutlaka ilerleme sağlanmalıdır.
        start = max(next_start, start + 1)

    return chunks


MARKDOWN_HEADING_PATTERN = re.compile(
    r"^\s*(#{1,6})\s+(.+?)\s*$"
)

NUMBERED_HEADING_PATTERN = re.compile(
    r"^\s*(\d{1,3}(?:\.\d{1,3})*)[.)]?\s+(.+?)\s*$"
)

DOCUMENT_TITLE_PATTERN = re.compile(
    r"^\s*(?:FinAi\s+AI\s+)?BİLGİ\s+TABANI\b",
    flags=re.IGNORECASE,
)


def clean_heading_text(value: str) -> str:
    """Başlık metnindeki Markdown ve gereksiz boşlukları temizler."""

    cleaned = value.strip()
    cleaned = re.sub(r"^#+\s*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" -–—")


def detect_heading(line: str) -> tuple[int, str] | None:
    """
    Bir satırın Markdown veya numaralı başlık olup olmadığını belirler.

    Returns:
        ``(level, heading_text)`` veya başlık değilse ``None``.
    """

    stripped = line.strip()

    if not stripped:
        return None

    markdown_match = MARKDOWN_HEADING_PATTERN.match(stripped)

    if markdown_match:
        level = len(markdown_match.group(1))
        heading = clean_heading_text(markdown_match.group(2))

        if heading:
            return level, heading

    numbered_match = NUMBERED_HEADING_PATTERN.match(stripped)

    if numbered_match:
        numbering = numbered_match.group(1)
        heading_text = clean_heading_text(numbered_match.group(2))

        if not heading_text:
            return None

        level = min(numbering.count(".") + 2, 4)
        return level, f"{numbering}. {heading_text}"

    return None


def split_text_into_heading_sections(
    raw_text: str,
) -> list[HeadingSection]:
    """
    TXT belgesini başlık hiyerarşisini koruyarak bölümlere ayırır.

    Markdown başlıkları ve ``1. Başlık`` biçimindeki numaralı başlıklar
    desteklenir. Başlıksız belgelerde tek bir genel bölüm döndürülür.
    """

    if not raw_text:
        return []

    normalized_lines = raw_text.replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    ).splitlines()

    document_title = ""
    section_title = ""
    subsection_title = ""

    sections: list[HeadingSection] = []
    body_lines: list[str] = []

    def flush_section() -> None:
        nonlocal body_lines

        raw_body = "\n".join(body_lines).strip()
        body_lines = []

        cleaned_body = normalize_text(raw_body)

        if not cleaned_body:
            return

        sections.append(
            HeadingSection(
                document_title=document_title,
                section_title=section_title,
                subsection_title=subsection_title,
                text=cleaned_body,
            )
        )

    for raw_line in normalized_lines:
        stripped = raw_line.strip()

        if (
            not document_title
            and stripped
            and (
                DOCUMENT_TITLE_PATTERN.search(stripped)
                or "—" in stripped
            )
        ):
            if len(stripped) <= 180:
                document_title = clean_heading_text(stripped)
                continue

        detected = detect_heading(raw_line)

        if detected is None:
            body_lines.append(raw_line)
            continue

        flush_section()

        level, heading = detected

        if level <= 2:
            section_title = heading
            subsection_title = ""
        else:
            subsection_title = heading

    flush_section()

    if sections:
        return sections

    cleaned_text = normalize_text(raw_text)

    if not cleaned_text:
        return []

    return [
        HeadingSection(
            document_title=document_title,
            section_title="",
            subsection_title="",
            text=cleaned_text,
        )
    ]


def build_chunk_prefix(
    section: HeadingSection,
) -> str:
    """Chunk içine eklenecek başlık bağlamını üretir."""

    headings: list[str] = []

    if section.document_title:
        headings.append(f"Belge: {section.document_title}")

    if section.section_title:
        headings.append(f"Bölüm: {section.section_title}")

    if section.subsection_title:
        headings.append(f"Alt başlık: {section.subsection_title}")

    return "\n".join(headings)


def split_heading_section(
    section: HeadingSection,
) -> list[str]:
    """
    Tek bir başlık bölümünü chunklara ayırır ve başlık bağlamını ekler.
    """

    prefix = build_chunk_prefix(section)

    available_size = CHUNK_SIZE

    if prefix:
        available_size = max(
            MIN_CHUNK_LENGTH,
            CHUNK_SIZE - len(prefix) - 2,
        )

    original_chunk_size = CHUNK_SIZE

    if len(section.text) <= available_size:
        body_chunks = [section.text]
    else:
        body_chunks = split_text(section.text)

    output: list[str] = []

    for body_chunk in body_chunks:
        if prefix:
            combined = f"{prefix}\n\n{body_chunk}".strip()
        else:
            combined = body_chunk.strip()

        if combined:
            output.append(combined)

    return output




def read_text_file(path: Path) -> str:
    """
    TXT dosyasını desteklenen kodlamalardan biriyle okur.

    Args:
        path:
            Okunacak TXT dosyası.

    Raises:
        UnicodeError:
            Dosya desteklenen kodlamalarla okunamazsa.
        OSError:
            Dosya erişim hatalarında.
    """

    decoding_errors: list[str] = []

    for encoding in TEXT_ENCODINGS:
        try:
            text = path.read_text(encoding=encoding)
            LOGGER.debug(
                "TXT kodlaması belirlendi: %s (%s)",
                path.name,
                encoding,
            )
            return text
        except UnicodeDecodeError as exc:
            decoding_errors.append(f"{encoding}: {exc}")

    raise UnicodeError(
        f"Dosya desteklenen kodlamalarla okunamadı: {path}. "
        f"Denenen kodlamalar: {', '.join(TEXT_ENCODINGS)}. "
        f"Hatalar: {' | '.join(decoding_errors)}"
    )


def build_chunk_id(
    *,
    source: str,
    page_number: int,
    chunk_index: int,
    text: str,
) -> tuple[str, str]:
    """
    Bir metin parçası için deterministik kimlik ve içerik özeti üretir.

    Aynı kaynak, sayfa, chunk sırası ve içerik her çalıştırmada aynı
    kimliği üretir.

    Returns:
        ``(chunk_id, content_hash)`` ikilisi.
    """

    content_hash = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    identity = (
        f"{source}|page={page_number}|chunk={chunk_index}|"
        f"content={content_hash}"
    )

    chunk_id = hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()

    return chunk_id, content_hash


def create_document_chunk(
    *,
    path: Path,
    text: str,
    page_number: int,
    chunk_index: int,
    global_chunk_index: int,
    document_title: str = "",
    section_title: str = "",
    subsection_title: str = "",
) -> DocumentChunk:
    """Belge bilgileri ve metinden doğrulanmış DocumentChunk oluşturur."""

    try:
        relative_path = path.relative_to(DOCUMENTS_DIR)
    except ValueError:
        relative_path = Path(path.name)

    source = relative_path.as_posix()
    cleaned_chunk = text.strip()

    chunk_id, content_hash = build_chunk_id(
        source=source,
        page_number=page_number,
        chunk_index=chunk_index,
        text=cleaned_chunk,
    )

    return DocumentChunk(
        chunk_id=chunk_id,
        text=cleaned_chunk,
        source=source,
        file_name=path.name,
        file_type=path.suffix.lower().lstrip("."),
        page_number=page_number,
        chunk_index=chunk_index,
        global_chunk_index=global_chunk_index,
        character_count=len(cleaned_chunk),
        content_hash=content_hash,
        document_title=document_title.strip(),
        section_title=section_title.strip(),
        subsection_title=subsection_title.strip(),
    )


def extract_txt_chunks(
    path: Path,
    statistics: IngestionStatistics,
) -> list[DocumentChunk]:
    """
    TXT dosyasını başlık hiyerarşisini koruyarak chunk listesine dönüştürür.
    """

    raw_text = read_text_file(path)
    sections = split_text_into_heading_sections(raw_text)

    if not sections:
        LOGGER.warning("Metin içermeyen TXT dosyası atlandı: %s", path)
        statistics.skipped_files += 1
        return []

    document_chunks: list[DocumentChunk] = []
    global_chunk_index = 0

    for section_index, section in enumerate(sections):
        text_chunks = split_heading_section(section)

        for section_chunk_index, chunk_text in enumerate(text_chunks):
            document_chunks.append(
                create_document_chunk(
                    path=path,
                    text=chunk_text,
                    page_number=1,
                    chunk_index=section_chunk_index,
                    global_chunk_index=global_chunk_index,
                    document_title=section.document_title,
                    section_title=section.section_title,
                    subsection_title=section.subsection_title,
                )
            )

            global_chunk_index += 1

        LOGGER.debug(
            "TXT bölümü işlendi | dosya=%s | bölüm=%s | chunk=%d",
            path.name,
            section.section_title or f"bölüm-{section_index}",
            len(text_chunks),
        )

    statistics.extracted_pages += 1
    return document_chunks


def extract_pdf_chunks(
    path: Path,
    statistics: IngestionStatistics,
) -> list[DocumentChunk]:
    """
    PDF sayfalarından metin çıkarıp DocumentChunk listesine dönüştürür.

    Not:
        pypdf yalnızca PDF içinde gerçek metin katmanı varsa metin çıkarabilir.
        Taranmış görüntü tabanlı PDF'ler OCR gerektirir.
    """

    try:
        reader = PdfReader(str(path), strict=False)
    except (PdfReadError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"PDF açılamadı: {path.name}. Hata: {exc}"
        ) from exc

    if reader.is_encrypted:
        try:
            decrypt_result = reader.decrypt("")
        except Exception as exc:
            raise RuntimeError(
                f"Åifreli PDF açılamadı: {path.name}"
            ) from exc

        if decrypt_result == 0:
            raise RuntimeError(
                f"PDF şifreli ve boş parola ile açılamıyor: {path.name}"
            )

    document_chunks: list[DocumentChunk] = []
    global_chunk_index = 0

    for page_number, page in enumerate(reader.pages, start=1):
        statistics.extracted_pages += 1

        try:
            raw_text = page.extract_text() or ""
        except Exception as exc:
            LOGGER.warning(
                "Sayfa metni çıkarılamadı: %s, sayfa %d. Hata: %s",
                path.name,
                page_number,
                exc,
            )
            statistics.empty_pages += 1
            continue

        text_chunks = split_text(raw_text)

        if not text_chunks:
            statistics.empty_pages += 1
            LOGGER.debug(
                "Boş veya metin katmanı olmayan sayfa: %s, sayfa %d",
                path.name,
                page_number,
            )
            continue

        for page_chunk_index, chunk_text in enumerate(text_chunks):
            document_chunks.append(
                create_document_chunk(
                    path=path,
                    text=chunk_text,
                    page_number=page_number,
                    chunk_index=page_chunk_index,
                    global_chunk_index=global_chunk_index,
                )
            )
            global_chunk_index += 1

    if not document_chunks:
        LOGGER.warning(
            "PDF'den kullanılabilir metin çıkarılamadı: %s. "
            "Belge taranmış görüntü içeriyorsa OCR gerekir.",
            path,
        )
        statistics.skipped_files += 1

    return document_chunks


def discover_documents(directory: Path) -> list[Path]:
    """
    Desteklenen belgeleri alt klasörler dahil deterministik sırada bulur.
    """

    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
        return []

    if not directory.is_dir():
        raise NotADirectoryError(
            f"DOCUMENTS_DIR bir klasör değil: {directory}"
        )

    documents = [
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not path.name.startswith(".")
        and not path.name.startswith("~$")
    ]

    return sorted(
        documents,
        key=lambda item: item.relative_to(directory).as_posix().lower(),
    )


def extract_document_chunks(
    path: Path,
    statistics: IngestionStatistics,
) -> list[DocumentChunk]:
    """
    Dosya uzantısına göre uygun çıkarıcıyı çalıştırır.
    """

    suffix = path.suffix.lower()

    if suffix == ".txt":
        return extract_txt_chunks(path, statistics)

    if suffix == ".pdf":
        return extract_pdf_chunks(path, statistics)

    LOGGER.warning("Desteklenmeyen dosya türü atlandı: %s", path)
    statistics.skipped_files += 1
    return []


def deduplicate_chunks(
    chunks: Sequence[DocumentChunk],
) -> list[DocumentChunk]:
    """
    Aynı çalıştırma içinde oluşabilecek yinelenen kimlikleri kaldırır.

    Normal şartlarda chunk kimlikleri kaynak ve konum bilgisi içerdiği için
    benzersizdir. Bu kontrol, beklenmeyen tekrarlar için koruma sağlar.
    """

    unique_chunks: list[DocumentChunk] = []
    seen_ids: set[str] = set()

    for chunk in chunks:
        if chunk.chunk_id in seen_ids:
            LOGGER.warning(
                "Yinelenen chunk kimliği atlandı: %s (%s)",
                chunk.chunk_id,
                chunk.source,
            )
            continue

        seen_ids.add(chunk.chunk_id)
        unique_chunks.append(chunk)

    return unique_chunks


def batched(
    items: Sequence[DocumentChunk],
    batch_size: int,
) -> Iterator[Sequence[DocumentChunk]]:
    """Sequence içeriğini belirtilen boyutta batch'lere böler."""

    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def insert_chunks(
    *,
    vector_store: VectorStore,
    chunks: Sequence[DocumentChunk],
    batch_size: int,
    statistics: IngestionStatistics,
) -> None:
    """
    Chunk'lar için embedding üretir ve ChromaDB'ye batch halinde ekler.

    Embedding işlemi bütün belge koleksiyonu yerine batch bazında yapılır.
    Bu yaklaşım büyük doküman kümelerinde bellek tüketimini sınırlar.
    """

    total_chunks = len(chunks)

    if total_chunks == 0:
        return

    processed = 0

    for batch_number, chunk_batch in enumerate(
        batched(chunks, batch_size),
        start=1,
    ):
        documents = [chunk.text for chunk in chunk_batch]

        LOGGER.info(
            "Embedding batch %d hazırlanıyor: %d metin",
            batch_number,
            len(documents),
        )

        try:
            embeddings = embed_texts(documents)
        except Exception as exc:
            raise RuntimeError(
                f"Embedding üretilemedi. Batch: {batch_number}"
            ) from exc

        if len(embeddings) != len(chunk_batch):
            raise RuntimeError(
                "Embedding sayısı ile belge sayısı eşleşmiyor. "
                f"Belgeler: {len(chunk_batch)}, "
                f"embeddingler: {len(embeddings)}"
            )

        ids = [chunk.chunk_id for chunk in chunk_batch]
        metadatas = [
            chunk.to_metadata()
            for chunk in chunk_batch
        ]

        try:
            vector_store.add_documents(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                 metadatas=metadatas,
         )
        except Exception as exc:
            raise RuntimeError(
                "Chunk batch'i ChromaDB'ye eklenemedi. "
                f"Batch: {batch_number}, "
                f"ilk kaynak: {chunk_batch[0].source}"
            ) from exc

        inserted_count = len(chunk_batch)
        processed += inserted_count
        statistics.inserted_chunks += inserted_count

        LOGGER.info(
            "ChromaDB ilerlemesi: %d/%d chunk",
            processed,
            total_chunks,
        )


def collect_all_chunks(
    document_paths: Sequence[Path],
    statistics: IngestionStatistics,
) -> list[DocumentChunk]:
    """
    Bütün belgeleri işler ve tek bir doğrulanmış chunk listesi döndürür.

    Bir dosyada hata oluşması diğer belgelerin işlenmesini durdurmaz.
    Hatalı dosya loglanır ve kalan belgelerle devam edilir.
    """

    all_chunks: list[DocumentChunk] = []

    for file_number, path in enumerate(document_paths, start=1):
        try:
            relative_path = path.relative_to(DOCUMENTS_DIR)
        except ValueError:
            relative_path = Path(path.name)

        LOGGER.info(
            "Belge işleniyor [%d/%d]: %s",
            file_number,
            len(document_paths),
            relative_path.as_posix(),
        )

        try:
            document_chunks = extract_document_chunks(
                path=path,
                statistics=statistics,
            )
        except Exception:
            statistics.failed_files += 1
            LOGGER.exception(
                "Belge işlenirken hata oluştu: %s",
                relative_path.as_posix(),
            )
            continue

        if not document_chunks:
            continue

        statistics.processed_files += 1
        statistics.generated_chunks += len(document_chunks)
        all_chunks.extend(document_chunks)

        LOGGER.info(
            "Belge tamamlandı: %s | %d chunk",
            relative_path.as_posix(),
            len(document_chunks),
        )

    return deduplicate_chunks(all_chunks)


def run_ingestion(
    *,
    reset: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> IngestionStatistics:
    """
    Tam belge indeksleme sürecini çalıştırır.

    Args:
        reset:
            True olduğunda mevcut ChromaDB koleksiyonunu siler ve yeniden
            oluşturur. Mevcut VectorStore API'sinde upsert veya duplicate
            kontrolü bulunmadığı için güvenli varsayılan True'dur.
        batch_size:
            Tek seferde embedding üretilecek ve ChromaDB'ye eklenecek
            metin sayısı.

    Returns:
        İndeksleme istatistikleri.

    Raises:
        FileNotFoundError:
            documents/ klasöründe desteklenen belge bulunamazsa.
        RuntimeError:
            İndeksleme veya veri tabanı işlemi başarısız olduğunda.
    """

    validate_settings(batch_size)

    started_at = time.perf_counter()
    statistics = IngestionStatistics()

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    document_paths = discover_documents(DOCUMENTS_DIR)
    statistics.discovered_files = len(document_paths)

    if not document_paths:
        supported_types = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise FileNotFoundError(
            "İndekslenecek belge bulunamadı. "
            f"TXT veya PDF belgelerini şu klasöre ekleyin: "
            f"{DOCUMENTS_DIR.resolve()} "
            f"Desteklenen türler: {supported_types}"
        )

    LOGGER.info("=" * 68)
    LOGGER.info("FinAi belge indeksleme işlemi başlatıldı")
    LOGGER.info("Belge klasörü : %s", DOCUMENTS_DIR.resolve())
    LOGGER.info("Bulunan belge : %d", statistics.discovered_files)
    LOGGER.info("Chunk boyutu  : %d", CHUNK_SIZE)
    LOGGER.info("Chunk overlap : %d", CHUNK_OVERLAP)
    LOGGER.info("Batch boyutu  : %d", batch_size)
    LOGGER.info("Reset modu    : %s", "açık" if reset else "kapalı")
    LOGGER.info("=" * 68)

    chunks = collect_all_chunks(
        document_paths=document_paths,
        statistics=statistics,
    )

    if not chunks:
        raise RuntimeError(
            "Belgeler bulundu ancak hiçbir kullanılabilir metin parçası "
            "oluşturulamadı. PDF belgeleri taranmış görüntülerden oluşuyorsa "
            "OCR uygulanması gerekir."
        )

    LOGGER.info(
        "Toplam %d benzersiz chunk oluşturuldu.",
        len(chunks),
    )

    try:
        vector_store = VectorStore()
    except Exception as exc:
        raise RuntimeError(
            "ChromaDB bağlantısı oluşturulamadı."
        ) from exc

    if reset:
        LOGGER.info("Mevcut ChromaDB koleksiyonu sıfırlanıyor...")

        try:
            vector_store.reset()
        except Exception as exc:
            raise RuntimeError(
                "ChromaDB koleksiyonu sıfırlanamadı."
            ) from exc
    elif vector_store.count() > 0:
        LOGGER.warning(
            "Koleksiyon sıfırlanmadı ve içinde %d kayıt bulunuyor. "
            "Mevcut VectorStore.add() metodu upsert kullanmadığı için aynı "
            "belgelerin yeniden eklenmesi duplicate ID hatası oluşturabilir.",
            vector_store.count(),
        )

    insert_chunks(
        vector_store=vector_store,
        chunks=chunks,
        batch_size=batch_size,
        statistics=statistics,
    )

    collection_count = vector_store.count()
    elapsed_seconds = time.perf_counter() - started_at

    LOGGER.info("=" * 68)
    LOGGER.info("İndeksleme başarıyla tamamlandı")
    LOGGER.info("Bulunan dosya       : %d", statistics.discovered_files)
    LOGGER.info("İşlenen dosya       : %d", statistics.processed_files)
    LOGGER.info("Atlanan dosya       : %d", statistics.skipped_files)
    LOGGER.info("Hatalı dosya        : %d", statistics.failed_files)
    LOGGER.info("İşlenen sayfa       : %d", statistics.extracted_pages)
    LOGGER.info("Boş sayfa           : %d", statistics.empty_pages)
    LOGGER.info("Oluşturulan chunk   : %d", statistics.generated_chunks)
    LOGGER.info("Eklenen chunk       : %d", statistics.inserted_chunks)
    LOGGER.info("Koleksiyon kayıtları: %d", collection_count)
    LOGGER.info("Toplam süre          : %.2f saniye", elapsed_seconds)
    LOGGER.info("=" * 68)

    if statistics.inserted_chunks != len(chunks):
        raise RuntimeError(
            "Eklenen chunk sayısı beklenen değerle eşleşmiyor. "
            f"Beklenen: {len(chunks)}, "
            f"eklenen: {statistics.inserted_chunks}"
        )

    return statistics


def build_argument_parser() -> argparse.ArgumentParser:
    """Komut satırı argümanlarını tanımlar."""

    parser = argparse.ArgumentParser(
        description=(
            "FinAi için TXT ve PDF belgelerini işleyerek "
            "ChromaDB vektör veritabanına kaydeder."
        )
    )

    reset_group = parser.add_mutually_exclusive_group()

    reset_group.add_argument(
        "--reset",
        dest="reset",
        action="store_true",
        help=(
            "Mevcut ChromaDB koleksiyonunu silip yeniden oluşturur. "
            "Varsayılan davranıştır."
        ),
    )

    reset_group.add_argument(
        "--no-reset",
        dest="reset",
        action="store_false",
        help=(
            "Mevcut koleksiyonu silmeden yeni kayıt eklemeyi dener. "
            "Duplicate kimlikler ChromaDB hatası oluşturabilir."
        ),
    )

    parser.set_defaults(reset=True)

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "Tek seferde embedding üretilecek metin sayısı. "
            f"Varsayılan: {DEFAULT_BATCH_SIZE}"
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
        run_ingestion(
            reset=args.reset,
            batch_size=args.batch_size,
        )
    except KeyboardInterrupt:
        LOGGER.warning("İndeksleme kullanıcı tarafından durduruldu.")
        return 130
    except Exception:
        LOGGER.exception("İndeksleme işlemi başarısız oldu.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

