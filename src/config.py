"""FinAi -- Central configuration values."""

from __future__ import annotations

from pathlib import Path


# Proje yolları
BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "documents"
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma"
MODEL_CACHE_DIR = DATA_DIR / "model_cache"


# ChromaDB
COLLECTION_NAME = "financial_literacy"


# Foundry Local
MODEL_NAME = "qwen3.5-2b-text"
MODEL_ALIAS = MODEL_NAME


# Embedding
#
# Türkçe dahil çok dilli semantic retrieval için kullanılır.
# Embedding modeli değiştirildiğinde ChromaDB indeksi mutlaka yeniden
# oluşturulmalıdır.
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

EMBEDDING_DEVICE = "cpu"
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_NORMALIZE = True


# Chunk ayarları
CHUNK_SIZE = 900
CHUNK_OVERLAP = 140
MIN_CHUNK_CHARACTERS = 80


# Retrieval
TOP_K = 4
RETRIEVAL_CANDIDATE_MULTIPLIER = 4
DEFAULT_MIN_SCORE = 0.55
MAX_CONTEXT_CHARACTERS = 5_000



# ---------------------------------------------------------------------
# Answer strategy
# ---------------------------------------------------------------------

# Güvenilir ve hızlı varsayılan davranış:
# Retrieval kaynaklarından deterministik cevap oluşturulur.
PREFER_DETERMINISTIC_ANSWERS = True

# Yerel LLM yalnızca açıkça istendiğinde kullanılır.
ENABLE_LLM_GENERATION = False

# En iyi retrieval skoru bunun altındaysa güçlü cevap üretilmez.
MIN_ANSWER_CONFIDENCE = 0.55

# Deterministik cevap sınırları.
MAX_ANSWER_SENTENCES = 5
MAX_ANSWER_CHARACTERS = 1_200
