from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from src.config import CHROMA_DIR, COLLECTION_NAME


class VectorStore:
    """
    ChromaDB katmanı.

    Bu sınıf;

    - Collection oluşturur
    - Collection sıfırlar
    - Toplu kayıt ekler
    - Arama yapar
    - Kayıt sayısını döndürür
    - Collection bilgilerini verir

    Diğer dosyalar (ingest, retriever, rag)
    yalnızca bu sınıfı kullanacaktır.
    """

    def __init__(self):

        Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(
                anonymized_telemetry=False
            )
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine"
            }
        )

    ####################################################################
    # Collection
    ####################################################################

    def reset(self) -> None:
        """
        Collection'ı tamamen silip yeniden oluşturur.
        """

        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine"
            }
        )

    ####################################################################
    # Add
    ####################################################################

    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """
        Toplu doküman ekler.
        """

        if len(ids) == 0:
            return

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    ####################################################################
    # Search
    ####################################################################

    def search(
        self,
        embedding: list[float],
        top_k: int,
    ) -> dict[str, Any]:
        """
        Embedding kullanarak benzer kayıtları getirir.
        """

        return self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

    ####################################################################
    # Count
    ####################################################################

    def count(self) -> int:

        return self.collection.count()

    ####################################################################
    # Collection Info
    ####################################################################

    def get_collection_name(self) -> str:

        return self.collection.name

    ####################################################################
    # Peek
    ####################################################################

    def peek(self, limit: int = 5):

        return self.collection.peek(limit=limit)

    ####################################################################
    # Exists
    ####################################################################

    def is_empty(self) -> bool:

        return self.count() == 0