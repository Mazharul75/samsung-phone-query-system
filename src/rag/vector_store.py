"""Hybrid vector store combining FAISS dense search with BM25 keyword search.

Why hybrid?
-----------
Dense embeddings capture meaning ("how long does the battery last" matches a
document about battery endurance) but blur exact tokens.  Phone questions are
full of exact tokens - ``Snapdragon 8 Gen 2``, ``120Hz``, ``S23 Ultra`` - where
a lexical scorer such as BM25 is far more reliable.

Running both and fusing their normalised scores gives noticeably better
retrieval than either alone: semantic questions still work, and a query naming
a specific chipset or model reliably surfaces the right document.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.config import VECTOR_STORE_DIR, settings
from src.rag.documents import Document
from src.rag.embeddings import EmbeddingModel, get_embedding_model

logger = logging.getLogger(__name__)

INDEX_FILE = "faiss.index"
DOCUMENTS_FILE = "documents.json"
BM25_FILE = "bm25.pkl"


def tokenize(text: str) -> list[str]:
    """Lower-case word tokenizer used for BM25.

    Alphanumeric runs are kept together so model numbers and units survive as
    single tokens (``s23``, ``5000mah``, ``120hz``).
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def _normalise(scores: np.ndarray) -> np.ndarray:
    """Scale scores to ``[0, 1]``; an all-equal array becomes all zeros."""
    if scores.size == 0:
        return scores
    lowest, highest = float(scores.min()), float(scores.max())
    if highest - lowest < 1e-9:
        return np.zeros_like(scores)
    return (scores - lowest) / (highest - lowest)


class HybridVectorStore:
    """Stores documents and retrieves them by combined dense + lexical score."""

    def __init__(self, embedding_model: EmbeddingModel | None = None) -> None:
        self.embedding_model = embedding_model or get_embedding_model()
        self.documents: list[Document] = []
        self._index = None
        self._bm25 = None

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------
    def build(self, documents: list[Document]) -> None:
        """Index a corpus, replacing anything previously held."""
        if not documents:
            raise ValueError("Cannot build a vector store from an empty corpus.")

        import faiss
        from rank_bm25 import BM25Okapi

        self.documents = documents
        texts = [doc.text for doc in documents]

        logger.info("Embedding %d documents...", len(texts))
        vectors = self.embedding_model.encode(texts)

        # Inner product over normalised vectors == cosine similarity.
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        self._index = index

        logger.info("Building BM25 index...")
        self._bm25 = BM25Okapi([tokenize(text) for text in texts])

        logger.info("Vector store ready: %d documents.", len(documents))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, directory: Path | None = None) -> Path:
        import faiss

        target = Path(directory or VECTOR_STORE_DIR)
        target.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(target / INDEX_FILE))
        (target / DOCUMENTS_FILE).write_text(
            json.dumps([asdict(doc) for doc in self.documents], ensure_ascii=False),
            encoding="utf-8",
        )
        with open(target / BM25_FILE, "wb") as handle:
            pickle.dump(self._bm25, handle)

        logger.info("Vector store saved to %s", target)
        return target

    def load(self, directory: Path | None = None) -> bool:
        """Load a previously saved store; return ``False`` when absent."""
        import faiss

        source = Path(directory or VECTOR_STORE_DIR)
        index_path = source / INDEX_FILE
        docs_path = source / DOCUMENTS_FILE
        bm25_path = source / BM25_FILE

        if not (index_path.exists() and docs_path.exists() and bm25_path.exists()):
            return False

        self._index = faiss.read_index(str(index_path))
        self.documents = [
            Document(**record)
            for record in json.loads(docs_path.read_text(encoding="utf-8"))
        ]
        with open(bm25_path, "rb") as handle:
            self._bm25 = pickle.load(handle)

        logger.info("Vector store loaded: %d documents.", len(self.documents))
        return True

    @property
    def is_ready(self) -> bool:
        return self._index is not None and bool(self.documents)

    def __len__(self) -> int:
        return len(self.documents)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int | None = None,
        phone_ids: set[int] | None = None,
        dense_weight: float | None = None,
    ) -> list[tuple[Document, float]]:
        """Return the ``top_k`` best documents with their fused scores.

        ``phone_ids`` restricts the result to specific phones, which the
        chatbot uses once it has identified which models a question is about.
        """
        if not self.is_ready:
            raise RuntimeError("Vector store is empty - build or load it first.")

        top_k = top_k or settings.retrieval_top_k
        weight = (
            settings.retrieval_dense_weight if dense_weight is None else dense_weight
        )

        # Dense scores over the whole corpus.
        query_vector = self.embedding_model.encode([query])
        dense_scores, dense_ids = self._index.search(query_vector, len(self.documents))
        dense = np.zeros(len(self.documents), dtype=np.float32)
        dense[dense_ids[0]] = dense_scores[0]

        # Lexical scores over the whole corpus.
        lexical = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float32)

        combined = weight * _normalise(dense) + (1.0 - weight) * _normalise(lexical)

        # Filtering happens after scoring so the normalisation stays stable.
        candidates = range(len(self.documents))
        if phone_ids:
            candidates = [
                i for i in candidates if self.documents[i].phone_id in phone_ids
            ]
            if not candidates:
                candidates = range(len(self.documents))

        ranked = sorted(candidates, key=lambda i: float(combined[i]), reverse=True)
        return [(self.documents[i], float(combined[i])) for i in ranked[:top_k]]


def get_vector_store(auto_build: bool = True) -> HybridVectorStore:
    """Load the saved store, optionally building it from the database.

    Returns a ready-to-query store.  When no saved index exists and
    ``auto_build`` is set, the corpus is rebuilt from the database so the API
    can start on a fresh checkout without a separate indexing step.
    """
    store = HybridVectorStore()
    if store.load():
        return store

    if not auto_build:
        raise RuntimeError(
            "No vector store found. Run 'python -m scripts.build_index' first."
        )

    logger.info("No saved index found - building one from the database.")
    from src.database.connection import session_scope
    from src.database.repository import PhoneRepository
    from src.rag.documents import build_corpus

    with session_scope() as session:
        documents = build_corpus(PhoneRepository(session))

    store.build(documents)
    store.save()
    return store
