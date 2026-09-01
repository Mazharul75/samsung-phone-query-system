"""Sentence-embedding model wrapper.

``all-MiniLM-L6-v2`` is used as the encoder: it is Apache-2.0 licensed, only
about 90 MB, and runs comfortably on CPU while still producing strong semantic
matches for short technical text.  Its 384-dimensional vectors keep the FAISS
index tiny.

The model is loaded lazily and cached process-wide, because loading it costs a
few seconds and every component that retrieves needs the same instance.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Encodes text into L2-normalised vectors suitable for cosine search."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            # Imported lazily so that merely importing the package does not
            # pull in torch and sentence-transformers.
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    @property
    def dimension(self) -> int:
        return int(self._ensure_loaded().get_sentence_embedding_dimension())

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Return a ``(len(texts), dimension)`` float32 matrix.

        Vectors are normalised so that an inner-product search is exactly a
        cosine-similarity search.
        """
        model = self._ensure_loaded()
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    """Return the process-wide embedding model singleton."""
    return EmbeddingModel()
