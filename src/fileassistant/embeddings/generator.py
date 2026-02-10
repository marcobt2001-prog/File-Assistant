"""Embedding generator for creating vector representations of text."""

import re
from dataclasses import dataclass, field
from typing import ClassVar

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""

    embedding: list[float]
    chunk_count: int
    token_estimate: int
    model_name: str
    success: bool = True
    error_message: str | None = None

    @classmethod
    def failure(cls, error_message: str) -> "EmbeddingResult":
        """Create a failed embedding result."""
        return cls(
            embedding=[],
            chunk_count=0,
            token_estimate=0,
            model_name="",
            success=False,
            error_message=error_message,
        )


class EmbeddingGenerator:
    """
    Generates vector embeddings from text using sentence-transformers.

    Features:
    - Uses all-MiniLM-L6-v2 model by default (fast, ~80MB)
    - Handles text chunking for long documents
    - Averages chunk embeddings into a single file embedding
    - Caches the model after first load for efficiency
    """

    # Class-level model cache to avoid reloading
    _model_cache: ClassVar[dict] = {}

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        """
        Initialize the embedding generator.

        Args:
            model_name: Name of the sentence-transformers model to use
            chunk_size: Target size of each text chunk (in estimated tokens)
            chunk_overlap: Number of tokens to overlap between chunks
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._model = None

    def _get_model(self):
        """
        Get the sentence transformer model, loading from cache if available.

        The model is cached at the class level to avoid reloading across instances.
        """
        if self.model_name not in self._model_cache:
            logger.info(f"Loading embedding model: {self.model_name}")
            try:
                from sentence_transformers import SentenceTransformer

                self._model_cache[self.model_name] = SentenceTransformer(self.model_name)
                logger.info(f"Embedding model loaded: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise

        return self._model_cache[self.model_name]

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in text.

        Uses a simple heuristic: ~4 characters per token for English.
        """
        return len(text) // 4

    def _split_into_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences.

        Uses a simple regex-based approach that handles common cases.
        """
        # Split on sentence-ending punctuation followed by whitespace
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _chunk_text(self, text: str) -> list[str]:
        """
        Split text into chunks of approximately chunk_size tokens.

        Splits on sentence boundaries when possible to maintain coherence.
        Includes overlap between chunks for context continuity.
        """
        if not text or not text.strip():
            return []

        # If text is short enough, return as single chunk
        if self._estimate_tokens(text) <= self.chunk_size:
            return [text.strip()]

        sentences = self._split_into_sentences(text)
        if not sentences:
            return [text.strip()]

        chunks = []
        current_chunk: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)

            # If single sentence exceeds chunk size, add it as its own chunk
            if sentence_tokens > self.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                chunks.append(sentence)
                continue

            # Check if adding this sentence would exceed chunk size
            if current_tokens + sentence_tokens > self.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))

                    # Calculate overlap: keep last few sentences that fit in overlap budget
                    overlap_sentences = []
                    overlap_tokens = 0
                    for s in reversed(current_chunk):
                        s_tokens = self._estimate_tokens(s)
                        if overlap_tokens + s_tokens <= self.chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_tokens += s_tokens
                        else:
                            break

                    current_chunk = overlap_sentences
                    current_tokens = overlap_tokens

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        # Add remaining sentences as final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def generate(self, text: str) -> EmbeddingResult:
        """
        Generate an embedding vector for the given text.

        For long text, chunks it and averages the embeddings.

        Args:
            text: The text to generate an embedding for

        Returns:
            EmbeddingResult with the embedding vector
        """
        if not text or not text.strip():
            return EmbeddingResult.failure("Empty text provided")

        try:
            model = self._get_model()

            # Chunk the text
            chunks = self._chunk_text(text)
            if not chunks:
                return EmbeddingResult.failure("No valid chunks generated from text")

            logger.debug(f"Generating embeddings for {len(chunks)} chunk(s)")

            # Generate embeddings for each chunk
            chunk_embeddings = model.encode(chunks, convert_to_numpy=True)

            # Average the embeddings if multiple chunks
            if len(chunks) > 1:
                import numpy as np

                averaged = np.mean(chunk_embeddings, axis=0)
                final_embedding = averaged.tolist()
            else:
                final_embedding = chunk_embeddings[0].tolist()

            return EmbeddingResult(
                embedding=final_embedding,
                chunk_count=len(chunks),
                token_estimate=self._estimate_tokens(text),
                model_name=self.model_name,
                success=True,
            )

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return EmbeddingResult.failure(str(e))

    def generate_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to generate embeddings for

        Returns:
            List of EmbeddingResult objects
        """
        results = []
        for text in texts:
            results.append(self.generate(text))
        return results

    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the model."""
        model = self._get_model()
        return model.get_sentence_embedding_dimension()

    @classmethod
    def clear_model_cache(cls):
        """Clear the model cache (useful for testing or memory management)."""
        cls._model_cache.clear()
        logger.info("Embedding model cache cleared")
