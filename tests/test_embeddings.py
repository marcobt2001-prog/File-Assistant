"""Tests for the embedding generator."""

from unittest.mock import MagicMock, patch

import pytest

from fileassistant.embeddings.generator import EmbeddingGenerator, EmbeddingResult


class TestEmbeddingResult:
    """Tests for EmbeddingResult dataclass."""

    def test_success_result(self):
        """Test creating a successful result."""
        result = EmbeddingResult(
            embedding=[0.1, 0.2, 0.3],
            chunk_count=1,
            token_estimate=100,
            model_name="test-model",
            success=True,
        )
        assert result.success is True
        assert result.error_message is None
        assert len(result.embedding) == 3

    def test_failure_result(self):
        """Test creating a failed result."""
        result = EmbeddingResult.failure("Test error")
        assert result.success is False
        assert result.error_message == "Test error"
        assert result.embedding == []
        assert result.chunk_count == 0


class TestEmbeddingGenerator:
    """Tests for EmbeddingGenerator."""

    @pytest.fixture
    def generator(self):
        """Create a generator instance."""
        return EmbeddingGenerator(
            model_name="all-MiniLM-L6-v2",
            chunk_size=512,
            chunk_overlap=50,
        )

    def test_initialization(self, generator):
        """Test generator initialization."""
        assert generator.model_name == "all-MiniLM-L6-v2"
        assert generator.chunk_size == 512
        assert generator.chunk_overlap == 50

    def test_estimate_tokens(self, generator):
        """Test token estimation."""
        # ~4 chars per token
        text = "a" * 100
        assert generator._estimate_tokens(text) == 25

        text = "a" * 400
        assert generator._estimate_tokens(text) == 100

    def test_split_into_sentences(self, generator):
        """Test sentence splitting."""
        text = "First sentence. Second sentence! Third sentence?"
        sentences = generator._split_into_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "First sentence."
        assert sentences[1] == "Second sentence!"
        assert sentences[2] == "Third sentence?"

    def test_split_into_sentences_handles_abbreviations(self, generator):
        """Test that sentence splitting handles common cases."""
        text = "Dr. Smith went to the store. He bought milk."
        sentences = generator._split_into_sentences(text)
        # This simple splitter will split on ". " so Dr. Smith stays together
        # but "store. He" will split
        assert len(sentences) >= 2

    def test_chunk_text_short(self, generator):
        """Test chunking with short text that fits in one chunk."""
        short_text = "This is a short text."
        chunks = generator._chunk_text(short_text)
        assert len(chunks) == 1
        assert chunks[0] == short_text

    def test_chunk_text_empty(self, generator):
        """Test chunking with empty text."""
        assert generator._chunk_text("") == []
        assert generator._chunk_text("   ") == []

    def test_chunk_text_long(self, generator):
        """Test chunking with long text that needs multiple chunks."""
        # Create text that exceeds chunk size
        # 512 tokens * 4 chars = ~2048 chars
        long_text = ". ".join(["This is sentence number " + str(i) for i in range(100)])
        chunks = generator._chunk_text(long_text)
        assert len(chunks) > 1

    def test_chunk_text_preserves_content(self, generator):
        """Test that chunking doesn't lose content."""
        sentences = [f"Sentence {i}." for i in range(20)]
        text = " ".join(sentences)
        chunks = generator._chunk_text(text)

        # All sentences should appear in at least one chunk
        combined = " ".join(chunks)
        for sentence in sentences:
            assert sentence in combined

    def test_generate_empty_text(self, generator):
        """Test generating embedding for empty text."""
        result = generator.generate("")
        assert result.success is False
        assert "Empty text" in result.error_message

        result = generator.generate("   ")
        assert result.success is False

    @patch.object(EmbeddingGenerator, '_get_model')
    def test_generate_success(self, mock_get_model, generator):
        """Test successful embedding generation."""
        import numpy as np

        # Mock the model
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_get_model.return_value = mock_model

        result = generator.generate("Test text for embedding.")

        assert result.success is True
        assert len(result.embedding) == 3
        assert result.chunk_count == 1
        assert result.model_name == "all-MiniLM-L6-v2"
        mock_model.encode.assert_called_once()

    @patch.object(EmbeddingGenerator, '_get_model')
    def test_generate_multiple_chunks_averages(self, mock_get_model, generator):
        """Test that multiple chunk embeddings are averaged."""
        import numpy as np

        # Mock model with deterministic embeddings
        mock_model = MagicMock()
        # Return different embeddings for each chunk
        mock_model.encode.return_value = np.array([
            [1.0, 2.0, 3.0],
            [3.0, 4.0, 5.0],
        ])
        mock_get_model.return_value = mock_model

        # Create text that will be split into multiple chunks
        long_text = ". ".join([f"Sentence number {i} with more words" for i in range(100)])
        result = generator.generate(long_text)

        assert result.success is True
        assert result.chunk_count > 1
        # Average of [1,2,3] and [3,4,5] is [2,3,4]
        assert result.embedding == [2.0, 3.0, 4.0]

    @patch.object(EmbeddingGenerator, '_get_model')
    def test_generate_handles_model_error(self, mock_get_model, generator):
        """Test handling of model errors."""
        mock_get_model.side_effect = Exception("Model loading failed")

        result = generator.generate("Test text")

        assert result.success is False
        assert "Model loading failed" in result.error_message

    @patch.object(EmbeddingGenerator, '_get_model')
    def test_generate_batch(self, mock_get_model, generator):
        """Test batch embedding generation."""
        import numpy as np

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_get_model.return_value = mock_model

        texts = ["Text one.", "Text two.", "Text three."]
        results = generator.generate_batch(texts)

        assert len(results) == 3
        for result in results:
            assert result.success is True

    @patch.object(EmbeddingGenerator, '_get_model')
    def test_embedding_dimension(self, mock_get_model, generator):
        """Test getting embedding dimension."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_get_model.return_value = mock_model

        assert generator.embedding_dimension == 384

    def test_clear_model_cache(self):
        """Test clearing the model cache."""
        # Add something to cache
        EmbeddingGenerator._model_cache["test"] = "dummy"
        assert "test" in EmbeddingGenerator._model_cache

        EmbeddingGenerator.clear_model_cache()
        assert len(EmbeddingGenerator._model_cache) == 0


class TestEmbeddingGeneratorIntegration:
    """Integration tests that use the real model (slower, optional)."""

    @pytest.fixture
    def real_generator(self):
        """Create a generator with the real model."""
        return EmbeddingGenerator()

    @pytest.mark.slow
    def test_real_embedding_generation(self, real_generator):
        """Test with real sentence-transformers model."""
        text = "This is a test document about machine learning and AI."
        result = real_generator.generate(text)

        assert result.success is True
        assert len(result.embedding) > 0
        assert result.model_name == "all-MiniLM-L6-v2"
        # all-MiniLM-L6-v2 produces 384-dimensional embeddings
        assert len(result.embedding) == 384

    @pytest.mark.slow
    def test_real_embedding_similarity(self, real_generator):
        """Test that similar texts produce similar embeddings."""
        import numpy as np

        text1 = "The quick brown fox jumps over the lazy dog."
        text2 = "A fast brown fox leaps over a sleepy dog."
        text3 = "Quantum mechanics describes subatomic particles."

        result1 = real_generator.generate(text1)
        result2 = real_generator.generate(text2)
        result3 = real_generator.generate(text3)

        # Compute cosine similarities
        def cosine_sim(a, b):
            a = np.array(a)
            b = np.array(b)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        sim_1_2 = cosine_sim(result1.embedding, result2.embedding)
        sim_1_3 = cosine_sim(result1.embedding, result3.embedding)

        # Similar texts should have higher similarity
        assert sim_1_2 > sim_1_3
        assert sim_1_2 > 0.8  # Should be quite similar
