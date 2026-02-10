"""Integration tests for the search pipeline."""

import sys
from pathlib import Path

import pytest

from fileassistant.analyzer import FileAnalyzer
from fileassistant.embeddings import EmbeddingGenerator
from fileassistant.search import IndexManager


def check_chromadb_available():
    """Check if ChromaDB can be initialized without errors."""
    if sys.version_info >= (3, 14):
        return False
    try:
        import chromadb
        client = chromadb.Client()
        return True
    except Exception:
        return False


CHROMADB_AVAILABLE = check_chromadb_available()


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not available")
class TestSearchPipelineIntegration:
    """
    Integration tests for the full search pipeline:
    File → Analyzer → EmbeddingGenerator → IndexManager → Search
    """

    @pytest.fixture
    def analyzer(self):
        """Create a file analyzer."""
        return FileAnalyzer(max_file_size_mb=10)

    @pytest.fixture
    def embedding_generator(self):
        """Create an embedding generator."""
        return EmbeddingGenerator()

    @pytest.fixture
    def index_manager(self, tmp_path):
        """Create an index manager with temporary storage."""
        manager = IndexManager(persist_directory=tmp_path / "chromadb")
        yield manager
        manager.close()

    @pytest.fixture
    def sample_text_file(self, tmp_path):
        """Create a sample text file for testing."""
        file_path = tmp_path / "sample_document.txt"
        file_path.write_text(
            """
            Machine Learning and Artificial Intelligence

            This document discusses the fundamentals of machine learning,
            a subset of artificial intelligence that enables systems to
            learn and improve from experience without being explicitly
            programmed.

            Key topics covered:
            - Supervised Learning
            - Unsupervised Learning
            - Neural Networks
            - Deep Learning

            Machine learning applications include image recognition,
            natural language processing, and recommendation systems.
            """
        )
        return file_path

    @pytest.fixture
    def sample_pdf_file(self, tmp_path):
        """Create a sample PDF-like file for testing."""
        # We'll use a text file since PDF creation requires additional dependencies
        # In real tests, you might want to include an actual PDF fixture
        file_path = tmp_path / "report.txt"
        file_path.write_text(
            """
            Q4 2024 Financial Report

            Revenue increased by 15% compared to Q3.
            Operating expenses remained stable.
            Net profit margin improved to 12%.

            Key metrics:
            - Total Revenue: $1.5M
            - Operating Costs: $900K
            - Net Profit: $180K
            """
        )
        return file_path

    @pytest.mark.slow
    def test_full_pipeline_index_and_retrieve(
        self, analyzer, embedding_generator, index_manager, sample_text_file
    ):
        """Test the complete pipeline: analyze → embed → index → retrieve."""
        # Step 1: Analyze the file
        analysis = analyzer.analyze(sample_text_file)
        assert analysis.success is True
        assert analysis.text is not None
        assert len(analysis.text) > 0

        # Step 2: Generate embedding
        embedding_result = embedding_generator.generate(analysis.text)
        assert embedding_result.success is True
        assert len(embedding_result.embedding) == 384

        # Step 3: Index the file
        file_id = f"file_{sample_text_file.stem}"
        success = index_manager.index_file(
            file_id=file_id,
            file_path=sample_text_file,
            text=analysis.text,
            embedding=embedding_result.embedding,
            tags=["ml", "ai", "document"],
            content_summary=analysis.text[:200],
            file_type="document",
            size_bytes=analysis.metadata.size_bytes,
        )
        assert success is True
        assert index_manager.get_indexed_count() == 1

        # Step 4: Retrieve and verify
        metadata, document = index_manager.get_file(file_id)
        assert metadata is not None
        assert metadata.filename == sample_text_file.name
        assert metadata.tags == ["ml", "ai", "document"]
        assert "Machine Learning" in document or "machine learning" in document.lower()

    @pytest.mark.slow
    def test_search_finds_relevant_document(
        self, analyzer, embedding_generator, index_manager, sample_text_file, sample_pdf_file
    ):
        """Test that search returns relevant documents."""
        # Index multiple files
        files = [sample_text_file, sample_pdf_file]
        for i, file_path in enumerate(files):
            analysis = analyzer.analyze(file_path)
            embedding_result = embedding_generator.generate(analysis.text)

            index_manager.index_file(
                file_id=f"file_{i}",
                file_path=file_path,
                text=analysis.text,
                embedding=embedding_result.embedding,
                tags=["test"],
            )

        assert index_manager.get_indexed_count() == 2

        # Search for ML-related content
        query = "neural networks and deep learning"
        query_embedding_result = embedding_generator.generate(query)
        assert query_embedding_result.success is True

        results = index_manager.search(
            query_embedding=query_embedding_result.embedding,
            n_results=2,
        )

        assert len(results) == 2

        # The ML document should be more relevant (lower distance)
        # Results are sorted by distance (ascending)
        top_result = results[0]
        assert "sample_document" in top_result[0].filename or "machine" in top_result[2].lower()

    @pytest.mark.slow
    def test_is_indexed_with_content_change_detection(
        self, analyzer, embedding_generator, index_manager, tmp_path
    ):
        """Test that content changes are detected via hash."""
        # Create a file
        file_path = tmp_path / "changing_file.txt"
        file_path.write_text("Original content about cats and dogs.")

        # Index it
        analysis = analyzer.analyze(file_path)
        embedding_result = embedding_generator.generate(analysis.text)
        original_hash = IndexManager.compute_content_hash(analysis.text)

        index_manager.index_file(
            file_id="changing_file",
            file_path=file_path,
            text=analysis.text,
            embedding=embedding_result.embedding,
        )

        # Check it's indexed with same hash
        assert index_manager.is_indexed("changing_file", original_hash) is True

        # Modify the file
        file_path.write_text("Completely new content about quantum physics.")

        # Re-analyze
        new_analysis = analyzer.analyze(file_path)
        new_hash = IndexManager.compute_content_hash(new_analysis.text)

        # Should detect change
        assert index_manager.is_indexed("changing_file", new_hash) is False

        # Re-index with new content
        new_embedding = embedding_generator.generate(new_analysis.text)
        index_manager.index_file(
            file_id="changing_file",
            file_path=file_path,
            text=new_analysis.text,
            embedding=new_embedding.embedding,
        )

        # Now should be indexed with new hash
        assert index_manager.is_indexed("changing_file", new_hash) is True

    @pytest.mark.slow
    def test_embedding_consistency(self, embedding_generator):
        """Test that the same text produces consistent embeddings."""
        text = "This is a test document for embedding consistency."

        result1 = embedding_generator.generate(text)
        result2 = embedding_generator.generate(text)

        assert result1.success is True
        assert result2.success is True

        # Embeddings should be identical for same input
        assert result1.embedding == result2.embedding

    @pytest.mark.slow
    def test_chunking_produces_valid_embeddings(self, embedding_generator):
        """Test that long text chunking produces valid embeddings."""
        # Create long text that will require chunking
        long_text = " ".join([
            f"This is paragraph {i}. " * 10
            for i in range(50)
        ])

        result = embedding_generator.generate(long_text)

        assert result.success is True
        assert result.chunk_count > 1
        assert len(result.embedding) == 384

    @pytest.mark.slow
    def test_search_empty_query(self, embedding_generator, index_manager, tmp_path):
        """Test search with minimal/edge case queries."""
        # Index a file
        file_path = tmp_path / "test.txt"
        file_path.write_text("Some test content.")

        embedding_result = embedding_generator.generate("Some test content.")
        index_manager.index_file(
            file_id="test",
            file_path=file_path,
            text="Some test content.",
            embedding=embedding_result.embedding,
        )

        # Search with very short query
        short_query_embedding = embedding_generator.generate("test")
        assert short_query_embedding.success is True

        results = index_manager.search(
            query_embedding=short_query_embedding.embedding,
            n_results=5,
        )

        # Should still find the document
        assert len(results) == 1


class TestSearchConfigIntegration:
    """Test search components with config integration."""

    def test_embedding_generator_with_config_settings(self):
        """Test embedding generator uses config chunk settings."""
        from fileassistant.config.models import SearchSettings

        settings = SearchSettings(chunk_size=256, chunk_overlap=25)

        generator = EmbeddingGenerator(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        assert generator.chunk_size == 256
        assert generator.chunk_overlap == 25

    def test_index_manager_with_config_path(self, tmp_path):
        """Test index manager uses config path."""
        from fileassistant.config.models import DatabaseSettings

        settings = DatabaseSettings(vector_store_path=tmp_path / "custom_chromadb")

        manager = IndexManager(persist_directory=settings.vector_store_path)
        assert manager.persist_directory == tmp_path / "custom_chromadb"
        manager.close()
