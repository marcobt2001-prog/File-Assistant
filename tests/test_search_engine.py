"""Tests for the search engine."""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fileassistant.search.engine import SearchEngine, SearchResult
from fileassistant.search.index_manager import IndexedFileMetadata


class TestSearchResult:
    """Tests for the SearchResult dataclass."""

    def test_search_result_creation(self):
        """Test creating a SearchResult."""
        result = SearchResult(
            file_path="/path/to/file.pdf",
            filename="file.pdf",
            relevance_score=0.85,
            content_snippet="This is a test document...",
            tags=["work", "report"],
            file_type="document",
            modified_at=datetime(2025, 1, 15),
            size_bytes=1024,
            extension=".pdf",
        )

        assert result.filename == "file.pdf"
        assert result.relevance_score == 0.85
        assert result.tags == ["work", "report"]
        assert result.extension == ".pdf"

    def test_from_index_result_high_relevance(self):
        """Test creating SearchResult from index result with high relevance."""
        metadata = IndexedFileMetadata(
            file_id="test123",
            file_path="/path/to/file.pdf",
            filename="file.pdf",
            extension=".pdf",
            file_type="document",
            tags=["work"],
            content_summary="Summary",
            content_hash="abc123",
            created_at=datetime(2025, 1, 10),
            modified_at=datetime(2025, 1, 15),
            indexed_at=datetime.now(),
            size_bytes=2048,
            source_folder="Documents",
        )

        # Low distance = high relevance
        result = SearchResult.from_index_result(
            metadata=metadata,
            distance=0.2,  # Very close match
            document="This is the full document content for testing purposes.",
        )

        assert result.filename == "file.pdf"
        assert result.relevance_score >= 0.9  # High relevance
        assert result.content_snippet == "This is the full document content for testing purposes."
        assert result.tags == ["work"]

    def test_from_index_result_low_relevance(self):
        """Test creating SearchResult from index result with low relevance."""
        metadata = IndexedFileMetadata(
            file_id="test123",
            file_path="/path/to/file.txt",
            filename="file.txt",
            extension=".txt",
            file_type="document",
            tags=[],
            content_summary="",
            content_hash="abc123",
            created_at=None,
            modified_at=None,
            indexed_at=datetime.now(),
            size_bytes=100,
            source_folder="",
        )

        # High distance = low relevance
        result = SearchResult.from_index_result(
            metadata=metadata,
            distance=1.8,  # Far match
            document="Some content",
        )

        assert result.relevance_score <= 0.2  # Low relevance

    def test_from_index_result_snippet_truncation(self):
        """Test that long content is truncated for snippet."""
        metadata = IndexedFileMetadata(
            file_id="test",
            file_path="/test",
            filename="test.txt",
            extension=".txt",
            file_type="document",
            tags=[],
            content_summary="",
            content_hash="",
            created_at=None,
            modified_at=None,
            indexed_at=datetime.now(),
            size_bytes=0,
            source_folder="",
        )

        # Create content longer than 200 chars
        long_content = "word " * 100  # 500 chars

        result = SearchResult.from_index_result(
            metadata=metadata,
            distance=0.5,
            document=long_content,
        )

        assert len(result.content_snippet) <= 210  # 200 + "..."
        assert result.content_snippet.endswith("...")


class TestSearchEngineFilters:
    """Tests for SearchEngine filter building."""

    @pytest.fixture
    def mock_engine(self):
        """Create a search engine with mocked dependencies."""
        with patch("fileassistant.search.engine.IndexManager") as mock_im:
            with patch("fileassistant.search.engine.EmbeddingGenerator") as mock_eg:
                mock_im_instance = MagicMock()
                mock_eg_instance = MagicMock()
                mock_im.return_value = mock_im_instance
                mock_eg.return_value = mock_eg_instance

                engine = SearchEngine()
                yield engine

    def test_build_chroma_filter_single_extension(self, mock_engine):
        """Test building filter with single extension."""
        filters = {"extension": ".pdf"}
        result = mock_engine._build_chroma_filter(filters)

        assert result == {"extension": ".pdf"}

    def test_build_chroma_filter_extension_without_dot(self, mock_engine):
        """Test that extension without dot gets normalized."""
        filters = {"extension": "pdf"}
        result = mock_engine._build_chroma_filter(filters)

        assert result == {"extension": ".pdf"}

    def test_build_chroma_filter_multiple_extensions(self, mock_engine):
        """Test building filter with multiple extensions."""
        filters = {"extension": [".pdf", ".docx", ".txt"]}
        result = mock_engine._build_chroma_filter(filters)

        assert result == {"extension": {"$in": [".pdf", ".docx", ".txt"]}}

    def test_build_chroma_filter_file_type(self, mock_engine):
        """Test building filter with file type."""
        filters = {"file_type": "document"}
        result = mock_engine._build_chroma_filter(filters)

        assert result == {"file_type": "document"}

    def test_build_chroma_filter_combined(self, mock_engine):
        """Test building filter with multiple criteria."""
        filters = {"extension": ".pdf", "file_type": "document"}
        result = mock_engine._build_chroma_filter(filters)

        assert "$and" in result
        assert {"extension": ".pdf"} in result["$and"]
        assert {"file_type": "document"} in result["$and"]

    def test_build_chroma_filter_empty(self, mock_engine):
        """Test building filter with no applicable filters."""
        # Date filters aren't handled by ChromaDB
        filters = {"after": datetime(2025, 1, 1)}
        result = mock_engine._build_chroma_filter(filters)

        assert result is None


class TestSearchEnginePostFilters:
    """Tests for SearchEngine post-retrieval filtering."""

    @pytest.fixture
    def mock_engine(self):
        """Create a search engine with mocked dependencies."""
        with patch("fileassistant.search.engine.IndexManager"):
            with patch("fileassistant.search.engine.EmbeddingGenerator"):
                yield SearchEngine()

    def test_passes_post_filters_no_filters(self, mock_engine):
        """Test that files pass when no filters are set."""
        metadata = IndexedFileMetadata(
            file_id="test",
            file_path="/test",
            filename="test.txt",
            extension=".txt",
            file_type="document",
            tags=["work"],
            content_summary="",
            content_hash="",
            created_at=None,
            modified_at=datetime(2025, 1, 15),
            indexed_at=datetime.now(),
            size_bytes=100,
            source_folder="",
        )

        assert mock_engine._passes_post_filters(metadata, {}) is True

    def test_passes_post_filters_after_date_pass(self, mock_engine):
        """Test after date filter - file passes."""
        metadata = IndexedFileMetadata(
            file_id="test",
            file_path="/test",
            filename="test.txt",
            extension=".txt",
            file_type="document",
            tags=[],
            content_summary="",
            content_hash="",
            created_at=None,
            modified_at=datetime(2025, 2, 1),
            indexed_at=datetime.now(),
            size_bytes=0,
            source_folder="",
        )

        filters = {"after": datetime(2025, 1, 1)}
        assert mock_engine._passes_post_filters(metadata, filters) is True

    def test_passes_post_filters_after_date_fail(self, mock_engine):
        """Test after date filter - file fails."""
        metadata = IndexedFileMetadata(
            file_id="test",
            file_path="/test",
            filename="test.txt",
            extension=".txt",
            file_type="document",
            tags=[],
            content_summary="",
            content_hash="",
            created_at=None,
            modified_at=datetime(2024, 12, 1),
            indexed_at=datetime.now(),
            size_bytes=0,
            source_folder="",
        )

        filters = {"after": datetime(2025, 1, 1)}
        assert mock_engine._passes_post_filters(metadata, filters) is False

    def test_passes_post_filters_before_date_pass(self, mock_engine):
        """Test before date filter - file passes."""
        metadata = IndexedFileMetadata(
            file_id="test",
            file_path="/test",
            filename="test.txt",
            extension=".txt",
            file_type="document",
            tags=[],
            content_summary="",
            content_hash="",
            created_at=None,
            modified_at=datetime(2024, 12, 1),
            indexed_at=datetime.now(),
            size_bytes=0,
            source_folder="",
        )

        filters = {"before": datetime(2025, 1, 1)}
        assert mock_engine._passes_post_filters(metadata, filters) is True

    def test_passes_post_filters_before_date_fail(self, mock_engine):
        """Test before date filter - file fails."""
        metadata = IndexedFileMetadata(
            file_id="test",
            file_path="/test",
            filename="test.txt",
            extension=".txt",
            file_type="document",
            tags=[],
            content_summary="",
            content_hash="",
            created_at=None,
            modified_at=datetime(2025, 2, 1),
            indexed_at=datetime.now(),
            size_bytes=0,
            source_folder="",
        )

        filters = {"before": datetime(2025, 1, 1)}
        assert mock_engine._passes_post_filters(metadata, filters) is False

    def test_passes_post_filters_tag_match(self, mock_engine):
        """Test tag filter - file has matching tag."""
        metadata = IndexedFileMetadata(
            file_id="test",
            file_path="/test",
            filename="test.txt",
            extension=".txt",
            file_type="document",
            tags=["work", "report"],
            content_summary="",
            content_hash="",
            created_at=None,
            modified_at=None,
            indexed_at=datetime.now(),
            size_bytes=0,
            source_folder="",
        )

        filters = {"tag": "work"}
        assert mock_engine._passes_post_filters(metadata, filters) is True

    def test_passes_post_filters_tag_no_match(self, mock_engine):
        """Test tag filter - file doesn't have tag."""
        metadata = IndexedFileMetadata(
            file_id="test",
            file_path="/test",
            filename="test.txt",
            extension=".txt",
            file_type="document",
            tags=["personal"],
            content_summary="",
            content_hash="",
            created_at=None,
            modified_at=None,
            indexed_at=datetime.now(),
            size_bytes=0,
            source_folder="",
        )

        filters = {"tag": "work"}
        assert mock_engine._passes_post_filters(metadata, filters) is False

    def test_passes_post_filters_tag_case_insensitive(self, mock_engine):
        """Test tag filter is case insensitive."""
        metadata = IndexedFileMetadata(
            file_id="test",
            file_path="/test",
            filename="test.txt",
            extension=".txt",
            file_type="document",
            tags=["Work", "Report"],
            content_summary="",
            content_hash="",
            created_at=None,
            modified_at=None,
            indexed_at=datetime.now(),
            size_bytes=0,
            source_folder="",
        )

        filters = {"tag": "work"}
        assert mock_engine._passes_post_filters(metadata, filters) is True

    def test_passes_post_filters_date_as_string(self, mock_engine):
        """Test that date filters work with ISO string format."""
        metadata = IndexedFileMetadata(
            file_id="test",
            file_path="/test",
            filename="test.txt",
            extension=".txt",
            file_type="document",
            tags=[],
            content_summary="",
            content_hash="",
            created_at=None,
            modified_at=datetime(2025, 2, 1),
            indexed_at=datetime.now(),
            size_bytes=0,
            source_folder="",
        )

        filters = {"after": "2025-01-01"}
        assert mock_engine._passes_post_filters(metadata, filters) is True


class TestSearchEngineSearch:
    """Tests for SearchEngine.search method."""

    def test_search_empty_query(self):
        """Test search with empty query."""
        with patch("fileassistant.search.engine.IndexManager") as mock_im:
            with patch("fileassistant.search.engine.EmbeddingGenerator"):
                mock_im.return_value.get_indexed_count.return_value = 10
                engine = SearchEngine()

                results = engine.search("")
                assert results == []

                results = engine.search("   ")
                assert results == []

    def test_search_query_too_short(self):
        """Test search with query that's too short."""
        with patch("fileassistant.search.engine.IndexManager") as mock_im:
            with patch("fileassistant.search.engine.EmbeddingGenerator"):
                mock_im.return_value.get_indexed_count.return_value = 10
                engine = SearchEngine()

                results = engine.search("a")
                assert results == []

    def test_search_empty_index(self):
        """Test search on empty index."""
        with patch("fileassistant.search.engine.IndexManager") as mock_im:
            with patch("fileassistant.search.engine.EmbeddingGenerator"):
                mock_im.return_value.get_indexed_count.return_value = 0
                engine = SearchEngine()

                results = engine.search("test query")
                assert results == []

    def test_search_embedding_failure(self):
        """Test search when embedding generation fails."""
        with patch("fileassistant.search.engine.IndexManager") as mock_im:
            with patch("fileassistant.search.engine.EmbeddingGenerator") as mock_eg:
                mock_im.return_value.get_indexed_count.return_value = 10

                # Simulate embedding failure
                mock_result = MagicMock()
                mock_result.success = False
                mock_result.error_message = "Model error"
                mock_eg.return_value.generate.return_value = mock_result

                engine = SearchEngine()
                results = engine.search("test query")

                assert results == []

    def test_search_success(self):
        """Test successful search."""
        with patch("fileassistant.search.engine.IndexManager") as mock_im:
            with patch("fileassistant.search.engine.EmbeddingGenerator") as mock_eg:
                # Setup mocks
                mock_im_instance = MagicMock()
                mock_im_instance.get_indexed_count.return_value = 10
                mock_im.return_value = mock_im_instance

                # Successful embedding
                mock_embed_result = MagicMock()
                mock_embed_result.success = True
                mock_embed_result.embedding = [0.1] * 384
                mock_eg.return_value.generate.return_value = mock_embed_result

                # Search results
                metadata = IndexedFileMetadata(
                    file_id="file1",
                    file_path="/path/to/file.pdf",
                    filename="file.pdf",
                    extension=".pdf",
                    file_type="document",
                    tags=["test"],
                    content_summary="Test summary",
                    content_hash="abc123",
                    created_at=None,
                    modified_at=datetime(2025, 1, 15),
                    indexed_at=datetime.now(),
                    size_bytes=1024,
                    source_folder="Documents",
                )
                mock_im_instance.search.return_value = [
                    (metadata, 0.3, "Test document content")
                ]

                engine = SearchEngine()
                results = engine.search("test query", limit=5)

                assert len(results) == 1
                assert results[0].filename == "file.pdf"
                assert results[0].relevance_score > 0.5

    def test_search_with_filters(self):
        """Test search with extension filter."""
        with patch("fileassistant.search.engine.IndexManager") as mock_im:
            with patch("fileassistant.search.engine.EmbeddingGenerator") as mock_eg:
                mock_im_instance = MagicMock()
                mock_im_instance.get_indexed_count.return_value = 10
                mock_im_instance.search.return_value = []
                mock_im.return_value = mock_im_instance

                mock_embed_result = MagicMock()
                mock_embed_result.success = True
                mock_embed_result.embedding = [0.1] * 384
                mock_eg.return_value.generate.return_value = mock_embed_result

                engine = SearchEngine()
                engine.search("test", filters={"extension": ".pdf"})

                # Verify ChromaDB was called with filter
                mock_im_instance.search.assert_called_once()
                call_args = mock_im_instance.search.call_args
                assert call_args.kwargs["where"] == {"extension": ".pdf"}

    def test_is_index_empty(self):
        """Test is_index_empty method."""
        with patch("fileassistant.search.engine.IndexManager") as mock_im:
            with patch("fileassistant.search.engine.EmbeddingGenerator"):
                mock_im.return_value.get_indexed_count.return_value = 0
                engine = SearchEngine()
                assert engine.is_index_empty() is True

                mock_im.return_value.get_indexed_count.return_value = 5
                assert engine.is_index_empty() is False


# Skip ChromaDB-dependent integration tests on Python 3.14+
def check_chromadb_available():
    """Check if ChromaDB can be initialized."""
    if sys.version_info >= (3, 14):
        return False
    try:
        import chromadb
        chromadb.Client()
        return True
    except Exception:
        return False


CHROMADB_AVAILABLE = check_chromadb_available()


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not available")
class TestSearchEngineIntegration:
    """Integration tests requiring ChromaDB."""

    @pytest.fixture
    def populated_engine(self, tmp_path):
        """Create an engine with test data."""
        from fileassistant.embeddings import EmbeddingGenerator
        from fileassistant.search import IndexManager

        index_manager = IndexManager(persist_directory=tmp_path / "chromadb")
        embedding_generator = EmbeddingGenerator()
        engine = SearchEngine(
            index_manager=index_manager,
            embedding_generator=embedding_generator,
        )

        # Index test files
        test_files = [
            ("Machine learning is a subset of artificial intelligence.", "ml_paper.pdf", [".pdf"], ["research", "ai"]),
            ("Quarterly financial report for Q4 2024.", "q4_report.pdf", [".pdf"], ["finance", "report"]),
            ("Python programming tutorial for beginners.", "python_tutorial.txt", [".txt"], ["programming"]),
            ("Meeting notes from team standup.", "meeting_notes.md", [".md"], ["work", "notes"]),
        ]

        for i, (content, filename, tags, _) in enumerate(test_files):
            # Generate embedding
            result = embedding_generator.generate(content)
            if result.success:
                test_file = tmp_path / filename
                test_file.write_text(content)
                index_manager.index_file(
                    file_id=f"file_{i}",
                    file_path=test_file,
                    text=content,
                    embedding=result.embedding,
                    tags=_,
                )

        yield engine
        engine.close()

    @pytest.mark.slow
    def test_semantic_search(self, populated_engine):
        """Test that semantic search finds relevant documents."""
        results = populated_engine.search("AI and machine learning")

        assert len(results) > 0
        # The ML paper should be most relevant
        assert "ml_paper" in results[0].filename or "artificial" in results[0].content_snippet.lower()

    @pytest.mark.slow
    def test_search_with_type_filter(self, populated_engine):
        """Test search with extension filter."""
        results = populated_engine.search("report", filters={"extension": ".pdf"})

        for result in results:
            assert result.extension == ".pdf"

    @pytest.mark.slow
    def test_search_with_tag_filter(self, populated_engine):
        """Test search with tag filter."""
        results = populated_engine.search("notes", filters={"tag": "work"})

        for result in results:
            assert "work" in [t.lower() for t in result.tags]
