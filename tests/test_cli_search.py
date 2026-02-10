"""Tests for the search CLI command."""

import json
import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from fileassistant.cli.search import (
    format_file_size,
    get_relevance_color,
    parse_date,
    parse_extensions,
    search_command,
)
from fileassistant.search import SearchResult


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_parse_extensions_single(self):
        """Test parsing single extension."""
        result = parse_extensions("pdf")
        assert result == [".pdf"]

    def test_parse_extensions_with_dot(self):
        """Test parsing extension with dot prefix."""
        result = parse_extensions(".pdf")
        assert result == [".pdf"]

    def test_parse_extensions_multiple(self):
        """Test parsing multiple extensions."""
        result = parse_extensions("pdf,docx,txt")
        assert result == [".pdf", ".docx", ".txt"]

    def test_parse_extensions_with_spaces(self):
        """Test parsing extensions with spaces."""
        result = parse_extensions("pdf, docx , txt")
        assert result == [".pdf", ".docx", ".txt"]

    def test_parse_extensions_empty(self):
        """Test parsing empty string."""
        result = parse_extensions("")
        assert result == []

    def test_parse_extensions_mixed_format(self):
        """Test parsing mixed formats."""
        result = parse_extensions(".pdf,docx,.TXT")
        assert result == [".pdf", ".docx", ".txt"]

    def test_parse_date_valid(self):
        """Test parsing valid date."""
        result = parse_date("2025-01-15")
        assert result == datetime(2025, 1, 15)

    def test_parse_date_empty(self):
        """Test parsing empty date."""
        result = parse_date("")
        assert result is None

        result = parse_date(None)
        assert result is None

    def test_parse_date_invalid(self):
        """Test parsing invalid date raises error."""
        with pytest.raises(Exception):  # click.BadParameter
            parse_date("not-a-date")

    def test_format_file_size_bytes(self):
        """Test formatting small file sizes."""
        assert format_file_size(512) == "512 B"

    def test_format_file_size_kb(self):
        """Test formatting KB sizes."""
        assert format_file_size(1536) == "1.5 KB"

    def test_format_file_size_mb(self):
        """Test formatting MB sizes."""
        assert format_file_size(1.5 * 1024 * 1024) == "1.5 MB"

    def test_format_file_size_gb(self):
        """Test formatting GB sizes."""
        assert format_file_size(2.5 * 1024 * 1024 * 1024) == "2.5 GB"

    def test_get_relevance_color_high(self):
        """Test color for high relevance."""
        assert get_relevance_color(0.9) == "green"
        assert get_relevance_color(0.8) == "green"

    def test_get_relevance_color_medium(self):
        """Test color for medium relevance."""
        assert get_relevance_color(0.7) == "yellow"
        assert get_relevance_color(0.5) == "yellow"

    def test_get_relevance_color_low(self):
        """Test color for low relevance."""
        assert get_relevance_color(0.4) == "red"
        assert get_relevance_color(0.1) == "red"


class TestSearchCommandUnit:
    """Unit tests for search command (no ChromaDB required)."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_search_help(self, runner):
        """Test search command help."""
        result = runner.invoke(search_command, ["--help"])

        assert result.exit_code == 0
        assert "Search indexed files" in result.output
        assert "--type" in result.output
        assert "--after" in result.output
        assert "--before" in result.output
        assert "--tag" in result.output
        assert "--limit" in result.output
        assert "--json" in result.output

    def test_search_empty_query(self, runner):
        """Test search with no query fails."""
        result = runner.invoke(search_command, [])

        assert result.exit_code != 0

    def test_search_query_too_short(self, runner):
        """Test search with very short query."""
        result = runner.invoke(search_command, ["a"])

        assert result.exit_code == 1
        assert "at least 2 characters" in result.output

    def test_search_empty_index(self, runner):
        """Test search on empty index shows helpful message."""
        with patch("fileassistant.cli.search.get_config_manager") as mock_cm:
            with patch("fileassistant.cli.search.SearchEngine") as mock_engine:
                mock_cm.return_value.load.side_effect = FileNotFoundError()

                mock_engine_instance = MagicMock()
                mock_engine_instance.is_index_empty.return_value = True
                mock_engine.return_value = mock_engine_instance

                result = runner.invoke(search_command, ["test query"])

                assert result.exit_code == 0
                assert "No files have been indexed" in result.output
                assert "fileassistant index" in result.output

    def test_search_no_results(self, runner):
        """Test search with no matching results."""
        with patch("fileassistant.cli.search.get_config_manager") as mock_cm:
            with patch("fileassistant.cli.search.SearchEngine") as mock_engine:
                mock_cm.return_value.load.side_effect = FileNotFoundError()

                mock_engine_instance = MagicMock()
                mock_engine_instance.is_index_empty.return_value = False
                mock_engine_instance.search.return_value = []
                mock_engine.return_value = mock_engine_instance

                result = runner.invoke(search_command, ["obscure query"])

                assert result.exit_code == 0
                assert "No matching files found" in result.output

    def test_search_with_results(self, runner):
        """Test search with results."""
        with patch("fileassistant.cli.search.get_config_manager") as mock_cm:
            with patch("fileassistant.cli.search.SearchEngine") as mock_engine:
                mock_cm.return_value.load.side_effect = FileNotFoundError()

                # Create mock search result
                mock_result = SearchResult(
                    file_path="/path/to/document.pdf",
                    filename="document.pdf",
                    relevance_score=0.85,
                    content_snippet="This is a test document about machine learning...",
                    tags=["research", "ml"],
                    file_type="document",
                    modified_at=datetime(2025, 1, 15),
                    size_bytes=1024,
                    extension=".pdf",
                )

                mock_engine_instance = MagicMock()
                mock_engine_instance.is_index_empty.return_value = False
                mock_engine_instance.search.return_value = [mock_result]
                mock_engine.return_value = mock_engine_instance

                result = runner.invoke(search_command, ["machine learning"])

                assert result.exit_code == 0
                assert "document.pdf" in result.output
                assert "1 result" in result.output

    def test_search_json_output(self, runner):
        """Test search with JSON output."""
        with patch("fileassistant.cli.search.get_config_manager") as mock_cm:
            with patch("fileassistant.cli.search.SearchEngine") as mock_engine:
                mock_cm.return_value.load.side_effect = FileNotFoundError()

                mock_result = SearchResult(
                    file_path="/path/to/doc.pdf",
                    filename="doc.pdf",
                    relevance_score=0.75,
                    content_snippet="Test content",
                    tags=["test"],
                    file_type="document",
                    modified_at=datetime(2025, 1, 15),
                    size_bytes=512,
                    extension=".pdf",
                )

                mock_engine_instance = MagicMock()
                mock_engine_instance.is_index_empty.return_value = False
                mock_engine_instance.search.return_value = [mock_result]
                mock_engine.return_value = mock_engine_instance

                result = runner.invoke(search_command, ["test", "--json"])

                assert result.exit_code == 0
                # Parse JSON output
                output = json.loads(result.output)
                assert len(output) == 1
                assert output[0]["filename"] == "doc.pdf"
                assert output[0]["relevance_score"] == 0.75

    def test_search_compact_output(self, runner):
        """Test search with compact table output."""
        with patch("fileassistant.cli.search.get_config_manager") as mock_cm:
            with patch("fileassistant.cli.search.SearchEngine") as mock_engine:
                mock_cm.return_value.load.side_effect = FileNotFoundError()

                mock_result = SearchResult(
                    file_path="/path/to/doc.pdf",
                    filename="doc.pdf",
                    relevance_score=0.75,
                    content_snippet="Test content",
                    tags=["test"],
                    file_type="document",
                    modified_at=datetime(2025, 1, 15),
                    size_bytes=512,
                    extension=".pdf",
                )

                mock_engine_instance = MagicMock()
                mock_engine_instance.is_index_empty.return_value = False
                mock_engine_instance.search.return_value = [mock_result]
                mock_engine.return_value = mock_engine_instance

                result = runner.invoke(search_command, ["test", "--compact"])

                assert result.exit_code == 0
                assert "doc.pdf" in result.output

    def test_search_with_type_filter(self, runner):
        """Test search with type filter."""
        with patch("fileassistant.cli.search.get_config_manager") as mock_cm:
            with patch("fileassistant.cli.search.SearchEngine") as mock_engine:
                mock_cm.return_value.load.side_effect = FileNotFoundError()

                mock_engine_instance = MagicMock()
                mock_engine_instance.is_index_empty.return_value = False
                mock_engine_instance.search.return_value = []
                mock_engine.return_value = mock_engine_instance

                result = runner.invoke(search_command, ["test", "--type", "pdf"])

                assert result.exit_code == 0
                # Verify the filter was passed
                mock_engine_instance.search.assert_called_once()
                call_args = mock_engine_instance.search.call_args
                filters = call_args.kwargs.get("filters", {})
                assert "extension" in filters

    def test_search_with_date_filter(self, runner):
        """Test search with date filter."""
        with patch("fileassistant.cli.search.get_config_manager") as mock_cm:
            with patch("fileassistant.cli.search.SearchEngine") as mock_engine:
                mock_cm.return_value.load.side_effect = FileNotFoundError()

                mock_engine_instance = MagicMock()
                mock_engine_instance.is_index_empty.return_value = False
                mock_engine_instance.search.return_value = []
                mock_engine.return_value = mock_engine_instance

                result = runner.invoke(search_command, ["test", "--after", "2024-01-01"])

                assert result.exit_code == 0
                call_args = mock_engine_instance.search.call_args
                filters = call_args.kwargs.get("filters", {})
                assert "after" in filters

    def test_search_with_invalid_date(self, runner):
        """Test search with invalid date format."""
        result = runner.invoke(search_command, ["test", "--after", "not-a-date"])

        assert result.exit_code == 1
        assert "Invalid date format" in result.output

    def test_search_with_tag_filter(self, runner):
        """Test search with tag filter."""
        with patch("fileassistant.cli.search.get_config_manager") as mock_cm:
            with patch("fileassistant.cli.search.SearchEngine") as mock_engine:
                mock_cm.return_value.load.side_effect = FileNotFoundError()

                mock_engine_instance = MagicMock()
                mock_engine_instance.is_index_empty.return_value = False
                mock_engine_instance.search.return_value = []
                mock_engine.return_value = mock_engine_instance

                result = runner.invoke(search_command, ["test", "--tag", "work"])

                assert result.exit_code == 0
                call_args = mock_engine_instance.search.call_args
                filters = call_args.kwargs.get("filters", {})
                assert filters.get("tag") == "work"

    def test_search_with_limit(self, runner):
        """Test search with limit option."""
        with patch("fileassistant.cli.search.get_config_manager") as mock_cm:
            with patch("fileassistant.cli.search.SearchEngine") as mock_engine:
                mock_cm.return_value.load.side_effect = FileNotFoundError()

                mock_engine_instance = MagicMock()
                mock_engine_instance.is_index_empty.return_value = False
                mock_engine_instance.search.return_value = []
                mock_engine.return_value = mock_engine_instance

                result = runner.invoke(search_command, ["test", "--limit", "5"])

                assert result.exit_code == 0
                call_args = mock_engine_instance.search.call_args
                assert call_args.kwargs.get("limit") == 5

    def test_search_multi_word_query(self, runner):
        """Test search with multi-word query."""
        with patch("fileassistant.cli.search.get_config_manager") as mock_cm:
            with patch("fileassistant.cli.search.SearchEngine") as mock_engine:
                mock_cm.return_value.load.side_effect = FileNotFoundError()

                mock_engine_instance = MagicMock()
                mock_engine_instance.is_index_empty.return_value = False
                mock_engine_instance.search.return_value = []
                mock_engine.return_value = mock_engine_instance

                result = runner.invoke(search_command, ["machine", "learning", "papers"])

                assert result.exit_code == 0
                call_args = mock_engine_instance.search.call_args
                # Query should be joined
                assert call_args.args[0] == "machine learning papers"


# Skip ChromaDB-dependent tests on Python 3.14+
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
class TestSearchCommandIntegration:
    """Integration tests requiring ChromaDB."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def indexed_files(self, tmp_path):
        """Create and index some test files."""
        from fileassistant.embeddings import EmbeddingGenerator
        from fileassistant.search import IndexManager

        index_dir = tmp_path / "chromadb"
        manager = IndexManager(persist_directory=index_dir)
        generator = EmbeddingGenerator()

        # Create test files
        files = [
            ("doc1.pdf", "Machine learning and AI research paper"),
            ("doc2.txt", "Meeting notes from standup"),
            ("code.py", "Python code for data processing"),
        ]

        for filename, content in files:
            file_path = tmp_path / filename
            file_path.write_text(content)

            result = generator.generate(content)
            if result.success:
                manager.index_file(
                    file_id=f"file_{filename}",
                    file_path=file_path,
                    text=content,
                    embedding=result.embedding,
                    tags=["test"],
                    modified_at=datetime.now(),
                )

        manager.close()
        return tmp_path, index_dir

    @pytest.mark.slow
    def test_end_to_end_search(self, runner, indexed_files):
        """Test complete search flow with real index."""
        tmp_path, index_dir = indexed_files

        with patch("fileassistant.cli.search.get_config_manager") as mock_cm:
            # Setup config to point to test index
            mock_config = MagicMock()
            mock_config.database.vector_store_path = index_dir
            mock_cm.return_value.load.return_value = mock_config

            result = runner.invoke(search_command, ["machine learning"])

            assert result.exit_code == 0
            # Should find the ML document
            assert "doc1.pdf" in result.output or "result" in result.output.lower()
