"""Tests for the index CLI command."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from fileassistant.cli.index import (
    INDEXABLE_EXTENSIONS,
    collect_files,
    extract_text_for_indexing,
    get_indexable_extensions,
    index_command,
    should_skip_path,
)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_indexable_extensions(self):
        """Test that indexable extensions includes both analyzer and additional types."""
        extensions = get_indexable_extensions()
        # Should include document types
        assert ".pdf" in extensions
        assert ".docx" in extensions
        assert ".txt" in extensions
        assert ".md" in extensions
        # Should include code types
        assert ".py" in extensions
        assert ".js" in extensions
        assert ".ts" in extensions
        # Should include config types
        assert ".json" in extensions
        assert ".yaml" in extensions
        assert ".yml" in extensions

    def test_should_skip_path_hidden_file(self, tmp_path):
        """Test that hidden files are skipped."""
        hidden = tmp_path / ".hidden"
        hidden.touch()
        assert should_skip_path(hidden) is True

    def test_should_skip_path_hidden_folder(self, tmp_path):
        """Test that files in hidden folders are skipped."""
        hidden_dir = tmp_path / ".hidden_dir"
        hidden_dir.mkdir()
        file_in_hidden = hidden_dir / "file.txt"
        file_in_hidden.touch()
        assert should_skip_path(file_in_hidden) is True

    def test_should_skip_path_normal_file(self, tmp_path):
        """Test that normal files are not skipped."""
        normal = tmp_path / "normal.txt"
        normal.touch()
        assert should_skip_path(normal) is False

    def test_should_skip_path_git_folder(self, tmp_path):
        """Test that .git folder contents are skipped."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        git_file = git_dir / "config"
        git_file.touch()
        assert should_skip_path(git_file) is True


class TestCollectFiles:
    """Tests for collect_files function."""

    @pytest.fixture
    def test_directory(self, tmp_path):
        """Create a test directory structure."""
        # Create some test files
        (tmp_path / "doc.txt").write_text("text file")
        (tmp_path / "code.py").write_text("print('hello')")
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")  # Not indexable
        (tmp_path / ".hidden.txt").write_text("hidden")

        # Create subdirectory
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "nested.md").write_text("# Markdown")
        (sub / ".git").mkdir()
        (sub / ".git" / "config").write_text("git config")

        return tmp_path

    def test_collect_files_recursive(self, test_directory):
        """Test recursive file collection."""
        files = collect_files(
            test_directory,
            recursive=True,
            max_size_mb=50,
            extensions=get_indexable_extensions(),
        )

        filenames = [f.name for f in files]
        assert "doc.txt" in filenames
        assert "code.py" in filenames
        assert "config.json" in filenames
        assert "nested.md" in filenames
        # Should skip
        assert "image.png" not in filenames
        assert ".hidden.txt" not in filenames
        assert "config" not in filenames  # .git/config

    def test_collect_files_non_recursive(self, test_directory):
        """Test non-recursive file collection."""
        files = collect_files(
            test_directory,
            recursive=False,
            max_size_mb=50,
            extensions=get_indexable_extensions(),
        )

        filenames = [f.name for f in files]
        assert "doc.txt" in filenames
        assert "code.py" in filenames
        assert "nested.md" not in filenames  # In subdirectory

    def test_collect_files_size_filter(self, tmp_path):
        """Test that large files are filtered out."""
        # Create a file that exceeds max size
        small_file = tmp_path / "small.txt"
        small_file.write_text("small")

        large_file = tmp_path / "large.txt"
        large_file.write_text("x" * (2 * 1024 * 1024))  # 2MB

        files = collect_files(
            tmp_path,
            recursive=True,
            max_size_mb=1,  # 1MB limit
            extensions={".txt"},
        )

        filenames = [f.name for f in files]
        assert "small.txt" in filenames
        assert "large.txt" not in filenames


class TestExtractTextForIndexing:
    """Tests for extract_text_for_indexing function."""

    @pytest.fixture
    def analyzer(self):
        """Create a mock analyzer."""
        from fileassistant.analyzer import FileAnalyzer
        return FileAnalyzer()

    def test_extract_text_txt_file(self, tmp_path, analyzer):
        """Test extracting text from a .txt file."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Hello, world!")

        text, error = extract_text_for_indexing(file_path, analyzer)

        assert error is None
        assert text == "Hello, world!"

    def test_extract_text_python_file(self, tmp_path, analyzer):
        """Test extracting text from a Python file."""
        file_path = tmp_path / "test.py"
        file_path.write_text('print("Hello, Python!")')

        text, error = extract_text_for_indexing(file_path, analyzer)

        assert error is None
        assert 'print("Hello, Python!")' in text

    def test_extract_text_json_file(self, tmp_path, analyzer):
        """Test extracting text from a JSON file."""
        file_path = tmp_path / "test.json"
        file_path.write_text('{"key": "value"}')

        text, error = extract_text_for_indexing(file_path, analyzer)

        assert error is None
        assert '"key": "value"' in text

    def test_extract_text_unsupported_file(self, tmp_path, analyzer):
        """Test extracting text from an unsupported file."""
        file_path = tmp_path / "test.xyz"
        file_path.write_text("some content")

        text, error = extract_text_for_indexing(file_path, analyzer)

        assert text is None
        assert "Unsupported" in error


# Skip CLI tests on Python 3.14+ due to ChromaDB issues
CHROMADB_AVAILABLE = sys.version_info < (3, 14)


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not available on Python 3.14+")
class TestIndexCommand:
    """Tests for the index CLI command."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def test_files(self, tmp_path):
        """Create test files for indexing."""
        (tmp_path / "doc1.txt").write_text("This is document one about machine learning.")
        (tmp_path / "doc2.txt").write_text("This is document two about data science.")
        (tmp_path / "code.py").write_text("def hello(): print('Hello, World!')")

        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "nested.md").write_text("# Nested Markdown\n\nSome content here.")

        return tmp_path

    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create a mock config."""
        config = MagicMock()
        config.database.path = tmp_path / "db.sqlite"
        config.database.vector_store_path = tmp_path / "chromadb"
        config.ai_settings.embedding_model = "all-MiniLM-L6-v2"
        config.search.chunk_size = 512
        config.search.chunk_overlap = 50
        return config

    def test_index_dry_run(self, runner, test_files):
        """Test index command with dry-run flag."""
        with patch("fileassistant.cli.index.get_config_manager") as mock_cm:
            mock_config_manager = MagicMock()
            mock_cm.return_value = mock_config_manager
            mock_config_manager.load.return_value = MagicMock()

            result = runner.invoke(
                index_command,
                [str(test_files), "--dry-run"],
                obj={},
            )

            assert result.exit_code == 0
            assert "Dry run" in result.output or "DRY RUN" in result.output
            assert "would be indexed" in result.output

    def test_index_non_recursive(self, runner, test_files):
        """Test index command without recursion."""
        with patch("fileassistant.cli.index.get_config_manager") as mock_cm:
            mock_config_manager = MagicMock()
            mock_cm.return_value = mock_config_manager
            mock_config_manager.load.return_value = MagicMock()

            result = runner.invoke(
                index_command,
                [str(test_files), "--no-recursive", "--dry-run"],
                obj={},
            )

            assert result.exit_code == 0
            # Should not include nested.md
            assert "nested.md" not in result.output

    def test_index_no_files_found(self, runner, tmp_path):
        """Test index command with empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with patch("fileassistant.cli.index.get_config_manager") as mock_cm:
            mock_config_manager = MagicMock()
            mock_cm.return_value = mock_config_manager
            mock_config_manager.load.return_value = MagicMock()

            result = runner.invoke(
                index_command,
                [str(empty_dir)],
                obj={},
            )

            assert result.exit_code == 0
            assert "No supported files found" in result.output


class TestIndexCommandUnit:
    """Unit tests that don't require ChromaDB."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_index_missing_config(self, runner, tmp_path):
        """Test index command when config is missing."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        with patch("fileassistant.cli.index.get_config_manager") as mock_cm:
            mock_cm.return_value.load.side_effect = FileNotFoundError("No config")

            result = runner.invoke(
                index_command,
                [str(tmp_path)],
                obj={},
            )

            assert result.exit_code == 1
            assert "No configuration found" in result.output

    def test_index_help(self, runner):
        """Test index command help."""
        result = runner.invoke(index_command, ["--help"])

        assert result.exit_code == 0
        assert "Index files for search" in result.output
        assert "--recursive" in result.output
        assert "--force" in result.output
        assert "--dry-run" in result.output
