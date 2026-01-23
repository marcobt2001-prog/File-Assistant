"""Tests for the processor component."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fileassistant.analyzer import AnalysisResult, FileMetadata
from fileassistant.classifier import ClassificationResult
from fileassistant.config.models import FileAssistantConfig
from fileassistant.core import FileProcessor, ProcessingResult, UserDecision
from fileassistant.mover import MoveResult


class TestUserDecision:
    """Tests for UserDecision enum."""

    def test_values(self):
        """Test UserDecision enum values."""
        assert UserDecision.ACCEPT.value == "accept"
        assert UserDecision.EDIT.value == "edit"
        assert UserDecision.SKIP.value == "skip"


class TestProcessingResult:
    """Tests for ProcessingResult."""

    def test_final_destination_from_classification(self):
        """Test final_destination returns classification destination."""
        result = ProcessingResult(
            file_path=Path("/test.txt"),
            filename="test.txt",
            classification=ClassificationResult(
                file_path=Path("/test.txt"),
                filename="test.txt",
                destination_folder="Documents/Work",
            ),
        )
        assert result.final_destination == "Documents/Work"

    def test_final_destination_prefers_edited(self):
        """Test final_destination prefers edited destination."""
        result = ProcessingResult(
            file_path=Path("/test.txt"),
            filename="test.txt",
            classification=ClassificationResult(
                file_path=Path("/test.txt"),
                filename="test.txt",
                destination_folder="Documents/Work",
            ),
            edited_destination="Projects/Personal",
        )
        assert result.final_destination == "Projects/Personal"

    def test_final_destination_none_without_classification(self):
        """Test final_destination is None without classification."""
        result = ProcessingResult(
            file_path=Path("/test.txt"),
            filename="test.txt",
        )
        assert result.final_destination is None


class TestFileProcessor:
    """Tests for FileProcessor."""

    @pytest.fixture
    def config(self, tmp_path):
        """Create a test configuration."""
        return FileAssistantConfig(
            inbox_folders=[tmp_path / "inbox"],
            organized_base_path=tmp_path / "organized",
        )

    @pytest.fixture
    def processor(self, config):
        """Create a FileProcessor instance."""
        return FileProcessor(config=config)

    @pytest.fixture
    def test_file(self, tmp_path):
        """Create a test file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, this is a test document.")
        return test_file

    def test_initialization(self, processor, config):
        """Test processor initialization."""
        assert processor.config == config
        assert processor.analyzer is not None
        assert processor.classifier is not None
        assert processor.mover is not None

    @patch("fileassistant.classifier.OllamaClient._check_connection")
    @patch("fileassistant.classifier.OllamaClient._check_model_available")
    def test_check_system_ready_success(
        self, mock_model, mock_conn, processor, tmp_path
    ):
        """Test system readiness check when everything is ready."""
        mock_conn.return_value = True
        mock_model.return_value = True

        # Ensure organized path is writable
        processor.config.organized_base_path.mkdir(parents=True, exist_ok=True)

        is_ready, issues = processor.check_system_ready()

        assert is_ready
        assert len(issues) == 0

    @patch("fileassistant.classifier.OllamaClient._check_connection")
    def test_check_system_ready_ollama_down(self, mock_conn, processor):
        """Test system readiness when Ollama is not available."""
        mock_conn.return_value = False

        is_ready, issues = processor.check_system_ready()

        assert not is_ready
        assert any("Ollama" in issue for issue in issues)

    @patch("fileassistant.core.processor.FileAnalyzer.analyze")
    def test_process_file_analysis_failure(self, mock_analyze, processor, test_file):
        """Test processing when analysis fails."""
        mock_analyze.return_value = AnalysisResult(
            file_path=test_file,
            metadata=None,  # type: ignore
            content="",
            content_preview="",
            success=False,
            error_message="Could not read file",
        )

        result = processor.process_file(test_file, interactive=False)

        assert not result.success
        assert "Analysis failed" in result.error_message

    @patch("fileassistant.core.processor.FileClassifier.classify")
    @patch("fileassistant.core.processor.FileAnalyzer.analyze")
    def test_process_file_classification_failure(
        self, mock_analyze, mock_classify, processor, test_file
    ):
        """Test processing when classification fails."""
        from datetime import datetime

        mock_analyze.return_value = AnalysisResult(
            file_path=test_file,
            metadata=FileMetadata(
                path=test_file,
                filename="test.txt",
                extension=".txt",
                size_bytes=100,
                created_at=datetime.now(),
                modified_at=datetime.now(),
                hash_md5="abc123",
            ),
            content="Test content",
            content_preview="Test content",
            success=True,
        )

        mock_classify.return_value = ClassificationResult(
            file_path=test_file,
            filename="test.txt",
            destination_folder="Unsorted",
            success=False,
            error_message="Ollama not available",
        )

        result = processor.process_file(test_file, interactive=False)

        assert not result.success
        assert "Classification failed" in result.error_message

    @patch("fileassistant.core.processor.FileMover.move")
    @patch("fileassistant.core.processor.FileClassifier.classify")
    @patch("fileassistant.core.processor.FileAnalyzer.analyze")
    def test_process_file_success_non_interactive(
        self, mock_analyze, mock_classify, mock_move, processor, test_file, tmp_path
    ):
        """Test successful file processing in non-interactive mode."""
        from datetime import datetime

        dest_path = tmp_path / "organized" / "Documents" / "test.txt"

        mock_analyze.return_value = AnalysisResult(
            file_path=test_file,
            metadata=FileMetadata(
                path=test_file,
                filename="test.txt",
                extension=".txt",
                size_bytes=100,
                created_at=datetime.now(),
                modified_at=datetime.now(),
                hash_md5="abc123",
            ),
            content="Test content",
            content_preview="Test content",
            success=True,
            word_count=2,
        )

        mock_classify.return_value = ClassificationResult(
            file_path=test_file,
            filename="test.txt",
            destination_folder="Documents",
            tags=["document"],
            confidence=0.85,
            reasoning="Looks like a document",
            success=True,
        )

        mock_move.return_value = MoveResult(
            source_path=test_file,
            destination_path=dest_path,
            filename="test.txt",
            success=True,
        )

        result = processor.process_file(test_file, interactive=False)

        assert result.success
        assert result.analysis is not None
        assert result.classification is not None
        assert result.move_result is not None

    @patch("fileassistant.core.processor.FileMover.move")
    @patch("fileassistant.core.processor.FileClassifier.classify")
    @patch("fileassistant.core.processor.FileAnalyzer.analyze")
    def test_process_file_move_failure(
        self, mock_analyze, mock_classify, mock_move, processor, test_file
    ):
        """Test processing when move fails."""
        from datetime import datetime

        mock_analyze.return_value = AnalysisResult(
            file_path=test_file,
            metadata=FileMetadata(
                path=test_file,
                filename="test.txt",
                extension=".txt",
                size_bytes=100,
                created_at=datetime.now(),
                modified_at=datetime.now(),
                hash_md5="abc123",
            ),
            content="Test content",
            content_preview="Test content",
            success=True,
        )

        mock_classify.return_value = ClassificationResult(
            file_path=test_file,
            filename="test.txt",
            destination_folder="Documents",
            success=True,
        )

        mock_move.return_value = MoveResult(
            source_path=test_file,
            destination_path=test_file,
            filename="test.txt",
            success=False,
            error_message="Permission denied",
        )

        result = processor.process_file(test_file, interactive=False)

        assert not result.success
        assert "Move failed" in result.error_message
