"""Tests for the classifier component."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fileassistant.analyzer import AnalysisResult, FileMetadata
from fileassistant.classifier import ClassificationResult, FileClassifier, OllamaClient
from fileassistant.config.models import AISettings


class TestOllamaClient:
    """Tests for OllamaClient."""

    def test_initialization(self):
        """Test client initialization with default values."""
        client = OllamaClient()
        assert client.base_url == "http://localhost:11434"
        assert client.model_name == "qwen2.5:latest"
        assert client.temperature == 0.1
        assert client.max_retries == 3

    def test_custom_initialization(self):
        """Test client initialization with custom values."""
        client = OllamaClient(
            base_url="http://custom:1234",
            model_name="llama3",
            temperature=0.5,
            max_retries=5,
        )
        assert client.base_url == "http://custom:1234"
        assert client.model_name == "llama3"
        assert client.temperature == 0.5
        assert client.max_retries == 5

    def test_base_url_strips_trailing_slash(self):
        """Test that trailing slash is stripped from base URL."""
        client = OllamaClient(base_url="http://localhost:11434/")
        assert client.base_url == "http://localhost:11434"

    @patch("httpx.Client")
    def test_check_connection_success(self, mock_client_class):
        """Test successful connection check."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        client = OllamaClient()
        assert client._check_connection() is True

    @patch("httpx.Client")
    def test_check_connection_failure(self, mock_client_class):
        """Test failed connection check."""
        mock_client_class.return_value.__enter__.side_effect = Exception("Connection refused")

        client = OllamaClient()
        assert client._check_connection() is False


class TestClassificationResult:
    """Tests for ClassificationResult."""

    def test_confidence_level_high(self):
        """Test high confidence level."""
        result = ClassificationResult(
            file_path=Path("/test.txt"),
            filename="test.txt",
            destination_folder="Documents",
            confidence=0.95,
        )
        assert result.confidence_level == "high"

    def test_confidence_level_medium(self):
        """Test medium confidence level."""
        result = ClassificationResult(
            file_path=Path("/test.txt"),
            filename="test.txt",
            destination_folder="Documents",
            confidence=0.75,
        )
        assert result.confidence_level == "medium"

    def test_confidence_level_low(self):
        """Test low confidence level."""
        result = ClassificationResult(
            file_path=Path("/test.txt"),
            filename="test.txt",
            destination_folder="Documents",
            confidence=0.3,
        )
        assert result.confidence_level == "low"


class TestFileClassifier:
    """Tests for FileClassifier."""

    @pytest.fixture
    def classifier(self):
        """Create a FileClassifier instance."""
        return FileClassifier()

    @pytest.fixture
    def mock_analysis(self, tmp_path):
        """Create a mock AnalysisResult."""
        from datetime import datetime

        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        return AnalysisResult(
            file_path=test_file,
            metadata=FileMetadata(
                path=test_file,
                filename="test.txt",
                extension=".txt",
                size_bytes=12,
                created_at=datetime.now(),
                modified_at=datetime.now(),
                hash_md5="abc123",
            ),
            content="Test content",
            content_preview="Test content",
            success=True,
        )

    def test_initialization_with_defaults(self, classifier):
        """Test classifier initializes with default settings."""
        assert classifier.ai_settings.model_name == "qwen2.5:latest"
        assert classifier.ollama is not None

    def test_initialization_with_custom_settings(self):
        """Test classifier initializes with custom settings."""
        settings = AISettings(model_name="llama3", temperature=0.5)
        classifier = FileClassifier(ai_settings=settings)
        assert classifier.ai_settings.model_name == "llama3"

    def test_build_prompt(self, classifier, mock_analysis):
        """Test prompt building."""
        prompt = classifier._build_prompt(mock_analysis)

        assert "test.txt" in prompt
        assert ".txt" in prompt
        assert "Test content" in prompt

    def test_parse_valid_response(self, classifier, tmp_path):
        """Test parsing a valid LLM response."""
        response = json.dumps({
            "destination_folder": "Documents/Work",
            "tags": ["document", "work"],
            "confidence": 0.85,
            "reasoning": "This is a work document",
        })

        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        result = classifier._parse_response(response, test_file)

        assert result.success
        assert result.destination_folder == "Documents/Work"
        assert result.tags == ["document", "work"]
        assert result.confidence == 0.85
        assert result.reasoning == "This is a work document"

    def test_parse_response_with_extra_text(self, classifier, tmp_path):
        """Test parsing response that has extra text around JSON."""
        response = """Here is the classification:
        {"destination_folder": "Projects", "tags": ["code"], "confidence": 0.9, "reasoning": "Code file"}
        Hope this helps!"""

        test_file = tmp_path / "test.py"
        test_file.write_text("test")

        result = classifier._parse_response(response, test_file)

        assert result.success
        assert result.destination_folder == "Projects"

    def test_parse_invalid_response(self, classifier, tmp_path):
        """Test parsing an invalid LLM response."""
        response = "I don't know how to classify this file."

        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        result = classifier._parse_response(response, test_file)

        assert not result.success
        assert "Unsorted" in result.destination_folder
        assert "parse" in result.error_message.lower()

    def test_parse_response_clamps_confidence(self, classifier, tmp_path):
        """Test that confidence is clamped to [0, 1] range."""
        response = json.dumps({
            "destination_folder": "Documents",
            "tags": [],
            "confidence": 1.5,  # Invalid, should be clamped
            "reasoning": "Test",
        })

        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        result = classifier._parse_response(response, test_file)

        assert result.confidence == 1.0

    def test_parse_response_sanitizes_destination(self, classifier, tmp_path):
        """Test that destination folder is sanitized."""
        response = json.dumps({
            "destination_folder": "/Documents\\Work/",
            "tags": [],
            "confidence": 0.8,
            "reasoning": "Test",
        })

        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        result = classifier._parse_response(response, test_file)

        assert result.destination_folder == "Documents/Work"

    def test_classify_failed_analysis(self, classifier, tmp_path):
        """Test classifying a file with failed analysis."""
        failed_analysis = AnalysisResult(
            file_path=tmp_path / "test.txt",
            metadata=None,  # type: ignore
            content="",
            content_preview="",
            success=False,
            error_message="File not found",
        )

        result = classifier.classify(failed_analysis)

        assert not result.success
        assert "Analysis failed" in result.error_message

    @patch.object(OllamaClient, "generate")
    def test_classify_ollama_failure(self, mock_generate, classifier, mock_analysis):
        """Test classification when Ollama fails."""
        mock_generate.return_value = None

        result = classifier.classify(mock_analysis)

        assert not result.success
        assert "Failed to get response" in result.error_message

    @patch.object(OllamaClient, "generate")
    def test_classify_success(self, mock_generate, classifier, mock_analysis):
        """Test successful classification."""
        mock_generate.return_value = json.dumps({
            "destination_folder": "Documents/Notes",
            "tags": ["note", "text"],
            "confidence": 0.9,
            "reasoning": "This appears to be a text note",
        })

        result = classifier.classify(mock_analysis)

        assert result.success
        assert result.destination_folder == "Documents/Notes"
        assert "note" in result.tags
        assert result.confidence == 0.9
