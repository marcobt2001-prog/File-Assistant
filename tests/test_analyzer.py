"""Tests for the analyzer component."""

import tempfile
from pathlib import Path

import pytest

from fileassistant.analyzer import (
    AnalysisResult,
    ExtractionError,
    FileAnalyzer,
    PlainTextExtractor,
    get_extractor,
    get_supported_extensions,
)


class TestPlainTextExtractor:
    """Tests for PlainTextExtractor."""

    def test_supported_extensions(self):
        """Test that extractor reports correct extensions."""
        extractor = PlainTextExtractor()
        assert ".txt" in extractor.supported_extensions
        assert ".md" in extractor.supported_extensions

    def test_can_handle_txt(self):
        """Test that extractor can handle .txt files."""
        extractor = PlainTextExtractor()
        assert extractor.can_handle(Path("test.txt"))
        assert extractor.can_handle(Path("TEST.TXT"))
        assert not extractor.can_handle(Path("test.pdf"))

    def test_extract_txt_file(self, tmp_path):
        """Test extracting text from a .txt file."""
        test_file = tmp_path / "test.txt"
        test_content = "Hello, World!\nThis is a test."
        test_file.write_text(test_content)

        extractor = PlainTextExtractor()
        result = extractor.extract(test_file)
        assert result == test_content

    def test_extract_md_file(self, tmp_path):
        """Test extracting text from a .md file."""
        test_file = tmp_path / "test.md"
        test_content = "# Heading\n\nParagraph with **bold** text."
        test_file.write_text(test_content)

        extractor = PlainTextExtractor()
        result = extractor.extract(test_file)
        assert result == test_content

    def test_extract_nonexistent_file(self, tmp_path):
        """Test that extracting from nonexistent file raises error."""
        extractor = PlainTextExtractor()
        with pytest.raises(ExtractionError):
            extractor.extract(tmp_path / "nonexistent.txt")


class TestGetExtractor:
    """Tests for get_extractor function."""

    def test_get_txt_extractor(self):
        """Test getting extractor for .txt files."""
        extractor = get_extractor(Path("test.txt"))
        assert extractor is not None
        assert isinstance(extractor, PlainTextExtractor)

    def test_get_md_extractor(self):
        """Test getting extractor for .md files."""
        extractor = get_extractor(Path("test.md"))
        assert extractor is not None
        assert isinstance(extractor, PlainTextExtractor)

    def test_get_pdf_extractor(self):
        """Test getting extractor for .pdf files."""
        extractor = get_extractor(Path("test.pdf"))
        assert extractor is not None
        assert ".pdf" in extractor.supported_extensions

    def test_get_docx_extractor(self):
        """Test getting extractor for .docx files."""
        extractor = get_extractor(Path("test.docx"))
        assert extractor is not None
        assert ".docx" in extractor.supported_extensions

    def test_get_unsupported_extractor(self):
        """Test that unsupported extensions return None."""
        assert get_extractor(Path("test.xyz")) is None
        assert get_extractor(Path("test.exe")) is None


class TestGetSupportedExtensions:
    """Tests for get_supported_extensions function."""

    def test_includes_basic_types(self):
        """Test that basic file types are supported."""
        extensions = get_supported_extensions()
        assert ".txt" in extensions
        assert ".md" in extensions
        assert ".pdf" in extensions
        assert ".docx" in extensions


class TestFileAnalyzer:
    """Tests for FileAnalyzer class."""

    def test_can_analyze_txt(self, tmp_path):
        """Test can_analyze returns True for supported files."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        analyzer = FileAnalyzer()
        assert analyzer.can_analyze(test_file)

    def test_cannot_analyze_unsupported(self, tmp_path):
        """Test can_analyze returns False for unsupported files."""
        test_file = tmp_path / "test.xyz"
        test_file.write_text("test content")

        analyzer = FileAnalyzer()
        assert not analyzer.can_analyze(test_file)

    def test_cannot_analyze_nonexistent(self, tmp_path):
        """Test can_analyze returns False for nonexistent files."""
        analyzer = FileAnalyzer()
        assert not analyzer.can_analyze(tmp_path / "nonexistent.txt")

    def test_analyze_txt_file(self, tmp_path):
        """Test analyzing a text file."""
        test_file = tmp_path / "test.txt"
        test_content = "Hello, World!\nThis is a test file."
        test_file.write_text(test_content)

        analyzer = FileAnalyzer()
        result = analyzer.analyze(test_file)

        assert result.success
        assert result.content == test_content
        assert result.metadata.filename == "test.txt"
        assert result.metadata.extension == ".txt"
        assert result.word_count == 6
        assert result.has_content

    def test_analyze_empty_file(self, tmp_path):
        """Test analyzing an empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        analyzer = FileAnalyzer()
        result = analyzer.analyze(test_file)

        assert result.success
        assert result.content == ""
        assert not result.has_content

    def test_analyze_nonexistent_file(self, tmp_path):
        """Test analyzing a nonexistent file."""
        analyzer = FileAnalyzer()
        result = analyzer.analyze(tmp_path / "nonexistent.txt")

        assert not result.success
        assert "not found" in result.error_message.lower()

    def test_analyze_file_too_large(self, tmp_path):
        """Test that files exceeding size limit are rejected."""
        test_file = tmp_path / "large.txt"
        # Create a file larger than 1KB for testing (set small limit)
        test_file.write_text("x" * 2048)

        analyzer = FileAnalyzer(max_file_size_mb=0.001)  # ~1KB limit
        result = analyzer.analyze(test_file)

        assert not result.success
        assert "too large" in result.error_message.lower()

    def test_analyze_metadata_extraction(self, tmp_path):
        """Test that metadata is correctly extracted."""
        test_file = tmp_path / "metadata_test.txt"
        test_file.write_text("test content for metadata")

        analyzer = FileAnalyzer()
        result = analyzer.analyze(test_file)

        assert result.success
        assert result.metadata.filename == "metadata_test.txt"
        assert result.metadata.extension == ".txt"
        assert result.metadata.size_bytes > 0
        assert result.metadata.hash_md5  # MD5 should be computed
        assert len(result.metadata.hash_md5) == 32  # MD5 hex length

    def test_analyze_multiple(self, tmp_path):
        """Test analyzing multiple files."""
        files = []
        for i in range(3):
            f = tmp_path / f"test{i}.txt"
            f.write_text(f"Content {i}")
            files.append(f)

        analyzer = FileAnalyzer()
        results = analyzer.analyze_multiple(files)

        assert len(results) == 3
        assert all(r.success for r in results)

    def test_content_preview_truncation(self, tmp_path):
        """Test that content preview is truncated for long files."""
        test_file = tmp_path / "long.txt"
        long_content = "word " * 200  # More than 500 chars
        test_file.write_text(long_content)

        analyzer = FileAnalyzer()
        result = analyzer.analyze(test_file)

        assert result.success
        assert len(result.content_preview) < len(result.content)
        assert result.content_preview.endswith("...")
