"""File analyzer component for content and metadata extraction."""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..utils.logging import get_logger
from .extractors import ExtractionError, get_extractor, get_supported_extensions

logger = get_logger(__name__)


@dataclass
class FileMetadata:
    """Basic file metadata."""

    path: Path
    filename: str
    extension: str
    size_bytes: int
    created_at: datetime
    modified_at: datetime
    hash_md5: str


@dataclass
class AnalysisResult:
    """Result of file analysis."""

    # File identification
    file_path: Path
    metadata: FileMetadata

    # Content extraction
    content: str
    content_preview: str  # First N characters for display

    # Analysis status
    success: bool
    error_message: str | None = None

    # Additional extracted info (for future use)
    word_count: int = 0
    line_count: int = 0

    @property
    def has_content(self) -> bool:
        """Check if content was successfully extracted."""
        return bool(self.content.strip())


class FileAnalyzer:
    """
    Analyzes files to extract content and metadata.

    Supports multiple file types through pluggable extractors.
    """

    PREVIEW_LENGTH = 500  # Characters to include in preview

    def __init__(self, max_file_size_mb: int = 100):
        """
        Initialize the file analyzer.

        Args:
            max_file_size_mb: Maximum file size to process in megabytes
        """
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.supported_extensions = get_supported_extensions()

    def _compute_md5(self, file_path: Path) -> str:
        """Compute MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except OSError as e:
            logger.warning(f"Could not compute MD5 for {file_path}: {e}")
            return ""

    def _extract_metadata(self, file_path: Path) -> FileMetadata:
        """Extract file metadata."""
        stat = file_path.stat()

        return FileMetadata(
            path=file_path,
            filename=file_path.name,
            extension=file_path.suffix.lower(),
            size_bytes=stat.st_size,
            created_at=datetime.fromtimestamp(stat.st_ctime),
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            hash_md5=self._compute_md5(file_path),
        )

    def can_analyze(self, file_path: Path) -> bool:
        """Check if a file can be analyzed."""
        if not file_path.exists():
            return False

        if not file_path.is_file():
            return False

        if file_path.suffix.lower() not in self.supported_extensions:
            return False

        return True

    def analyze(self, file_path: Path) -> AnalysisResult:
        """
        Analyze a file to extract content and metadata.

        Args:
            file_path: Path to the file to analyze

        Returns:
            AnalysisResult with extracted content and metadata
        """
        file_path = Path(file_path).resolve()

        # Validate file exists
        if not file_path.exists():
            return AnalysisResult(
                file_path=file_path,
                metadata=None,  # type: ignore
                content="",
                content_preview="",
                success=False,
                error_message=f"File not found: {file_path}",
            )

        # Extract metadata first
        try:
            metadata = self._extract_metadata(file_path)
        except OSError as e:
            return AnalysisResult(
                file_path=file_path,
                metadata=None,  # type: ignore
                content="",
                content_preview="",
                success=False,
                error_message=f"Could not read file metadata: {e}",
            )

        # Check file size
        if metadata.size_bytes > self.max_file_size_bytes:
            return AnalysisResult(
                file_path=file_path,
                metadata=metadata,
                content="",
                content_preview="",
                success=False,
                error_message=f"File too large: {metadata.size_bytes / 1024 / 1024:.1f}MB exceeds limit of {self.max_file_size_bytes / 1024 / 1024:.0f}MB",
            )

        # Get appropriate extractor
        extractor = get_extractor(file_path)
        if extractor is None:
            return AnalysisResult(
                file_path=file_path,
                metadata=metadata,
                content="",
                content_preview="",
                success=False,
                error_message=f"No extractor available for extension: {metadata.extension}",
            )

        # Extract content
        try:
            content = extractor.extract(file_path)
            content_preview = content[: self.PREVIEW_LENGTH]
            if len(content) > self.PREVIEW_LENGTH:
                content_preview += "..."

            # Calculate stats
            word_count = len(content.split())
            line_count = content.count("\n") + 1 if content else 0

            logger.info(
                f"Analyzed {file_path.name}: {word_count} words, {metadata.size_bytes} bytes"
            )

            return AnalysisResult(
                file_path=file_path,
                metadata=metadata,
                content=content,
                content_preview=content_preview,
                success=True,
                word_count=word_count,
                line_count=line_count,
            )

        except ExtractionError as e:
            logger.error(f"Extraction failed for {file_path}: {e}")
            return AnalysisResult(
                file_path=file_path,
                metadata=metadata,
                content="",
                content_preview="",
                success=False,
                error_message=str(e),
            )

        except Exception as e:
            logger.exception(f"Unexpected error analyzing {file_path}")
            return AnalysisResult(
                file_path=file_path,
                metadata=metadata,
                content="",
                content_preview="",
                success=False,
                error_message=f"Unexpected error: {e}",
            )

    def analyze_multiple(self, file_paths: list[Path]) -> list[AnalysisResult]:
        """
        Analyze multiple files.

        Args:
            file_paths: List of file paths to analyze

        Returns:
            List of AnalysisResult objects
        """
        results: list[AnalysisResult] = []
        for file_path in file_paths:
            results.append(self.analyze(file_path))
        return results
