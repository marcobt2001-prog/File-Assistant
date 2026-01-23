"""Analyzer module for file content and metadata extraction."""

from .analyzer import AnalysisResult, FileAnalyzer, FileMetadata
from .extractors import (
    BaseExtractor,
    DOCXExtractor,
    ExtractionError,
    PDFExtractor,
    PlainTextExtractor,
    get_extractor,
    get_supported_extensions,
)

__all__ = [
    "FileAnalyzer",
    "AnalysisResult",
    "FileMetadata",
    "BaseExtractor",
    "PlainTextExtractor",
    "PDFExtractor",
    "DOCXExtractor",
    "ExtractionError",
    "get_extractor",
    "get_supported_extensions",
]
