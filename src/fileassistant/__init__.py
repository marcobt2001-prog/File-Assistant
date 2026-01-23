"""
FileAssistant - A local, privacy-first AI file organizer.

An intelligent assistant that learns how you organize files and helps automate
the process, keeping your digital life organized without compromising privacy.
"""

__version__ = "0.1.0"
__author__ = "FileAssistant Team"
__license__ = "MIT"

from .config import get_config
from .database import get_database
from .utils.logging import get_logger

# Phase 1 components
from .analyzer import AnalysisResult, FileAnalyzer
from .classifier import ClassificationResult, FileClassifier
from .core import FileProcessor, ProcessingResult
from .mover import FileMover, MoveResult
from .watcher import FileWatcher

__all__ = [
    "get_config",
    "get_database",
    "get_logger",
    # Watcher
    "FileWatcher",
    # Analyzer
    "FileAnalyzer",
    "AnalysisResult",
    # Classifier
    "FileClassifier",
    "ClassificationResult",
    # Mover
    "FileMover",
    "MoveResult",
    # Core processor
    "FileProcessor",
    "ProcessingResult",
]
