"""Core module containing the main processing orchestrator."""

from .processor import FileProcessor, ProcessingResult, UserDecision

__all__ = [
    "FileProcessor",
    "ProcessingResult",
    "UserDecision",
]
