"""Search module for vector-based file search using ChromaDB."""

from .index_manager import IndexManager, IndexedFileMetadata
from .engine import SearchEngine, SearchResult

__all__ = [
    "IndexManager",
    "IndexedFileMetadata",
    "SearchEngine",
    "SearchResult",
]
