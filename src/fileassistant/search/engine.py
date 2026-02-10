"""Search engine for natural language file search."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..embeddings import EmbeddingGenerator
from ..utils.logging import get_logger
from .index_manager import IndexManager, IndexedFileMetadata

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """A single search result with relevance information."""

    file_path: str
    filename: str
    relevance_score: float  # 0.0 to 1.0, where 1.0 is perfect match
    content_snippet: str  # First ~200 chars of content
    tags: list[str]
    file_type: str
    modified_at: datetime | None
    size_bytes: int
    extension: str = ""

    @classmethod
    def from_index_result(
        cls,
        metadata: IndexedFileMetadata,
        distance: float,
        document: str,
        max_distance: float = 2.0,
    ) -> "SearchResult":
        """
        Create a SearchResult from IndexManager search output.

        Args:
            metadata: The indexed file metadata
            distance: ChromaDB distance (lower = more similar)
            document: The stored document text
            max_distance: Maximum expected distance for normalization

        Returns:
            SearchResult with normalized relevance score
        """
        # Normalize distance to relevance score (0-1, higher = better)
        # ChromaDB uses L2 distance by default, typical range 0-2
        relevance = max(0.0, min(1.0, 1.0 - (distance / max_distance)))

        # Create snippet from document (first ~200 chars)
        snippet = document[:200].strip() if document else ""
        if len(document) > 200:
            # Try to break at word boundary
            last_space = snippet.rfind(" ")
            if last_space > 150:
                snippet = snippet[:last_space]
            snippet += "..."

        return cls(
            file_path=metadata.file_path,
            filename=metadata.filename,
            relevance_score=round(relevance, 3),
            content_snippet=snippet,
            tags=metadata.tags,
            file_type=metadata.file_type,
            modified_at=metadata.modified_at,
            size_bytes=metadata.size_bytes,
            extension=metadata.extension,
        )


class SearchEngine:
    """
    Natural language search engine for indexed files.

    Uses embeddings to find semantically similar documents and supports
    post-retrieval filtering by file type, date range, and tags.
    """

    def __init__(
        self,
        index_manager: IndexManager | None = None,
        embedding_generator: EmbeddingGenerator | None = None,
        persist_directory: Path | str | None = None,
    ):
        """
        Initialize the search engine.

        Args:
            index_manager: Optional pre-configured IndexManager
            embedding_generator: Optional pre-configured EmbeddingGenerator
            persist_directory: ChromaDB storage path (used if index_manager not provided)
        """
        self.index_manager = index_manager or IndexManager(persist_directory=persist_directory)
        self.embedding_generator = embedding_generator or EmbeddingGenerator()

    def search(
        self,
        query: str,
        filters: dict | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """
        Search for files matching the natural language query.

        Args:
            query: Natural language search query
            filters: Optional filters:
                - extension: str or list[str] (e.g., ".pdf" or [".pdf", ".docx"])
                - after: datetime - only files modified after this date
                - before: datetime - only files modified before this date
                - tag: str or list[str] - files must have these tags
                - file_type: str - filter by file type (document, image, etc.)
            limit: Maximum number of results to return

        Returns:
            List of SearchResult objects sorted by relevance
        """
        filters = filters or {}

        # Validate query
        if not query or not query.strip():
            logger.warning("Empty search query provided")
            return []

        query = query.strip()
        if len(query) < 2:
            logger.warning(f"Query too short: '{query}'")
            return []

        # Check if index is empty
        indexed_count = self.index_manager.get_indexed_count()
        if indexed_count == 0:
            logger.info("Search attempted on empty index")
            return []

        logger.info(f"Searching for: '{query}' (limit={limit})")

        # Generate query embedding
        embedding_result = self.embedding_generator.generate(query)
        if not embedding_result.success:
            logger.error(f"Failed to generate query embedding: {embedding_result.error_message}")
            return []

        # Build ChromaDB where filter for extension/file_type (these are efficient in ChromaDB)
        chroma_where = self._build_chroma_filter(filters)

        # Fetch more results than needed to allow for post-filtering
        fetch_limit = min(limit * 2, 100)

        # Query the index
        raw_results = self.index_manager.search(
            query_embedding=embedding_result.embedding,
            n_results=fetch_limit,
            where=chroma_where,
        )

        if not raw_results:
            logger.info(f"No results found for query: '{query}'")
            return []

        # Convert to SearchResult and apply post-filters
        results = []
        for metadata, distance, document in raw_results:
            # Apply post-retrieval filters (date, tags)
            if not self._passes_post_filters(metadata, filters):
                continue

            result = SearchResult.from_index_result(metadata, distance, document)
            results.append(result)

            # Stop if we have enough results
            if len(results) >= limit:
                break

        logger.info(f"Found {len(results)} results for query: '{query}'")
        return results

    def _build_chroma_filter(self, filters: dict) -> dict | None:
        """
        Build ChromaDB where filter from user filters.

        Only uses filters that ChromaDB can handle efficiently.
        """
        conditions = []

        # Extension filter
        if "extension" in filters:
            ext = filters["extension"]
            if isinstance(ext, str):
                # Normalize extension format
                ext = ext if ext.startswith(".") else f".{ext}"
                conditions.append({"extension": ext.lower()})
            elif isinstance(ext, list) and ext:
                # Multiple extensions - use $in operator
                exts = [e if e.startswith(".") else f".{e}" for e in ext]
                exts = [e.lower() for e in exts]
                conditions.append({"extension": {"$in": exts}})

        # File type filter
        if "file_type" in filters:
            conditions.append({"file_type": filters["file_type"]})

        if not conditions:
            return None

        if len(conditions) == 1:
            return conditions[0]

        return {"$and": conditions}

    def _passes_post_filters(self, metadata: IndexedFileMetadata, filters: dict) -> bool:
        """
        Check if metadata passes filters that can't be done in ChromaDB.

        Args:
            metadata: The file metadata to check
            filters: User-provided filters

        Returns:
            True if file passes all filters
        """
        # Date range filters
        if "after" in filters and filters["after"]:
            after_date = filters["after"]
            if isinstance(after_date, str):
                after_date = datetime.fromisoformat(after_date)
            if metadata.modified_at and metadata.modified_at < after_date:
                return False

        if "before" in filters and filters["before"]:
            before_date = filters["before"]
            if isinstance(before_date, str):
                before_date = datetime.fromisoformat(before_date)
            if metadata.modified_at and metadata.modified_at > before_date:
                return False

        # Tag filter
        if "tag" in filters and filters["tag"]:
            required_tags = filters["tag"]
            if isinstance(required_tags, str):
                required_tags = [required_tags]
            # Check if file has any of the required tags
            file_tags = set(t.lower() for t in metadata.tags)
            required_set = set(t.lower() for t in required_tags)
            if not file_tags.intersection(required_set):
                return False

        return True

    def get_indexed_count(self) -> int:
        """Get the number of indexed files."""
        return self.index_manager.get_indexed_count()

    def is_index_empty(self) -> bool:
        """Check if the index is empty."""
        return self.get_indexed_count() == 0

    def close(self):
        """Close the search engine and release resources."""
        self.index_manager.close()
