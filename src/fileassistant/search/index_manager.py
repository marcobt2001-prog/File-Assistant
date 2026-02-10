"""Index manager for ChromaDB vector storage."""

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IndexedFileMetadata:
    """Metadata stored alongside file embeddings in ChromaDB."""

    file_id: str
    file_path: str
    filename: str
    extension: str
    file_type: str
    tags: list[str]
    content_summary: str
    content_hash: str
    created_at: datetime | None
    modified_at: datetime | None
    indexed_at: datetime
    size_bytes: int
    source_folder: str

    def to_chroma_metadata(self) -> dict:
        """Convert to ChromaDB metadata format (string values only)."""
        return {
            "file_id": self.file_id,
            "file_path": self.file_path,
            "filename": self.filename,
            "extension": self.extension,
            "file_type": self.file_type,
            "tags": ",".join(self.tags) if self.tags else "",
            "content_summary": self.content_summary[:1000],  # Limit summary length
            "content_hash": self.content_hash,
            "created_at": self.created_at.isoformat() if self.created_at else "",
            "modified_at": self.modified_at.isoformat() if self.modified_at else "",
            "indexed_at": self.indexed_at.isoformat(),
            "size_bytes": self.size_bytes,
            "source_folder": self.source_folder,
        }

    @classmethod
    def from_chroma_metadata(cls, metadata: dict) -> "IndexedFileMetadata":
        """Create from ChromaDB metadata format."""
        return cls(
            file_id=metadata["file_id"],
            file_path=metadata["file_path"],
            filename=metadata["filename"],
            extension=metadata["extension"],
            file_type=metadata["file_type"],
            tags=metadata["tags"].split(",") if metadata.get("tags") else [],
            content_summary=metadata.get("content_summary", ""),
            content_hash=metadata.get("content_hash", ""),
            created_at=datetime.fromisoformat(metadata["created_at"]) if metadata.get("created_at") else None,
            modified_at=datetime.fromisoformat(metadata["modified_at"]) if metadata.get("modified_at") else None,
            indexed_at=datetime.fromisoformat(metadata["indexed_at"]) if metadata.get("indexed_at") else datetime.now(),
            size_bytes=int(metadata.get("size_bytes", 0)),
            source_folder=metadata.get("source_folder", ""),
        )


class IndexManager:
    """
    Manages file embeddings in ChromaDB.

    Provides CRUD operations for file embeddings with rich metadata
    for filtering and display.
    """

    COLLECTION_NAME = "fileassistant_files"

    def __init__(
        self,
        persist_directory: Path | str | None = None,
    ):
        """
        Initialize the index manager.

        Args:
            persist_directory: Path to ChromaDB storage directory.
                              Defaults to ~/.fileassistant/chromadb/
        """
        if persist_directory is None:
            persist_directory = Path.home() / ".fileassistant" / "chromadb"
        self.persist_directory = Path(persist_directory)
        self._client = None
        self._collection = None

    def _ensure_directory(self):
        """Ensure the persist directory exists."""
        self.persist_directory.mkdir(parents=True, exist_ok=True)

    def _get_client(self):
        """Get or create the ChromaDB client."""
        if self._client is None:
            import chromadb

            self._ensure_directory()

            logger.info(f"Initializing ChromaDB at {self.persist_directory}")
            # Use simple PersistentClient without Settings to avoid Pydantic v1 issues
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
            )
        return self._client

    def _get_collection(self):
        """Get or create the files collection."""
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"description": "FileAssistant indexed files"},
            )
            logger.info(f"ChromaDB collection ready: {self.COLLECTION_NAME}")
        return self._collection

    @staticmethod
    def compute_content_hash(text: str) -> str:
        """Compute a hash of the content for change detection."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def index_file(
        self,
        file_id: str,
        file_path: Path | str,
        text: str,
        embedding: list[float],
        tags: list[str] | None = None,
        content_summary: str | None = None,
        file_type: str = "document",
        created_at: datetime | None = None,
        modified_at: datetime | None = None,
        size_bytes: int = 0,
    ) -> bool:
        """
        Add or update a file in the index.

        Args:
            file_id: Unique identifier for the file
            file_path: Path to the file
            text: Extracted text content (for snippet storage)
            embedding: Vector embedding of the content
            tags: List of tags for filtering
            content_summary: Brief summary of content
            file_type: Type of file (document, image, etc.)
            created_at: File creation time
            modified_at: File modification time
            size_bytes: File size in bytes

        Returns:
            True if successful, False otherwise
        """
        try:
            collection = self._get_collection()
            file_path = Path(file_path)

            # Build metadata
            metadata = IndexedFileMetadata(
                file_id=file_id,
                file_path=str(file_path),
                filename=file_path.name,
                extension=file_path.suffix.lower(),
                file_type=file_type,
                tags=tags or [],
                content_summary=content_summary or text[:500],
                content_hash=self.compute_content_hash(text),
                created_at=created_at,
                modified_at=modified_at,
                indexed_at=datetime.now(),
                size_bytes=size_bytes,
                source_folder=file_path.parent.name,
            )

            # Use upsert to add or update
            collection.upsert(
                ids=[file_id],
                embeddings=[embedding],
                documents=[text[:2000]],  # Store first 2000 chars for snippet display
                metadatas=[metadata.to_chroma_metadata()],
            )

            logger.debug(f"Indexed file: {file_path.name} (id={file_id})")
            return True

        except Exception as e:
            logger.error(f"Failed to index file {file_path}: {e}")
            return False

    def remove_file(self, file_id: str) -> bool:
        """
        Remove a file from the index.

        Args:
            file_id: The file ID to remove

        Returns:
            True if successful, False otherwise
        """
        try:
            collection = self._get_collection()
            collection.delete(ids=[file_id])
            logger.debug(f"Removed file from index: {file_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove file {file_id}: {e}")
            return False

    def get_indexed_count(self) -> int:
        """
        Get the number of indexed files.

        Returns:
            Count of indexed files
        """
        try:
            collection = self._get_collection()
            return collection.count()
        except Exception as e:
            logger.error(f"Failed to get indexed count: {e}")
            return 0

    def is_indexed(self, file_id: str, content_hash: str | None = None) -> bool:
        """
        Check if a file is already indexed.

        If content_hash is provided, also checks if the content has changed.

        Args:
            file_id: The file ID to check
            content_hash: Optional hash of current content

        Returns:
            True if indexed (and unchanged if hash provided), False otherwise
        """
        try:
            collection = self._get_collection()
            result = collection.get(ids=[file_id], include=["metadatas"])

            if not result["ids"]:
                return False

            if content_hash is None:
                return True

            # Check if content has changed
            stored_hash = result["metadatas"][0].get("content_hash", "")
            return stored_hash == content_hash

        except Exception as e:
            logger.error(f"Failed to check if indexed: {e}")
            return False

    def get_file(self, file_id: str) -> tuple[IndexedFileMetadata | None, str | None]:
        """
        Get a file's metadata and stored text snippet.

        Args:
            file_id: The file ID to retrieve

        Returns:
            Tuple of (metadata, document_text) or (None, None) if not found
        """
        try:
            collection = self._get_collection()
            result = collection.get(
                ids=[file_id],
                include=["metadatas", "documents"],
            )

            if not result["ids"]:
                return None, None

            metadata = IndexedFileMetadata.from_chroma_metadata(result["metadatas"][0])
            document = result["documents"][0] if result["documents"] else None

            return metadata, document

        except Exception as e:
            logger.error(f"Failed to get file {file_id}: {e}")
            return None, None

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict | None = None,
    ) -> list[tuple[IndexedFileMetadata, float, str]]:
        """
        Search for similar files.

        Args:
            query_embedding: The query vector
            n_results: Maximum number of results to return
            where: Optional ChromaDB where filter

        Returns:
            List of (metadata, distance, document) tuples sorted by relevance
        """
        try:
            collection = self._get_collection()

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["metadatas", "documents", "distances"],
            )

            if not results["ids"] or not results["ids"][0]:
                return []

            output = []
            for i, file_id in enumerate(results["ids"][0]):
                metadata = IndexedFileMetadata.from_chroma_metadata(results["metadatas"][0][i])
                distance = results["distances"][0][i] if results["distances"] else 0.0
                document = results["documents"][0][i] if results["documents"] else ""
                output.append((metadata, distance, document))

            return output

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_all_file_ids(self) -> list[str]:
        """
        Get all indexed file IDs.

        Returns:
            List of file IDs
        """
        try:
            collection = self._get_collection()
            result = collection.get(include=[])
            return result["ids"]
        except Exception as e:
            logger.error(f"Failed to get all file IDs: {e}")
            return []

    def clear(self) -> bool:
        """
        Clear all indexed files.

        Returns:
            True if successful, False otherwise
        """
        try:
            client = self._get_client()
            client.delete_collection(self.COLLECTION_NAME)
            self._collection = None
            logger.info("Cleared all indexed files")
            return True
        except Exception as e:
            logger.error(f"Failed to clear index: {e}")
            return False

    def close(self):
        """Close the ChromaDB connection."""
        self._collection = None
        self._client = None
        logger.debug("ChromaDB connection closed")
