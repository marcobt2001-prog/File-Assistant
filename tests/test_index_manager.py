"""Tests for the index manager."""

import sys
from datetime import datetime
from pathlib import Path

import pytest

from fileassistant.search.index_manager import IndexManager, IndexedFileMetadata

# ChromaDB has compatibility issues with Python 3.14+ due to Pydantic v1 usage
CHROMADB_PYTHON_COMPAT_ISSUE = sys.version_info >= (3, 14)


def check_chromadb_available():
    """Check if ChromaDB can be initialized without errors."""
    if CHROMADB_PYTHON_COMPAT_ISSUE:
        return False
    try:
        import chromadb
        # Try to create a simple client to check for Pydantic issues
        client = chromadb.Client()
        return True
    except Exception:
        return False


CHROMADB_AVAILABLE = check_chromadb_available()
skip_if_chromadb_unavailable = pytest.mark.skipif(
    not CHROMADB_AVAILABLE,
    reason="ChromaDB not available or has Python 3.14+ compatibility issues"
)


class TestIndexedFileMetadata:
    """Tests for IndexedFileMetadata dataclass."""

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata."""
        return IndexedFileMetadata(
            file_id="test123",
            file_path="/path/to/file.pdf",
            filename="file.pdf",
            extension=".pdf",
            file_type="document",
            tags=["tag1", "tag2"],
            content_summary="This is a test document.",
            content_hash="abc123",
            created_at=datetime(2025, 1, 15, 10, 30),
            modified_at=datetime(2025, 1, 15, 11, 0),
            indexed_at=datetime(2025, 2, 9, 14, 0),
            size_bytes=45000,
            source_folder="Downloads",
        )

    def test_to_chroma_metadata(self, sample_metadata):
        """Test conversion to ChromaDB format."""
        chroma_meta = sample_metadata.to_chroma_metadata()

        assert chroma_meta["file_id"] == "test123"
        assert chroma_meta["file_path"] == "/path/to/file.pdf"
        assert chroma_meta["filename"] == "file.pdf"
        assert chroma_meta["extension"] == ".pdf"
        assert chroma_meta["tags"] == "tag1,tag2"
        assert chroma_meta["size_bytes"] == 45000
        assert "2025-01-15" in chroma_meta["created_at"]

    def test_to_chroma_metadata_empty_tags(self):
        """Test conversion with empty tags."""
        metadata = IndexedFileMetadata(
            file_id="test",
            file_path="/test",
            filename="test",
            extension=".txt",
            file_type="document",
            tags=[],
            content_summary="",
            content_hash="",
            created_at=None,
            modified_at=None,
            indexed_at=datetime.now(),
            size_bytes=0,
            source_folder="",
        )
        chroma_meta = metadata.to_chroma_metadata()
        assert chroma_meta["tags"] == ""
        assert chroma_meta["created_at"] == ""

    def test_from_chroma_metadata(self, sample_metadata):
        """Test round-trip conversion."""
        chroma_meta = sample_metadata.to_chroma_metadata()
        restored = IndexedFileMetadata.from_chroma_metadata(chroma_meta)

        assert restored.file_id == sample_metadata.file_id
        assert restored.file_path == sample_metadata.file_path
        assert restored.tags == sample_metadata.tags
        assert restored.size_bytes == sample_metadata.size_bytes


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not available")
class TestIndexManager:
    """Tests for IndexManager."""

    @pytest.fixture
    def index_manager(self, tmp_path):
        """Create an index manager with temporary storage."""
        manager = IndexManager(persist_directory=tmp_path / "chromadb")
        yield manager
        manager.close()

    @pytest.fixture
    def sample_embedding(self):
        """Create a sample embedding vector."""
        return [0.1] * 384  # all-MiniLM-L6-v2 dimension

    def test_initialization(self, tmp_path):
        """Test manager initialization."""
        manager = IndexManager(persist_directory=tmp_path / "chromadb")
        assert manager.persist_directory == tmp_path / "chromadb"
        manager.close()

    def test_initialization_default_path(self):
        """Test manager uses default path when not specified."""
        manager = IndexManager()
        expected = Path.home() / ".fileassistant" / "chromadb"
        assert manager.persist_directory == expected

    def test_compute_content_hash(self):
        """Test content hash computation."""
        hash1 = IndexManager.compute_content_hash("test content")
        hash2 = IndexManager.compute_content_hash("test content")
        hash3 = IndexManager.compute_content_hash("different content")

        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 16  # Truncated hash

    def test_index_file(self, index_manager, sample_embedding, tmp_path):
        """Test indexing a file."""
        test_file = tmp_path / "test.pdf"
        test_file.touch()

        success = index_manager.index_file(
            file_id="file1",
            file_path=test_file,
            text="This is test content for the file.",
            embedding=sample_embedding,
            tags=["test", "document"],
            content_summary="Test document summary",
            file_type="document",
            size_bytes=1024,
        )

        assert success is True
        assert index_manager.get_indexed_count() == 1

    def test_index_file_upsert(self, index_manager, sample_embedding, tmp_path):
        """Test that indexing same file ID updates instead of duplicating."""
        test_file = tmp_path / "test.pdf"
        test_file.touch()

        # Index once
        index_manager.index_file(
            file_id="file1",
            file_path=test_file,
            text="Original content",
            embedding=sample_embedding,
        )

        # Index again with same ID
        index_manager.index_file(
            file_id="file1",
            file_path=test_file,
            text="Updated content",
            embedding=sample_embedding,
        )

        # Should still only have one entry
        assert index_manager.get_indexed_count() == 1

        # Check it was updated
        metadata, document = index_manager.get_file("file1")
        assert "Updated" in document

    def test_remove_file(self, index_manager, sample_embedding, tmp_path):
        """Test removing a file from the index."""
        test_file = tmp_path / "test.pdf"
        test_file.touch()

        index_manager.index_file(
            file_id="file1",
            file_path=test_file,
            text="Content",
            embedding=sample_embedding,
        )
        assert index_manager.get_indexed_count() == 1

        success = index_manager.remove_file("file1")
        assert success is True
        assert index_manager.get_indexed_count() == 0

    def test_remove_nonexistent_file(self, index_manager):
        """Test removing a file that doesn't exist."""
        success = index_manager.remove_file("nonexistent")
        # ChromaDB doesn't error on deleting non-existent IDs
        assert success is True

    def test_get_indexed_count_empty(self, index_manager):
        """Test count on empty index."""
        assert index_manager.get_indexed_count() == 0

    def test_is_indexed_not_indexed(self, index_manager):
        """Test checking if a non-indexed file is indexed."""
        assert index_manager.is_indexed("nonexistent") is False

    def test_is_indexed_exists(self, index_manager, sample_embedding, tmp_path):
        """Test checking if an indexed file is indexed."""
        test_file = tmp_path / "test.pdf"
        test_file.touch()

        index_manager.index_file(
            file_id="file1",
            file_path=test_file,
            text="Content",
            embedding=sample_embedding,
        )

        assert index_manager.is_indexed("file1") is True
        assert index_manager.is_indexed("file2") is False

    def test_is_indexed_with_hash_unchanged(self, index_manager, sample_embedding, tmp_path):
        """Test checking indexed with content hash - unchanged."""
        test_file = tmp_path / "test.pdf"
        test_file.touch()

        content = "Content"
        content_hash = IndexManager.compute_content_hash(content)

        index_manager.index_file(
            file_id="file1",
            file_path=test_file,
            text=content,
            embedding=sample_embedding,
        )

        # Same hash should return True
        assert index_manager.is_indexed("file1", content_hash) is True

    def test_is_indexed_with_hash_changed(self, index_manager, sample_embedding, tmp_path):
        """Test checking indexed with content hash - changed."""
        test_file = tmp_path / "test.pdf"
        test_file.touch()

        index_manager.index_file(
            file_id="file1",
            file_path=test_file,
            text="Original content",
            embedding=sample_embedding,
        )

        # Different hash should return False
        new_hash = IndexManager.compute_content_hash("Different content")
        assert index_manager.is_indexed("file1", new_hash) is False

    def test_get_file(self, index_manager, sample_embedding, tmp_path):
        """Test retrieving file metadata."""
        test_file = tmp_path / "test.pdf"
        test_file.touch()

        index_manager.index_file(
            file_id="file1",
            file_path=test_file,
            text="This is the document content.",
            embedding=sample_embedding,
            tags=["test", "pdf"],
            content_summary="A test summary",
            size_bytes=2048,
        )

        metadata, document = index_manager.get_file("file1")

        assert metadata is not None
        assert metadata.file_id == "file1"
        assert metadata.filename == "test.pdf"
        assert metadata.extension == ".pdf"
        assert metadata.tags == ["test", "pdf"]
        assert metadata.size_bytes == 2048
        assert "document content" in document

    def test_get_file_not_found(self, index_manager):
        """Test getting a non-existent file."""
        metadata, document = index_manager.get_file("nonexistent")
        assert metadata is None
        assert document is None

    def test_search(self, index_manager, sample_embedding, tmp_path):
        """Test searching the index."""
        # Index some files
        for i in range(3):
            test_file = tmp_path / f"file{i}.pdf"
            test_file.touch()
            # Slightly modify embedding for each file
            embedding = [0.1 + (i * 0.01)] * 384
            index_manager.index_file(
                file_id=f"file{i}",
                file_path=test_file,
                text=f"Content for file {i}",
                embedding=embedding,
            )

        # Search with query similar to first file
        query_embedding = [0.1] * 384
        results = index_manager.search(query_embedding, n_results=2)

        assert len(results) == 2
        # Results should be tuples of (metadata, distance, document)
        for metadata, distance, document in results:
            assert metadata is not None
            assert isinstance(distance, float)
            assert document is not None

    def test_search_empty_index(self, index_manager):
        """Test searching an empty index."""
        query_embedding = [0.1] * 384
        results = index_manager.search(query_embedding)
        assert results == []

    def test_get_all_file_ids(self, index_manager, sample_embedding, tmp_path):
        """Test getting all file IDs."""
        # Index some files
        for i in range(3):
            test_file = tmp_path / f"file{i}.pdf"
            test_file.touch()
            index_manager.index_file(
                file_id=f"file{i}",
                file_path=test_file,
                text=f"Content {i}",
                embedding=sample_embedding,
            )

        file_ids = index_manager.get_all_file_ids()
        assert len(file_ids) == 3
        assert set(file_ids) == {"file0", "file1", "file2"}

    def test_clear(self, index_manager, sample_embedding, tmp_path):
        """Test clearing the index."""
        test_file = tmp_path / "test.pdf"
        test_file.touch()

        index_manager.index_file(
            file_id="file1",
            file_path=test_file,
            text="Content",
            embedding=sample_embedding,
        )
        assert index_manager.get_indexed_count() == 1

        success = index_manager.clear()
        assert success is True
        assert index_manager.get_indexed_count() == 0

    def test_close(self, tmp_path):
        """Test closing the manager."""
        manager = IndexManager(persist_directory=tmp_path / "chromadb")
        manager._get_collection()  # Initialize
        manager.close()
        assert manager._client is None
        assert manager._collection is None


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not available")
class TestIndexManagerWithFiltering:
    """Tests for index filtering capabilities."""

    @pytest.fixture
    def populated_index(self, tmp_path):
        """Create an index with test data."""
        manager = IndexManager(persist_directory=tmp_path / "chromadb")

        # Add various files
        files_data = [
            ("file1", "report.pdf", ".pdf", "document", ["work", "report"]),
            ("file2", "photo.jpg", ".jpg", "image", ["personal", "photo"]),
            ("file3", "notes.txt", ".txt", "document", ["work", "notes"]),
            ("file4", "vacation.jpg", ".jpg", "image", ["personal", "vacation"]),
        ]

        for file_id, filename, ext, ftype, tags in files_data:
            test_file = tmp_path / filename
            test_file.touch()
            embedding = [0.1] * 384
            manager.index_file(
                file_id=file_id,
                file_path=test_file,
                text=f"Content for {filename}",
                embedding=embedding,
                tags=tags,
                file_type=ftype,
            )

        yield manager
        manager.close()

    def test_search_with_extension_filter(self, populated_index):
        """Test searching with extension filter."""
        query_embedding = [0.1] * 384
        results = populated_index.search(
            query_embedding,
            where={"extension": ".pdf"},
        )

        assert len(results) == 1
        assert results[0][0].filename == "report.pdf"

    def test_search_with_file_type_filter(self, populated_index):
        """Test searching with file type filter."""
        query_embedding = [0.1] * 384
        results = populated_index.search(
            query_embedding,
            where={"file_type": "image"},
        )

        assert len(results) == 2
        filenames = [r[0].filename for r in results]
        assert "photo.jpg" in filenames
        assert "vacation.jpg" in filenames
