"""Tests for the watcher component."""

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fileassistant.watcher import SUPPORTED_EXTENSIONS, DebouncedFileHandler, FileWatcher
from fileassistant.watcher.handler import IGNORED_EXTENSIONS, IGNORED_PATTERNS


class TestDebouncedFileHandler:
    """Tests for DebouncedFileHandler."""

    def test_should_ignore_hidden_files(self):
        """Test that hidden files are ignored."""
        handler = DebouncedFileHandler(callback=MagicMock())
        assert handler._should_ignore(Path(".hidden"))
        assert handler._should_ignore(Path(".DS_Store"))

    def test_should_ignore_system_files(self):
        """Test that system files are ignored."""
        handler = DebouncedFileHandler(callback=MagicMock())
        assert handler._should_ignore(Path("Thumbs.db"))
        assert handler._should_ignore(Path("desktop.ini"))

    def test_should_ignore_temp_files(self):
        """Test that temp files are ignored."""
        handler = DebouncedFileHandler(callback=MagicMock())
        assert handler._should_ignore(Path("file.tmp"))
        assert handler._should_ignore(Path("file.part"))
        assert handler._should_ignore(Path("file.crdownload"))

    def test_should_not_ignore_supported_files(self):
        """Test that supported files are not ignored."""
        handler = DebouncedFileHandler(callback=MagicMock())
        assert not handler._should_ignore(Path("document.pdf"))
        assert not handler._should_ignore(Path("readme.md"))
        assert not handler._should_ignore(Path("notes.txt"))

    def test_is_supported_extension(self):
        """Test supported extension checking."""
        handler = DebouncedFileHandler(callback=MagicMock())
        assert handler._is_supported(Path("file.txt"))
        assert handler._is_supported(Path("FILE.TXT"))  # Case insensitive
        assert handler._is_supported(Path("file.pdf"))
        assert handler._is_supported(Path("file.docx"))
        assert handler._is_supported(Path("file.md"))
        assert not handler._is_supported(Path("file.exe"))
        assert not handler._is_supported(Path("file.jpg"))

    def test_custom_supported_extensions(self):
        """Test using custom supported extensions."""
        handler = DebouncedFileHandler(
            callback=MagicMock(),
            supported_extensions={".xyz", ".abc"},
        )
        assert handler._is_supported(Path("file.xyz"))
        assert handler._is_supported(Path("file.abc"))
        assert not handler._is_supported(Path("file.txt"))

    def test_debounce_callback(self, tmp_path):
        """Test that callback is called after debounce period."""
        callback = MagicMock()
        handler = DebouncedFileHandler(
            callback=callback,
            debounce_seconds=0.1,  # Short debounce for testing
        )

        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        # Simulate file creation event
        from watchdog.events import FileCreatedEvent

        event = FileCreatedEvent(str(test_file))
        handler.on_created(event)

        # Wait for debounce
        time.sleep(0.2)

        # Callback should have been called
        callback.assert_called_once()
        called_path = callback.call_args[0][0]
        assert called_path == test_file

    def test_debounce_reschedule_on_modification(self, tmp_path):
        """Test that callback is rescheduled on file modification."""
        callback = MagicMock()
        handler = DebouncedFileHandler(
            callback=callback,
            debounce_seconds=0.15,
        )

        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("initial")

        # Simulate file creation
        from watchdog.events import FileCreatedEvent, FileModifiedEvent

        handler.on_created(FileCreatedEvent(str(test_file)))

        # Wait a bit, then modify
        time.sleep(0.05)
        test_file.write_text("modified")
        handler.on_modified(FileModifiedEvent(str(test_file)))

        # Wait for original debounce (shouldn't fire)
        time.sleep(0.12)
        assert callback.call_count == 0  # Not called yet due to reschedule

        # Wait for full debounce after modification
        time.sleep(0.1)
        assert callback.call_count == 1

    def test_stop_cancels_pending_timers(self, tmp_path):
        """Test that stop() cancels pending timers."""
        callback = MagicMock()
        handler = DebouncedFileHandler(
            callback=callback,
            debounce_seconds=1.0,  # Long debounce
        )

        # Create test file and trigger event
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        from watchdog.events import FileCreatedEvent

        handler.on_created(FileCreatedEvent(str(test_file)))

        # Stop immediately
        handler.stop()

        # Wait and ensure callback was NOT called
        time.sleep(0.2)
        callback.assert_not_called()

    def test_ignores_directory_events(self, tmp_path):
        """Test that directory events are ignored."""
        callback = MagicMock()
        handler = DebouncedFileHandler(callback=callback, debounce_seconds=0.1)

        from watchdog.events import DirCreatedEvent

        event = DirCreatedEvent(str(tmp_path / "subdir"))
        handler.on_created(event)

        time.sleep(0.2)
        callback.assert_not_called()


class TestSupportedExtensions:
    """Tests for supported extensions constant."""

    def test_includes_phase1_extensions(self):
        """Test that Phase 1 MVP extensions are included."""
        assert ".txt" in SUPPORTED_EXTENSIONS
        assert ".md" in SUPPORTED_EXTENSIONS
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".docx" in SUPPORTED_EXTENSIONS


class TestIgnoredExtensions:
    """Tests for ignored extensions constant."""

    def test_includes_common_temp_extensions(self):
        """Test that common temp file extensions are ignored."""
        assert ".tmp" in IGNORED_EXTENSIONS
        assert ".part" in IGNORED_EXTENSIONS
        assert ".crdownload" in IGNORED_EXTENSIONS
        assert ".swp" in IGNORED_EXTENSIONS


class TestIgnoredPatterns:
    """Tests for ignored file patterns."""

    def test_includes_system_files(self):
        """Test that system files are in ignored patterns."""
        assert ".DS_Store" in IGNORED_PATTERNS
        assert "Thumbs.db" in IGNORED_PATTERNS
        assert "desktop.ini" in IGNORED_PATTERNS


class TestFileWatcher:
    """Tests for FileWatcher class."""

    def test_watched_folders_property(self, tmp_path):
        """Test watched_folders property."""
        from fileassistant.config import FileAssistantConfig

        config = FileAssistantConfig(inbox_folders=[tmp_path / "inbox1", tmp_path / "inbox2"])

        watcher = FileWatcher(config=config, on_file_ready=MagicMock())
        folders = watcher.watched_folders

        assert len(folders) == 2
        assert tmp_path / "inbox1" in folders
        assert tmp_path / "inbox2" in folders

    def test_is_running_property(self, tmp_path):
        """Test is_running property."""
        from fileassistant.config import FileAssistantConfig

        inbox = tmp_path / "inbox"
        inbox.mkdir()

        config = FileAssistantConfig(inbox_folders=[inbox])
        watcher = FileWatcher(config=config, on_file_ready=MagicMock())

        assert not watcher.is_running
        watcher.start()
        assert watcher.is_running
        watcher.stop()
        assert not watcher.is_running

    def test_context_manager(self, tmp_path):
        """Test context manager usage."""
        from fileassistant.config import FileAssistantConfig

        inbox = tmp_path / "inbox"
        inbox.mkdir()

        config = FileAssistantConfig(inbox_folders=[inbox])

        with FileWatcher(config=config, on_file_ready=MagicMock()) as watcher:
            assert watcher.is_running

        assert not watcher.is_running

    def test_scan_existing_finds_supported_files(self, tmp_path):
        """Test that scan_existing finds supported files."""
        from fileassistant.config import FileAssistantConfig

        inbox = tmp_path / "inbox"
        inbox.mkdir()

        # Create some test files
        (inbox / "doc.txt").write_text("text")
        (inbox / "doc.pdf").write_bytes(b"fake pdf")
        (inbox / "doc.docx").write_bytes(b"fake docx")
        (inbox / "doc.md").write_text("# markdown")
        (inbox / "image.jpg").write_bytes(b"fake jpg")  # Should be ignored
        (inbox / ".hidden").write_text("hidden")  # Should be ignored

        config = FileAssistantConfig(inbox_folders=[inbox])
        watcher = FileWatcher(config=config, on_file_ready=MagicMock())

        existing = watcher.scan_existing()

        # Should find 4 supported files
        assert len(existing) == 4
        filenames = {f.name for f in existing}
        assert "doc.txt" in filenames
        assert "doc.pdf" in filenames
        assert "doc.docx" in filenames
        assert "doc.md" in filenames
        assert "image.jpg" not in filenames
        assert ".hidden" not in filenames

    def test_creates_missing_inbox_folder(self, tmp_path):
        """Test that start() creates missing inbox folders."""
        from fileassistant.config import FileAssistantConfig

        inbox = tmp_path / "new_inbox"
        assert not inbox.exists()

        config = FileAssistantConfig(inbox_folders=[inbox])
        watcher = FileWatcher(config=config, on_file_ready=MagicMock())

        watcher.start()
        assert inbox.exists()
        watcher.stop()
