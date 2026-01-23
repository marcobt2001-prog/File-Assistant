"""File system event handler with debouncing."""

import threading
import time
from collections.abc import Callable
from pathlib import Path

from watchdog.events import FileCreatedEvent, FileModifiedEvent, FileSystemEventHandler

from ..utils.logging import get_logger

logger = get_logger(__name__)


# File extensions to ignore (temp files, partial downloads, etc.)
IGNORED_EXTENSIONS = {
    ".tmp",
    ".temp",
    ".part",
    ".partial",
    ".crdownload",  # Chrome partial download
    ".download",  # Safari partial download
    ".opdownload",  # Opera partial download
    ".aria2",  # aria2 partial download
    ".unconfirmed",
    ".swp",  # Vim swap files
    ".swo",
    ".swn",
    "~",  # Backup files
    ".bak",
    ".lock",
}

# File name patterns to ignore
IGNORED_PATTERNS = {
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
    ".gitignore",
    ".gitkeep",
}

# Supported file extensions for Phase 1 MVP
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


class DebouncedFileHandler(FileSystemEventHandler):
    """
    File system event handler with debouncing.

    Waits for files to be fully written before triggering callbacks.
    Filters to only supported file types.
    """

    def __init__(
        self,
        callback: Callable[[Path], None],
        debounce_seconds: float = 2.0,
        supported_extensions: set[str] | None = None,
    ):
        """
        Initialize the debounced file handler.

        Args:
            callback: Function to call when a file is ready for processing
            debounce_seconds: Time to wait after last modification before processing
            supported_extensions: Set of file extensions to process (lowercase, with dot)
        """
        super().__init__()
        self.callback = callback
        self.debounce_seconds = debounce_seconds
        self.supported_extensions = supported_extensions or SUPPORTED_EXTENSIONS

        # Track pending files: path -> (timer, last_size)
        self._pending: dict[str, tuple[threading.Timer, int]] = {}
        self._lock = threading.Lock()

    def _should_ignore(self, path: Path) -> bool:
        """Check if a file should be ignored based on name or extension."""
        # Ignore hidden files
        if path.name.startswith("."):
            return True

        # Ignore known system files
        if path.name in IGNORED_PATTERNS:
            return True

        # Ignore temp file extensions
        suffix = path.suffix.lower()
        if suffix in IGNORED_EXTENSIONS:
            return True

        # Ignore files that end with ~ (backup files)
        if path.name.endswith("~"):
            return True

        return False

    def _is_supported(self, path: Path) -> bool:
        """Check if file extension is supported."""
        return path.suffix.lower() in self.supported_extensions

    def _get_file_size(self, path: Path) -> int:
        """Get file size, returns -1 if file doesn't exist or can't be read."""
        try:
            return path.stat().st_size
        except (OSError, FileNotFoundError):
            return -1

    def _schedule_callback(self, path_str: str):
        """Schedule a callback for a file after debounce period."""
        path = Path(path_str)

        with self._lock:
            # Cancel existing timer if any
            if path_str in self._pending:
                old_timer, _ = self._pending[path_str]
                old_timer.cancel()

            current_size = self._get_file_size(path)

            def check_and_process():
                """Check if file is stable and process it."""
                with self._lock:
                    if path_str not in self._pending:
                        return

                    _, last_size = self._pending[path_str]
                    new_size = self._get_file_size(path)

                    # File was deleted or can't be read
                    if new_size == -1:
                        logger.debug(f"File no longer accessible: {path}")
                        del self._pending[path_str]
                        return

                    # File size changed, reschedule
                    if new_size != last_size:
                        logger.debug(f"File still changing: {path} ({last_size} -> {new_size})")
                        del self._pending[path_str]
                        # Release lock before recursive call
                        self._schedule_callback(path_str)
                        return

                    # File is stable, process it
                    del self._pending[path_str]

                # Call callback outside of lock
                logger.info(f"File ready for processing: {path}")
                try:
                    self.callback(path)
                except Exception as e:
                    logger.error(f"Error processing file {path}: {e}")

            timer = threading.Timer(self.debounce_seconds, check_and_process)
            self._pending[path_str] = (timer, current_size)
            timer.start()

    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory:
            return

        path = Path(event.src_path)

        if self._should_ignore(path):
            logger.debug(f"Ignoring file (system/temp): {path}")
            return

        if not self._is_supported(path):
            logger.debug(f"Ignoring file (unsupported extension): {path}")
            return

        logger.debug(f"File created: {path}")
        self._schedule_callback(event.src_path)

    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return

        path = Path(event.src_path)

        if self._should_ignore(path):
            return

        if not self._is_supported(path):
            return

        # Only reschedule if we're already tracking this file
        with self._lock:
            if event.src_path in self._pending:
                logger.debug(f"File modified (rescheduling): {path}")
                self._schedule_callback(event.src_path)

    def stop(self):
        """Cancel all pending timers."""
        with self._lock:
            for timer, _ in self._pending.values():
                timer.cancel()
            self._pending.clear()
