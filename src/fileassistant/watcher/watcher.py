"""File watcher component for monitoring inbox folders."""

import threading
from collections.abc import Callable
from pathlib import Path

from watchdog.observers import Observer

from ..config import FileAssistantConfig
from ..utils.logging import get_logger
from .handler import SUPPORTED_EXTENSIONS, DebouncedFileHandler

logger = get_logger(__name__)


class FileWatcher:
    """
    Watches configured inbox folders for new files.

    Uses watchdog for cross-platform file system monitoring.
    Debounces events to ensure files are fully written before processing.
    """

    def __init__(
        self,
        config: FileAssistantConfig,
        on_file_ready: Callable[[Path], None],
    ):
        """
        Initialize the file watcher.

        Args:
            config: Application configuration
            on_file_ready: Callback when a file is ready for processing
        """
        self.config = config
        self.on_file_ready = on_file_ready

        self._observer: Observer | None = None
        self._handlers: list[DebouncedFileHandler] = []
        self._running = False
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        """Check if the watcher is currently running."""
        return self._running

    @property
    def watched_folders(self) -> list[Path]:
        """Get list of folders being watched."""
        return list(self.config.inbox_folders)

    def start(self):
        """Start watching all configured inbox folders."""
        with self._lock:
            if self._running:
                logger.warning("Watcher is already running")
                return

            self._observer = Observer()

            for folder in self.config.inbox_folders:
                if not folder.exists():
                    logger.warning(f"Inbox folder does not exist, creating: {folder}")
                    folder.mkdir(parents=True, exist_ok=True)

                handler = DebouncedFileHandler(
                    callback=self.on_file_ready,
                    debounce_seconds=self.config.processing.debounce_seconds,
                    supported_extensions=SUPPORTED_EXTENSIONS,
                )
                self._handlers.append(handler)

                self._observer.schedule(handler, str(folder), recursive=False)
                logger.info(f"Watching folder: {folder}")

            self._observer.start()
            self._running = True
            logger.info(f"File watcher started, monitoring {len(self.config.inbox_folders)} folder(s)")

    def stop(self):
        """Stop watching all folders."""
        with self._lock:
            if not self._running:
                return

            # Stop all handlers (cancel pending timers)
            for handler in self._handlers:
                handler.stop()
            self._handlers.clear()

            # Stop the observer
            if self._observer:
                self._observer.stop()
                self._observer.join(timeout=5.0)
                self._observer = None

            self._running = False
            logger.info("File watcher stopped")

    def scan_existing(self) -> list[Path]:
        """
        Scan inbox folders for existing files.

        Returns list of existing supported files (useful for initial processing).
        """
        existing_files: list[Path] = []

        for folder in self.config.inbox_folders:
            if not folder.exists():
                continue

            for file_path in folder.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    # Skip hidden files
                    if not file_path.name.startswith("."):
                        existing_files.append(file_path)
                        logger.debug(f"Found existing file: {file_path}")

        logger.info(f"Scan complete: found {len(existing_files)} existing file(s)")
        return existing_files

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
