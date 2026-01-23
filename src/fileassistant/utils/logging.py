"""Logging infrastructure with Rich console output and rotating file logs."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Custom theme for FileAssistant
FILEASSISTANT_THEME = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "red bold",
        "critical": "red bold reverse",
        "success": "green bold",
        "highlight": "magenta",
    }
)


class FileAssistantLogger:
    """Custom logger with Rich console and file output."""

    _instance: Optional["FileAssistantLogger"] = None
    _initialized: bool = False

    def __new__(cls):
        """Singleton pattern to ensure only one logger instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize logger (only once)."""
        if not self._initialized:
            self.console = Console(theme=FILEASSISTANT_THEME)
            self.logger = logging.getLogger("fileassistant")
            self._initialized = True

    def setup(
        self,
        level: str = "INFO",
        log_dir: Optional[Path] = None,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
        console_enabled: bool = True,
        file_enabled: bool = True,
    ):
        """
        Configure logging handlers and formatters.

        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files
            max_bytes: Maximum size of each log file before rotation
            backup_count: Number of rotated log files to keep
            console_enabled: Enable console (Rich) logging
            file_enabled: Enable file logging
        """
        # Clear existing handlers
        self.logger.handlers.clear()

        # Set logging level
        log_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.setLevel(log_level)

        # Prevent propagation to root logger
        self.logger.propagate = False

        # Console handler with Rich
        if console_enabled:
            console_handler = RichHandler(
                console=self.console,
                rich_tracebacks=True,
                tracebacks_show_locals=True,
                show_time=True,
                show_path=True,
                markup=True,
            )
            console_handler.setLevel(log_level)
            self.logger.addHandler(console_handler)

        # File handler with rotation
        if file_enabled:
            if log_dir is None:
                log_dir = Path("logs")

            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            log_file = log_dir / "fileassistant.log"

            file_handler = RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )

            file_formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(log_level)
            self.logger.addHandler(file_handler)

    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """
        Get a logger instance.

        Args:
            name: Optional logger name (creates child logger)

        Returns:
            Logger instance
        """
        if name:
            return self.logger.getChild(name)
        return self.logger

    def print_success(self, message: str):
        """Print a success message with green styling."""
        self.console.print(f"✓ {message}", style="success")

    def print_error(self, message: str):
        """Print an error message with red styling."""
        self.console.print(f"✗ {message}", style="error")

    def print_info(self, message: str):
        """Print an info message with cyan styling."""
        self.console.print(f"ℹ {message}", style="info")

    def print_warning(self, message: str):
        """Print a warning message with yellow styling."""
        self.console.print(f"⚠ {message}", style="warning")


# Global logger instance
_logger_instance: Optional[FileAssistantLogger] = None


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Optional logger name for component-specific logging

    Returns:
        Configured logger instance
    """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = FileAssistantLogger()
        # Set up with defaults if not already configured
        _logger_instance.setup()
    return _logger_instance.get_logger(name)


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    console_enabled: bool = True,
    file_enabled: bool = True,
):
    """
    Configure global logging settings.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        max_bytes: Maximum size of each log file before rotation
        backup_count: Number of rotated log files to keep
        console_enabled: Enable console (Rich) logging
        file_enabled: Enable file logging
    """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = FileAssistantLogger()
    _logger_instance.setup(level, log_dir, max_bytes, backup_count, console_enabled, file_enabled)


def get_console() -> Console:
    """Get the Rich console instance for custom printing."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = FileAssistantLogger()
    return _logger_instance.console
