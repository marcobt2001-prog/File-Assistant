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

__all__ = ["get_config", "get_database", "get_logger"]
