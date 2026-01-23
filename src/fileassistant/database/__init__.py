"""Database module for FileAssistant."""

from .migrations import Migration, MigrationManager, initialize_migrations
from .models import Database, get_database, get_session
from .schema import (
    Action,
    ActionType,
    Classification,
    ClassificationStatus,
    Correction,
    File,
    FileStatus,
    FileTag,
    Preference,
    Rule,
    SchemaVersion,
    Tag,
)

__all__ = [
    # Models
    "File",
    "Tag",
    "FileTag",
    "Classification",
    "Action",
    "Rule",
    "Preference",
    "Correction",
    "SchemaVersion",
    # Enums
    "FileStatus",
    "ClassificationStatus",
    "ActionType",
    # Database
    "Database",
    "get_database",
    "get_session",
    # Migrations
    "Migration",
    "MigrationManager",
    "initialize_migrations",
]
