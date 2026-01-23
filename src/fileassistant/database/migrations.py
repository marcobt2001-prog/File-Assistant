"""Simple migration system for database schema updates."""

from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..utils.logging import get_logger
from .models import Database
from .schema import SchemaVersion

logger = get_logger(__name__)


class Migration:
    """Represents a single database migration."""

    def __init__(
        self,
        version: int,
        description: str,
        up: Callable[[Session], None],
        down: Optional[Callable[[Session], None]] = None,
    ):
        """
        Initialize migration.

        Args:
            version: Migration version number (sequential)
            description: Human-readable description
            up: Function to apply migration
            down: Optional function to rollback migration
        """
        self.version = version
        self.description = description
        self.up = up
        self.down = down


class MigrationManager:
    """Manages database migrations."""

    def __init__(self, db: Database):
        """
        Initialize migration manager.

        Args:
            db: Database instance
        """
        self.db = db
        self.migrations: List[Migration] = []

    def register(self, migration: Migration):
        """Register a migration."""
        self.migrations.append(migration)
        # Keep migrations sorted by version
        self.migrations.sort(key=lambda m: m.version)

    def get_current_version(self, session: Session) -> int:
        """
        Get current schema version from database.

        Args:
            session: Database session

        Returns:
            Current version number (0 if no migrations applied)
        """
        try:
            latest = (
                session.query(SchemaVersion).order_by(SchemaVersion.version.desc()).first()
            )
            return latest.version if latest else 0
        except Exception:
            # Table might not exist yet
            return 0

    def apply_migrations(self, target_version: Optional[int] = None):
        """
        Apply pending migrations.

        Args:
            target_version: Target version to migrate to (None = latest)
        """
        session = self.db.get_session()

        try:
            # Ensure schema_version table exists
            self.db.create_all_tables()

            current_version = self.get_current_version(session)
            logger.info(f"Current database version: {current_version}")

            # Determine target version
            if target_version is None:
                target_version = self.migrations[-1].version if self.migrations else 0

            # Filter migrations to apply
            pending_migrations = [
                m for m in self.migrations if current_version < m.version <= target_version
            ]

            if not pending_migrations:
                logger.info("No pending migrations")
                return

            logger.info(f"Applying {len(pending_migrations)} migration(s)")

            # Apply each migration
            for migration in pending_migrations:
                logger.info(
                    f"Applying migration {migration.version}: {migration.description}"
                )

                try:
                    # Run migration
                    migration.up(session)

                    # Record migration
                    version_record = SchemaVersion(
                        version=migration.version,
                        description=migration.description,
                        applied_at=datetime.utcnow(),
                    )
                    session.add(version_record)
                    session.commit()

                    logger.info(f"Successfully applied migration {migration.version}")

                except Exception as e:
                    logger.error(f"Failed to apply migration {migration.version}: {e}")
                    session.rollback()
                    raise

        finally:
            session.close()

    def rollback(self, target_version: int):
        """
        Rollback migrations to a target version.

        Args:
            target_version: Version to rollback to
        """
        session = self.db.get_session()

        try:
            current_version = self.get_current_version(session)

            if target_version >= current_version:
                logger.info("Nothing to rollback")
                return

            # Get migrations to rollback (in reverse order)
            migrations_to_rollback = [
                m for m in reversed(self.migrations) if target_version < m.version <= current_version
            ]

            logger.info(f"Rolling back {len(migrations_to_rollback)} migration(s)")

            for migration in migrations_to_rollback:
                if migration.down is None:
                    raise ValueError(
                        f"Migration {migration.version} has no down() function - cannot rollback"
                    )

                logger.info(
                    f"Rolling back migration {migration.version}: {migration.description}"
                )

                try:
                    # Run rollback
                    migration.down(session)

                    # Remove version record
                    session.query(SchemaVersion).filter(
                        SchemaVersion.version == migration.version
                    ).delete()
                    session.commit()

                    logger.info(f"Successfully rolled back migration {migration.version}")

                except Exception as e:
                    logger.error(f"Failed to rollback migration {migration.version}: {e}")
                    session.rollback()
                    raise

        finally:
            session.close()


# Define migrations
def create_initial_schema(session: Session):
    """Migration 1: Create initial schema."""
    # Tables are created by SQLAlchemy's create_all()
    # This migration just marks the initial state
    pass


def migration_example_add_index(session: Session):
    """Migration 2: Example - add index to files table."""
    # Example of a future migration
    # session.execute(text("CREATE INDEX idx_files_created_at ON files(created_at)"))
    pass


# Initialize default migrations
def get_default_migrations() -> List[Migration]:
    """Get list of default migrations."""
    return [
        Migration(
            version=1,
            description="Create initial schema",
            up=create_initial_schema,
            down=None,  # Cannot rollback initial schema
        ),
        # Add more migrations here as the schema evolves
    ]


def initialize_migrations(db: Database) -> MigrationManager:
    """
    Initialize migration manager with default migrations.

    Args:
        db: Database instance

    Returns:
        Configured MigrationManager
    """
    manager = MigrationManager(db)

    for migration in get_default_migrations():
        manager.register(migration)

    return manager
