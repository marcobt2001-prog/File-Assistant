"""Database models and ORM operations."""

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .schema import Base


class Database:
    """Database connection and session management."""

    def __init__(self, db_path: Path):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create engine with SQLite
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            echo=False,  # Set to True for SQL debugging
            connect_args={"check_same_thread": False},  # Needed for SQLite
        )

        # Create session factory
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def create_all_tables(self):
        """Create all tables if they don't exist."""
        Base.metadata.create_all(bind=self.engine)

    def drop_all_tables(self):
        """Drop all tables (use with caution!)."""
        Base.metadata.drop_all(bind=self.engine)

    def get_session(self) -> Session:
        """
        Get a new database session.

        Returns:
            SQLAlchemy session
        """
        return self.SessionLocal()

    def close(self):
        """Close database connection."""
        self.engine.dispose()


# Global database instance
_database: Database | None = None


def get_database(db_path: Path | None = None) -> Database:
    """
    Get global database instance.

    Args:
        db_path: Path to database file (required on first call)

    Returns:
        Database instance
    """
    global _database
    if _database is None:
        if db_path is None:
            raise ValueError("db_path must be provided on first call to get_database()")
        _database = Database(db_path)
    return _database


def get_session() -> Session:
    """
    Get a database session from the global database instance.

    Returns:
        SQLAlchemy session
    """
    db = get_database()
    return db.get_session()
