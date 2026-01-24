"""SQLite database schema for FileAssistant."""

from datetime import datetime
from enum import Enum

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class FileStatus(str, Enum):
    """File processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    ERROR = "error"
    SKIPPED = "skipped"


class ClassificationStatus(str, Enum):
    """Classification decision status."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    MODIFIED = "modified"


class ActionType(str, Enum):
    """Type of file action."""

    MOVE = "move"
    TAG = "tag"
    CREATE_FOLDER = "create_folder"
    DELETE = "delete"
    RENAME = "rename"


class File(Base):
    """File record table."""

    __tablename__ = "files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    path = Column(Text, nullable=False, unique=True, index=True)
    filename = Column(String(255), nullable=False)
    extension = Column(String(50))
    size_bytes = Column(Integer)
    hash_md5 = Column(String(32), index=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    modified_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    processed_at = Column(DateTime)

    # Processing
    status = Column(String(20), default=FileStatus.PENDING, nullable=False)
    content_summary = Column(Text)
    embedding_id = Column(String(255))  # Reference to vector store

    # Relationships
    tags = relationship("FileTag", back_populates="file", cascade="all, delete-orphan")
    classifications = relationship(
        "Classification", back_populates="file", cascade="all, delete-orphan"
    )
    actions = relationship("Action", back_populates="file", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<File(id={self.id}, path='{self.path}', status='{self.status}')>"


class Tag(Base):
    """Tag taxonomy table."""

    __tablename__ = "tags"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True, index=True)
    description = Column(Text)
    color = Column(String(7))  # Hex color code
    parent_tag_id = Column(Integer, ForeignKey("tags.id"))
    auto_generated = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    parent = relationship("Tag", remote_side=[id], backref="children")
    file_tags = relationship("FileTag", back_populates="tag", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Tag(id={self.id}, name='{self.name}')>"


class FileTag(Base):
    """Many-to-many relationship between files and tags."""

    __tablename__ = "file_tags"

    file_id = Column(Integer, ForeignKey("files.id"), primary_key=True)
    tag_id = Column(Integer, ForeignKey("tags.id"), primary_key=True)
    confidence = Column(Float)
    source = Column(String(20))  # 'ai', 'user', 'rule'

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    file = relationship("File", back_populates="tags")
    tag = relationship("Tag", back_populates="file_tags")

    def __repr__(self):
        return (
            f"<FileTag(file_id={self.file_id}, tag_id={self.tag_id}, confidence={self.confidence})>"
        )


class Classification(Base):
    """Classification history table."""

    __tablename__ = "classifications"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_id = Column(Integer, ForeignKey("files.id"), nullable=False)

    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    # AI suggestions
    suggested_destination = Column(Text)
    suggested_tags = Column(JSON)  # JSON array of tag names
    confidence = Column(Float)
    reasoning = Column(Text)

    # User decision
    status = Column(String(20), default=ClassificationStatus.PENDING)
    final_destination = Column(Text)
    final_tags = Column(JSON)  # JSON array of tag names

    # Relationships
    file = relationship("File", back_populates="classifications")

    def __repr__(self):
        return f"<Classification(id={self.id}, file_id={self.file_id}, confidence={self.confidence}, status='{self.status}')>"


class Action(Base):
    """Action log for undo capability."""

    __tablename__ = "actions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    action_type = Column(String(20), nullable=False)
    file_id = Column(Integer, ForeignKey("files.id"))

    # State snapshots (JSON)
    before_state = Column(JSON)
    after_state = Column(JSON)

    # Undo tracking
    undone = Column(Boolean, default=False)
    undone_at = Column(DateTime)

    # Relationships
    file = relationship("File", back_populates="actions")

    def __repr__(self):
        return f"<Action(id={self.id}, type='{self.action_type}', undone={self.undone})>"


class Rule(Base):
    """User-defined rules table."""

    __tablename__ = "rules"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    priority = Column(Integer, default=0)

    # Rule definition (JSON)
    condition_json = Column(JSON, nullable=False)
    action_json = Column(JSON, nullable=False)

    enabled = Column(Boolean, default=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Rule(id={self.id}, name='{self.name}', enabled={self.enabled})>"


class Preference(Base):
    """User preferences key-value store."""

    __tablename__ = "preferences"

    key = Column(String(255), primary_key=True)
    value = Column(Text)

    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Preference(key='{self.key}', value='{self.value}')>"


class Correction(Base):
    """Learning data from user corrections."""

    __tablename__ = "corrections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_id = Column(Integer, ForeignKey("files.id"), nullable=False)

    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Correction data (JSON)
    original_classification = Column(JSON)
    corrected_classification = Column(JSON)

    learned = Column(Boolean, default=False)

    def __repr__(self):
        return f"<Correction(id={self.id}, file_id={self.file_id}, learned={self.learned})>"


class SchemaVersion(Base):
    """Schema version tracking for migrations."""

    __tablename__ = "schema_version"

    version = Column(Integer, primary_key=True)
    applied_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    description = Column(String(255))

    def __repr__(self):
        return f"<SchemaVersion(version={self.version}, applied_at={self.applied_at})>"
