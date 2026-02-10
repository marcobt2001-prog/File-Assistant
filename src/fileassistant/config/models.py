"""Configuration models using Pydantic for validation."""

from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class ConfidenceThresholds(BaseModel):
    """Confidence thresholds for classification decisions."""

    high: float = Field(default=0.9, ge=0.0, le=1.0, description="High confidence threshold")
    medium: float = Field(default=0.6, ge=0.0, le=1.0, description="Medium confidence threshold")
    low: float = Field(default=0.0, ge=0.0, le=1.0, description="Low confidence threshold")

    @field_validator("medium")
    @classmethod
    def medium_less_than_high(cls, v: float, info) -> float:
        """Ensure medium threshold is less than high threshold."""
        if "high" in info.data and v >= info.data["high"]:
            raise ValueError("medium threshold must be less than high threshold")
        return v


class ProcessingSettings(BaseModel):
    """Settings for file processing behavior."""

    idle_only: bool = Field(default=True, description="Only process files when system is idle")
    debounce_seconds: int = Field(
        default=2, ge=0, description="Wait time after file changes before processing"
    )
    max_file_size_mb: int = Field(
        default=100, ge=1, description="Maximum file size to process (in MB)"
    )
    batch_size: int = Field(default=10, ge=1, description="Number of files to process in one batch")


class AISettings(BaseModel):
    """AI model configuration."""

    model_name: str = Field(
        default="qwen2.5:latest", description="Ollama model name for classification"
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", description="Sentence transformer model for embeddings"
    )
    temperature: float = Field(
        default=0.1, ge=0.0, le=2.0, description="LLM temperature for classification"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Ollama API base URL"
    )
    max_retries: int = Field(default=3, ge=1, description="Maximum AI API retry attempts")


class LoggingSettings(BaseModel):
    """Logging configuration."""

    level: str = Field(
        default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    log_dir: Path = Field(default=Path("logs"), description="Directory for log files")
    max_bytes: int = Field(
        default=10 * 1024 * 1024, ge=1024, description="Max log file size before rotation (bytes)"
    )
    backup_count: int = Field(default=5, ge=1, description="Number of rotated log files to keep")
    console_enabled: bool = Field(default=True, description="Enable console logging")
    file_enabled: bool = Field(default=True, description="Enable file logging")

    @field_validator("level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is valid."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v_upper


class DatabaseSettings(BaseModel):
    """Database configuration."""

    path: Path = Field(
        default=Path("data/fileassistant.db"), description="Path to SQLite database file"
    )
    vector_store_path: Path = Field(
        default=Path("data/chromadb"), description="Path to ChromaDB vector store"
    )
    backup_enabled: bool = Field(default=True, description="Enable automatic database backups")
    backup_interval_hours: int = Field(
        default=24, ge=1, description="Hours between automatic backups"
    )


class FileAssistantConfig(BaseModel):
    """Main configuration for FileAssistant."""

    # Core settings
    inbox_folders: list[Path] = Field(
        default_factory=lambda: [
            Path.home() / "Downloads",
            Path.home() / "Desktop",
        ],
        description="Folders to monitor for new files",
    )

    organized_base_path: Path | None = Field(
        default=None, description="Base path for organized files (defaults to user's Documents)"
    )

    # Component settings
    confidence_thresholds: ConfidenceThresholds = Field(
        default_factory=ConfidenceThresholds, description="Classification confidence thresholds"
    )

    processing: ProcessingSettings = Field(
        default_factory=ProcessingSettings, description="File processing settings"
    )

    ai_settings: AISettings = Field(default_factory=AISettings, description="AI model settings")

    logging: LoggingSettings = Field(
        default_factory=LoggingSettings, description="Logging settings"
    )

    database: DatabaseSettings = Field(
        default_factory=DatabaseSettings, description="Database settings"
    )

    # Folder context for classification
    scan_folders_for_context: list[Path] | None = Field(
        default=None,
        description="Additional folders to scan for context (defaults to organized_base_path)",
    )

    folder_scan_depth: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Maximum depth to scan folders for context",
    )

    # Feature flags
    auto_process_enabled: bool = Field(
        default=False, description="Enable automatic file processing (vs. manual approval)"
    )

    learning_enabled: bool = Field(
        default=True, description="Learn from user corrections to improve classification"
    )

    def get_context_folders(self) -> list[Path]:
        """Get the list of folders to scan for classification context."""
        if self.scan_folders_for_context:
            return self.scan_folders_for_context
        # Default to organized_base_path
        return [self.organized_base_path]

    @field_validator("organized_base_path")
    @classmethod
    def set_default_organized_path(cls, v: Path | None) -> Path:
        """Set default organized path to user's Documents folder."""
        if v is None:
            return Path.home() / "Documents" / "FileAssistant"
        return v

    @field_validator("inbox_folders")
    @classmethod
    def validate_inbox_folders(cls, v: list[Path]) -> list[Path]:
        """Ensure inbox folders exist or can be created."""
        if not v:
            raise ValueError("At least one inbox folder must be specified")
        return v

    class Config:
        """Pydantic config."""

        validate_assignment = True
        extra = "forbid"  # Raise error on unknown fields
