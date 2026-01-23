"""Configuration module for FileAssistant."""

from .manager import ConfigManager, get_config, get_config_manager
from .models import (
    AISettings,
    ConfidenceThresholds,
    DatabaseSettings,
    FileAssistantConfig,
    LoggingSettings,
    ProcessingSettings,
)

__all__ = [
    "FileAssistantConfig",
    "ConfidenceThresholds",
    "ProcessingSettings",
    "AISettings",
    "LoggingSettings",
    "DatabaseSettings",
    "ConfigManager",
    "get_config",
    "get_config_manager",
]
