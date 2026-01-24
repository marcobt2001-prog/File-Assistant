"""Configuration management - loading, validation, and persistence."""

from pathlib import Path

import yaml
from pydantic import ValidationError

from .models import FileAssistantConfig


class ConfigManager:
    """Manages loading and saving configuration."""

    DEFAULT_CONFIG_LOCATIONS = [
        Path("config/default_config.yaml"),
        Path.home() / ".config" / "fileassistant" / "config.yaml",
        Path.home() / ".fileassistant" / "config.yaml",
    ]

    def __init__(self, config_path: Path | None = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Explicit path to config file. If None, searches default locations.
        """
        self.config_path = config_path
        self._config: FileAssistantConfig | None = None

    def load(self, create_if_missing: bool = False) -> FileAssistantConfig:
        """
        Load configuration from file.

        Args:
            create_if_missing: Create default config if no config file found.

        Returns:
            Loaded and validated configuration.

        Raises:
            FileNotFoundError: If no config found and create_if_missing is False.
            ValidationError: If config file is invalid.
        """
        config_file = self._find_config_file()

        if config_file is None:
            if create_if_missing:
                return self._create_default_config()
            raise FileNotFoundError(
                f"No configuration file found. Searched: {self.DEFAULT_CONFIG_LOCATIONS}"
            )

        try:
            with open(config_file, encoding="utf-8") as f:
                config_dict = yaml.safe_load(f) or {}

            self._config = FileAssistantConfig(**config_dict)
            self.config_path = config_file
            return self._config

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file {config_file}: {e}") from e
        except ValidationError as e:
            raise ValueError(f"Invalid configuration in {config_file}: {e}") from e

    def save(self, config: FileAssistantConfig | None = None, path: Path | None = None):
        """
        Save configuration to file.

        Args:
            config: Configuration to save. Uses current config if None.
            path: Path to save to. Uses current config_path if None.
        """
        config_to_save = config or self._config
        if config_to_save is None:
            raise ValueError("No configuration to save")

        save_path = path or self.config_path
        if save_path is None:
            # Default to user config location
            save_path = Path.home() / ".config" / "fileassistant" / "config.yaml"

        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and dump to YAML
        config_dict = config_to_save.model_dump(mode="python")

        # Convert Path objects to strings for YAML serialization
        config_dict = self._paths_to_strings(config_dict)

        with open(save_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                config_dict,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

        self.config_path = save_path
        self._config = config_to_save

    def _find_config_file(self) -> Path | None:
        """Find the first existing config file in default locations."""
        if self.config_path and self.config_path.exists():
            return self.config_path

        for location in self.DEFAULT_CONFIG_LOCATIONS:
            if location.exists():
                return location

        return None

    def _create_default_config(self) -> FileAssistantConfig:
        """Create and return default configuration."""
        default_config = FileAssistantConfig()
        self._config = default_config
        return default_config

    @staticmethod
    def _paths_to_strings(obj):
        """Recursively convert Path objects to strings in a nested dict/list structure."""
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {key: ConfigManager._paths_to_strings(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [ConfigManager._paths_to_strings(item) for item in obj]
        else:
            return obj

    @property
    def config(self) -> FileAssistantConfig:
        """Get current configuration."""
        if self._config is None:
            self._config = self.load(create_if_missing=True)
        return self._config


# Global config instance
_config_manager: ConfigManager | None = None


def get_config_manager(config_path: Path | None = None) -> ConfigManager:
    """
    Get global config manager instance.

    Args:
        config_path: Optional explicit config path.

    Returns:
        ConfigManager instance.
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager


def get_config(reload: bool = False) -> FileAssistantConfig:
    """
    Get current configuration.

    Args:
        reload: Force reload from file.

    Returns:
        Current configuration.
    """
    manager = get_config_manager()
    if reload:
        return manager.load()
    return manager.config
