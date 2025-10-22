from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from src.utils.logger import get_logger, setup_logger


class ConfigManager:
    _instance: "ConfigManager | None" = None
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    _defaults: Dict[str, Any] = {
        "download_directory": str((PROJECT_ROOT / "downloads").resolve()),
        "ollama_path": "",
        "theme": "System",
        "log_level": "INFO",
    }

    def __new__(cls) -> "ConfigManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "_initialized") and self._initialized:
            return
        self.logger = get_logger(__name__)
        self.config_file_path = self._determine_config_path()
        self.config: Dict[str, Any] = dict(self._defaults)
        self.load_config()
        setup_logger(__name__, self.config.get("log_level", "INFO"))
        self._initialized = True

    def _determine_config_path(self) -> Path:
        appdata = os.getenv("APPDATA")
        if appdata:
            config_dir = Path(appdata) / "OllamaReformat"
        else:
            config_dir = Path.home() / ".ollama_reformat"
        if not config_dir.exists():
            try:
                config_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                self.logger.warning("Failed to create config directory %s: %s", config_dir, exc)
                config_dir = self.PROJECT_ROOT
        return config_dir / "config.json"

    def load_config(self) -> None:
        if not self.config_file_path.exists():
            self.logger.info("Configuration file not found, using defaults")
            self.save_config()
            return
        try:
            with self.config_file_path.open("r", encoding="utf-8") as config_file:
                data = json.load(config_file)
            merged = dict(self._defaults)
            if isinstance(data, dict):
                merged.update({k: v for k, v in data.items() if k in merged})
            self.config = merged
            self.logger.info("Configuration loaded from %s", self.config_file_path)
        except (json.JSONDecodeError, OSError) as exc:
            self.logger.error("Failed to load config, reverting to defaults: %s", exc)
            self.config = dict(self._defaults)
            self.save_config()

    def save_config(self) -> None:
        try:
            self.config_file_path.parent.mkdir(parents=True, exist_ok=True)
            with self.config_file_path.open("w", encoding="utf-8") as config_file:
                json.dump(self.config, config_file, indent=2)
            self.logger.info("Configuration saved to %s", self.config_file_path)
        except OSError as exc:
            self.logger.error("Failed to save configuration: %s", exc)

    def get(self, key: str, default: Any | None = None) -> Any:
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        if key in self._defaults:
            self.config[key] = value
            if key == "log_level":
                setup_logger(__name__, value)
        else:
            self.logger.warning("Attempt to set unknown config key: %s", key)

    def reset_to_defaults(self) -> None:
        self.config = dict(self._defaults)
        self.save_config()
        self.logger.info("Configuration reset to defaults")
*** End***
