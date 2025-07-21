"""
Configuration settings for the One Ring platform.

This module handles all configuration settings, including environment variables,
file paths, and default values. It's the heart of the application's configuration.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    """
    Application settings and configuration.
    
    These settings can be overridden using environment variables with the prefix 'ONE_RING_'.
    For example, to set the log level: ONE_RING_LOG_LEVEL=DEBUG
    """
    
    # Application settings
    APP_NAME: str = "One Ring to Tune Them All"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # File paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = DATA_DIR / "models"
    DATASETS_DIR: Path = DATA_DIR / "datasets"
    LOGS_DIR: Path = BASE_DIR / "logs"
    CACHE_DIR: Path = BASE_DIR / ".cache"
    
    # Model settings
    DEFAULT_MODEL: str = "meta-llama/Llama-2-7b-chat-hf"
    MODEL_CACHE_DIR: Path = MODELS_DIR / "cache"
    MODEL_SAVE_DIR: Path = MODELS_DIR / "saved"
    
    # Training settings
    BATCH_SIZE: int = 4
    GRADIENT_ACCUMULATION_STEPS: int = 4
    LEARNING_RATE: float = 2e-5
    NUM_TRAIN_EPOCHS: int = 3
    MAX_SEQ_LENGTH: int = 2048
    
    # Hardware settings
    USE_CUDA: bool = True
    USE_MPS: bool = False  # For Apple Silicon
    USE_CPU: bool = False
    
    # Ollama settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_TIMEOUT: int = 300
    
    # Security
    SECRET_KEY: str = "your-secret-key-here"  # Change in production!
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    
    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "One Ring to Tune Them All"
    
    # CORS settings
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    class Config:
        env_prefix = "ONE_RING_"
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        """Parse CORS origins from a comma-separated string."""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    def setup_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.DATA_DIR,
            self.MODELS_DIR,
            self.DATASETS_DIR,
            self.LOGS_DIR,
            self.CACHE_DIR,
            self.MODEL_CACHE_DIR,
            self.MODEL_SAVE_DIR,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / ".gitkeep").touch(exist_ok=True)


# Create settings instance
settings = Settings()

# Export settings for easy access
__all__ = ["settings"]
