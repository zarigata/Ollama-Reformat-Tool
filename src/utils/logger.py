from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "app.log"

_configured_loggers: Dict[str, logging.Logger] = {}


def _coerce_log_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        normalized = level.upper()
        if normalized in logging._nameToLevel:
            return logging._nameToLevel[normalized]
    return logging.INFO


def setup_logger(name: str, log_level: str | int = "INFO") -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    level = _coerce_log_level(log_level)
    logger.setLevel(level)
    logger.propagate = False

    if name not in _configured_loggers:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.handlers.clear()
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        _configured_loggers[name] = logger
    else:
        logger.setLevel(level)
    return logger


def get_logger(name: str) -> logging.Logger:
    if name not in _configured_loggers:
        return setup_logger(name)
    return logging.getLogger(name)
