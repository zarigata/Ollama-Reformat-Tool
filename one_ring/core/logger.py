"""
Logging configuration for the One Ring platform.

This module provides a centralized logging configuration that can be used
throughout the application. It's designed to be both powerful and flexible,
allowing for different log levels and output formats.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger as loguru_logger
from loguru._defaults import LOGURU_FORMAT

from one_ring.core.config import settings


class InterceptHandler(logging.Handler):
    """
    Default handler used to intercept standard logging messages and route them to Loguru.
    
    This handler acts as a bridge between Python's standard logging and Loguru.
    """
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to Loguru."""
        try:
            level = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where the logged message originated
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def configure_logger(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    rotation: str = "10 MB",
    retention: str = "30 days",
    format: str = LOGURU_FORMAT,
) -> None:
    """
    Configure the logger with the specified settings.
    
    Args:
        log_level: The log level to use (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        log_file: Path to the log file. If None, logs will only be output to stderr.
        rotation: When to rotate the log file. Can be a size (e.g., '10 MB') or time (e.g., '1 day').
        retention: How long to keep log files before they're removed.
        format: The log format string.
    """
    # Remove default logger
    loguru_logger.remove()
    
    # Add stderr logger
    loguru_logger.add(
        sys.stderr,
        level=log_level,
        format=format,
        colorize=True,
        enqueue=True,
        backtrace=True,
        diagnose=settings.DEBUG,
    )
    
    # Add file logger if log_file is provided
    if log_file:
        loguru_logger.add(
            str(log_file),
            level=log_level,
            format=format,
            rotation=rotation,
            retention=retention,
            enqueue=True,
            backtrace=True,
            diagnose=settings.DEBUG,
            compression="zip",
        )
    
    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Set log levels for noisy libraries
    for _log in ["uvicorn", "uvicorn.error", "fastapi"]:
        logging.getLogger(_log).handlers = [InterceptHandler()]
        logging.getLogger(_log).propagate = False
    
    # Set log levels for other libraries
    logging.getLogger("httpx").setLevel("WARNING")
    logging.getLogger("httpcore").setLevel("WARNING")
    logging.getLogger("urllib3").setLevel("WARNING")
    logging.getLogger("asyncio").setLevel("WARNING")
    
    # Log configuration
    loguru_logger.info("Logging configured")


def get_logger(name: Optional[str] = None) -> loguru_logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: The name of the logger. If None, the root logger is returned.
        
    Returns:
        A configured logger instance.
    """
    return loguru_logger.bind(name=name)


# Configure logger on import
log_file = settings.LOGS_DIR / f"one_ring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
configure_logger(
    log_level=settings.LOG_LEVEL,
    log_file=log_file,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

# Export the logger for easy access
logger = get_logger(__name__)

# Log startup message
logger.info("The One Ring logger is ready")
logger.debug(f"Debug mode is {'enabled' if settings.DEBUG else 'disabled'}")

__all__ = ["logger", "get_logger", "configure_logger"]
