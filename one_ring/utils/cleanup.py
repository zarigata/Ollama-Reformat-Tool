"""
Cleanup utilities for the One Ring platform.

This module provides functionality to clean up resources when the application exits,
ensuring that all resources are properly released and temporary files are removed.
"""

import atexit
import gc
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Set, Optional, Callable

import torch
from loguru import logger

from one_ring.core.config import settings

# Set to store temporary files and directories that need cleanup
_temp_files: Set[Path] = set()
_temp_dirs: Set[Path] = set()
_cleanup_handlers: List[Callable[[], None]] = []


def register_temp_file(file_path: Path) -> None:
    """Register a temporary file for cleanup on exit.
    
    Args:
        file_path: Path to the temporary file.
    """
    _temp_files.add(Path(file_path).resolve())


def register_temp_dir(dir_path: Path) -> None:
    """Register a temporary directory for cleanup on exit.
    
    Args:
        dir_path: Path to the temporary directory.
    """
    _temp_dirs.add(Path(dir_path).resolve())


def create_temp_file(suffix: str = "", prefix: str = "one_ring_", dir: Optional[Path] = None) -> Path:
    """Create a temporary file that will be automatically cleaned up on exit.
    
    Args:
        suffix: File suffix.
        prefix: File prefix.
        dir: Directory to create the file in. If None, uses the system temp directory.
        
    Returns:
        Path to the created temporary file.
    """
    dir_path = Path(dir) if dir else settings.CACHE_DIR / "temp"
    dir_path.mkdir(parents=True, exist_ok=True)
    
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=str(dir_path))
    os.close(fd)  # Close the file descriptor as we'll reopen it later if needed
    
    file_path = Path(path)
    register_temp_file(file_path)
    return file_path


def create_temp_dir(suffix: str = "", prefix: str = "one_ring_", dir: Optional[Path] = None) -> Path:
    """Create a temporary directory that will be automatically cleaned up on exit.
    
    Args:
        suffix: Directory name suffix.
        prefix: Directory name prefix.
        dir: Parent directory. If None, uses the system temp directory.
        
    Returns:
        Path to the created temporary directory.
    """
    dir_path = Path(dir) if dir else settings.CACHE_DIR / "temp"
    dir_path.mkdir(parents=True, exist_ok=True)
    
    temp_dir = Path(tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=str(dir_path)))
    register_temp_dir(temp_dir)
    return temp_dir


def register_cleanup_handler(handler: Callable[[], None]) -> None:
    """Register a cleanup handler to be called on application exit.
    
    Args:
        handler: A callable that takes no arguments and returns None.
    """
    _cleanup_handlers.append(handler)


def cleanup_temp_files() -> None:
    """Clean up all registered temporary files."""
    for file_path in _temp_files:
        try:
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Removed temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to remove temporary file {file_path}: {e}")
    _temp_files.clear()


def cleanup_temp_dirs() -> None:
    """Clean up all registered temporary directories."""
    for dir_path in _temp_dirs:
        try:
            if dir_path.exists() and dir_path.is_dir():
                shutil.rmtree(dir_path, ignore_errors=True)
                logger.debug(f"Removed temporary directory: {dir_path}")
        except Exception as e:
            logger.warning(f"Failed to remove temporary directory {dir_path}: {e}")
    _temp_dirs.clear()


def cleanup_resources() -> None:
    """Clean up all resources."""
    logger.info("Cleaning up resources...")
    
    # Call registered cleanup handlers
    for handler in _cleanup_handlers:
        try:
            handler()
        except Exception as e:
            logger.error(f"Error in cleanup handler: {e}")
    
    # Clean up temporary files and directories
    cleanup_temp_files()
    cleanup_temp_dirs()
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            logger.debug("Cleared CUDA cache")
        except Exception as e:
            logger.warning(f"Failed to clear CUDA cache: {e}")
    
    # Run garbage collection
    gc.collect()
    logger.info("Cleanup complete")


# Register cleanup function to run on exit
def cleanup_on_exit() -> None:
    """Clean up resources when the application exits."""
    cleanup_resources()


# Register the cleanup function to run on normal program termination
atexit.register(cleanup_on_exit)

# Log when the module is loaded
logger.debug("Cleanup utilities initialized")

__all__ = [
    "register_temp_file",
    "register_temp_dir",
    "create_temp_file",
    "create_temp_dir",
    "register_cleanup_handler",
    "cleanup_temp_files",
    "cleanup_temp_dirs",
    "cleanup_resources",
    "cleanup_on_exit"
]
