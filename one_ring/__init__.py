"""
The One Ring to rule them all, One Ring to find them,
One Ring to bring them all, and in the darkness bind them.

This is the main package for the AI Fine-Tuning Platform.
"""

__version__ = "0.1.0"
__author__ = "Your Name <your.email@example.com>"

# Import core components to make them available at package level
from one_ring.core.config import settings
from one_ring.core.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Create necessary directories on import
settings.setup_directories()

# Log startup message
logger.info("The One Ring awakens...")

# Clean up when the application exits
import atexit
from one_ring.utils.cleanup import cleanup_on_exit

atexit.register(cleanup_on_exit)
