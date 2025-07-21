"""
The One Ring - Main Entry Point

"One Ring to rule them all, One Ring to find them,
One Ring to bring them all and in the darkness bind them."

This is the main entry point for the One Ring AI Fine-Tuning Platform.
"""

import argparse
import logging
import sys
from pathlib import Path

from loguru import logger

from one_ring import __version__
from one_ring.core.config import settings
from one_ring.core.hardware import hardware_manager
from one_ring.core.logger import configure_logger
from one_ring.utils.cleanup import cleanup_resources


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="One Ring - AI Fine-Tuning Platform")
    
    # General options
    parser.add_argument(
        "--version", 
        action="version", 
        version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode"
    )
    
    # Logging options
    log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    parser.add_argument(
        "--log-level", 
        choices=log_levels, 
        default="INFO",
        help="Set the logging level"
    )
    parser.add_argument(
        "--log-file", 
        type=Path,
        help="Path to the log file (default: logs/one_ring_<timestamp>.log)"
    )
    
    # Hardware options
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "rocm", "mps"],
        default="auto",
        help="Device to use for computation (default: auto-detect)"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the web interface")
    serve_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on (default: 7860)"
    )
    serve_parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public link for the interface"
    )
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "config",
        type=Path,
        help="Path to the training configuration file"
    )
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save the trained model (default: models/)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the One Ring application."""
    # Parse command line arguments
    args = parse_args()
    
    # Configure logging
    log_file = args.log_file or settings.LOGS_DIR / f"one_ring_{__version__.replace('.', '_')}.log"
    configure_logger(
        log_level=args.log_level,
        log_file=log_file,
    )
    
    # Log startup information
    logger.info(f"Starting One Ring v{__version__}")
    logger.debug(f"Command line arguments: {sys.argv}")
    logger.debug(f"Running on {sys.platform} with Python {sys.version}")
    
    # Log hardware information
    logger.info(f"Using {hardware_manager.info.device_type.name} with {hardware_manager.info.device_count} devices")
    
    try:
        # Execute the requested command
        if args.command == "serve":
            from one_ring.app.serve import serve_web_interface
            serve_web_interface(args.host, args.port, args.share)
        elif args.command == "train":
            from one_ring.training.trainer import train_model
            train_model(args.config, args.output_dir)
        else:
            # If no command is provided, start the web interface by default
            from one_ring.app.serve import serve_web_interface
            serve_web_interface()
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Ensure all resources are cleaned up
        cleanup_resources()


if __name__ == "__main__":
    main()
