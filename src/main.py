from __future__ import annotations

import sys

from src.core.config_manager import ConfigManager
from src.gui.app import App
from src.utils.logger import setup_logger


def main() -> None:
    config_manager = ConfigManager()
    log_level = config_manager.get("log_level", "INFO")
    logger = setup_logger("ollama_reformat", log_level)
    try:
        app = App()
        logger.info("Application initialized")
        app.mainloop()
        logger.info("Application closed")
    except Exception as exc:
        logger.critical("Unhandled exception during startup: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
