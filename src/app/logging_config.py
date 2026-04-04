from __future__ import annotations

import logging

from .config import AppConfig


def configure_logging(config: AppConfig) -> None:
    """
    Configure the logging system based on the provided application configuration.

    Args:
        config: An instance of AppConfig containing logging settings such as log level and format.
    """
    level = getattr(logging, config.log_level, logging.INFO)
    logging.basicConfig(level=level, format=config.log_format, force=True)
