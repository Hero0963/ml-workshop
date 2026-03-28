import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Get a pre-configured logger for the project."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(message)s"  # Simplified for clean hierarchical tree output
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
