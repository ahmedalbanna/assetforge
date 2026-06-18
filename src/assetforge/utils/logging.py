"""
Logging configuration for AssetForge.

Sets up structured logging with optional verbose/quiet modes.
"""

import sys
import logging


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure the root logger for AssetForge.

    Args:
        verbose: Enable debug-level logging
        quiet: Suppress all non-error output
    """
    logger = logging.getLogger("assetforge")

    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if verbose:
        fmt = "[%(levelname)s] %(name)s: %(message)s"
    else:
        fmt = "%(message)s"

    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Suppress noisy third-party logs
    if not verbose:
        for noisy in ("urllib3", "PIL", "huggingface_hub", "matplotlib"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
