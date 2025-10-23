"""
Logging configuration for brahe.

This module provides a unified logging interface using loguru for both CLI and
programmatic usage. Logs are formatted with colors and timestamps for easy reading.

Example:
    Configure logging programmatically::

        import brahe.logging
        brahe.logging.configure(level="DEBUG")

    CLI usage::

        # Default (WARNING level)
        brahe access compute 25544 --lat 40.7 --lon -74.0

        # Verbose (INFO level)
        brahe --verbose access compute 25544 --lat 40.7 --lon -74.0

        # Debug (DEBUG level)
        brahe --debug access compute 25544 --lat 40.7 --lon -74.0
"""

import sys
from loguru import logger

# Remove default handler to prevent duplicate logs
logger.remove()

# Track if we've configured logging to prevent duplicate configuration
_configured = False


def configure(level: str = "WARNING") -> None:
    """
    Configure logging for programmatic use.

    This function sets up loguru logging with standardized formatting for
    library users who want to see brahe's internal logging output.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Default is WARNING.

    Example:
        ```python
        import brahe
        import brahe.logging

        # Enable debug logging to see internal operations
        brahe.logging.configure(level="DEBUG")

        # Now use brahe - you'll see debug logs
        epoch = brahe.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0)
        ```

    Note:
        This configures logging globally. Call this early in your program,
        before importing other brahe modules if you want to capture all logs.
    """
    global _configured

    if _configured:
        # Remove existing handlers and reconfigure
        logger.remove()

    logger.add(
        sys.stderr,
        level=level.upper(),
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> <level>{message}</level>",
    )

    _configured = True
    logger.debug(f"Logging configured at {level.upper()} level")


def setup_cli_logging(verbose: bool = False, debug: bool = False) -> None:
    """
    Configure logging for CLI usage.

    This is called by the CLI entry point to set up logging based on
    command-line flags. Should not be called by library users.

    Args:
        verbose: If True, set log level to INFO
        debug: If True, set log level to DEBUG (overrides verbose)

    Note:
        Log levels:
        - Default (neither flag): WARNING
        - --verbose: INFO
        - --debug: DEBUG
    """
    if debug:
        level = "DEBUG"
    elif verbose:
        level = "INFO"
    else:
        level = "WARNING"

    configure(level=level)


__all__ = ["configure", "setup_cli_logging", "logger"]
