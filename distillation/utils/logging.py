"""Centralized logging configuration.

We use the standard library ``logging`` module (so third-party libs integrate
cleanly) but route console output through :mod:`rich` for readability, and
optionally tee everything to a rotating log file per experiment.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

_CONFIGURED = False
_DEFAULT_FORMAT = "%(name)s | %(message)s"
_FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(
    level: int | str = logging.INFO,
    *,
    log_file: Path | str | None = None,
    force: bool = False,
) -> None:
    """Configure the root logger once for the whole process.

    Args:
        level: Minimum level to emit (name or numeric).
        log_file: Optional path; if given, logs are also written there
            with rotation (5 files x 10 MB).
        force: Reconfigure even if logging was already set up. Useful in tests.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    root = logging.getLogger()
    root.setLevel(level)
    # Clear existing handlers to avoid duplicate lines on re-entry.
    root.handlers.clear()

    console = RichHandler(
        rich_tracebacks=True,
        show_path=False,
        markup=True,
        log_time_format="[%X]",
    )
    console.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
    root.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
        root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger, ensuring logging is configured.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger`.
    """
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
