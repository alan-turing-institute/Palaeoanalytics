"""
Console and file logging for page segmentation.

Mirrors the arrangement the analysis command uses, so both tools behave
the same way from a user's point of view: a readable, styled summary on
screen and a complete trace on disk.

The two handlers are filtered independently. The console shows INFO by
default, keeping per-page output to one line each. The file always
records DEBUG, because a split that looks wrong weeks later needs to be
explainable without reproducing the run — and on a batch of hundreds of
plates, reproducing one page's failure may not be possible.

The ``Console`` built here is shared with the progress display so live
output and log lines render cooperatively rather than overwriting one
another.
"""

import logging
import os
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

LOG_FILENAME = 'pylithics-pages.log'

# Marks handlers this module installed, so repeat calls replace only
# their own and leave any host or test handlers untouched.
_OWNED_FLAG = '_pylithics_pages_handler'

# Third-party DEBUG output buries the segmentation trace: PIL logs every
# PNG chunk it reads, matplotlib scores every font on the system.
_NOISY_LOGGERS = ('PIL', 'matplotlib', 'fontTools', 'asyncio')


def setup_logging(
    console_level: int = logging.INFO,
    output_dir: Optional[str] = None,
) -> Console:
    """
    Configure styled console logging and a full-trace log file.

    Parameters
    ----------
    console_level : int
        Level for terminal output. The log file is unaffected and always
        records DEBUG.
    output_dir : str, optional
        Directory to write ``pylithics-pages.log`` into. When omitted no
        file handler is added, so running from an arbitrary directory
        does not scatter log files.

    Returns
    -------
    Console
        The console to share with any progress display.
    """
    logger = logging.getLogger()
    _reset_handlers(logger)
    logger.setLevel(logging.DEBUG)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    console = Console()
    handler = RichHandler(
        level=console_level,
        console=console,
        show_time=False,
        show_path=False,
        rich_tracebacks=True,
        markup=False,
    )
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(_own(handler))

    if output_dir:
        _add_file_handler(logger, output_dir)
    return console


def _reset_handlers(logger: logging.Logger) -> None:
    """
    Remove handlers left by an earlier call to this function.

    Without this, invoking the CLI twice in one process would duplicate
    every line once per previous run. Only handlers installed here are
    removed: anything the surrounding process attached — a test
    framework's capture handler, or a host application's own logging —
    is left in place.
    """
    for handler in list(logger.handlers):
        if getattr(handler, _OWNED_FLAG, False):
            logger.removeHandler(handler)
            handler.close()


def _own(handler: logging.Handler) -> logging.Handler:
    """Mark a handler as ours, so a later call knows to replace it."""
    setattr(handler, _OWNED_FLAG, True)
    return handler


def _add_file_handler(logger: logging.Logger, output_dir: str) -> None:
    """
    Attach a plain-text file handler recording the full trace.

    Failure to create the file is not fatal: a read-only destination
    should not stop a run whose output is going somewhere else, and
    console logging is already in place.
    """
    path = os.path.join(output_dir, LOG_FILENAME)
    try:
        os.makedirs(output_dir, exist_ok=True)
        handler = logging.FileHandler(path)
    except OSError as exc:
        logging.warning("Cannot open the log file %s: %s", path, exc)
        return

    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    ))
    logger.addHandler(_own(handler))
