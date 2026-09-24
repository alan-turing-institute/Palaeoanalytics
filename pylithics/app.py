#!/usr/bin/env python3
"""
PyLithics Application Entry Point
=================================

Configuration management, error handling, and flexible command-line options.
"""

import sys

if len(sys.argv) == 1:
    from pylithics.cli_splash import print_splash
    from pylithics.update_check import check_for_update
    print_splash()
    check_for_update()
    sys.exit(0)

# Print an immediate "Starting…" line so the user knows the CLI is
# alive during the ~2s cold-import phase (Python interpreter +
# pandas/scipy/opencv/matplotlib/rich). Without it the terminal
# looks frozen until the first INFO log line emits — which won't
# happen until every top-level import below has resolved. Written
# to stderr with a carriage return so it can be overwritten by the
# first real log line; stdout stays clean for piping.
#
# Guarded against firing inside multiprocessing-spawn worker
# processes. Each Pool worker re-imports this module, which would
# otherwise have every worker write its own "Starting PyLithics…"
# to the user's terminal, garbling the output. The main entry
# point has ``sys.argv[0]`` pointing at the installed CLI script
# (e.g. ``/.../bin/pylithics``); a spawned worker has it pointing
# at the Python interpreter executing ``-c "from multiprocessing
# .spawn import ..."``.
import os as _os_for_startup_guard
_STARTUP_NOTICE_SHOWN = False
_is_main_cli = (
    _os_for_startup_guard.path.basename(sys.argv[0] or "")
    .startswith("pylithics")
)
if _is_main_cli and sys.stderr.isatty() and not any(
    a in sys.argv for a in ("--help", "-h", "--version", "--docs")
):
    sys.stderr.write("Starting PyLithics…\r")
    sys.stderr.flush()
    _STARTUP_NOTICE_SHOWN = True
del _os_for_startup_guard, _is_main_cli


def _clear_startup_notice() -> None:
    """Erase the "Starting…" line once real output is about to print."""
    if _STARTUP_NOTICE_SHOWN:
        sys.stderr.write("\r" + " " * 30 + "\r")
        sys.stderr.flush()


_EXPLORE_MODE = "--explore" in sys.argv
_EXPLORE_PROGRESS = None


def _start_explore_progress() -> None:
    """Show a rich spinner during slow module imports for ``--explore``."""
    global _EXPLORE_PROGRESS
    from rich.console import Console
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, TimeElapsedColumn,
    )

    console = Console()
    if not console.is_terminal:
        print(
            "Starting the PyLithics dashboard... "
            "(reading the modules and the data).",
            flush=True,
        )
        return

    _EXPLORE_PROGRESS = Progress(
        SpinnerColumn(style="cyan"),
        TextColumn(
            "[cyan]Starting the PyLithics dashboard...[/] "
            "reading the modules and the data"
        ),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )
    _EXPLORE_PROGRESS.start()
    _EXPLORE_PROGRESS.add_task("loading", total=None)


def _stop_explore_progress() -> None:
    """Halt the explore-mode progress spinner and clear its line."""
    if _EXPLORE_PROGRESS is not None:
        _EXPLORE_PROGRESS.stop()


if _EXPLORE_MODE:
    _start_explore_progress()

import argparse
import json
import logging
import os
from datetime import datetime
import subprocess
from PIL import Image
from typing import Optional, Dict, Any, Tuple

from pylithics.image_processing.config import (
    get_config_manager,
    ConfigurationManager,
)
from pylithics.image_processing.importer import (
    execute_preprocessing_pipeline,
    verify_image_dpi_and_scale,
)
from pylithics.image_processing.image_analysis import (
    debug_dir_for,
    process_and_save_contours,
)
from pylithics.image_processing.utils import read_metadata
from pylithics.image_processing.modules.scale_calibration import get_calibration_factor


_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')


# The analysis writes into results/ inside the project folder, beside
# the images/, scales/ and meta_data.csv it reads.
ANALYSIS_OUTPUT_DIRNAME = 'results'
METADATA_FILENAME = 'meta_data.csv'
# A metadata row whose flag column is not empty is analysed like any
# other, and reported at the end, so the user knows which crops
# pylithics-pages could not link to a scale bar or could not name.
_FLAG_COLUMN = 'flag'


def resolve_output_dir(data_dir: str) -> str:
    """
    Return the directory the analysis writes its results into.

    Parameters
    ----------
    data_dir : str
        Directory given on the command line.

    Returns
    -------
    str
        ``<data_dir>/results``.
    """
    return os.path.join(data_dir, ANALYSIS_OUTPUT_DIRNAME)


def default_meta_file(data_dir: str) -> Optional[str]:
    """
    Return ``<data_dir>/meta_data.csv`` when that file is there.

    ``pylithics-pages`` writes the metadata under this name, and a
    project prepared by hand usually follows the same convention, so
    ``--meta_file`` need not be typed.
    """
    candidate = os.path.join(data_dir, METADATA_FILENAME)
    return candidate if os.path.isfile(candidate) else None


def _new_batch_results(
    metadata: list, flagged: list, output_dir: str = ''
) -> Dict[str, Any]:
    """Return the empty results record for a batch."""
    return {
        'success': True,
        'total_images': len(metadata),
        'processed_successfully': 0,
        'failed_images': [],
        'processing_errors': [],
        'flagged': flagged,
        'output_dir': output_dir,
    }


# Debug flags: the configuration key that turns each on, and the folder
# it writes under results/. Named on the screen at the end of a run.
_DEBUG_FOLDERS = (
    ('thresholding', 'debug_output', 'threshold_debug', 'Threshold'),
    ('scale_calibration', 'debug_output', 'scale_debug', 'Scale bar'),
    ('arrow_detection', 'debug_enabled', 'arrow_debug', 'Arrow'),
)


def _ask_yes_no(question: str) -> bool:
    """Put a yes/no question to the person at the terminal. No is the default."""
    try:
        answer = input(question)
    except EOFError:
        return False
    return answer.strip().lower() in ('y', 'yes')


def flagged_rows(metadata: list) -> list:
    """
    List the metadata rows that carry a flag.

    A flag does not stop the analysis: the row runs like any other,
    in pixels when it has no scale value. The flags are reported at
    the end so the user knows which crops to examine in
    ``meta_data.csv``.

    Returns
    -------
    list of dict
        ``{'image_id', 'flag'}`` for each row whose flag is not empty.
    """
    return [
        {'image_id': entry.get('image_id', ''), 'flag': flag}
        for entry in metadata
        for flag in [(entry.get(_FLAG_COLUMN) or '').strip()]
        if flag
    ]


def resolve_images_dir(data_dir: str) -> str:
    """
    Locate the images to analyse within a data directory.

    ``--data_dir`` normally names a project holding ``images/`` and
    ``scales/``. Pointing it straight at a folder of images works too,
    so an existing collection can be analysed where it sits rather than
    being copied or symlinked into the expected layout.

    Parameters
    ----------
    data_dir : str
        Directory given on the command line.

    Returns
    -------
    str
        ``<data_dir>/images`` when that exists, otherwise ``data_dir``.
    """
    nested = os.path.join(data_dir, 'images')
    return nested if os.path.isdir(nested) else data_dir


def _resolve_image_path(images_dir: str, image_id: str) -> Optional[str]:
    """Return the resolved image path, trying common extensions if missing."""
    path = os.path.join(images_dir, image_id)
    if os.path.exists(path):
        return path
    for ext in _IMAGE_EXTENSIONS:
        candidate = os.path.join(images_dir, image_id + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def _parse_scale(scale_value, image_id: str) -> Optional[float]:
    """Parse a metadata scale cell; warn and return None if unusable."""
    try:
        return float(scale_value) if scale_value else None
    except (ValueError, TypeError):
        logging.warning(
            f"The scale for {image_id} is not valid. Pixel measurements are used."
        )
        return None


_RUN_SUMMARY_SCHEMA_VERSION = 2


def _read_image_dpi(image_path: str) -> Optional[float]:
    """
    Extract the DPI from an image file's metadata; ``None`` if absent.

    Mirrors :py:meth:`PyLithicsApplication._extract_image_dpi` but as a
    module-level helper so the manifest writer can use it without an app
    instance.
    """
    try:
        with Image.open(image_path) as img:
            dpi_info = img.info.get('dpi')
            if not dpi_info:
                return None
            return round(float(dpi_info[0]))
    except (OSError, ValueError, TypeError):
        return None


def _calibration_suffix(
    method: str, conversion_factor: Optional[float],
) -> str:
    """Render a one-shot per-image calibration summary suffix."""
    if method == "scale_bar" and conversion_factor:
        return f"{conversion_factor:.2f} px/mm"
    if method == "pixels_detection_failed":
        return "pixels (scale bar not found — see the log)"
    return "pixels (no scale given)"


def _write_run_summary(
    processed_dir: str,
    images_dir: str,
    results: Dict[str, Any],
    metadata: list,
) -> None:
    """
    Write ``results/run_summary.json`` recording the run.

    The dashboard reads this file to populate its data-quality tiles. Each
    successful entry carries the image_id and the source DPI (or ``null`` if
    PIL could not extract it). Failures are listed by image_id with a generic
    reason; the underlying error detail is in ``pylithics.log``.
    """
    failed_ids = set(results.get('failed_images', []) or [])
    successful = []
    for entry in metadata:
        image_id = entry['image_id']
        if image_id in failed_ids:
            continue
        image_path = _resolve_image_path(images_dir, image_id)
        dpi = _read_image_dpi(image_path) if image_path else None
        successful.append({"image_id": image_id, "dpi": dpi})

    failed = [
        {"image_id": image_id, "reason": "Processing failed"}
        for image_id in results.get('failed_images', []) or []
    ]

    summary = {
        "schema_version": _RUN_SUMMARY_SCHEMA_VERSION,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "total_images": results.get('total_images', 0),
        "processed_successfully": results.get('processed_successfully', 0),
        "successful": successful,
        "failed": failed,
        "flagged": results.get('flagged', []),
        "not_analysed": results.get('not_analysed', []),
    }

    summary_path = os.path.join(processed_dir, "run_summary.json")
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logging.debug("Wrote run summary to %s", summary_path)
    except OSError as e:
        logging.warning("Cannot write the run summary: %s", e)


# ---------------------------------------------------------------------------
# Parallel batch processing — module-level helpers
# ---------------------------------------------------------------------------
#
# When --workers > 1, the per-image pipeline runs in a multiprocessing Pool.
# Each worker writes to a unique per-image partial CSV under
# <processed_dir>/_partial/, then the main process concatenates them into
# the canonical processed_metrics.csv. Visualizations and per-lithic JSONs
# are per-image files so they don't need any contention-avoidance treatment.
# These functions must live at module scope because multiprocessing pickles
# them by qualified name when shipping work to child processes.


# Set once per child process by _init_worker; reused across imap_unordered
# work items so we don't re-load config / re-setup matplotlib per image.
_WORKER_APP: Optional["PyLithicsApplication"] = None


def _init_worker(
    config_file: Optional[str],
    data_dir: Optional[str],
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Pool initializer — runs once per child process at spawn.

    Forces matplotlib to its non-GUI Agg backend (must happen before
    any pyplot import in the child) and creates a per-process
    ``PyLithicsApplication`` so each worker has its own
    ConfigurationManager.

    Logging strategy in workers:
      - File handler: kept, at DEBUG level. Worker per-image trace
        lands in pylithics.log just like sequential mode — that's
        where the detailed analysis info belongs. Multiple workers
        append to the same file; lines interleave but every record
        carries a timestamp so chronology is recoverable.
      - Console handler: stripped. Workers must not write to the
        terminal — the main process owns it (progress bar + per-image
        OK/FAIL status). Worker stderr chatter would collide with
        the live progress bar and clutter the user's view.
      - NullHandler added as a tombstone so stdlib logging's
        autoconfigure-on-first-log path doesn't quietly add a default
        StreamHandler back in.
    """
    global _WORKER_APP
    import matplotlib
    matplotlib.use("Agg", force=True)

    _WORKER_APP = PyLithicsApplication(config_file=config_file)
    if config is not None:
        # The main process merged the CLI overrides; use them, not the file.
        _WORKER_APP.config_manager.replace(config)

    # No FileHandler attached at the worker level — that would funnel
    # every worker's events to the same pylithics.log simultaneously
    # and interleave lithic-A's lines with lithic-B's by timestamp.
    # Instead, _worker_process_image opens a per-image FileHandler
    # for the duration of one image; the main process concatenates
    # those per-image partial logs into pylithics.log in metadata
    # order after the pool finishes, so each lithic's debug trace
    # stays grouped (the sequential mode's structure is preserved).
    #
    # data_dir is kept in the signature for forward compatibility
    # but is not currently used here.
    del data_dir  # silence unused-arg linters

    # Strip every console-bound handler attached by PyLithicsApplication
    # setup_logging. Workers must not write to the terminal — the
    # main process owns it (progress bar + per-image OK/FAIL lines).
    from rich.logging import RichHandler
    root = logging.getLogger()
    for handler in list(root.handlers):
        if isinstance(handler, logging.FileHandler):
            continue
        if isinstance(handler, (RichHandler, logging.StreamHandler)):
            root.removeHandler(handler)

    # Tombstone — prevents stdlib's lazy basicConfig from quietly
    # adding a default StreamHandler if any log call fires before
    # _worker_process_image attaches its per-image FileHandler.
    root.addHandler(logging.NullHandler())


def _worker_process_image(args_tuple) -> tuple:
    """Pool worker — process one image; return ``(image_id, success, suffix)``.

    Attaches a per-image FileHandler at ``log_path`` for the duration
    of the call so this lithic's debug events are buffered to their
    own file rather than interleaved with other workers' events in
    the shared pylithics.log. The main process concatenates these
    per-image partials in metadata order after the pool finishes.

    Returns the calibration suffix string (e.g. ``"25.20 px/mm"``) so
    the main process can echo a per-lithic INFO line above the live
    progress bar, matching the look of the sequential TTY flow.
    Suffix is None when the worker fails.

    Exceptions are caught and converted to a False return so a single
    bad image can't kill the whole batch.
    """
    (image_id, scale_mm, images_dir, processed_dir,
     entry, csv_path, log_path) = args_tuple
    global _WORKER_APP

    root = logging.getLogger()
    log_handler = logging.FileHandler(log_path, mode="w")
    log_handler.setLevel(logging.DEBUG)
    log_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(log_handler)

    # Reach back into the worker app's last-image bookkeeping for
    # the calibration suffix. process_single_image already builds
    # this string for its own console log line; capturing it via an
    # attribute set on the app instance avoids re-deriving it here
    # and keeps the suffix format authoritative in one place.
    _WORKER_APP._last_calibration_suffix = None
    try:
        success = _WORKER_APP.process_single_image(
            image_id, scale_mm, images_dir, processed_dir, entry,
            csv_path=csv_path,
        )
    except Exception:
        logging.exception("Unhandled exception processing %s", image_id)
        success = False
    finally:
        log_handler.flush()
        root.removeHandler(log_handler)
        log_handler.close()

    suffix = (
        getattr(_WORKER_APP, "_last_calibration_suffix", None)
        if success else None
    )
    return image_id, success, suffix


def _resolve_worker_count(workers_arg, n_images: int) -> int:
    """Parse the ``--workers`` CLI value into a concrete worker count.

    ``auto`` (or ``None``) → ``min(cpu_count - 1, n_images, 8)`` with a
    floor of 1. A positive integer is taken at face value but still
    capped by ``n_images`` (no point spawning more workers than there
    is work). Invalid input falls back to 1 with a warning.
    """
    cap = 8  # diminishing returns past ~8 workers for IO-bound pipelines
    if workers_arg is None or str(workers_arg).lower() == "auto":
        cpu = os.cpu_count() or 1
        return max(1, min(cpu - 1, n_images, cap))
    try:
        n = int(workers_arg)
    except (TypeError, ValueError):
        logging.warning(
            "The --workers value %r is not valid. One worker is used.",
            workers_arg,
        )
        return 1
    if n < 1:
        return 1
    return min(n, n_images)


def _concatenate_partial_csvs(processed_dir: str) -> None:
    """Merge per-image partial CSVs into ``processed_metrics.csv``.

    Called by the parallel batch path after the pool has finished.
    Reads every ``*.csv`` in ``<processed_dir>/_partial/``, concatenates
    them into one dataframe preserving column order, writes the final
    CSV with ``na_rep="NA"`` to keep the file format identical to the
    sequential path, then removes the partials.
    """
    import glob
    import pandas as pd

    partial_dir = os.path.join(processed_dir, "_partial")
    final_path = os.path.join(processed_dir, "processed_metrics.csv")
    partials = sorted(glob.glob(os.path.join(partial_dir, "*.csv")))
    if not partials:
        return

    frames = []
    for p in partials:
        try:
            frames.append(pd.read_csv(p, na_values=["NA"]))
        except Exception:
            logging.exception("Could not read partial CSV %s", p)

    if not frames:
        return

    merged = pd.concat(frames, ignore_index=True, sort=False)

    # Re-coerce count-like columns to nullable Int64. pd.read_csv
    # promotes them to float64 here because every partial has some
    # NaN rows (non-dorsal surfaces) — without this, scar_count and
    # voronoi_num_cells render as "10.0" instead of "10".
    from pylithics.image_processing.modules.visualization import (
        _coerce_integer_columns,
    )
    merged = _coerce_integer_columns(merged)

    if os.path.exists(final_path):
        os.remove(final_path)
    merged.to_csv(final_path, index=False, na_rep="NA")

    for p in partials:
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.rmdir(partial_dir)
    except OSError:
        pass


def _concatenate_partial_logs(
    processed_dir: str, ordered_image_ids: list,
) -> None:
    """Append per-image worker logs to the main pylithics.log in order.

    Each worker writes its lithic's debug trace to
    ``<processed>/_partial/<image_id>.log``. After the pool finishes,
    this walks ``ordered_image_ids`` (metadata order — same order
    sequential mode would produce) and appends each partial's contents
    to ``pylithics.log``, then deletes the partial. This restores the
    per-lithic grouping in the log file: image A's full trace, then
    image B's full trace, instead of A and B interleaved by timestamp.

    Main-process log lines written before the pool ran (config, batch
    start) stay at the top of pylithics.log; lines written after the
    pool ran (per-image OK/FAIL summary, totals) come naturally after
    the per-image blocks since they're appended later by the running
    main process's FileHandler.
    """
    partial_dir = os.path.join(processed_dir, "_partial")
    if not os.path.isdir(partial_dir):
        return
    main_log = os.path.join(processed_dir, "pylithics.log")

    # Flush the main-process FileHandler first so the boundary
    # between pre-pool main lines and the appended per-image blocks
    # is stable in the file before we write into it directly.
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                handler.flush()
            except Exception:
                pass

    with open(main_log, "a", encoding="utf-8") as out:
        for image_id in ordered_image_ids:
            partial = os.path.join(partial_dir, f"{image_id}.log")
            if not os.path.exists(partial):
                continue
            try:
                with open(partial, "r", encoding="utf-8") as f:
                    out.write(f.read())
            except OSError:
                logging.exception(
                    "Could not read partial log %s", partial,
                )
                continue
            try:
                os.remove(partial)
            except OSError:
                pass

    # _partial may still hold the directory itself; _concatenate_partial_csvs
    # tries to rmdir it too, so this is a no-op if CSV concat ran first.
    try:
        os.rmdir(partial_dir)
    except OSError:
        pass


class PyLithicsApplication:
    """
    Main application class for PyLithics with enhanced functionality.
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the PyLithics application.

        Parameters
        ----------
        config_file : str, optional
            Path to configuration file
        """
        self.config_file = config_file
        self.workers_arg = "auto"
        self.config_manager = get_config_manager(config_file)
        self.setup_logging()

    def setup_logging(self, data_dir: Optional[str] = None) -> None:
        """Set up logging configuration from config manager.

        Console output goes through ``rich.logging.RichHandler`` for
        coloured level icons and syntax-highlighted tracebacks. The file
        handler keeps a plain text format for grep-friendly logs.

        Default split:
            - Console: INFO (concise — shows per-image summaries, warnings,
              errors). ``--verbose`` flips this to DEBUG.
            - File: always at DEBUG so the full per-step trace is preserved
              for reproducibility regardless of console verbosity.

        Parameters
        ----------
        data_dir : str, optional
            When provided and the config has no explicit ``logging.log_file``
            entry, the log file is written to
            ``<data_dir>/results/pylithics.log``. This avoids
            creating stray ``results/`` folder trees wherever the
            user happens to launch the command.

        The console ``rich.console.Console`` is stored on ``self.rich_console``
        so other code (e.g. the batch-loop progress bar) can share the same
        Console instance — without sharing, ``Progress`` and ``RichHandler``
        collide on stdout and the live bar renders inline with log lines.
        """
        from rich.console import Console
        from rich.logging import RichHandler

        logging_config = self.config_manager.get_section('logging')
        configured_level = logging_config.get('level', 'INFO').upper()
        console_level = logging_config.get(
            'console_level', configured_level,
        )
        if isinstance(console_level, str):
            console_level = console_level.upper()

        # Remove existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logger = logging.getLogger()
        # Root level low enough that all handlers can filter independently.
        logger.setLevel(logging.DEBUG)

        # Suppress noisy third-party DEBUG output so the log file stays
        # focused on lithic-processing events. PIL dumps every PNG chunk;
        # matplotlib logs every font it scores; both bury the actual
        # pipeline trace under hundreds of irrelevant lines.
        for noisy in ("PIL", "matplotlib", "fontTools", "asyncio"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        # Single Console shared with the batch-loop Progress bar so live
        # output and log lines render cooperatively.
        self.rich_console = Console()
        console_handler = RichHandler(
            level=console_level,
            console=self.rich_console,
            show_time=False,
            show_path=False,
            rich_tracebacks=True,
            markup=False,
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console_handler)

        # File handler — plain text, always captures full trace.
        self.log_file_path: Optional[str] = None

        class _ConsoleOnlyFilter(logging.Filter):
            """Drop records flagged ``console_only`` from the file handler."""

            def filter(self, record: logging.LogRecord) -> bool:
                return not getattr(record, "console_only", False)

        if logging_config.get('log_to_file', True):
            # Resolve log file path. Priority:
            #   1. Explicit ``logging.log_file`` in config (honour as-is).
            #   2. Derived from ``data_dir`` when provided
            #      (``<data_dir>/results/pylithics.log``).
            #   3. Otherwise skip the file handler entirely (avoids
            #      creating stray ``results/`` trees in
            #      whatever directory the user happens to launch from).
            log_file = logging_config.get('log_file')
            if not log_file and data_dir:
                log_file = os.path.join(
                    data_dir, ANALYSIS_OUTPUT_DIRNAME, 'pylithics.log',
                )

            if log_file:
                # Best-effort: if the log directory can't be created
                # (read-only filesystem, permission denied, a fake path
                # used during tests, etc.) skip the file handler rather
                # than aborting the run. Console logging is always set
                # up above, so the user still sees output.
                try:
                    log_dir = os.path.dirname(log_file)
                    if log_dir:
                        os.makedirs(log_dir, exist_ok=True)

                    file_handler = logging.FileHandler(log_file)
                except OSError:
                    pass
                else:
                    file_handler.setLevel(logging.DEBUG)
                    file_handler.setFormatter(logging.Formatter(
                        "%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                    ))
                    file_handler.addFilter(_ConsoleOnlyFilter())
                    logger.addHandler(file_handler)
                    self.log_file_path = log_file

    def validate_inputs(self, data_dir: str, meta_file: str) -> bool:
        """
        Validate input parameters.

        Parameters
        ----------
        data_dir : str
            Directory containing images and scale files
        meta_file : str
            Path to the metadata CSV file

        Returns
        -------
        bool
            True if inputs are valid, False otherwise
        """
        # Check data directory
        if not os.path.exists(data_dir):
            logging.error(f"Data directory does not exist: {data_dir}")
            return False

        images_dir = resolve_images_dir(data_dir)
        if not os.path.exists(images_dir):
            logging.error(
                f"There is no images/ directory in {data_dir}, and the "
                f"directory contains no images"
            )
            return False

        # Check metadata file
        if not os.path.exists(meta_file):
            logging.error(f"Metadata file does not exist: {meta_file}")
            return False

        # Validate metadata format
        try:
            metadata = read_metadata(meta_file)
            if not metadata:
                logging.error("The metadata file is empty or not valid")
                return False

            # Check required columns
            required_columns = ['image_id', 'scale']
            first_entry = metadata[0]
            for col in required_columns:
                if col not in first_entry:
                    logging.error(f"Column missing in the metadata: {col}")
                    return False

        except (FileNotFoundError, KeyError, ValueError) as e:
            logging.error(f"Error reading metadata file: {e}")
            return False

        logging.info("The input is correct")
        return True

    def process_single_image(self,
                           image_id: str,
                           real_world_scale_mm: Optional[float],
                           images_dir: str,
                           processed_dir: str,
                           scale_data: Optional[Dict] = None,
                           progress_index: Optional[int] = None,
                           progress_total: Optional[int] = None,
                           csv_path: Optional[str] = None) -> bool:
        """
        Process a single image through the complete pipeline.

        Parameters
        ----------
        image_id : str
            Image identifier
        real_world_scale_mm : float, optional
            Real world scale in millimeters
        images_dir : str
            Directory containing images
        processed_dir : str
            Directory for processed outputs
        scale_data : dict, optional
            Full metadata entry including scale_id for scale calibration
        progress_index, progress_total : int, optional
            1-based image index and total. When supplied and stdout is not
            a TTY, the per-image summary line is prefixed with ``N/TOTAL``
            so CI logs still show per-image progress without a live bar.

        Returns
        -------
        bool
            True if processing succeeded, False otherwise
        """
        image_path = _resolve_image_path(images_dir, image_id)
        if image_path is None:
            logging.error(
                f"Image file does not exist: {os.path.join(images_dir, image_id)}"
            )
            return False

        logging.debug(f"Processing image: {image_id}")

        try:
            processed_image = execute_preprocessing_pipeline(
                image_path, self.config_manager.config,
            )
            if processed_image is None:
                logging.error(f"Preprocessing error for {image_id}")
                return False

            self._write_threshold_debug(processed_image, image_id, processed_dir)
            image_dpi = self._extract_image_dpi(image_path)
            conversion_factor, calibration_method, scale_confidence = (
                self._resolve_calibration(
                    image_path, scale_data or {}, processed_dir
                )
            )

            # Keep the CSV's calibration_method column on the legacy
            # two-value convention ("scale_bar" / "pixels") so downstream
            # analysis scripts and the dashboard's unit_suffix() filter
            # still work. The three-way status survives only as the
            # per-image summary suffix below.
            csv_method = (
                "scale_bar" if calibration_method == "scale_bar" else "pixels"
            )

            process_and_save_contours(
                processed_image,
                conversion_factor,
                processed_dir,
                image_id,
                image_dpi,
                csv_method,
                scale_confidence,
                csv_path=csv_path,
            )

            suffix = _calibration_suffix(calibration_method, conversion_factor)
            # Stash for parallel-mode callers (workers): the pool
            # initializer reads this back via getattr to forward the
            # calibration suffix to the main-process progress display.
            # Harmless in sequential mode — the attribute is just
            # written and never read.
            self._last_calibration_suffix = suffix
            if progress_index is not None and progress_total is not None:
                logging.info(
                    f"{progress_index}/{progress_total} {image_id} · {suffix}"
                )
            else:
                logging.info(f"{image_id} · {suffix}")
            return True

        except (FileNotFoundError, ValueError, IOError) as e:
            logging.error(f"Error processing {image_id}: {e}")
            return False
        except Exception:
            logging.exception(f"Unexpected error processing {image_id}")
            return False

    def _write_threshold_debug(
        self, processed_image, image_id: str, processed_dir: str
    ) -> None:
        """Write the black-and-white image to ``results/threshold_debug/``."""
        if not self.config_manager.get_section('thresholding').get('debug_output'):
            return
        import cv2
        folder = debug_dir_for(processed_dir, 'threshold')
        os.makedirs(folder, exist_ok=True)
        stem = os.path.splitext(image_id)[0]
        cv2.imwrite(os.path.join(folder, f"{stem}.png"), processed_image)

    def _resolve_calibration(
        self, image_path: str, scale_data: Dict, processed_dir: str,
    ) -> "tuple[float, str, Optional[float]]":
        """Get conversion factor with fallback to pixel measurements."""
        conversion_factor, calibration_method, scale_confidence = (
            get_calibration_factor(
                image_path, scale_data, self.config_manager.config,
                debug_dir_for(processed_dir, 'scale'),
            )
        )
        if conversion_factor:
            logging.debug(
                f"Using {calibration_method} calibration: "
                f"{conversion_factor:.3f} pixels/mm"
            )
            return conversion_factor, calibration_method, scale_confidence

        logging.debug("No calibration available, using pixel measurements")
        return 1.0, calibration_method, scale_confidence

    def _extract_image_dpi(self, image_path: str) -> Optional[float]:
        """
        Extract DPI information from image.

        Parameters
        ----------
        image_path : str
            Path to image file

        Returns
        -------
        float or None
            Image DPI if available, None otherwise
        """
        try:
            with Image.open(image_path) as img:
                dpi_info = img.info.get('dpi')
                if dpi_info:
                    image_dpi = round(float(dpi_info[0]))
                    logging.debug(f"Image DPI detected: {image_dpi}")
                    return image_dpi
                else:
                    logging.warning(f"No DPI in {image_path}")
                    return None
        except Exception as e:
            logging.warning(f"Cannot read the DPI from {image_path}: {e}")
            return None

    def run_batch_analysis(
        self, data_dir: str, meta_file: str, show_thresholded_images: bool = False
    ) -> Dict[str, Any]:
        """
        Run batch analysis on all images in the dataset.

        Parameters
        ----------
        data_dir : str
            Directory containing images and scale files
        meta_file : str
            Path to the metadata CSV file
        show_thresholded_images : bool
            Write each thresholded image to ``results/threshold_debug/``.

        Returns
        -------
        dict : processing results summary.
        """
        if not self.validate_inputs(data_dir, meta_file):
            return {'success': False, 'error': 'Input validation failed'}
        if show_thresholded_images:
            self.update_configuration(**{'thresholding.debug_output': True})

        images_dir, processed_dir = self._prepare_dirs(data_dir)
        metadata, not_analysed = self._rows_to_analyse(
            read_metadata(meta_file), meta_file
        )
        results = _new_batch_results(metadata, flagged_rows(metadata), processed_dir)
        results['not_analysed'] = not_analysed
        logging.debug(f"Starting batch processing of {len(metadata)} images")

        workers = _resolve_worker_count(self.workers_arg, len(metadata))
        if not metadata:
            logging.warning("No rows to analyse in %s", meta_file)
        elif workers > 1 and len(metadata) > 1:
            logging.info(
                f"Parallel batch: {workers} worker processes"
            )
            self._run_batch_loop_parallel(
                metadata, images_dir, processed_dir, results, workers,
            )
        else:
            self._run_batch_loop(metadata, images_dir, processed_dir, results)

        self._log_batch_summary(results)
        _write_run_summary(processed_dir, images_dir, results, metadata)

        return results

    @staticmethod
    def _prepare_dirs(data_dir: str) -> Tuple[str, str]:
        """Resolve the images folder and make the results folder."""
        images_dir = resolve_images_dir(data_dir)
        processed_dir = resolve_output_dir(data_dir)
        logging.info(f"Reading images from {images_dir}")
        os.makedirs(processed_dir, exist_ok=True)
        logging.info(f"Output directory: {processed_dir}")
        return images_dir, processed_dir

    def _rows_to_analyse(self, metadata: list, meta_file: str) -> Tuple[list, list]:
        """
        Ask, when a person is there, before measuring in pixels.

        Rows with no scale value are measured in pixels. That is the
        right result when the user wants it and a silent surprise when
        they forgot to fill in the scale, so on a terminal the command
        asks once. A script cannot answer, so it gets a warning and
        continues. ``--force_pixels`` and ``--disable_scale_calibration``
        mean pixels on purpose, so they are not asked.

        Returns
        -------
        tuple of (list, list)
            The rows to analyse, and the image_ids of the rows left out.
        """
        without = [e for e in metadata if not (e.get('scale') or '').strip()]
        calibration = self.config_manager.get_section('scale_calibration')
        if not without or not calibration.get('enabled', True):
            return metadata, []

        notice = f"{len(without)} of {len(metadata)} images have no scale value."
        if not sys.stdin.isatty():
            logging.warning(f"{notice} They are measured in pixels.")
            return metadata, []
        if _ask_yes_no(f"{notice} Measure them in pixels? [y/N] "):
            return metadata, []

        logging.warning(
            f"{len(without)} images with no scale value are not analysed. "
            f"Fill in the scale column of {meta_file}."
        )
        left_out = {id(e) for e in without}
        kept = [e for e in metadata if id(e) not in left_out]
        return kept, [e.get('image_id', '') for e in without]

    def _run_batch_loop(
        self,
        metadata: list,
        images_dir: str,
        processed_dir: str,
        results: Dict[str, Any],
    ) -> None:
        """Iterate the batch with a rich progress bar on TTY, plain on CI."""
        total = len(metadata)
        use_progress = sys.stdout.isatty()

        if use_progress:
            from rich.progress import (
                BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
                TextColumn, TimeElapsedColumn, TimeRemainingColumn,
            )

            with Progress(
                SpinnerColumn(style="cyan"),
                TextColumn("[cyan]Processing[/]"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("[dim]{task.fields[image]}[/]"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.rich_console,
            ) as progress:
                task = progress.add_task(
                    "processing", total=total, image="",
                )
                for i, entry in enumerate(metadata, 1):
                    image_id = entry['image_id']
                    progress.update(task, image=image_id)
                    self._process_one_in_batch(
                        i, total, entry, images_dir, processed_dir, results,
                        include_index_prefix=False,
                    )
                    progress.advance(task)
        else:
            for i, entry in enumerate(metadata, 1):
                self._process_one_in_batch(
                    i, total, entry, images_dir, processed_dir, results,
                    include_index_prefix=True,
                )

    def _run_batch_loop_parallel(
        self,
        metadata: list,
        images_dir: str,
        processed_dir: str,
        results: Dict[str, Any],
        workers: int,
    ) -> None:
        """Dispatch the batch across a multiprocessing Pool of ``workers``.

        Each worker writes its per-image CSV to ``<processed>/_partial/``;
        the main process concatenates them into the canonical
        ``processed_metrics.csv`` after the pool finishes. Progress is
        reported in the main process as workers complete (via
        ``imap_unordered``) so the bar advances in completion order, not
        submission order.
        """
        import glob
        from multiprocessing import Pool

        partial_dir = os.path.join(processed_dir, "_partial")
        os.makedirs(partial_dir, exist_ok=True)
        # Clear any stale partials from a previous interrupted run.
        for stale in glob.glob(os.path.join(partial_dir, "*.csv")):
            try:
                os.remove(stale)
            except OSError:
                pass

        total = len(metadata)
        work = []
        ordered_image_ids = []
        for entry in metadata:
            image_id = entry['image_id']
            scale_mm = _parse_scale(entry.get('scale'), image_id)
            csv_partial = os.path.join(partial_dir, f"{image_id}.csv")
            log_partial = os.path.join(partial_dir, f"{image_id}.log")
            ordered_image_ids.append(image_id)
            work.append((
                image_id, scale_mm, images_dir, processed_dir, entry,
                csv_partial, log_partial,
            ))

        # setup_logging needs data_dir to resolve the FileHandler path.
        # processed_dir is always <data_dir>/results, so walking
        # one directory up recovers it.
        worker_data_dir = os.path.dirname(processed_dir)

        use_progress = sys.stdout.isatty()

        try:
            if use_progress:
                from rich.progress import (
                    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
                    TextColumn, TimeElapsedColumn, TimeRemainingColumn,
                )
                with Progress(
                    SpinnerColumn(style="cyan"),
                    TextColumn(f"[cyan]Processing ({workers} workers)[/]"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TextColumn("[dim]{task.fields[image]}[/]"),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    console=self.rich_console,
                ) as progress:
                    task = progress.add_task(
                        "processing", total=total, image="",
                    )
                    with Pool(
                        processes=workers,
                        initializer=_init_worker,
                        initargs=(
                        self.config_file, worker_data_dir,
                        self.config_manager.config,
                    ),
                    ) as pool:
                        for image_id, success, suffix in pool.imap_unordered(
                            _worker_process_image, work,
                        ):
                            progress.update(task, image=image_id)
                            self._tally_worker_result(
                                image_id, success, results,
                            )
                            # Per-lithic line scrolls above the live
                            # progress bar (RichHandler + Progress
                            # share self.rich_console). Matches the
                            # sequential TTY look.
                            if success and suffix:
                                logging.info(f"{image_id} · {suffix}")
                            elif not success:
                                logging.error(f"{image_id} · ERROR")
                            progress.advance(task)
            else:
                with Pool(
                    processes=workers,
                    initializer=_init_worker,
                    initargs=(
                        self.config_file, worker_data_dir,
                        self.config_manager.config,
                    ),
                ) as pool:
                    for i, (image_id, success, suffix) in enumerate(
                        pool.imap_unordered(_worker_process_image, work), 1,
                    ):
                        self._tally_worker_result(image_id, success, results)
                        if success and suffix:
                            logging.info(
                                f"{i}/{total} {image_id} · {suffix}"
                            )
                        else:
                            prefix = "OK" if success else "ERROR"
                            logging.info(
                                f"{i}/{total} [{prefix}] {image_id}"
                            )
        finally:
            # Order matters: append per-image log blocks to pylithics.log
            # FIRST, then merge CSV partials. The log concat tries to
            # rmdir _partial as a courtesy; CSV concat will also try
            # and the second OSError is swallowed. Running CSV first
            # would delete _partial before the logs get appended.
            _concatenate_partial_logs(processed_dir, ordered_image_ids)
            _concatenate_partial_csvs(processed_dir)

    @staticmethod
    def _tally_worker_result(
        image_id: str, success: bool, results: Dict[str, Any],
    ) -> None:
        """Bump success / failure counters for one parallel result."""
        if success:
            results['processed_successfully'] += 1
        else:
            results['failed_images'].append(image_id)
            results['processing_errors'].append(
                f"Failed to process {image_id}"
            )

    def _process_one_in_batch(
        self,
        index: int,
        total: int,
        entry: Dict,
        images_dir: str,
        processed_dir: str,
        results: Dict[str, Any],
        include_index_prefix: bool,
    ) -> None:
        """Run one image through the pipeline and tally success/failure."""
        image_id = entry['image_id']
        scale_mm = _parse_scale(entry.get('scale'), image_id)
        success = self.process_single_image(
            image_id, scale_mm, images_dir, processed_dir, entry,
            progress_index=index if include_index_prefix else None,
            progress_total=total if include_index_prefix else None,
        )
        if success:
            results['processed_successfully'] += 1
        else:
            results['failed_images'].append(image_id)
            results['processing_errors'].append(
                f"Failed to process {image_id}"
            )

    def _log_batch_summary(self, results: Dict[str, Any]) -> None:
        """Print the end-of-batch summary lines with a pointer to the log."""
        total = results['total_images']
        done = results['processed_successfully']
        log_path = self.log_file_path or "the log file"
        console_only = {"console_only": True}
        if total > 0 and done == total:
            logging.info(f"{done}/{total} images analysed with no errors.")
            logging.info(
                f"See the log at {log_path}", extra=console_only,
            )
        else:
            logging.info(f"{done}/{total} images analysed with no errors.")
            logging.info(
                f"See the log at {log_path} for the errors.",
                extra=console_only,
            )
            if results['failed_images']:
                logging.warning(
                    f"Images with errors: {', '.join(results['failed_images'])}"
                )
        if results.get('not_analysed'):
            logging.warning(
                f"{len(results['not_analysed'])} images with no scale value "
                f"not analysed."
            )
        self._log_flagged_rows(results.get('flagged', []))
        self._log_debug_folders(results.get('output_dir', ''))
        if hasattr(self, 'rich_console'):
            self.rich_console.print()

    def _log_debug_folders(self, output_dir: str) -> None:
        """Name the folder of each debug flag that was on."""
        for section, key, folder, label in _DEBUG_FOLDERS:
            if not self.config_manager.get_section(section).get(key, False):
                continue
            path = os.path.join(output_dir, folder)
            if os.path.isdir(path):
                logging.info(f"{label} debug images: {path}")
            else:
                logging.info(f"No {label.lower()} debug images were written.")

    @staticmethod
    def _log_flagged_rows(flagged: list) -> None:
        """Give one count for each flag on the screen, the rows in the log."""
        if not flagged:
            return
        counts: Dict[str, int] = {}
        for entry in flagged:
            for flag in entry['flag'].split(';'):
                counts[flag] = counts.get(flag, 0) + 1
        summary = ', '.join(f"{flag} {n}" for flag, n in sorted(counts.items()))
        logging.warning(
            f"{len(flagged)} row(s) in the metadata have a flag: {summary}. "
            f"The images were analysed. Examine each row in meta_data.csv."
        )
        for entry in flagged:
            logging.debug(f"  flagged: {entry['image_id']}: {entry['flag']}")

    def update_configuration(self, **kwargs) -> None:
        """
        Update configuration values at runtime.

        Parameters
        ----------
        **kwargs
            Configuration key-value pairs to update
        """
        for key, value in kwargs.items():
            if '.' in key:
                section, config_key = key.split('.', 1)
                self.config_manager.update_value(section, config_key, value)
            else:
                logging.warning(
                    f"The configuration key {key} is not valid. "
                    "Use the format 'section.key'."
                )

        # Do NOT call clear_config_cache() here. The update_value() calls
        # above mutate the cached singleton in place; clearing the cache would
        # cause the next get_config_manager() call to reload from disk and
        # silently discard every override.


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog='PyLithics',
        description='PyLithics v2.0.0: analysis of stone tool images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Related command:\n'
            '  pylithics-pages   Cut the scanned plates in <data_dir>/pages/ into\n'
            '                    one image for each artefact, ready for this command.\n'
            '                    See: pylithics-pages --help\n\n'
            'Use --docs to open the full documentation.'
        )
    )

    _add_required_args(parser)
    _add_config_args(parser)
    _add_processing_args(parser)
    _add_arrow_args(parser)
    _add_scale_args(parser)
    _add_cortex_args(parser)
    _add_scar_args(parser)
    _add_output_args(parser)
    _add_explore_args(parser)
    _add_help_args(parser)

    return parser


def _add_required_args(parser: argparse.ArgumentParser) -> None:
    """Add required argument group."""
    group = parser.add_argument_group('NECESSARY ARGUMENTS')
    group.add_argument(
        '--data_dir', required=False, metavar='PATH',
        help='The project folder that contains images/, scales/ and '
             'meta_data.csv. The results go to <data_dir>/results/.'
    )
    group.add_argument(
        '--meta_file', required=False, metavar='FILE',
        help='The metadata CSV file (columns: image_id, scale_id, scale, '
             'flag). Default: <data_dir>/meta_data.csv. If some rows have '
             'no scale value, the command asks once whether to measure '
             'those images in pixels. Flags are reported.'
    )


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    """Add configuration argument group."""
    group = parser.add_argument_group('CONFIGURATION OPTIONS')
    group.add_argument(
        '--config_file', metavar='FILE',
        help='A YAML configuration file to use in place of the default.'
    )
    group.add_argument(
        '--threshold_method',
        choices=["adaptive", "simple", "otsu", "default"],
        metavar='METHOD',
        help='The threshold method: simple, otsu, adaptive or default.'
    )
    group.add_argument(
        '--log_level',
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        metavar='LEVEL',
        help='The logging level for the screen and the log file '
             '(default: INFO on the screen; the log file always has DEBUG).'
    )
    group.add_argument(
        '--verbose', '-v', action='store_true',
        help='Show the full trace for each step on the screen. This is '
             'the same as --log_level DEBUG for the screen only. The log '
             'file always has the full trace.'
    )


def _add_processing_args(parser: argparse.ArgumentParser) -> None:
    """Add processing options argument group."""
    group = parser.add_argument_group('PROCESSING OPTIONS')
    group.add_argument(
        '--workers', default='auto', metavar='N',
        help='The number of parallel worker processes for a batch. '
             '"auto" (default) uses cpu_count - 1, with a maximum of 8 '
             'and a maximum of the batch size. "1" uses one process; use '
             'it to find a problem with one image. A positive integer '
             'sets the count. The images and the CSV are identical in '
             'parallel and in sequential mode.'
    )
    group.add_argument(
        '--threshold_debug', '--show_thresholded_images', action='store_true',
        dest='show_thresholded_images',
        help='Write the black-and-white image of each lithic to '
             'results/threshold_debug/.'
    )
    group.add_argument(
        '--closing', type=bool, default=True, metavar='BOOL',
        help='Apply morphological closing (default: True).'
    )
    group.add_argument(
        '--enable_dpi_scaling', action='store_true',
        help='Set DPI-aware kernel scaling on for the preprocessing.'
    )
    group.add_argument(
        '--dpi_reference', type=float, metavar='DPI',
        help='The reference DPI for kernel scaling (default: 300.0).'
    )
    group.add_argument(
        '--dpi_max_scale', type=float, metavar='FACTOR',
        help='The maximum DPI scaling factor (default: 1.5).'
    )
    group.add_argument(
        '--dpi_scaling_mode',
        choices=['conservative', 'standard', 'aggressive'],
        metavar='MODE',
        help='The DPI scaling mode (default: standard).'
    )


def _add_arrow_args(parser: argparse.ArgumentParser) -> None:
    """Add arrow detection argument group."""
    group = parser.add_argument_group('ARROW DETECTION OPTIONS')
    group.add_argument(
        '--disable_arrow_detection', action='store_true',
        help='Set arrow detection off.'
    )
    group.add_argument(
        '--arrow_debug', action='store_true',
        help='Write the arrow found in each scar, and the steps, to '
             'results/arrow_debug/<image>/.'
    )
    group.add_argument(
        '--show-arrow-lines', action='store_true',
        help='Draw a red line on each arrow found.'
    )


def _add_scale_args(parser: argparse.ArgumentParser) -> None:
    """Add scale calibration argument group."""
    group = parser.add_argument_group('SCALE CALIBRATION OPTIONS')
    group.add_argument(
        '--disable_scale_calibration', action='store_true',
        help='Set scale bar calibration off.'
    )
    group.add_argument(
        '--scale_debug', action='store_true',
        help='Write the scale image with the bar that was found to '
             'results/scale_debug/.'
    )
    group.add_argument(
        '--force_pixels', action='store_true',
        help='Use pixel measurements only. The command does not ask about '
             'images with no scale value.'
    )


def _add_cortex_args(parser: argparse.ArgumentParser) -> None:
    """Add cortex detection argument group."""
    group = parser.add_argument_group('CORTEX DETECTION OPTIONS')
    group.add_argument(
        '--disable_cortex_detection', action='store_true',
        help='Set cortex detection off.'
    )
    group.add_argument(
        '--cortex_sensitivity', type=str,
        choices=['low', 'medium', 'high'],
        help='The cortex detection sensitivity (default: medium).'
    )


def _add_scar_args(parser: argparse.ArgumentParser) -> None:
    """Add scar complexity argument group."""
    group = parser.add_argument_group('SCAR COMPLEXITY OPTIONS')
    group.add_argument(
        '--disable_scar_complexity', action='store_true',
        help='Set scar complexity analysis off.'
    )
    group.add_argument(
        '--scar_complexity_distance_threshold',
        type=float, metavar='PIXELS',
        help='The adjacency distance in pixels (default: 10.0).'
    )


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    """Add output options argument group."""
    group = parser.add_argument_group('OUTPUT OPTIONS')
    group.add_argument(
        '--export_json', action='store_true',
        help=(
            'Also write one JSON file for each lithic to '
            'results/json/{image_stem}.json.'
        )
    )
    group.add_argument(
        '--save_visualizations', action='store_true',
        default=True,
        help='Write the labelled images (default: True).'
    )


def _add_explore_args(parser: argparse.ArgumentParser) -> None:
    """Add interactive dashboard argument group."""
    group = parser.add_argument_group('EXPLORE OPTIONS')
    group.add_argument(
        '--explore', nargs='?', const=True, default=False, metavar='PATH',
        help=(
            'Open the dashboard. With no PATH: do the analysis, then open '
            'the dashboard for <data_dir>/results/. With PATH: open the '
            'dashboard for that results folder. No analysis.'
        )
    )


def _add_help_args(parser: argparse.ArgumentParser) -> None:
    """Add extended help argument group."""
    group = parser.add_argument_group('EXTENDED HELP OPTIONS')
    group.add_argument(
        '--help-config', action='store_true',
        help='Show the documentation of the configuration file.'
    )
    group.add_argument(
        '--help-examples', action='store_true',
        help='Show examples of use.'
    )
    group.add_argument(
        '--help-troubleshooting', action='store_true',
        help='Show common problems and their procedures.'
    )
    group.add_argument(
        '--docs', action='store_true',
        help='Start the documentation server (http://127.0.0.1:8000).'
    )


def show_config_help() -> None:
    """Display configuration help summary."""
    print("""
    PYLITHICS CONFIGURATION HELP
    ============================

    PyLithics reads a YAML configuration file. To change the settings:
      1. Copy pylithics/config/config.yaml
      2. Change the values that you want
      3. Give the file with --config_file path/to/your/config.yaml

    The main sections: thresholding, arrow_detection, cortex_detection,
    scar_complexity, logging, contour_filtering, data_export

    Full documentation: pylithics --docs
    """)


def show_examples_help() -> None:
    """Display usage examples summary."""
    print("""
    PYLITHICS USAGE EXAMPLES
    ========================

    Basic analysis. The project folder holds images/, scales/ and
    meta_data.csv. The results go to ./project/results/:
      pylithics --data_dir ./project

    A metadata file with a different name or place:
      pylithics --data_dir ./project --meta_file ./metadata.csv

    With Otsu thresholding:
      pylithics --data_dir ./project \\
          --threshold_method otsu

    Arrow detection debug output:
      pylithics --data_dir ./project \\
          --arrow_debug --log_level DEBUG

    A fast batch (no arrow detection):
      pylithics --data_dir ./project \\
          --disable_arrow_detection

    Also write one JSON file for each lithic:
      pylithics --data_dir ./project \\
          --export_json

    Analyse, then open the dashboard:
      pylithics --data_dir ./project --explore

    Open the dashboard for a previous analysis (no new analysis). Give
    the folder that contains processed_metrics.csv:
      pylithics --explore ./project/results

    Full documentation: pylithics --docs
    """)


def show_troubleshooting_help() -> None:
    """Display troubleshooting summary."""
    print("""
    PYLITHICS TROUBLESHOOTING
    =========================

    Common problems:
    - "Directory does not exist": examine the --data_dir path
    - "Column missing": the CSV must have image_id, scale_id, scale
    - "images have no scale value. Measure them in pixels?": y analyses
      all images, in pixels where there is no scale. n analyses only the
      images with a scale value. Fill in the scale column of
      meta_data.csv, then start the command again.
    - "row(s) in the metadata have a flag": examine those rows in
      meta_data.csv. Their images were analysed.
    - Contours not correct: use --threshold_method otsu
    - Slow analysis: use --disable_arrow_detection
    - Arrow problems: use --arrow_debug --log_level DEBUG

    Debug output:
      pylithics --data_dir ./data \\
          --log_level DEBUG --threshold_debug --scale_debug --arrow_debug

    Each flag writes to its own folder in results/, and the run names
    the folders at the end:
      results/threshold_debug/<image>.png       the black-and-white image
      results/scale_debug/<scale image>.png     the bar that was found
      results/arrow_debug/<image>/<scar>.png    the arrow found, and .txt
    pylithics-pages --debug writes pages_debug/<page>.png in the project.

    The log: <data_dir>/results/pylithics.log

    Full documentation: pylithics --docs
    """)


def launch_docs_server() -> None:
    """Launch the MkDocs development server."""
    try:
        print("\nStarting the documentation server...")
        print("URL: http://127.0.0.1:8000/Palaeoanalytics/")
        print("Press Ctrl+C to stop\n")

        try:
            subprocess.run(
                ['mkdocs', '--version'],
                capture_output=True, check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: MkDocs is not installed.")
            print("Install it with: pip install mkdocs mkdocs-material")
            sys.exit(1)

        subprocess.run(['mkdocs', 'serve'])

    except KeyboardInterrupt:
        print("\nThe documentation server stopped.")
    except OSError as e:
        print(f"Error when the documentation server started: {e}")
        sys.exit(1)

def _apply_config_overrides(
    app: 'PyLithicsApplication',
    args: argparse.Namespace
) -> None:
    """
    Map CLI arguments to configuration overrides.

    Parameters
    ----------
    app : PyLithicsApplication
        Application instance to update
    args : argparse.Namespace
        Parsed command-line arguments
    """
    overrides: Dict[str, Any] = {}

    if args.threshold_method:
        overrides['thresholding.method'] = args.threshold_method
    if args.log_level:
        overrides['logging.level'] = args.log_level
        overrides['logging.console_level'] = args.log_level
    if getattr(args, 'verbose', False):
        overrides['logging.console_level'] = 'DEBUG'
    if args.disable_arrow_detection:
        overrides['arrow_detection.enabled'] = False
    if args.arrow_debug:
        overrides['arrow_detection.debug_enabled'] = True
    if getattr(args, 'show_thresholded_images', False):
        overrides['thresholding.debug_output'] = True
    if args.show_arrow_lines:
        overrides['arrow_detection.show_arrow_lines'] = True

    _apply_scale_overrides(args, overrides)
    _apply_cortex_overrides(args, overrides)
    _apply_scar_overrides(args, overrides)
    _apply_dpi_overrides(args, overrides)
    _apply_export_overrides(args, overrides)

    if overrides:
        app.update_configuration(**overrides)
        logging.info(f"Configuration changes applied: {overrides}")


def _apply_scale_overrides(
    args: argparse.Namespace, overrides: Dict[str, Any]
) -> None:
    """Map scale calibration CLI args to config overrides."""
    if args.disable_scale_calibration:
        overrides['scale_calibration.enabled'] = False
    if args.scale_debug:
        overrides['scale_calibration.debug_output'] = True
    if args.force_pixels:
        overrides['scale_calibration.enabled'] = False


def _apply_cortex_overrides(
    args: argparse.Namespace, overrides: Dict[str, Any]
) -> None:
    """Map cortex detection CLI args to config overrides."""
    if getattr(args, 'disable_cortex_detection', False):
        overrides['cortex_detection.enabled'] = False

    sensitivity = getattr(args, 'cortex_sensitivity', None)
    if sensitivity == 'low':
        overrides['cortex_detection.stippling_density_threshold'] = 0.4
        overrides['cortex_detection.texture_variance_threshold'] = 200
        overrides['cortex_detection.edge_density_threshold'] = 0.1
    elif sensitivity == 'high':
        overrides['cortex_detection.stippling_density_threshold'] = 0.1
        overrides['cortex_detection.texture_variance_threshold'] = 50
        overrides['cortex_detection.edge_density_threshold'] = 0.02


def _apply_scar_overrides(
    args: argparse.Namespace, overrides: Dict[str, Any]
) -> None:
    """Map scar complexity CLI args to config overrides."""
    if getattr(args, 'disable_scar_complexity', False):
        overrides['scar_complexity.enabled'] = False
    threshold = getattr(args, 'scar_complexity_distance_threshold', None)
    if threshold:
        overrides['scar_complexity.distance_threshold'] = threshold


def _apply_dpi_overrides(
    args: argparse.Namespace, overrides: Dict[str, Any]
) -> None:
    """Map DPI processing CLI args to config overrides."""
    if args.enable_dpi_scaling:
        overrides['dpi_processing.enabled'] = True
    if args.dpi_reference:
        overrides['dpi_processing.reference_dpi'] = args.dpi_reference
    if args.dpi_max_scale:
        overrides['dpi_processing.max_scale_factor'] = args.dpi_max_scale


def _apply_export_overrides(
    args: argparse.Namespace, overrides: Dict[str, Any]
) -> None:
    """Map output / export CLI args to config overrides."""
    if getattr(args, 'export_json', False):
        overrides['data_export.json_per_lithic'] = True
    if args.dpi_scaling_mode:
        overrides['dpi_processing.scaling_mode'] = args.dpi_scaling_mode


_HELP_FLAGS = (
    ('help_config', show_config_help),
    ('help_examples', show_examples_help),
    ('help_troubleshooting', show_troubleshooting_help),
    ('docs', launch_docs_server),
)


def _handle_help_flags(args) -> bool:
    """Run whichever help/docs command was requested. Return True if handled."""
    for attr, action in _HELP_FLAGS:
        if getattr(args, attr, False):
            action()
            return True
    return False


def main() -> int:
    """Main entry point for PyLithics CLI."""
    _stop_explore_progress()
    args = create_argument_parser().parse_args()

    if _handle_help_flags(args):
        return 0
    _offer_update()

    explore = getattr(args, 'explore', False)
    if isinstance(explore, str):
        return _launch_explore(explore)
    if not _resolve_inputs(args, explore):
        return 1

    try:
        app = PyLithicsApplication(args.config_file)
        app.workers_arg = getattr(args, 'workers', 'auto')
        _apply_config_overrides(app, args)
        # Re-configure logging now that CLI overrides (--verbose,
        # --log_level) are merged into the config. data_dir puts the log
        # file in the user's results/ folder, not the shell's cwd.
        app.setup_logging(data_dir=args.data_dir)
        # Erase the "Starting PyLithics…" stderr line written at
        # import time before the first real INFO line scrolls past it.
        _clear_startup_notice()

        logging.info(f"Config: {args.config_file or 'default'}")
        logging.info(f"Data directory: {args.data_dir}")

        if args.meta_file:
            logging.info(f"Metadata file: {args.meta_file}")
            results = app.run_batch_analysis(
                args.data_dir, args.meta_file, args.show_thresholded_images,
            )
            if not results['success']:
                logging.error("The batch stopped with an error")
                return 1

        if explore:
            return _launch_explore(_resolve_explore_dir(args.data_dir))
        return 0

    except KeyboardInterrupt:
        logging.info("Stopped by the user")
        return 1
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Input error: {e}")
        return 1


def _offer_update() -> None:
    """Once a day, tell the user about a newer release and offer it."""
    from pylithics.update_check import check_for_update
    enabled = get_config_manager().get_section('update_check').get('enabled', True)
    check_for_update(enabled)


def _resolve_inputs(args: argparse.Namespace, explore: bool) -> bool:
    """
    Fill the metadata default and report a missing argument.

    Prints the error, since logging is not yet set up at this point.
    """
    if not args.data_dir:
        print("Error: --data_dir is necessary.")
        print("Use 'pylithics --help' or 'pylithics --docs'.")
        return False
    if not args.meta_file:
        args.meta_file = default_meta_file(args.data_dir)
    if not args.meta_file and not explore:
        print(f"Error: there is no {METADATA_FILENAME} in {args.data_dir}. "
              f"Give --meta_file.")
        print("Use 'pylithics --help' or 'pylithics --docs'.")
        return False
    return True


def _resolve_explore_dir(data_dir: str) -> str:
    """Resolve ``--data_dir`` to the results folder for a bare ``--explore``.

    After an analysis the results are in ``data_dir/results/``. A
    project folder with no metadata, given with a bare ``--explore``,
    is accepted as the results folder itself, so an older command line
    still opens the dashboard.
    """
    candidate = os.path.join(data_dir, ANALYSIS_OUTPUT_DIRNAME)
    if os.path.isfile(os.path.join(candidate, 'processed_metrics.csv')):
        return candidate
    return data_dir


def _launch_explore(processed_dir: str) -> int:
    """Open the dashboard against ``processed_dir`` (the folder containing
    ``processed_metrics.csv``).
    """
    from pylithics.image_processing.modules.dashboard.runner import (
        launch_dashboard,
    )

    csv_path = os.path.join(processed_dir, "processed_metrics.csv")
    if not os.path.exists(csv_path):
        logging.error(
            "There is no processed_metrics.csv in %s. "
            "Give --explore the folder that contains it, or give "
            "--data_dir to do the analysis first.",
            processed_dir,
        )
        return 1
    return launch_dashboard(processed_dir)


if __name__ == "__main__":
    sys.exit(main())