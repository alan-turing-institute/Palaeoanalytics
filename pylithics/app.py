#!/usr/bin/env python3
"""
PyLithics Application Entry Point
=================================

Configuration management, error handling, and flexible command-line options.
"""

import sys

if len(sys.argv) == 1:
    from pylithics.cli_splash import print_splash
    print_splash()
    sys.exit(0)

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
            "Starting PyLithics data explorer... "
            "(loading modules and PyLithics data).",
            flush=True,
        )
        return

    _EXPLORE_PROGRESS = Progress(
        SpinnerColumn(style="cyan"),
        TextColumn(
            "[cyan]Starting PyLithics data explorer...[/] "
            "loading modules and PyLithics data"
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
from typing import Optional, Dict, Any

from pylithics.image_processing.config import (
    get_config_manager,
    ConfigurationManager,
)
from pylithics.image_processing.importer import (
    execute_preprocessing_pipeline,
    verify_image_dpi_and_scale,
)
from pylithics.image_processing.image_analysis import process_and_save_contours
from pylithics.image_processing.utils import read_metadata
from pylithics.image_processing.modules.scale_calibration import get_calibration_factor


_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')


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
            f"Invalid scale for {image_id}, using pixel measurements"
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
        return "pixels (scale detection failed — see log)"
    return "pixels (no scale provided)"


def _write_run_summary(
    processed_dir: str,
    images_dir: str,
    results: Dict[str, Any],
    metadata: list,
) -> None:
    """
    Write ``processed/run_summary.json`` with a structured record of the run.

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
    }

    summary_path = os.path.join(processed_dir, "run_summary.json")
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logging.debug("Wrote run summary to %s", summary_path)
    except OSError as e:
        logging.warning("Could not write run summary: %s", e)


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
        self.config_manager = get_config_manager(config_file)
        self.setup_logging()

    def setup_logging(self) -> None:
        """Set up logging configuration from config manager.

        Console output goes through ``rich.logging.RichHandler`` for
        coloured level icons and syntax-highlighted tracebacks. The file
        handler keeps a plain text format for grep-friendly logs.

        Default split:
            - Console: INFO (concise — shows per-image summaries, warnings,
              errors). ``--verbose`` flips this to DEBUG.
            - File: always at DEBUG so the full per-step trace is preserved
              for reproducibility regardless of console verbosity.

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
            log_file = logging_config.get(
                'log_file', 'pylithics/data/processed/pylithics.log',
            )
            log_dir = os.path.dirname(log_file)

            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
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

        images_dir = os.path.join(data_dir, 'images')
        if not os.path.exists(images_dir):
            logging.error(f"Images directory does not exist: {images_dir}")
            return False

        # Check metadata file
        if not os.path.exists(meta_file):
            logging.error(f"Metadata file does not exist: {meta_file}")
            return False

        # Validate metadata format
        try:
            metadata = read_metadata(meta_file)
            if not metadata:
                logging.error("Metadata file is empty or invalid")
                return False

            # Check required columns
            required_columns = ['image_id', 'scale']
            first_entry = metadata[0]
            for col in required_columns:
                if col not in first_entry:
                    logging.error(f"Missing required column in metadata: {col}")
                    return False

        except (FileNotFoundError, KeyError, ValueError) as e:
            logging.error(f"Error reading metadata file: {e}")
            return False

        logging.info("Input validation passed")
        return True

    def process_single_image(self,
                           image_id: str,
                           real_world_scale_mm: Optional[float],
                           images_dir: str,
                           processed_dir: str,
                           scale_data: Optional[Dict] = None,
                           progress_index: Optional[int] = None,
                           progress_total: Optional[int] = None) -> bool:
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
                logging.error(f"Preprocessing failed for {image_id}")
                return False

            image_dpi = self._extract_image_dpi(image_path)
            conversion_factor, calibration_method, scale_confidence = (
                self._resolve_calibration(image_path, scale_data or {})
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
            )

            suffix = _calibration_suffix(calibration_method, conversion_factor)
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

    def _resolve_calibration(
        self, image_path: str, scale_data: Dict,
    ) -> "tuple[float, str, Optional[float]]":
        """Get conversion factor with fallback to pixel measurements."""
        conversion_factor, calibration_method, scale_confidence = (
            get_calibration_factor(
                image_path, scale_data, self.config_manager.config,
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
                    logging.warning(f"No DPI information found in {image_path}")
                    return None
        except Exception as e:
            logging.warning(f"Could not extract DPI from {image_path}: {e}")
            return None

    def run_batch_analysis(self,
                          data_dir: str,
                          meta_file: str,
                          show_thresholded_images: bool = False) -> Dict[str, Any]:
        """
        Run batch analysis on all images in the dataset.

        Parameters
        ----------
        data_dir : str
            Directory containing images and scale files
        meta_file : str
            Path to the metadata CSV file
        show_thresholded_images : bool
            Whether to display thresholded images

        Returns
        -------
        dict
            Processing results summary
        """
        if not self.validate_inputs(data_dir, meta_file):
            return {'success': False, 'error': 'Input validation failed'}

        images_dir = os.path.join(data_dir, 'images')
        processed_dir = os.path.join(data_dir, 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        logging.info(f"Output directory: {processed_dir}")

        metadata = read_metadata(meta_file)
        results = {
            'success': True,
            'total_images': len(metadata),
            'processed_successfully': 0,
            'failed_images': [],
            'processing_errors': [],
        }

        logging.debug(f"Starting batch processing of {len(metadata)} images")

        self._run_batch_loop(metadata, images_dir, processed_dir, results)

        self._log_batch_summary(results)
        _write_run_summary(processed_dir, images_dir, results, metadata)

        return results

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
            logging.info(f"{done}/{total} images processed without errors.")
            logging.info(
                f"Please check logs at {log_path}", extra=console_only,
            )
        else:
            logging.info(f"{done}/{total} images processed successfully.")
            logging.info(
                f"Please check logs at {log_path} for errors.",
                extra=console_only,
            )
            if results['failed_images']:
                logging.warning(
                    f"Failed images: {', '.join(results['failed_images'])}"
                )
        if hasattr(self, 'rich_console'):
            self.rich_console.print()

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
                logging.warning(f"Invalid config key format: {key}. Use 'section.key' format.")

        # Do NOT call clear_config_cache() here. The update_value() calls
        # above mutate the cached singleton in place; clearing the cache would
        # cause the next get_config_manager() call to reload from disk and
        # silently discard every override.


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog='PyLithics',
        description='PyLithics v2.0.0: Stone Tool Image Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Use --docs to launch full documentation.'
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
    group = parser.add_argument_group('REQUIRED ARGUMENTS')
    group.add_argument(
        '--data_dir', required=False, metavar='PATH',
        help='Directory containing images/ and scale files'
    )
    group.add_argument(
        '--meta_file', required=False, metavar='FILE',
        help='CSV metadata file (columns: image_id, scale_id, scale)'
    )


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    """Add configuration argument group."""
    group = parser.add_argument_group('CONFIGURATION OPTIONS')
    group.add_argument(
        '--config_file', metavar='FILE',
        help='Custom YAML configuration file'
    )
    group.add_argument(
        '--threshold_method',
        choices=["adaptive", "simple", "otsu", "default"],
        metavar='METHOD',
        help='Thresholding method: simple, otsu, adaptive, default'
    )
    group.add_argument(
        '--log_level',
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        metavar='LEVEL',
        help='Logging level for both console and file handlers '
             '(default: INFO on console, DEBUG always in file)'
    )
    group.add_argument(
        '--verbose', '-v', action='store_true',
        help='Show the full per-step pipeline trace on screen. '
             'Equivalent to --log_level DEBUG for the console only; the '
             'log file always captures the full trace.'
    )


def _add_processing_args(parser: argparse.ArgumentParser) -> None:
    """Add processing options argument group."""
    group = parser.add_argument_group('PROCESSING OPTIONS')
    group.add_argument(
        '--show_thresholded_images', action='store_true',
        help='Display processed images during analysis'
    )
    group.add_argument(
        '--closing', type=bool, default=True, metavar='BOOL',
        help='Apply morphological closing (default: True)'
    )
    group.add_argument(
        '--enable_dpi_scaling', action='store_true',
        help='Enable DPI-aware kernel scaling for preprocessing'
    )
    group.add_argument(
        '--dpi_reference', type=float, metavar='DPI',
        help='Reference DPI for kernel scaling (default: 300.0)'
    )
    group.add_argument(
        '--dpi_max_scale', type=float, metavar='FACTOR',
        help='Maximum DPI scaling factor (default: 1.5)'
    )
    group.add_argument(
        '--dpi_scaling_mode',
        choices=['conservative', 'standard', 'aggressive'],
        metavar='MODE',
        help='DPI scaling strategy (default: standard)'
    )


def _add_arrow_args(parser: argparse.ArgumentParser) -> None:
    """Add arrow detection argument group."""
    group = parser.add_argument_group('ARROW DETECTION OPTIONS')
    group.add_argument(
        '--disable_arrow_detection', action='store_true',
        help='Disable arrow detection analysis'
    )
    group.add_argument(
        '--arrow_debug', action='store_true',
        help='Enable arrow detection debug output'
    )
    group.add_argument(
        '--show-arrow-lines', action='store_true',
        help='Draw red arrow lines on detected arrows'
    )


def _add_scale_args(parser: argparse.ArgumentParser) -> None:
    """Add scale calibration argument group."""
    group = parser.add_argument_group('SCALE CALIBRATION OPTIONS')
    group.add_argument(
        '--disable_scale_calibration', action='store_true',
        help='Disable scale bar calibration'
    )
    group.add_argument(
        '--scale_debug', action='store_true',
        help='Enable scale bar detection debug output'
    )
    group.add_argument(
        '--force_pixels', action='store_true',
        help='Force pixel measurements only'
    )


def _add_cortex_args(parser: argparse.ArgumentParser) -> None:
    """Add cortex detection argument group."""
    group = parser.add_argument_group('CORTEX DETECTION OPTIONS')
    group.add_argument(
        '--disable_cortex_detection', action='store_true',
        help='Disable cortex detection analysis'
    )
    group.add_argument(
        '--cortex_sensitivity', type=str,
        choices=['low', 'medium', 'high'],
        help='Cortex detection sensitivity (default: medium)'
    )


def _add_scar_args(parser: argparse.ArgumentParser) -> None:
    """Add scar complexity argument group."""
    group = parser.add_argument_group('SCAR COMPLEXITY OPTIONS')
    group.add_argument(
        '--disable_scar_complexity', action='store_true',
        help='Disable scar complexity analysis'
    )
    group.add_argument(
        '--scar_complexity_distance_threshold',
        type=float, metavar='PIXELS',
        help='Adjacency distance threshold in pixels (default: 10.0)'
    )


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    """Add output options argument group."""
    group = parser.add_argument_group('OUTPUT OPTIONS')
    group.add_argument(
        '--export_json', action='store_true',
        help=(
            'Also write a per-lithic JSON file to processed/json/'
            '{image_stem}.json (in addition to the CSV).'
        )
    )
    group.add_argument(
        '--save_visualizations', action='store_true',
        default=True,
        help='Generate visualization images (default: True)'
    )


def _add_explore_args(parser: argparse.ArgumentParser) -> None:
    """Add interactive dashboard argument group."""
    group = parser.add_argument_group('EXPLORE OPTIONS')
    group.add_argument(
        '--explore', action='store_true',
        help=(
            'Run analysis (if --meta_file is provided) and then launch the '
            'PyLithics Explorer. Without --meta_file, point --data_dir at '
            'the folder containing processed_metrics.csv (commonly '
            '<project_root>/processed/).'
        )
    )


def _add_help_args(parser: argparse.ArgumentParser) -> None:
    """Add extended help argument group."""
    group = parser.add_argument_group('EXTENDED HELP OPTIONS')
    group.add_argument(
        '--help-config', action='store_true',
        help='Show configuration file documentation'
    )
    group.add_argument(
        '--help-examples', action='store_true',
        help='Show usage examples'
    )
    group.add_argument(
        '--help-troubleshooting', action='store_true',
        help='Show common problems and solutions'
    )
    group.add_argument(
        '--docs', action='store_true',
        help='Launch documentation server (http://127.0.0.1:8000)'
    )


def show_config_help() -> None:
    """Display configuration help summary."""
    print("""
    PYLITHICS CONFIGURATION HELP
    ============================

    PyLithics uses YAML configuration files. To customise:
      1. Copy pylithics/config/config.yaml
      2. Edit values as needed
      3. Use --config_file path/to/your/config.yaml

    Key sections: thresholding, arrow_detection, cortex_detection,
    scar_complexity, logging, contour_filtering, data_export

    For full documentation: pylithics --docs
    """)


def show_examples_help() -> None:
    """Display usage examples summary."""
    print("""
    PYLITHICS USAGE EXAMPLES
    ========================

    Basic analysis:
      pylithics --data_dir ./artifacts --meta_file ./metadata.csv

    With Otsu thresholding:
      pylithics --data_dir ./artifacts --meta_file ./metadata.csv \\
          --threshold_method otsu

    Debug arrow detection:
      pylithics --data_dir ./artifacts --meta_file ./metadata.csv \\
          --arrow_debug --log_level DEBUG

    Fast batch (no arrows):
      pylithics --data_dir ./artifacts --meta_file ./metadata.csv \\
          --disable_arrow_detection

    Also export per-lithic JSON files (in addition to CSV):
      pylithics --data_dir ./artifacts --meta_file ./metadata.csv \\
          --export_json

    Analyze and immediately launch the interactive dashboard:
      pylithics --data_dir ./artifacts --meta_file ./metadata.csv --explore

    Re-open the dashboard later (no re-analysis). --data_dir is the
    folder that actually contains processed_metrics.csv (the folder name
    doesn't have to be 'processed/' — it can be any folder you've moved
    or renamed):
      pylithics --data_dir ./artifacts/processed --explore

    For full documentation: pylithics --docs
    """)


def show_troubleshooting_help() -> None:
    """Display troubleshooting summary."""
    print("""
    PYLITHICS TROUBLESHOOTING
    =========================

    Common fixes:
    - "Directory does not exist": Check --data_dir path
    - "Missing required column": CSV needs image_id, scale_id, scale
    - Poor contour detection: Try --threshold_method otsu
    - Slow processing: Use --disable_arrow_detection
    - Arrow issues: Use --arrow_debug --log_level DEBUG

    Debug mode:
      pylithics --data_dir ./data --meta_file ./meta.csv \\
          --log_level DEBUG --arrow_debug

    Check logs: data_dir/processed/pylithics.log

    For full documentation: pylithics --docs
    """)


def launch_docs_server() -> None:
    """Launch the MkDocs development server."""
    try:
        print("\nStarting documentation server...")
        print("URL: http://127.0.0.1:8000/Palaeoanalytics/")
        print("Press Ctrl+C to stop\n")

        try:
            subprocess.run(
                ['mkdocs', '--version'],
                capture_output=True, check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: MkDocs is not installed.")
            print("Install with: pip install mkdocs mkdocs-material")
            sys.exit(1)

        subprocess.run(['mkdocs', 'serve'])

    except KeyboardInterrupt:
        print("\nDocumentation server stopped.")
    except OSError as e:
        print(f"Error launching documentation server: {e}")
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
    if args.show_arrow_lines:
        overrides['arrow_detection.show_arrow_lines'] = True

    _apply_scale_overrides(args, overrides)
    _apply_cortex_overrides(args, overrides)
    _apply_scar_overrides(args, overrides)
    _apply_dpi_overrides(args, overrides)
    _apply_export_overrides(args, overrides)

    if overrides:
        app.update_configuration(**overrides)
        logging.info(f"Applied config overrides: {overrides}")


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

    if not args.data_dir:
        print("Error: --data_dir is required.")
        print("Use 'pylithics --help' or 'pylithics --docs'.")
        return 1

    explore = getattr(args, 'explore', False)
    if not args.meta_file and not explore:
        print("Error: --meta_file is required (or pass --explore to open the "
              "dashboard against an existing run).")
        print("Use 'pylithics --help' or 'pylithics --docs'.")
        return 1

    try:
        app = PyLithicsApplication(args.config_file)
        _apply_config_overrides(app, args)
        # Re-configure logging now that CLI overrides (e.g. --verbose,
        # --log_level) have been merged into the config.
        app.setup_logging()

        logging.info(f"Config: {args.config_file or 'default'}")
        logging.info(f"Data directory: {args.data_dir}")

        if args.meta_file:
            logging.info(f"Metadata file: {args.meta_file}")
            results = app.run_batch_analysis(
                args.data_dir, args.meta_file, args.show_thresholded_images,
            )
            if not results['success']:
                logging.error("Batch processing failed")
                return 1

        if explore:
            if args.meta_file:
                processed_dir = os.path.join(args.data_dir, 'processed')
            else:
                processed_dir = args.data_dir
            return _launch_explore(processed_dir)
        return 0

    except KeyboardInterrupt:
        logging.info("Processing interrupted by user")
        return 1
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Input error: {e}")
        return 1


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
            "No processed_metrics.csv found in %s. "
            "Point --data_dir at the folder that contains it, or pass "
            "--meta_file to run analysis first.",
            processed_dir,
        )
        return 1
    return launch_dashboard(processed_dir)


if __name__ == "__main__":
    sys.exit(main())