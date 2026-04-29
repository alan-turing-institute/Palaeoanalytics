#!/usr/bin/env python3
"""
PyLithics Application Entry Point
=================================

Configuration management, error handling, and flexible command-line options.
"""

import argparse
import logging
import os
import sys
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
        """Set up logging configuration from config manager."""
        logging_config = self.config_manager.get_section('logging')
        log_level = logging_config.get('level', 'INFO').upper()

        # Remove existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Create formatters
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        # Set up logger
        logger = logging.getLogger()
        logger.setLevel(log_level)
        logger.addHandler(console_handler)

        # File handler if enabled
        if logging_config.get('log_to_file', True):
            log_file = logging_config.get('log_file', 'pylithics/data/processed/pylithics.log')
            log_dir = os.path.dirname(log_file)

            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            logging.info(f"Logging to file: {log_file}")

        logging.info(f"Logging level set to {log_level}")

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
                           scale_data: Optional[Dict] = None) -> bool:
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

        logging.info(f"Processing image: {image_id}")

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

            process_and_save_contours(
                processed_image,
                conversion_factor,
                processed_dir,
                image_id,
                image_dpi,
                calibration_method,
                scale_confidence,
            )

            logging.info(f"Successfully processed {image_id}")
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
            logging.info(
                f"Using {calibration_method} calibration: "
                f"{conversion_factor:.3f} pixels/mm"
            )
            return conversion_factor, calibration_method, scale_confidence

        logging.info("No calibration available, using pixel measurements")
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
                    logging.info(f"Image DPI detected: {image_dpi}")
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

        metadata = read_metadata(meta_file)
        results = {
            'success': True,
            'total_images': len(metadata),
            'processed_successfully': 0,
            'failed_images': [],
            'processing_errors': [],
        }

        logging.info(f"Starting batch processing of {len(metadata)} images")

        for i, entry in enumerate(metadata, 1):
            image_id = entry['image_id']
            logging.info(f"Processing image {i}/{len(metadata)}: {image_id}")
            scale_mm = _parse_scale(entry.get('scale'), image_id)
            success = self.process_single_image(
                image_id, scale_mm, images_dir, processed_dir, entry,
            )
            if success:
                results['processed_successfully'] += 1
            else:
                results['failed_images'].append(image_id)
                results['processing_errors'].append(
                    f"Failed to process {image_id}"
                )

        self._log_batch_summary(results)

        return results

    @staticmethod
    def _log_batch_summary(results: Dict[str, Any]) -> None:
        """Log a human-readable summary line after a batch run."""
        total = results['total_images']
        done = results['processed_successfully']
        rate = (done / total) * 100 if total else 0.0
        logging.info(
            f"Batch processing completed: {done}/{total} ({rate:.1f}%)"
        )
        if results['failed_images']:
            logging.warning(
                f"Failed images: {', '.join(results['failed_images'])}"
            )

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
        metavar='LEVEL', help='Logging level (default: INFO)'
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
    args = create_argument_parser().parse_args()

    if _handle_help_flags(args):
        return 0

    if not args.data_dir or not args.meta_file:
        print("Error: --data_dir and --meta_file are required.")
        print("Use 'pylithics --help' or 'pylithics --docs'.")
        return 1

    try:
        app = PyLithicsApplication(args.config_file)
        _apply_config_overrides(app, args)

        logging.info(f"Config: {args.config_file or 'default'}")
        logging.info(f"Data directory: {args.data_dir}")
        logging.info(f"Metadata file: {args.meta_file}")

        results = app.run_batch_analysis(
            args.data_dir, args.meta_file, args.show_thresholded_images,
        )
        if not results['success']:
            logging.error("Batch processing failed")
            return 1

        if results['processed_successfully'] < results['total_images']:
            logging.warning("Some images failed to process")
        else:
            logging.info("All images processed successfully!")
        return 0

    except KeyboardInterrupt:
        logging.info("Processing interrupted by user")
        return 1
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Input error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())