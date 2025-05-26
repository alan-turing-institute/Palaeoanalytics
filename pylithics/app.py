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
from pathlib import Path
from PIL import Image
from typing import Optional, Dict, Any

from pylithics.image_processing.config import (
    get_config_manager,
    ConfigurationManager,
    clear_config_cache
)
from pylithics.image_processing.importer import (
    execute_preprocessing_pipeline,
    verify_image_dpi_and_scale,
)
from pylithics.image_processing.image_analysis import process_and_save_contours
from pylithics.image_processing.utils import read_metadata


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
            log_file = logging_config.get('log_file', 'logs/pylithics.log')
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

        except Exception as e:
            logging.error(f"Error reading metadata file: {e}")
            return False

        logging.info("Input validation passed")
        return True

    def process_single_image(self,
                           image_id: str,
                           real_world_scale_mm: Optional[float],
                           images_dir: str,
                           processed_dir: str) -> bool:
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

        Returns
        -------
        bool
            True if processing succeeded, False otherwise
        """
        image_path = os.path.join(images_dir, image_id)

        if not os.path.exists(image_path):
            logging.error(f"Image file does not exist: {image_path}")
            return False

        logging.info(f"Processing image: {image_id}")

        try:
            # Step 1: Preprocess the image
            processed_image = execute_preprocessing_pipeline(
                image_path, self.config_manager.config
            )
            if processed_image is None:
                logging.error(f"Preprocessing failed for {image_id}")
                return False

            # Step 2: Extract and validate DPI information
            image_dpi = self._extract_image_dpi(image_path)

            # Step 3: Verify image scale and calculate conversion factor
            conversion_factor = verify_image_dpi_and_scale(image_path, real_world_scale_mm)
            if conversion_factor is None:
                logging.error(f"DPI validation failed for {image_id}")
                return False

            # Step 4: Run complete analysis pipeline
            process_and_save_contours(
                processed_image,
                conversion_factor,
                processed_dir,
                image_id,
                image_dpi
            )

            logging.info(f"Successfully processed {image_id}")
            return True

        except Exception as e:
            logging.error(f"Error processing {image_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

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
            'processing_errors': []
        }

        logging.info(f"Starting batch processing of {len(metadata)} images")

        for i, entry in enumerate(metadata, 1):
            image_id = entry['image_id']
            real_world_scale_mm = float(entry['scale']) if entry['scale'] else None

            logging.info(f"Processing image {i}/{len(metadata)}: {image_id}")

            success = self.process_single_image(
                image_id, real_world_scale_mm, images_dir, processed_dir
            )

            if success:
                results['processed_successfully'] += 1
            else:
                results['failed_images'].append(image_id)
                results['processing_errors'].append(f"Failed to process {image_id}")

        # Log final summary
        success_rate = (results['processed_successfully'] / results['total_images']) * 100
        logging.info(f"Batch processing completed: {results['processed_successfully']}/{results['total_images']} successful ({success_rate:.1f}%)")

        if results['failed_images']:
            logging.warning(f"Failed images: {', '.join(results['failed_images'])}")

        return results

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

        # Clear cache to ensure new values are used
        clear_config_cache()


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="PyLithics: Enhanced Stone Tool Image Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --data_dir ./data --meta_file ./metadata.csv
  %(prog)s --data_dir ./data --meta_file ./metadata.csv --config_file custom_config.yaml
  %(prog)s --data_dir ./data --meta_file ./metadata.csv --threshold_method otsu --log_level DEBUG
        """
    )

    # Required arguments
    parser.add_argument(
        '--data_dir',
        required=True,
        help="Directory containing images and scale files"
    )
    parser.add_argument(
        '--meta_file',
        required=True,
        help="Path to the metadata CSV file"
    )

    # Configuration arguments
    parser.add_argument(
        '--config_file',
        default=None,
        help="Path to the configuration file (default: use built-in config)"
    )

    # Processing options
    parser.add_argument(
        '--threshold_method',
        choices=["adaptive", "simple", "otsu", "default"],
        help="Thresholding method to override config"
    )
    parser.add_argument(
        '--log_level',
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level to override config"
    )
    parser.add_argument(
        '--show_thresholded_images',
        action='store_true',
        help="Display processed images after preprocessing"
    )

    # Morphological processing
    parser.add_argument(
        "--closing",
        type=bool,
        default=True,
        help="Apply morphological closing (default: True)"
    )

    # Arrow detection options
    parser.add_argument(
        '--disable_arrow_detection',
        action='store_true',
        help="Disable arrow detection"
    )
    parser.add_argument(
        '--arrow_debug',
        action='store_true',
        help="Enable arrow detection debug output"
    )

    # Output options
    parser.add_argument(
        '--output_format',
        choices=['csv', 'json'],
        default='csv',
        help="Output format for results (default: csv)"
    )
    parser.add_argument(
        '--save_visualizations',
        action='store_true',
        default=True,
        help="Save visualization images (default: True)"
    )

    return parser


def main() -> int:
    """
    Main function to parse command-line arguments and run PyLithics analysis.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure)
    """
    parser = create_argument_parser()
    args = parser.parse_args()

    try:
        # Initialize application
        app = PyLithicsApplication(args.config_file)

        # Apply command-line overrides
        config_overrides = {}

        if args.threshold_method:
            config_overrides['thresholding.method'] = args.threshold_method

        if args.log_level:
            config_overrides['logging.level'] = args.log_level

        if args.disable_arrow_detection:
            config_overrides['arrow_detection.enabled'] = False

        if args.arrow_debug:
            config_overrides['arrow_detection.debug_enabled'] = True

        # Apply overrides
        if config_overrides:
            app.update_configuration(**config_overrides)
            logging.info(f"Applied configuration overrides: {config_overrides}")

        # Log configuration summary
        logging.info(f"Configuration loaded from: {args.config_file or 'default'}")
        logging.info(f"Data directory: {args.data_dir}")
        logging.info(f"Metadata file: {args.meta_file}")

        # Run batch analysis
        results = app.run_batch_analysis(
            args.data_dir,
            args.meta_file,
            args.show_thresholded_images
        )

        if results['success']:
            if results['processed_successfully'] == results['total_images']:
                logging.info("All images processed successfully!")
                return 0
            else:
                logging.warning("Some images failed to process")
                return 0  # Partial success
        else:
            logging.error("Batch processing failed")
            return 1

    except KeyboardInterrupt:
        logging.info("Processing interrupted by user")
        return 1
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())