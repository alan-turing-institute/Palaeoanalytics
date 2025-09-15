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
from pylithics.image_processing.modules.scale_calibration import get_calibration_factor


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

        except Exception as e:
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
        image_path = os.path.join(images_dir, image_id)

        # If file doesn't exist, try adding common extensions
        if not os.path.exists(image_path):
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
                test_path = os.path.join(images_dir, image_id + ext)
                if os.path.exists(test_path):
                    image_path = test_path
                    break

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

            # Step 3: Get conversion factor using scale calibration with fallback
            if scale_data is None:
                scale_data = {}  # Empty dict if no scale data provided

            conversion_factor, calibration_method, scale_confidence = get_calibration_factor(
                image_path, scale_data, self.config_manager.config
            )

            # Log calibration method used
            if conversion_factor:
                logging.info(f"Using {calibration_method} calibration: "
                           f"{conversion_factor:.3f} pixels/mm")
            else:
                logging.info(f"No calibration available, using pixel measurements")
                conversion_factor = 1.0  # Use 1.0 for pixel measurements

            # Step 4: Run complete analysis pipeline
            process_and_save_contours(
                processed_image,
                conversion_factor,
                processed_dir,
                image_id,
                image_dpi,
                calibration_method,
                scale_confidence
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
                image_id, real_world_scale_mm, images_dir, processed_dir, entry
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
    """Create and configure the comprehensive argument parser with detailed help."""

    # Main parser with enhanced description
    parser = argparse.ArgumentParser(
        prog='PyLithics',
        description="""
        PyLithics v2.0.0: Professional Stone Tool Image Analysis Software
        ================================================================

        PyLithics is a comprehensive archaeological tool for analyzing lithic artifacts using
        computer vision and advanced image processing techniques. It provides:

        • Automated contour detection and geometric analysis
        • Advanced arrow detection with DPI-aware scaling
        • Surface classification (Dorsal, Ventral, Platform, Lateral)
        • Symmetry analysis for dorsal surfaces
        • Voronoi diagram spatial analysis
        • Comprehensive metric calculation and CSV export
        • Professional error handling and batch processing

        This enhanced version includes enterprise-grade configuration management,
        robust error handling, and flexible command-line control for research workflows.
        """.strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        USAGE EXAMPLES:
        ==============

        Basic Analysis:
        %(prog)s --data_dir ./artifacts --meta_file ./metadata.csv

        Advanced Configuration:
        %(prog)s --data_dir ./artifacts --meta_file ./metadata.csv \\
                --threshold_method otsu --log_level DEBUG \\
                --arrow_debug --config_file custom_settings.yaml

        Batch Processing with Custom Settings:
        %(prog)s --data_dir ./large_assemblage --meta_file ./assemblage.csv \\
                --threshold_method adaptive --output_format csv \\
                --log_level INFO

        Troubleshooting Mode:
        %(prog)s --data_dir ./problem_images --meta_file ./metadata.csv \\
                --log_level DEBUG --arrow_debug --show_thresholded_images

        HELP OPTIONS:
        ============
        Use -h or --help to see this help message
        Use --help-config for configuration file documentation
        Use --help-examples for detailed usage examples
        Use --help-troubleshooting for common problem solutions

        For more information: https://github.com/alan-turing-institute/Palaeoanalytics
        """)

    # Required Arguments Group
    required_group = parser.add_argument_group(
        'REQUIRED ARGUMENTS',
        'These arguments must be provided for PyLithics to run'
    )

    required_group.add_argument(
        '--data_dir',
        required=True,
        metavar='PATH',
        help="""Directory containing your artifact images and associated scale files.
             Must contain an 'images/' subdirectory with your artifact photos.
             Example: --data_dir ./my_artifacts"""
    )

    required_group.add_argument(
        '--meta_file',
        required=True,
        metavar='FILE',
        help="""Path to CSV metadata file containing image information.
             Must have columns: image_id, scale_id, scale
             Example: --meta_file ./artifact_metadata.csv"""
    )

    # Configuration Arguments Group
    config_group = parser.add_argument_group(
        'CONFIGURATION OPTIONS',
        'Control PyLithics behavior and processing methods'
    )

    config_group.add_argument(
        '--config_file',
        metavar='FILE',
        help="""Custom configuration file (YAML format).
             If not provided, uses built-in default settings.
             Example: --config_file ./custom_config.yaml"""
    )

    config_group.add_argument(
        '--threshold_method',
        choices=["adaptive", "simple", "otsu", "default"],
        metavar='METHOD',
        help="""Image thresholding method for contour detection:
             • simple: Fixed threshold value (fast, basic)
             • otsu: Automatic optimal threshold (recommended)
             • adaptive: Local threshold adjustment (for varied lighting)
             • default: Falls back to simple method
             Default: Uses setting from configuration file"""
    )

    config_group.add_argument(
        '--log_level',
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        metavar='LEVEL',
        help="""Logging detail level:
             • DEBUG: Detailed technical information (for troubleshooting)
             • INFO: General progress information (recommended)
             • WARNING: Only important warnings
             • ERROR: Only critical errors
             Default: INFO"""
    )

    # Processing Options Group
    processing_group = parser.add_argument_group(
        'PROCESSING OPTIONS',
        'Control specific analysis features and behavior'
    )

    processing_group.add_argument(
        '--show_thresholded_images',
        action='store_true',
        help="""Display processed images during analysis.
             Useful for debugging image processing issues.
             Note: Requires display capability (not for headless servers)"""
    )

    processing_group.add_argument(
        "--closing",
        type=bool,
        default=True,
        metavar='BOOL',
        help="""Apply morphological closing to clean up contours.
             Helps connect broken contour lines.
             Default: True (recommended for most images)"""
    )

    # Arrow Detection Group
    arrow_group = parser.add_argument_group(
        'ARROW DETECTION OPTIONS',
        'Control advanced arrow detection features'
    )

    arrow_group.add_argument(
        '--disable_arrow_detection',
        action='store_true',
        help="""Completely disable arrow detection analysis.
             Use if you only need basic geometric measurements.
             Significantly speeds up processing for large batches."""
    )

    arrow_group.add_argument(
        '--arrow_debug',
        action='store_true',
        help="""Enable detailed arrow detection debugging.
             Creates debug images and detailed logs for each detection attempt.
             Output saved to [data_dir]/processed/arrow_debug/
             Useful for troubleshooting detection issues."""
    )

    arrow_group.add_argument(
        '--show-arrow-lines',
        action='store_true',
        help="""Draw red arrow lines on detected arrows in visualization.
             By default, only light blue contour outlines are shown for arrows.
             Enable this flag to also draw the traditional red arrow lines
             showing direction and angle. Useful for detailed directional analysis."""
    )

    # Scale Calibration Options Group
    scale_group = parser.add_argument_group(
        'SCALE CALIBRATION OPTIONS',
        'Control scale bar detection and measurement conversion'
    )

    scale_group.add_argument(
        '--disable_scale_calibration',
        action='store_true',
        help="""Disable scale bar calibration.
             Uses pixel measurements only (no real-world units).
             Use if scale bar detection is failing or not needed."""
    )

    scale_group.add_argument(
        '--scale_debug',
        action='store_true',
        help="""Enable scale bar detection debugging.
             Creates debug images showing detected scale bars.
             Output saved to [data_dir]/processed/scale_debug/
             Useful for verifying scale detection accuracy."""
    )

    scale_group.add_argument(
        '--force_pixels',
        action='store_true',
        help="""Force pixel measurements only (no scale calibration).
             Output will be in pixels only regardless of scale data.
             Use to ensure consistent pixel-based measurements."""
    )

    # Cortex Detection Options Group
    cortex_group = parser.add_argument_group(
        'CORTEX DETECTION OPTIONS',
        'Control cortex detection and sensitivity settings'
    )
    cortex_group.add_argument(
        '--disable_cortex_detection',
        action='store_true',
        help="""Completely disable cortex detection analysis.
             Use if you only need basic scar detection without cortex identification.
             Slightly speeds up processing for large batches."""
    )
    cortex_group.add_argument(
        '--cortex_sensitivity',
        type=str,
        choices=['low', 'medium', 'high'],
        help="""Set cortex detection sensitivity level.
             - low: Only detect very obvious cortex (strict thresholds)
             - medium: Standard detection (default settings)
             - high: Detect subtle cortex patterns (permissive thresholds)
             Overrides config file settings."""
    )

    # Scar Complexity Options Group
    scar_group = parser.add_argument_group(
        'SCAR COMPLEXITY OPTIONS',
        'Control scar complexity analysis and adjacency detection'
    )
    scar_group.add_argument(
        '--disable_scar_complexity',
        action='store_true',
        help="""Completely disable scar complexity analysis.
             Use if you only need basic geometric measurements without adjacency counts.
             Slightly speeds up processing for large batches."""
    )
    scar_group.add_argument(
        '--scar_complexity_distance_threshold',
        type=float,
        metavar='PIXELS',
        help="""Distance threshold in pixels for scar adjacency detection.
             Scars within this distance are considered adjacent.
             Typical range: 3.0-20.0 pixels
             Lower values = stricter adjacency detection
             Higher values = more permissive adjacency detection
             Default: 10.0 pixels"""
    )

    # Output Options Group
    output_group = parser.add_argument_group(
        'OUTPUT OPTIONS',
        'Control output format and file generation'
    )

    output_group.add_argument(
        '--output_format',
        choices=['csv', 'json'],
        default='csv',
        metavar='FORMAT',
        help="""Output data format:
             • csv: Comma-separated values (recommended for Excel/R/Python)
             • json: JavaScript Object Notation (for web applications)
             Default: csv"""
    )

    output_group.add_argument(
        '--save_visualizations',
        action='store_true',
        default=True,
        help="""Generate visualization images showing detected features.
             Creates *_labeled.png and *_voronoi.png files.
             Default: True (highly recommended for result verification)"""
    )

    # Help Extensions Group
    help_group = parser.add_argument_group(
        'EXTENDED HELP OPTIONS',
        'Additional documentation and examples'
    )

    help_group.add_argument(
        '--help-config',
        action='store_true',
        help='Show detailed configuration file documentation'
    )

    help_group.add_argument(
        '--help-examples',
        action='store_true',
        help='Show comprehensive usage examples for different scenarios'
    )

    help_group.add_argument(
        '--help-troubleshooting',
        action='store_true',
        help='Show common problems and solutions'
    )

    return parser


def show_config_help():
    """Display detailed configuration file help."""
    help_text = """
    PYLITHICS CONFIGURATION FILE HELP
    =================================

    PyLithics uses YAML configuration files to control processing behavior.
    The default configuration is built-in, but you can customize it with --config_file.

    CONFIGURATION FILE STRUCTURE:
    ----------------------------

    # Image Processing Settings
    thresholding:
    method: simple          # simple, otsu, adaptive, default
    threshold_value: 127    # Used with 'simple' method (0-255)
    max_value: 255         # Maximum pixel value after thresholding

    normalization:
    enabled: true          # Apply contrast normalization
    method: minmax         # minmax, zscore, custom
    clip_values: [0, 255]  # Output range for minmax method

    grayscale_conversion:
    enabled: true          # Convert to grayscale before processing
    method: standard       # standard, clahe

    morphological_closing:
    enabled: true          # Clean up contour breaks
    kernel_size: 3         # Size of morphological kernel

    # Arrow Detection Settings
    arrow_detection:
    enabled: true                      # Enable arrow detection
    reference_dpi: 300.0              # DPI for parameter calibration
    min_area_scale_factor: 0.7        # Detection sensitivity (0-1)
    min_defect_depth_scale_factor: 0.8
    min_triangle_height_scale_factor: 0.8
    debug_enabled: false              # Create debug visualizations

    # Logging Configuration
    logging:
    level: INFO                       # DEBUG, INFO, WARNING, ERROR
    log_to_file: true                 # Save logs to file
    log_file: data/processed/pylithics.log
    
    # Cortex Detection Settings
    cortex_detection:
    enabled: true                      # Enable cortex detection
    stippling_density_threshold: 0.2   # Minimum stippling density (0.1-1.0)
    texture_variance_threshold: 100    # Minimum texture variance (50-500)
    edge_density_threshold: 0.05       # Minimum edge density (0.01-0.2)

    # Analysis Parameters
    contour_filtering:
    min_area: 50.0         # Minimum contour size (pixels)
    exclude_border: true   # Ignore border-touching contours

    CREATING CUSTOM CONFIGURATIONS:
    ------------------------------
    1. Copy the default config.yaml from pylithics/config/
    2. Modify values as needed
    3. Use --config_file path/to/your/config.yaml

    COMMON CUSTOMIZATIONS:
    ---------------------
    • More sensitive arrow detection: Set scale_factors to 0.5-0.6
    • Less sensitive detection: Set scale_factors to 0.8-0.9
    • Debug mode: Set arrow_detection.debug_enabled: true
    • Verbose logging: Set logging.level: DEBUG
    • Different thresholding: Change thresholding.method
    """
    print(help_text)


def show_examples_help():
    """Display comprehensive usage examples."""
    help_text = """
    PYLITHICS USAGE EXAMPLES
    ========================

    BASIC USAGE:
    -----------
    # Analyze artifacts with default settings
    python app.py --data_dir ./my_artifacts --meta_file ./metadata.csv

    # Same with explicit log level
    python app.py --data_dir ./artifacts --meta_file ./data.csv --log_level INFO

    ADVANCED IMAGE PROCESSING:
    -------------------------
    # Use Otsu thresholding (recommended for varied lighting)
    python app.py --data_dir ./artifacts --meta_file ./metadata.csv \\
                --threshold_method otsu

    # Use adaptive thresholding (for very uneven lighting)
    python app.py --data_dir ./scans --meta_file ./data.csv \\
                --threshold_method adaptive --log_level DEBUG

    ARROW DETECTION OPTIONS:
    -----------------------
    # Enable arrow detection debugging
    python app.py --data_dir ./artifacts --meta_file ./metadata.csv \\
                --arrow_debug --log_level DEBUG

    # Show red arrow lines on detected arrows (default: only blue contours)
    python app.py --data_dir ./artifacts --meta_file ./metadata.csv \\
                --show-arrow-lines

    # Disable arrow detection (faster processing)
    python app.py --data_dir ./large_collection --meta_file ./metadata.csv \\
                --disable_arrow_detection

    CORTEX DETECTION OPTIONS:
    ------------------------
    # Enable high sensitivity cortex detection
    python app.py --data_dir ./artifacts --meta_file ./metadata.csv \\
                --cortex_sensitivity high
    
    # Disable cortex detection (faster processing)
    python app.py --data_dir ./large_collection --meta_file ./metadata.csv \\
                --disable_cortex_detection
    
    # Low sensitivity for obvious cortex only
    python app.py --data_dir ./clean_artifacts --meta_file ./metadata.csv \\
                --cortex_sensitivity low
    
    CUSTOM CONFIGURATIONS:
    ---------------------
    # Use custom configuration file
    python app.py --data_dir ./artifacts --meta_file ./metadata.csv \\
                --config_file ./my_settings.yaml

    # Override specific settings
    python app.py --data_dir ./artifacts --meta_file ./metadata.csv \\
                --threshold_method otsu --log_level DEBUG --arrow_debug

    BATCH PROCESSING:
    ----------------
    # Process large dataset with progress tracking
    python app.py --data_dir ./assemblage_2023 --meta_file ./assemblage.csv \\
                --log_level INFO --output_format csv

    # Silent processing (minimal output)
    python app.py --data_dir ./artifacts --meta_file ./metadata.csv \\
                --log_level WARNING

    TROUBLESHOOTING MODES:
    ---------------------
    # Maximum debug information
    python app.py --data_dir ./problem_images --meta_file ./metadata.csv \\
                --log_level DEBUG --arrow_debug --show_thresholded_images

    # Test configuration without arrow detection
    python app.py --data_dir ./test_images --meta_file ./test.csv \\
                --disable_arrow_detection --log_level DEBUG

    REAL-WORLD SCENARIOS:
    --------------------
    # Publication-quality analysis
    python app.py --data_dir ./final_dataset --meta_file ./publication_data.csv \\
                --threshold_method otsu --log_level INFO --save_visualizations

    # Quick preliminary analysis
    python app.py --data_dir ./preliminary --meta_file ./quick_test.csv \\
                --disable_arrow_detection --log_level WARNING

    # Comparative method testing
    python app.py --data_dir ./comparison --meta_file ./test.csv --threshold_method simple
    python app.py --data_dir ./comparison --meta_file ./test.csv --threshold_method otsu
    python app.py --data_dir ./comparison --meta_file ./test.csv --threshold_method adaptive
"""
    print(help_text)


def show_troubleshooting_help():
    """Display troubleshooting guide."""
    help_text = """
    PYLITHICS TROUBLESHOOTING GUIDE
    ===============================

    COMMON PROBLEMS AND SOLUTIONS:

    ERROR: "Data directory does not exist"
    -------------------------------------
    Problem: PyLithics can't find your data folder
    Solutions:
    • Check the path: --data_dir ./correct/path/to/data
    • Use absolute paths: --data_dir /full/path/to/artifacts
    • Ensure the directory exists and is accessible

    ERROR: "Images directory does not exist"
    ---------------------------------------
    Problem: Missing 'images' subdirectory in your data folder
    Solutions:
    • Create subdirectory: mkdir your_data_dir/images
    • Move images into: your_data_dir/images/
    • Check folder structure matches: data_dir/images/your_images.png

    ERROR: "Missing required column in metadata"
    ------------------------------------------
    Problem: CSV file doesn't have required headers
    Solutions:
    • Ensure CSV has headers: image_id, scale_id, scale
    • Check for typos in column names (case-sensitive)
    • Verify CSV format with a text editor

    ERROR: "Image file does not exist"
    ---------------------------------
    Problem: Image files referenced in CSV don't exist
    Solutions:
    • Check image filenames match CSV exactly (including extensions)
    • Verify images are in the images/ subdirectory
    • Check for case sensitivity issues (image.PNG vs image.png)

    ARROW DETECTION ISSUES:
    ----------------------
    Problem: Low arrow detection rates
    Solutions:
    • Enable debug mode: --arrow_debug --log_level DEBUG
    • Try different thresholding: --threshold_method otsu
    • Check image quality and resolution
    • Adjust sensitivity in config file (lower scale_factors)

    Problem: Too many false arrow detections
    Solutions:
    • Increase sensitivity thresholds in config file
    • Use stricter thresholding method
    • Check for image artifacts or noise

    PERFORMANCE ISSUES:
    ------------------
    Problem: Processing too slow
    Solutions:
    • Disable arrow detection: --disable_arrow_detection
    • Use simple thresholding: --threshold_method simple
    • Process smaller batches
    • Check available system memory

    Problem: Running out of memory
    Solutions:
    • Process images in smaller batches
    • Use lower resolution images
    • Close other applications
    • Increase system virtual memory

    IMAGE PROCESSING ISSUES:
    -----------------------
    Problem: Poor contour detection
    Solutions:
    • Try different thresholding methods
    • Adjust lighting/contrast in original images
    • Enable debug mode to see processed images
    • Check image DPI information

    Problem: Inconsistent results
    Solutions:
    • Ensure consistent image DPI across dataset
    • Use same thresholding method for all images
    • Check for variations in lighting/background

    GETTING HELP:
    ------------
    1. Enable debug logging: --log_level DEBUG
    2. Check log files in: data_dir/processed/pylithics.log
    3. Use arrow debug mode: --arrow_debug
    4. Report issues with specific error messages and log files

    For technical support: https://github.com/alan-turing-institute/Palaeoanalytics/issues
"""
    print(help_text)

# Enhanced main function to handle extended help
def main() -> int:
    """
    Main function with extended help system.
    """
    parser = create_argument_parser()
    args = parser.parse_args()

    # Handle extended help options
    if hasattr(args, 'help_config') and args.help_config:
        show_config_help()
        return 0

    if hasattr(args, 'help_examples') and args.help_examples:
        show_examples_help()
        return 0

    if hasattr(args, 'help_troubleshooting') and args.help_troubleshooting:
        show_troubleshooting_help()
        return 0

    # Continue with normal processing...
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

        if args.show_arrow_lines:
            config_overrides['arrow_detection.show_arrow_lines'] = True

        # Scale calibration overrides
        if args.disable_scale_calibration:
            config_overrides['scale_calibration.enabled'] = False
        if args.scale_debug:
            config_overrides['scale_calibration.debug_output'] = True
        if args.force_pixels:
            config_overrides['scale_calibration.enabled'] = False

        # Cortex detection overrides
        if hasattr(args, 'disable_cortex_detection') and args.disable_cortex_detection:
            config_overrides['cortex_detection.enabled'] = False
        if hasattr(args, 'cortex_sensitivity') and args.cortex_sensitivity:
            # Apply sensitivity presets
            if args.cortex_sensitivity == 'low':
                config_overrides['cortex_detection.stippling_density_threshold'] = 0.4
                config_overrides['cortex_detection.texture_variance_threshold'] = 200
                config_overrides['cortex_detection.edge_density_threshold'] = 0.1
            elif args.cortex_sensitivity == 'high':
                config_overrides['cortex_detection.stippling_density_threshold'] = 0.1
                config_overrides['cortex_detection.texture_variance_threshold'] = 50
                config_overrides['cortex_detection.edge_density_threshold'] = 0.02
            # Medium sensitivity uses default config values

        # Scar complexity overrides
        if hasattr(args, 'disable_scar_complexity') and args.disable_scar_complexity:
            config_overrides['scar_complexity.enabled'] = False
        if hasattr(args, 'scar_complexity_distance_threshold') and args.scar_complexity_distance_threshold:
            config_overrides['scar_complexity.distance_threshold'] = args.scar_complexity_distance_threshold

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