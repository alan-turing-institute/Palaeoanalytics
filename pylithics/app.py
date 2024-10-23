"""
PyLithics: Stone Tool Image Analysis CLI

This script serves as the command-line interface (CLI) for the PyLithics package.
It processes lithic images, applying various thresholding methods and other
preprocessing steps, and saves or displays the processed images.

Usage:
    python app.py --data_dir <path_to_data_directory> --meta_file <path_to_metadata_csv>
                  [--config_file <path_to_config>]
                  [--threshold_method {adaptive,simple,otsu,default}]
                  [--log_level {DEBUG,INFO,WARNING,ERROR}]
                  [--show_thresholded_images]

Options:
    --data_dir: Directory containing the images and scale files.
    --meta_file: Path to the metadata CSV file.
    --config_file: Optional, path to the configuration file.
    --threshold_method: Optional, choose a thresholding method to override the config file.
    --log_level: Optional, set the logging level.
    --show_thresholded_images: Optional, display thresholded images after processing.
"""

import argparse
import logging
from pylithics.image_processing.importer import preprocess_images, load_preprocessing_config


def setup_logging(level):
    """
    Set up logging configuration with the specified logging level.
    If logging is already set up, it will reset the logging configuration.

    :param level: The logging level as a string (e.g., 'DEBUG', 'INFO').
    """
    # Reset any previously configured loggers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    """
    Main function to parse command-line arguments and initiate the image preprocessing process.
    """
    parser = argparse.ArgumentParser(description="PyLithics: Stone Tool Image Analysis")

    # Arguments for input data directory and metadata file
    parser.add_argument('--data_dir', required=True,
                        help="Directory containing images and scale files.")
    parser.add_argument('--meta_file', required=True,
                        help="Path to the metadata CSV file.")

    # Optional argument to specify a custom config file
    parser.add_argument('--config_file', default="pylithics/config/config.yaml",
                        help="Path to the configuration file.")

    # Command-line overrides for thresholding and logging
    parser.add_argument('--threshold_method', choices=["adaptive", "simple", "otsu", "default"],
                        help="Thresholding method to override config.")
    parser.add_argument('--log_level', choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level to override config.")

    # Option to save processed images
    parser.add_argument('--show_thresholded_images', action='store_true',
                        help="Display processed images after preprocessing.")

    args = parser.parse_args()

    # Load the config file (default or custom)
    config = load_preprocessing_config(args.config_file)

    # Override config settings with command-line arguments
    if args.threshold_method:
        config['thresholding']['method'] = args.threshold_method

    # Set logging level based on config or command-line override
    log_level = args.log_level or config['logging']['level']
    setup_logging(log_level.upper())

    # Log the current configuration
    logging.info("Logging level set to %s", log_level.upper())
    logging.info("Configuration loaded from: %s", args.config_file)

    # Call the preprocess_images function with the modified config
    preprocess_images(args.data_dir, args.meta_file,
                      show_thresholded_images=args.show_thresholded_images)


if __name__ == "__main__":
    main()
