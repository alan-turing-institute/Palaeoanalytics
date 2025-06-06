import argparse
import logging
import os
from PIL import Image

from pylithics.image_processing.importer import (
    execute_preprocessing_pipeline,
    verify_image_dpi_and_scale,
    load_preprocessing_config,
)
from pylithics.image_processing.image_analysis import (
    process_and_save_contours,
    visualize_contours_with_hierarchy,
)

from pylithics.image_processing.utils import read_metadata


def setup_logging(level: str) -> None:
    """
    Set up logging configuration with the specified logging level.

    Args:
        level (str): Logging level (e.g., "DEBUG", "INFO", etc.).
    """
    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def preprocess_and_analyze_images(
    data_dir: str, meta_file: str, config: dict, show_thresholded_images: bool
) -> None:
    """
    Preprocess images and perform contour analysis.

    Args:
        data_dir (str): Directory containing images and scale files.
        meta_file (str): Path to the metadata CSV file.
        config (dict): Preprocessing configuration settings.
        show_thresholded_images (bool): Whether to display thresholded images.
    """
    images_dir = os.path.join(data_dir, 'images')
    processed_dir = os.path.join(data_dir, 'processed')  # Directory for processed outputs
    metadata = read_metadata(meta_file)

    os.makedirs(processed_dir, exist_ok=True)  # Ensure output directory exists

    for entry in metadata:
        image_id = entry['image_id']
        real_world_scale_mm = float(entry['scale']) if entry['scale'] else None
        image_path = os.path.join(images_dir, image_id)

        # Step 1: Preprocess the image
        logging.info("Processing image: %s", image_id)
        processed_image = execute_preprocessing_pipeline(image_path, config)
        if processed_image is None:
            logging.error("Skipping analysis for %s due to preprocessing failure.", image_id)
            continue

        # Extract DPI information
        image_dpi = None
        try:
            with Image.open(image_path) as img:
                dpi_info = img.info.get('dpi')
                if dpi_info:
                    image_dpi = round(float(dpi_info[0]))  # Use horizontal DPI
                    logging.info(f"Image DPI detected: {image_dpi}")
        except Exception as e:
            logging.warning(f"Could not extract DPI from image: {e}")

        conversion_factor = verify_image_dpi_and_scale(image_path, real_world_scale_mm)
        if conversion_factor is None:
            logging.error(
                "Skipping analysis for %s due to DPI mismatch or missing information.", image_id
            )
            continue

        # Step 2: Analyze and save the processed image (passing image_dpi)
        try:
            logging.info("Analyzing contours and hierarchy in: %s", image_id)
            process_and_save_contours(
                processed_image,
                conversion_factor,
                processed_dir,
                image_id,      # Pass image_id explicitly
                image_dpi      # Pass image_dpi to enable scaled detection
            )
            logging.info("Analysis complete for image: %s", image_id)
        except Exception as e:
            logging.error("Error analyzing image %s: %s", image_id, str(e))


def main() -> None:
    """
    Main function to parse command-line arguments and perform preprocessing and analysis.
    """
    parser = argparse.ArgumentParser(description="PyLithics: Stone Tool Image Analysis")

    # Arguments for input data directory and metadata file
    parser.add_argument('--data_dir', required=True, help="Directory containing images and scale files.")
    parser.add_argument('--meta_file', required=True, help="Path to the metadata CSV file.")

    # Optional argument to specify a custom config file
    parser.add_argument('--config_file', default="pylithics/config/config.yaml", help="Path to the configuration file.")

    # Command-line overrides for thresholding
    parser.add_argument('--threshold_method', choices=["adaptive", "simple", "otsu", "default"],
                        help="Thresholding method to override config.")

    # Command-line overrides for logging
    parser.add_argument('--log_level', choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level to override config.")

    # Option to show processed images
    parser.add_argument('--show_thresholded_images', action='store_true',
                        help="Display processed images after preprocessing.")

    # Turn on/off morphological closing (default: True)
    parser.add_argument("--closing", type=bool, default=True,
                        help="Apply morphological closing (default: True)")

    args = parser.parse_args()

    # Load the config file (default or custom)
    config = load_preprocessing_config(args.config_file)
    if args.threshold_method:
        config['thresholding']['method'] = args.threshold_method

    log_level = args.log_level or config['logging']['level']
    setup_logging(log_level.upper())
    logging.info("Logging level set to %s", log_level.upper())
    logging.info("Configuration loaded from: %s", args.config_file)

    # Run the preprocessing and analysis steps
    preprocess_and_analyze_images(args.data_dir, args.meta_file, config, args.show_thresholded_images)

if __name__ == "__main__":
    main()
