import argparse
import logging
import os
from pylithics.image_processing.importer import execute_preprocessing_pipeline, verify_image_dpi_and_scale,preprocess_images, load_preprocessing_config
from pylithics.image_processing.image_analysis import process_and_save_contours
from pylithics.image_processing.image_analysis import visualize_contours_with_hierarchy

from pylithics.image_processing.utils import read_metadata

def setup_logging(level):
    """Set up logging configuration with the specified logging level."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def preprocess_and_analyze_images(data_dir, meta_file, config, show_thresholded_images):
    """Preprocess images and perform contour analysis."""
    from pylithics.image_processing.importer import execute_preprocessing_pipeline, verify_image_dpi_and_scale
    from pylithics.image_processing.image_analysis import process_and_save_contours

    images_dir = os.path.join(data_dir, 'images')
    processed_dir = os.path.join(data_dir, 'processed')  # Directory for processed outputs
    metadata = read_metadata(meta_file)

    os.makedirs(processed_dir, exist_ok=True)  # Ensure output directory exists

    for entry in metadata:
        image_id = entry['image_id']
        real_world_scale_mm = float(entry['scale']) if entry['scale'] else None
        image_path = os.path.join(images_dir, image_id)

        # Step 1: Preprocess the image
        logging.info(f"Processing image: {image_id}")
        processed_image = execute_preprocessing_pipeline(image_path, config)
        if processed_image is None:
            logging.error("Skipping analysis for %s due to preprocessing failure.", image_id)
            continue

        conversion_factor = verify_image_dpi_and_scale(image_path, real_world_scale_mm)
        if conversion_factor is None:
            logging.error("Skipping analysis for %s due to DPI mismatch or missing information.", image_id)
            continue

        # Step 2: Analyze and save the processed image
        try:
            logging.info("Analyzing contours and hierarchy in: %s", image_id)
            process_and_save_contours(
                processed_image,
                conversion_factor,
                processed_dir,
                image_id  # Pass image_id explicitly
            )
            logging.info("Analysis complete for image: %s", image_id)
        except Exception as e:
            logging.error("Error analyzing image %s: %s", image_id, str(e))



def main():
    """Main function to parse command-line arguments and perform preprocessing and analysis."""
    parser = argparse.ArgumentParser(description="PyLithics: Stone Tool Image Analysis")

    # Arguments for input data directory and metadata file
    parser.add_argument('--data_dir', required=True, help="Directory containing images and scale files.")
    parser.add_argument('--meta_file', required=True, help="Path to the metadata CSV file.")

    # Optional argument to specify a custom config file
    parser.add_argument('--config_file', default="pylithics/config/config.yaml", help="Path to the configuration file.")

    # Command-line overrides for thresholding
    parser.add_argument('--threshold_method', choices=["adaptive", "simple", "otsu", "default"], help="Thresholding method to override config.")

    # Command-line overrides for logging
    parser.add_argument('--log_level', choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level to override config.")

    # Option to show processed images
    parser.add_argument('--show_thresholded_images', action='store_true', help="Display processed images after preprocessing.")

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

    logging.info("Processing and analysis complete.")

if __name__ == "__main__":
    main()
