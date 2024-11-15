import argparse
import logging
import os
from pylithics.image_processing.importer import preprocess_images, load_preprocessing_config
from pylithics.image_processing.image_analysis import analyze_image_contours, save_measurements_to_csv


def setup_logging(level):
    """
    Set up logging configuration with the specified logging level.
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def preprocess_and_analyze_images(data_dir, meta_file, config, show_thresholded_images):
    """
    Preprocess images and then perform contour analysis on the preprocessed images.

    :param data_dir: Directory containing the images and scale files.
    :param meta_file: Path to the metadata file.
    :param config: Configuration dictionary.
    :param show_thresholded_images: Boolean indicating whether to display thresholded images.
    """
    # Step 1: Preprocess the images
    logging.info("Starting image preprocessing...")
    preprocess_images(data_dir, meta_file, show_thresholded_images)

    # Step 2: Perform analysis on the preprocessed images
    logging.info("Starting image analysis...")
    processed_dir = os.path.join(data_dir, 'processed')

    all_measurements = []

    for image_file in os.listdir(processed_dir):
        image_path = os.path.join(processed_dir, image_file)
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            logging.info("Analyzing contours in: %s", image_file)
            df_measurements = analyze_image_contours(processed_image, image_id, conversion_factor)
            if not df_measurements.empty:
                all_measurements.append(df_measurements)

    # If there are measurements, export them to CSV
    if all_measurements:
        final_df = pd.concat(all_measurements, ignore_index=True)
        export_measurements_to_csv(final_df, data_dir)
        logging.info("Contour analysis complete and measurements saved.")
    else:
        logging.warning("No contour measurements were found in the processed images.")


def main():
    """
    Main function to parse command-line arguments and perform preprocessing and analysis.
    """
    parser = argparse.ArgumentParser(description="PyLithics: Stone Tool Image Analysis")

    # Arguments for input data directory and metadata file
    parser.add_argument('--data_dir', required=True, help="Directory containing images and scale files.")
    parser.add_argument('--meta_file', required=True, help="Path to the metadata CSV file.")

    # Optional argument to specify a custom config file
    parser.add_argument('--config_file', default="pylithics/config/config.yaml", help="Path to the configuration file.")

    # Command-line overrides for thresholding and logging
    parser.add_argument('--threshold_method', choices=["adaptive", "simple", "otsu", "default"], help="Thresholding method to override config.")
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
