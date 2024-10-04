import argparse
import logging
from pylithics.image_processing.importer import import_images
from pylithics.image_processing.utils import read_metadata
from pylithics.image_processing.importer import load_config

def main():
    parser = argparse.ArgumentParser(description="PyLithics: Stone Tool Image Analysis")

    # Arguments for input data directory and metadata file
    parser.add_argument('--data_dir', required=True, help="Directory containing images and scale files.")
    parser.add_argument('--meta_file', required=True, help="Path to the metadata CSV file.")

    # Optional argument to specify a custom config file
    parser.add_argument('--config_file', default="pylithics/config/config.yaml", help="Path to the configuration file.")

    # Command-line overrides for thresholding and logging
    parser.add_argument('--threshold_method', choices=["adaptive", "simple", "otsu", "default"], help="Thresholding method to override config.")
    parser.add_argument('--log_level', choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level to override config.")

    # Option to save processed images
    parser.add_argument('--show_thresholded_images', action='store_true', help="Save processed images to disk.")

    args = parser.parse_args()

    # Load the config file (default or custom)
    config = load_config(args.config_file)

    # Override config settings with command-line arguments
    if args.threshold_method:
        config['thresholding']['method'] = args.threshold_method
    if args.log_level:
        config['logging']['level'] = args.log_level

    # Set logging level based on config or command-line override
    logging.basicConfig(level=getattr(logging, config['logging']['level'].upper(), 'INFO'))

    # Call the import_images function with the modified config
    import_images(args.data_dir, args.meta_file, show_thresholded_images=args.show_thresholded_images)

if __name__ == "__main__":
    main()
