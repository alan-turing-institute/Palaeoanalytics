import argparse
from pylithics.image_processing.importer import import_images

def main():
    parser = argparse.ArgumentParser(description="Import and preprocess images for lithic analysis.")
    parser.add_argument('--data_dir', required=True, help='Directory containing images and scales.')
    parser.add_argument('--meta_file', required=True, help='Path to the metadata CSV file.')
    parser.add_argument('--save_processed_images', action='store_true', help='Flag to save preprocessed images.')

    args = parser.parse_args()
    import_images(args.data_dir, args.meta_file, save_processed_images=args.save_processed_images)

if __name__ == '__main__':
    main()
