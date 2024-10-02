import argparse
from pylithics.image_processing.importer import import_images

def main():
    parser = argparse.ArgumentParser(description="Pylithics Image Import Tool")
    parser.add_argument('--data_dir', required=True, help="Directory containing images and scales")
    parser.add_argument('--meta_file', required=True, help="Path to the metadata CSV file")

    args = parser.parse_args()

    import_images(args.data_dir, args.meta_file)

if __name__ == '__main__':
    main()
