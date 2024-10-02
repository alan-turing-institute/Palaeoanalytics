import csv
import logging

def read_metadata(meta_file_path):
    """
    Reads metadata from a CSV file.
    Returns a list of dictionaries with keys: image_id, scale_id, scale.
    """
    metadata = []
    try:
        with open(meta_file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                metadata.append(row)
        return metadata
    except FileNotFoundError:
        logging.error(f"Metadata file not found: {meta_file_path}")
        return []
    except Exception as e:
        logging.error(f"Error reading metadata file {meta_file_path}: {e}")
        return []
