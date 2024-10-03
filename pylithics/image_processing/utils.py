import csv
import logging
import json

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

def load_config(config_file="config.json"):
    """
    Load configuration settings from a JSON file.
    :param config_file: Path to the configuration file.
    :return: A dictionary containing configuration settings.
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        logging.info(f"Loaded configuration from {config_file}.")
        return config
    except Exception as e:
        logging.error(f"Failed to load config file {config_file}: {e}")
        return None
