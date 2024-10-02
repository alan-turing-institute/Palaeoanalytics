import os
import csv
import logging
from PIL import Image
from pylithics.image_processing.measurement import Measurement
from pylithics.image_processing.utils import read_metadata

# Load logging configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def validate_image_scale_dpi(image_path, scale_dpi):
    try:
        with Image.open(image_path) as img:
            dpi = img.info.get('dpi')
            if dpi is None:
                logging.warning(f"DPI information missing for {image_path}")
                return False
            if abs(dpi[0] - scale_dpi) > 1:  # Allowing for minor floating-point variance
                logging.error(f"Image DPI ({dpi[0]}) does not match expected scale ({scale_dpi}) for {image_path}")
                return False
        return True
    except Exception as e:
        logging.error(f"Failed to load image {image_path}: {e}")
        return False

def import_images(data_dir, meta_file):
    # Directories for images and scales
    images_dir = os.path.join(data_dir, 'images')
    scales_dir = os.path.join(data_dir, 'scales')

    # Load metadata
    metadata = read_metadata(meta_file)

    for entry in metadata:
        image_id = entry['image_id']
        scale_id = entry['scale_id']
        scale_dpi = float(entry['scale']) if entry['scale'] else None

        image_path = os.path.join(images_dir, image_id)
        scale_path = os.path.join(scales_dir, scale_id)

        # Validate if files exist
        if not os.path.exists(image_path):
            logging.error(f"Image file not found: {image_path}")
            continue
        if not os.path.exists(scale_path):
            logging.error(f"Scale file not found: {scale_path}")
            continue

        # Create a measurement in pixels (as an example, let's assume we're measuring width in pixels)
        # This will be retrieved based on actual use case later
        pixels = 500  # Example measurement in pixels

        # Create the Measurement object
        measurement = Measurement(pixels, scale_dpi)

        # Convert to millimeters (if possible) or leave as pixels
        result = measurement.to_millimeters()
        logging.info(f"Final measurement: {result} ({'mm' if measurement.is_scaled() else 'pixels'})")