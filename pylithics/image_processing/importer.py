import os
import logging
import cv2  # Using OpenCV for image preprocessing
from PIL import Image
from pylithics.image_processing.measurement import Measurement
from pylithics.image_processing.utils import read_metadata

# Load logging configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def preprocess_image(image_path):
    """
    Preprocess the image by converting it to grayscale, reducing noise,
    and applying adaptive thresholding.

    :param image_path: Path to the original color image.
    :return: The processed image ready for further analysis.
    """
    try:
        # Read the image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image at {image_path} could not be loaded.")

        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logging.info(f"Converted {image_path} to grayscale.")

        # Apply Gaussian blur for noise reduction
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        logging.info(f"Applied Gaussian blur to {image_path} for noise reduction.")

        # Apply adaptive thresholding for better segmentation of low-res images
        thresholded_image = cv2.adaptiveThreshold(
            blurred_image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        logging.info(f"Applied adaptive thresholding to {image_path}.")

        return thresholded_image

    except Exception as e:
        logging.error(f"Failed to preprocess image {image_path}: {e}")
        return None

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
    """
    Import images from the specified directory, preprocess each image,
    and measure features from the processed image.

    :param data_dir: Directory containing the images and scale images.
    :param meta_file: Path to the metadata file.
    """
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

        # Preprocess the image (grayscale + noise reduction + thresholding)
        processed_image = preprocess_image(image_path)
        if processed_image is None:
            logging.error(f"Skipping measurement for {image_id} due to preprocessing failure.")
            continue

        # Validate the image DPI against the scale DPI from metadata
        if scale_dpi is not None and not validate_image_scale_dpi(image_path, scale_dpi):
            logging.error(f"Skipping measurement for {image_id} due to DPI mismatch.")
            continue

        # Assuming we take measurements after preprocessing (here, just a placeholder measurement)
        # Normally, this would be based on actual features in the processed image.
        pixels = 500  # Example placeholder for a measurement in pixels

        # Create the Measurement object with the processed scale DPI
        measurement = Measurement(pixels, scale_dpi)

        # Convert to millimeters (if possible) or leave as pixels
        result = measurement.to_millimeters()
        logging.info(f"Final measurement for {image_id}: {result} ({'mm' if measurement.is_scaled() else 'pixels'})")
