import os
import logging
import cv2  # Using OpenCV for image preprocessing
from PIL import Image
from pylithics.image_processing.measurement import Measurement
from pylithics.image_processing.utils import read_metadata
import yaml


# Load logging configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


### CONFIGURATION LOADER ###

import os
import yaml
import logging

def load_config(config_file="config.yaml"):
    """
    Load configuration settings from a YAML file located in the 'pylithics/config' directory.
    :param config_file: Name of the configuration file.
    :return: A dictionary containing configuration settings.
    """
    try:
        # Get the directory of the current file and navigate up to the parent 'pylithics' directory
        base_dir = os.path.dirname(os.path.dirname(__file__))

        # Join the correct path to the 'config' directory
        config_file_path = os.path.join(base_dir, 'config', config_file)

        if not os.path.exists(config_file_path):
            logging.error(f"Configuration file {config_file_path} not found.")
            return None

        with open(config_file_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded configuration from {config_file_path}.")
        return config
    except Exception as e:
        logging.error(f"Failed to load config file {config_file_path}: {e}")
        return None



### IMAGE PROCESSING FUNCTIONS ###

def load_image(image_path):
    """
    Load an image from the specified path using OpenCV.
    :param image_path: Path to the image file.
    :return: The loaded image or None if it fails.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image at {image_path} could not be loaded.")
        logging.info(f"Loaded image: {image_path}")
        return image
    except Exception as e:
        logging.error(f"Failed to load image {image_path}: {e}")
        return None

def convert_to_grayscale(image, config):
    """
    Convert the input image to grayscale based on config settings.
    :param image: The loaded image.
    :param config: Configuration dictionary for grayscale conversion.
    :return: Grayscale image.
    """
    if not config.get('grayscale_conversion', {}).get('enabled', True):
        return image  # If grayscale conversion is disabled, return the original image

    method = config['grayscale_conversion'].get('method', 'standard')

    if method == "standard":
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logging.info("Converted image to standard grayscale.")
    elif method == "clahe":
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_image = clahe.apply(gray_image)
        logging.info("Converted image to CLAHE grayscale.")
    else:
        logging.error(f"Unsupported grayscale conversion method: {method}")
        return None

    return gray_image


def normalize_grayscale_image(gray_image, config):
    """
    Normalize the grayscale image by stretching contrast based on config settings.
    :param gray_image: The grayscale image.
    :param config: Configuration dictionary for normalization.
    :return: Normalized grayscale image.
    """
    if not config.get('normalization', {}).get('enabled', True):
        return gray_image  # Skip normalization if disabled

    method = config['normalization'].get('method', 'minmax')

    if method == "minmax":
        # Min-Max normalization
        min_intensity = gray_image.min()
        max_intensity = gray_image.max()
        normalized_image = cv2.normalize(
            gray_image, None, alpha=config['normalization'].get('clip_values', [0, 255])[0],
            beta=config['normalization'].get('clip_values', [0, 255])[1], norm_type=cv2.NORM_MINMAX
        )
        logging.info("Applied Min-Max normalization.")
    elif method == "zscore":
        # Z-score normalization
        mean = gray_image.mean()
        std = gray_image.std()
        normalized_image = (gray_image - mean) / std
        logging.info("Applied Z-score normalization.")
    else:
        logging.error(f"Unsupported normalization method: {method}")
        return None

    return normalized_image


### DYNAMIC THRESHOLDING BASED ON CONFIGURATION ###

def apply_threshold(normalized_image, config):
    """
    Apply various thresholding methods based on the configuration.

    :param normalized_image: The normalized grayscale image.
    :param config: Configuration dictionary with thresholding parameters.
    :return: Thresholded image.
    """
    try:
        method = config['thresholding'].get('method', 'default')
        max_value = config['thresholding'].get('max_value', 255)

        # Apply Gaussian blur for noise reduction before thresholding
        blurred_image = cv2.GaussianBlur(normalized_image, (5, 5), 0)
        logging.info("Applied Gaussian blur for noise reduction.")

        thresholded_image = None

        # Use a dictionary to map method names to the corresponding thresholding functions
        threshold_methods = {
            "adaptive": lambda: cv2.adaptiveThreshold(
                blurred_image, max_value,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            ),
            "simple": lambda: cv2.threshold(
                blurred_image,
                config['thresholding'].get('threshold_value', 127),
                max_value,
                cv2.THRESH_BINARY
            )[1],  # `cv2.threshold` returns a tuple, we only need the second element (the image)
            "otsu": lambda: cv2.threshold(
                blurred_image,
                0,
                max_value,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1],  # `cv2.threshold` returns a tuple, we only need the second element (the image)
            "default": lambda: cv2.adaptiveThreshold(
                blurred_image, max_value,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )
        }

        if method not in threshold_methods:
            raise ValueError(f"Unsupported thresholding method: {method}")

        thresholded_image = threshold_methods[method]()
        logging.info(f"Applied {method} thresholding.")

        return thresholded_image

    except Exception as e:
        logging.error(f"Failed to apply thresholding: {e}")
        return None


### IMAGE VALIDATION ###

def validate_image_scale_dpi(image_path, real_world_scale_mm):
    """
    Validate the DPI of the image and calculate the conversion factor between pixels and millimeters
    based on the real-world scale length (in millimeters).
    :param image_path: Path to the image.
    :param real_world_scale_mm: Real-world length of the scale bar in millimeters.
    :return: Conversion factor from pixels to millimeters, or None if validation fails.
    """
    try:
        with Image.open(image_path) as img:
            dpi = img.info.get('dpi')
            if dpi is None:
                logging.warning(f"DPI information missing for {image_path}")
                return None  # Continue but return None for the conversion factor

            # Calculate the conversion factor: how many pixels per millimeter
            # DPI is in dots per inch, so we convert it to dots per mm: (1 inch = 25.4 mm)
            pixels_per_mm = dpi[0] / 25.4

            # Calculate the length of the scale bar in pixels
            scale_length_pixels = real_world_scale_mm * pixels_per_mm
            logging.info(f"Image DPI: {dpi[0]}, Real-world scale (mm): {real_world_scale_mm}, "
                         f"Scale bar in pixels: {scale_length_pixels}")

            return pixels_per_mm  # Return the conversion factor
    except Exception as e:
        logging.error(f"Failed to load image {image_path}: {e}")
        return None


### FILE MANAGEMENT ###

COMMON_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif']

def find_image_file(image_dir, image_name):
    """
    Find the image file in the directory, allowing the file extension to be omitted in the input.
    This function checks for the image with common extensions.
    :param image_dir: Directory where images are stored.
    :param image_name: The name of the image (without extension).
    :return: Full path to the image if found, else None.
    """
    if '.' in image_name:
        image_path = os.path.join(image_dir, image_name)
        if os.path.exists(image_path):
            return image_path
    else:
        for ext in COMMON_EXTENSIONS:
            image_path = os.path.join(image_dir, image_name + ext)
            if os.path.exists(image_path):
                return image_path
    return None


### IMAGE PREPROCESSING PIPELINE ###

def preprocess_image(image_path, config):
    """
    A high-level function to preprocess an image by loading, converting to grayscale,
    normalizing, and applying thresholding.
    :param image_path: Path to the image file.
    :param config: Configuration dictionary for preprocessing.
    :return: The processed image or None if any step fails.
    """
    image = load_image(image_path)
    if image is None:
        return None

    # Pass the config object when calling convert_to_grayscale
    gray_image = convert_to_grayscale(image, config)
    if gray_image is None:
        return None

    normalized_image = normalize_grayscale_image(gray_image, config)  # Pass config here too
    if normalized_image is None:
        return None

    thresholded_image = apply_threshold(normalized_image, config)  # And here
    if thresholded_image is None:
        return None

    return thresholded_image


### MAIN IMAGE IMPORT FUNCTION ###

def import_images(data_dir, meta_file, save_processed_images=False):
    """
    Import images from the specified directory, preprocess each image, and measure features from the processed image.
    :param data_dir: Directory containing the images and scale images.
    :param meta_file: Path to the metadata file.
    :param save_processed_images: Flag to determine whether processed images should be saved.
    """
    images_dir = os.path.join(data_dir, 'images')
    scales_dir = os.path.join(data_dir, 'scales')

    # Load configuration from file
    config = load_config("config.yaml")
    if config is None:
        logging.error("Configuration could not be loaded. Exiting.")
        return

    # Load metadata
    metadata = read_metadata(meta_file)

    for entry in metadata:
        image_id = entry['image_id']
        scale_id = entry['scale_id']
        real_world_scale_mm = float(entry['scale']) if entry['scale'] else None  # This is in mm, not DPI

        # Find the image and scale files
        image_path = find_image_file(images_dir, image_id)
        scale_path = find_image_file(scales_dir, scale_id)

        # Validate if files exist
        if not image_path:
            logging.error(f"Image file not found: {os.path.join(images_dir, image_id)}")
            continue
        if not scale_path:
            logging.error(f"Scale file not found: {os.path.join(scales_dir, scale_id)}")
            continue

        # Preprocess the image
        processed_image = preprocess_image(image_path, config)
        if processed_image is None:
            logging.error(f"Skipping measurement for {image_id} due to preprocessing failure.")
            continue

        # Validate the image DPI and calculate the conversion factor
        conversion_factor = validate_image_scale_dpi(image_path, real_world_scale_mm)
        if conversion_factor is None:
            logging.error(f"Skipping measurement for {image_id} due to DPI mismatch or missing information.")
            continue

        # Further processing, such as measurements, would go here
        logging.info(f"Processing {image_id} with conversion factor: {conversion_factor} pixels per mm")

        # Optionally save the processed image
        if save_processed_images:
            processed_image_path = os.path.join(data_dir, 'processed', image_id + '_processed.png')
            os.makedirs(os.path.dirname(processed_image_path), exist_ok=True)
            cv2.imwrite(processed_image_path, processed_image)
            logging.info(f"Saved processed image: {processed_image_path}")
