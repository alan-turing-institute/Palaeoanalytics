"""
PyLithics: Image Import and Preprocessing Module

This module provides functions for importing, preprocessing, and validating
images of stone tools for the PyLithics project. It supports image loading,
grayscale conversion, contrast normalization, and various thresholding methods.

The module also includes functionality for reading a configuration file
and applying settings from it during image processing. Configuration can
be provided via command-line arguments, environment variables, or default
bundled configurations.

Usage:
    - load_preprocessing_config(): Load configuration from a YAML file.
    - read_image_from_path(): Load an image using OpenCV.
    - apply_grayscale_conversion(): Convert an image to grayscale.
    - apply_contrast_normalization(): Normalize a grayscale image.
    - perform_thresholding(): Apply various thresholding methods to an image.
    - verify_image_dpi_and_scale(): Validate image DPI and calculate pixel-to-mm conversion.
    - execute_preprocessing_pipeline(): Preprocess images by chaining the above functions.
    - preprocess_images(): Import images and apply the preprocessing pipeline.
"""

import os
import logging
import cv2  # Using OpenCV for image preprocessing
from PIL import Image
import yaml
from pkg_resources import resource_filename
from pylithics.image_processing.utils import read_metadata
from pylithics.image_processing.image_analysis import analyze_image_contours


### CONFIGURATION LOADER ###

def load_preprocessing_config(config_file=None):
    """
    Load configuration settings from a YAML file.
    """
    if config_file and os.path.isabs(config_file):
        config_path = config_file
    elif os.getenv('PYLITHICS_CONFIG'):
        config_path = os.getenv('PYLITHICS_CONFIG')
    else:
        config_path = resource_filename(__name__, '../config/config.yaml')

    logging.info("Attempting to load config file from: %s", config_path)

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logging.info("Loaded configuration from %s.", config_path)
        return config
    except FileNotFoundError:
        logging.error("Configuration file %s not found.", config_path)
        return None
    except yaml.YAMLError as yaml_error:
        logging.error("Failed to parse YAML file %s: %s", config_path, yaml_error)
        return None
    except OSError as os_error:
        logging.error("Failed to load config file %s due to OS error: %s", config_path, os_error)
        return None


### IMAGE PROCESSING FUNCTIONS ###

def read_image_from_path(image_path):
    """Load an image from the specified path using OpenCV."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image at %s could not be loaded." % image_path)
        logging.info("Loaded image: %s", image_path)
        return image
    except ValueError as value_error:
        logging.error("Image loading error: %s", value_error)
        return None
    except OSError as os_error:
        logging.error("Failed to load image %s due to OS error: %s", image_path, os_error)
        return None

def apply_grayscale_conversion(image, config):
    """Convert the input image to grayscale based on config settings."""
    if not config.get('grayscale_conversion', {}).get('enabled', True):
        return image

    method = config['grayscale_conversion'].get('method', 'standard')
    try:
        if method == "standard":
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            logging.info("Converted image to standard grayscale.")
        elif method == "clahe":
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_image = clahe.apply(gray_image)
            logging.info("Converted image to CLAHE grayscale.")
        else:
            raise ValueError(f"Unsupported grayscale conversion method: {method}")
    except ValueError as value_error:
        logging.error("Error in grayscale conversion: %s", value_error)
        return None

    return gray_image

def apply_contrast_normalization(gray_image, config):
    """Normalize the grayscale image by stretching contrast based on config settings."""
    if not config.get('normalization', {}).get('enabled', True):
        return gray_image

    method = config['normalization'].get('method', 'minmax')
    try:
        if method == "minmax":
            normalized_image = cv2.normalize(
                gray_image, None, alpha=config['normalization'].get('clip_values', [0, 255])[0],
                beta=config['normalization'].get('clip_values', [0, 255])[1], norm_type=cv2.NORM_MINMAX
            )
            logging.info("Applied Min-Max normalization.")
        elif method == "zscore":
            mean = gray_image.mean()
            std = gray_image.std()
            normalized_image = (gray_image - mean) / std
            logging.info("Applied Z-score normalization.")
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
    except ValueError as value_error:
        logging.error("Normalization error: %s", value_error)
        return None

    return normalized_image


def perform_thresholding(normalized_image, config):
    """Apply various thresholding methods based on the configuration."""
    try:
        method = config['thresholding'].get('method', 'default')
        max_value = config['thresholding'].get('max_value', 255)

        blurred_image = cv2.GaussianBlur(normalized_image, (5, 5), 0)
        logging.info("Applied Gaussian blur for noise reduction.")

        threshold_methods = {
            "adaptive": lambda: cv2.adaptiveThreshold(
                blurred_image, max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
            "simple": lambda: cv2.threshold(blurred_image, config['thresholding'].get('threshold_value', 127),
                                            max_value, cv2.THRESH_BINARY)[1],
            "otsu": lambda: cv2.threshold(blurred_image, 0, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            "default": lambda: cv2.adaptiveThreshold(
                blurred_image, max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        }

        if method not in threshold_methods:
            raise ValueError(f"Unsupported thresholding method: {method}")

        thresholded_image = threshold_methods[method]()
        logging.info("Applied %s thresholding.", method)
        return thresholded_image

    except ValueError as value_error:
        logging.error("Thresholding error: %s", value_error)
        return None
    except OSError as os_error:
        logging.error("OS error during thresholding: %s", os_error)
        return None


### IMAGE VALIDATION ###

def verify_image_dpi_and_scale(image_path, real_world_scale_mm):
    """Validate the DPI of the image and calculate the conversion factor between pixels and millimeters."""
    try:
        with Image.open(image_path) as img:
            dpi = img.info.get('dpi')
            if dpi is None:
                logging.warning("DPI information missing for %s", image_path)
                return None

            pixels_per_mm = dpi[0] / 25.4
            scale_length_pixels = real_world_scale_mm * pixels_per_mm
            logging.info("Image DPI: %f, Real-world scale (mm): %f, Scale bar in pixels: %f",
                         dpi[0], real_world_scale_mm, scale_length_pixels)

            return pixels_per_mm
    except OSError as os_error:
        logging.error("OS error loading image %s: %s", image_path, os_error)
        return None


### FILE MANAGEMENT ###

COMMON_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif']

def locate_image_file(image_dir, image_name):
    """Find the image file in the directory, allowing the file extension to be omitted in the input."""
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

def execute_preprocessing_pipeline(image_path, config):
    """
    A high-level function to preprocess an image by loading, converting to grayscale,
    normalizing, and applying thresholding.
    :param image_path: Path to the image file.
    :param config: Configuration dictionary for preprocessing.
    :return: The processed image or None if any step fails.
    """
    image = read_image_from_path(image_path)
    if image is None:
        return None

    gray_image = apply_grayscale_conversion(image, config)
    if gray_image is None:
        return None

    normalized_image = apply_contrast_normalization(gray_image, config)
    if normalized_image is None:
        return None

    thresholded_image = perform_thresholding(normalized_image, config)

    if thresholded_image is not None and thresholded_image.any():
        return thresholded_image
    else:
        return None


### MAIN IMAGE IMPORT FUNCTION ###

def preprocess_images(data_dir, meta_file, show_thresholded_images):
    """
    Preprocess each image and return a dictionary of processed images and their conversion factors.
    """
    images_dir = os.path.join(data_dir, 'images')
    config = load_preprocessing_config("config.yaml")
    if config is None:
        logging.error("Configuration could not be loaded. Exiting.")
        return {}

    metadata = read_metadata(meta_file)
    preprocessed_images = {}

    for entry in metadata:
        image_id = entry['image_id']
        real_world_scale_mm = float(entry['scale']) if entry['scale'] else None
        image_path = os.path.join(images_dir, image_id)

        processed_image = execute_preprocessing_pipeline(image_path, config)
        if processed_image is None:
            logging.error("Skipping analysis for %s due to preprocessing failure.", image_id)
            continue

        conversion_factor = verify_image_dpi_and_scale(image_path, real_world_scale_mm)
        if conversion_factor is None:
            logging.error("Skipping analysis for %s due to DPI mismatch.", image_id)
            continue

        preprocessed_images[image_id] = (processed_image, conversion_factor)

    return preprocessed_images