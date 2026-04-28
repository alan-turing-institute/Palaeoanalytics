"""
Image preprocessing and pipeline management for PyLithics.

This module handles the complete image preprocessing pipeline
including grayscale conversion, normalization, thresholding,
and morphological operations.
"""

import logging
import os
import yaml
from typing import Optional, Dict, Tuple

import cv2
import numpy as np
from PIL import Image

from .config import load_preprocessing_config
from .utils import read_metadata


### DPI-AWARE PROCESSING FUNCTIONS ###

def get_image_dpi(image_path: str) -> float:
    """
    Extract DPI from image metadata, default to 300 if missing.

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Returns
    -------
    float
        DPI value, defaults to 300 if not found.
    """
    try:
        with Image.open(image_path) as img:
            dpi = img.info.get('dpi')
            if dpi is None:
                logging.warning(
                    f"DPI missing for {image_path}, "
                    f"defaulting to 300"
                )
                return 300.0
            image_dpi = (
                float(dpi[0]) if isinstance(dpi, tuple)
                else float(dpi)
            )
            logging.info(f"Image DPI detected: {image_dpi}")
            return image_dpi
    except Exception as e:
        logging.error(
            f"Error reading DPI from {image_path}: {e}, "
            f"defaulting to 300"
        )
        return 300.0


def calculate_dpi_scale_factor(
    image_dpi: float, config: Dict
) -> float:
    """
    Calculate kernel scaling factor for archaeological drawings.

    Parameters
    ----------
    image_dpi : float
        DPI of the current image.
    config : dict
        Configuration dictionary.

    Returns
    -------
    float
        Scale factor for kernel sizing.
    """
    dpi_config = config.get('dpi_processing', {})

    if not dpi_config.get('enabled', True):
        logging.info(
            "DPI-aware processing disabled, using 1.0"
        )
        return 1.0

    reference_dpi = dpi_config.get('reference_dpi', 300.0)
    max_scale = dpi_config.get('max_scale_factor', 1.5)
    mode = dpi_config.get('scaling_mode', 'standard')

    raw_scale = image_dpi / reference_dpi

    if mode == 'conservative':
        conservative = 1.0 + (raw_scale - 1.0) * 0.5
        scale = max(1.0, min(conservative, max_scale))
        label = "conservative"
    elif mode == 'standard':
        scale = max(1.0, min(raw_scale, max_scale))
        label = "standard"
    else:
        scale = max(1.0, raw_scale)
        label = "aggressive"

    if raw_scale > max_scale and mode != 'aggressive':
        logging.warning(
            f"DPI scale {raw_scale:.2f} "
            f"limited to {max_scale:.2f}"
        )

    logging.info(
        f"DPI scale: {scale:.2f} "
        f"(image: {image_dpi}, ref: {reference_dpi}, "
        f"mode: {label})"
    )
    return scale


def ensure_odd_kernel_size(size: float) -> int:
    """Ensure kernel size is odd and at least 1."""
    size = max(1, int(size))
    return size if size % 2 == 1 else size + 1


### IMAGE PROCESSING FUNCTIONS ###

def read_image_from_path(
    image_path: str
) -> Optional[np.ndarray]:
    """Load an image from the specified path using OpenCV."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(
                f"Image at {image_path} could not be loaded."
            )
        logging.info("Loaded image: %s", image_path)
        return image
    except ValueError as e:
        logging.error("Image loading error: %s", e)
        return None
    except OSError as e:
        logging.error(
            "Failed to load image %s: %s", image_path, e
        )
        return None


def apply_grayscale_conversion(
    image: np.ndarray, config: Dict
) -> Optional[np.ndarray]:
    """Convert the input image to grayscale."""
    grayscale_config = config.get('grayscale_conversion', {})
    if not grayscale_config.get('enabled', True):
        return image

    method = grayscale_config.get('method', 'standard')
    try:
        if method == "standard":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            logging.info("Converted to standard grayscale.")
        elif method == "clahe":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(
                clipLimit=2.0, tileGridSize=(8, 8)
            )
            gray = clahe.apply(gray)
            logging.info("Converted to CLAHE grayscale.")
        else:
            raise ValueError(
                f"Unsupported grayscale method: {method}"
            )
    except ValueError as e:
        logging.error("Grayscale conversion error: %s", e)
        return None

    return gray


def apply_contrast_normalization(
    gray_image: np.ndarray, config: Dict
) -> Optional[np.ndarray]:
    """Normalize grayscale image by stretching contrast."""
    norm_config = config.get('normalization', {})
    if not norm_config.get('enabled', True):
        return gray_image

    method = norm_config.get('method', 'minmax')
    clip = norm_config.get('clip_values', [0, 255])

    try:
        if method == "minmax":
            normalized = cv2.normalize(
                gray_image, None,
                alpha=clip[0], beta=clip[1],
                norm_type=cv2.NORM_MINMAX
            )
            logging.info("Applied Min-Max normalization.")
        elif method == "zscore":
            mean = gray_image.mean()
            std = gray_image.std()
            normalized = (gray_image - mean) / std
            logging.info("Applied Z-score normalization.")
        else:
            raise ValueError(
                f"Unsupported normalization method: {method}"
            )
    except ValueError as e:
        logging.error("Normalization error: %s", e)
        return None

    return normalized


def perform_thresholding(
    normalized_image: np.ndarray,
    config: Dict,
    dpi_scale: float = 1.0
) -> Optional[np.ndarray]:
    """Apply thresholding with DPI-aware kernel scaling."""
    try:
        thresh_config = config.get('thresholding', {})
        method = thresh_config.get('method', 'default')
        max_val = thresh_config.get('max_value', 255)
        thresh_val = thresh_config.get('threshold_value', 127)

        base_kernels = config.get('dpi_processing', {}).get('base_kernels', {})
        base_blur = base_kernels.get('gaussian_blur', 5)
        base_block = base_kernels.get('adaptive_block', 11)

        blurred = _gaussian_blur_with_log(
            normalized_image, base_blur, dpi_scale,
        )
        result = _apply_threshold_method(
            blurred, method, max_val, thresh_val, base_block, dpi_scale,
        )
        logging.info("Applied %s thresholding.", method)
        return result

    except ValueError as e:
        logging.error("Thresholding error: %s", e)
        return None
    except OSError as e:
        logging.error("OS error during thresholding: %s", e)
        return None


def _gaussian_blur_with_log(
    image: np.ndarray, base_blur: int, dpi_scale: float,
) -> np.ndarray:
    """Apply Gaussian blur with DPI-scaled kernel size."""
    blur_size = ensure_odd_kernel_size(base_blur * dpi_scale)
    blurred = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
    logging.info(
        f"Gaussian blur: {blur_size}x{blur_size} "
        f"(base: {base_blur}, scale: {dpi_scale:.2f})"
    )
    return blurred


def _apply_threshold_method(
    blurred: np.ndarray,
    method: str,
    max_val: int,
    thresh_val: int,
    base_block: int,
    dpi_scale: float,
) -> np.ndarray:
    """Dispatch to the requested thresholding algorithm."""
    if method == "adaptive":
        block = ensure_odd_kernel_size(base_block * dpi_scale)
        logging.info(
            f"Adaptive threshold: {block}x{block} "
            f"(base: {base_block}, scale: {dpi_scale:.2f})"
        )
        return cv2.adaptiveThreshold(
            blurred, max_val,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block, 2,
        )
    if method == "otsu":
        return cv2.threshold(
            blurred, 0, max_val,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[1]
    if method in ("simple", "default"):
        return cv2.threshold(
            blurred, thresh_val, max_val, cv2.THRESH_BINARY,
        )[1]
    raise ValueError(f"Unsupported thresholding method: {method}")


def invert_image(
    thresholded_image: np.ndarray
) -> np.ndarray:
    """Invert a thresholded image."""
    return cv2.bitwise_not(thresholded_image)


def morphological_closing(
    inverted_image: np.ndarray,
    config: Dict,
    dpi_scale: float = 1.0
) -> np.ndarray:
    """
    Apply DPI-aware morphological closing.

    Parameters
    ----------
    inverted_image : np.ndarray
        Input inverted binary image.
    config : dict
        Configuration dictionary.
    dpi_scale : float
        DPI scaling factor for kernel size.

    Returns
    -------
    np.ndarray
        Image after morphological closing.
    """
    base_kernels = config.get(
        'dpi_processing', {}
    ).get('base_kernels', {})
    morph_config = config.get('morphological_closing', {})
    base_size = base_kernels.get(
        'morphological', morph_config.get('kernel_size', 3)
    )

    scaled_size = max(1, int(base_size * dpi_scale))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (scaled_size, scaled_size)
    )
    closed = cv2.morphologyEx(
        inverted_image, cv2.MORPH_CLOSE, kernel
    )
    logging.info(
        f"Morphological closing: {scaled_size}x{scaled_size} "
        f"(base: {base_size}, scale: {dpi_scale:.2f})"
    )
    return closed


### IMAGE VALIDATION ###

def verify_image_dpi_and_scale(
    image_path: str,
    real_world_scale_mm: float
) -> Optional[float]:
    """
    Validate image DPI and calculate pixel-to-mm conversion.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    real_world_scale_mm : float
        Scale length in millimeters.

    Returns
    -------
    float or None
        Pixels per millimeter, or None if DPI unavailable.
    """
    try:
        with Image.open(image_path) as img:
            dpi = img.info.get('dpi')
            if dpi is None:
                logging.warning(
                    "DPI missing for %s", image_path
                )
                return None

            pixels_per_mm = dpi[0] / 25.4
            scale_px = real_world_scale_mm * pixels_per_mm
            logging.info(
                "DPI: %.2f, Scale (mm): %.2f, "
                "Scale (px): %.2f",
                round(dpi[0], 2),
                round(real_world_scale_mm, 2),
                round(scale_px, 2)
            )
            return pixels_per_mm
    except OSError as e:
        logging.error(
            "OS error loading image %s: %s",
            image_path, e
        )
        return None


### IMAGE PREPROCESSING PIPELINE ###

def execute_preprocessing_pipeline(
    image_path: str, config: Dict
) -> Optional[np.ndarray]:
    """
    Preprocess an image through the complete pipeline.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    config : dict
        Configuration dictionary.

    Returns
    -------
    np.ndarray or None
        Preprocessed image, or None on failure.
    """
    image_dpi = get_image_dpi(image_path)
    dpi_scale = calculate_dpi_scale_factor(image_dpi, config)

    image = read_image_from_path(image_path)
    if image is None:
        return None

    gray = apply_grayscale_conversion(image, config)
    if gray is None:
        return None

    normalized = apply_contrast_normalization(gray, config)
    if normalized is None:
        return None

    thresholded = perform_thresholding(
        normalized, config, dpi_scale
    )
    if thresholded is None:
        return None

    inverted = invert_image(thresholded)
    return morphological_closing(inverted, config, dpi_scale)


def preprocess_images(
    data_dir: str,
    meta_file: str,
    show_thresholded_images: bool
) -> Dict[str, Tuple[np.ndarray, float]]:
    """
    Preprocess all images and return with conversion factors.

    Parameters
    ----------
    data_dir : str
        Directory containing images.
    meta_file : str
        Path to metadata CSV.
    show_thresholded_images : bool
        Whether to display thresholded images.

    Returns
    -------
    dict
        Map of image_id to (processed_image, conversion_factor).
    """
    images_dir = os.path.join(data_dir, 'images')
    config = load_preprocessing_config("config.yaml")
    if config is None:
        logging.error("Configuration could not be loaded.")
        return {}

    metadata = read_metadata(meta_file)
    preprocessed = {}

    for entry in metadata:
        image_id = entry['image_id']
        scale_mm = (
            float(entry['scale']) if entry['scale']
            else None
        )
        image_path = os.path.join(images_dir, image_id)

        processed = execute_preprocessing_pipeline(
            image_path, config
        )
        if processed is None:
            logging.error(
                "Skipping %s: preprocessing failed.",
                image_id
            )
            continue

        conversion = verify_image_dpi_and_scale(
            image_path, scale_mm
        )
        if conversion is None:
            logging.error(
                "Skipping %s: DPI mismatch.", image_id
            )
            continue

        preprocessed[image_id] = (processed, conversion)

    return preprocessed
