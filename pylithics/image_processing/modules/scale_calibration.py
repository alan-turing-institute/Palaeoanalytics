"""
Scale bar calibration module for PyLithics.

This module provides functionality to detect and measure scale bars in images
to calculate pixel-to-millimeter conversion factors. It implements a simple
bounding box approach that works with various scale bar styles.
"""

import cv2
import numpy as np
import logging
import os
from typing import Dict, Optional, Tuple

from ..config import get_config_manager


def detect_scale_bar(scale_image_path: str, config: Dict) -> Optional[Tuple[int, float]]:
    """
    Detect and measure scale bar in scale image.

    Handles various scale bar styles: simple lines, segmented bars, bars with tick marks.
    Measures the full extent including all elements (segments, ticks, brackets).

    Args:
        scale_image_path: Path to the scale bar image
        config: Configuration dictionary for scale detection

    Returns:
        tuple[int, float]: (scale_length_pixels, confidence_score) or None if failed
    """
    try:
        # Load the scale image
        image = cv2.imread(scale_image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            logging.error(f"Failed to load scale image: {scale_image_path}")
            return None

        # Threshold to binary (scale bars are typically black on white)
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

        # Find all non-zero (black) pixels
        points = cv2.findNonZero(binary)
        if points is None:
            logging.warning(f"No black pixels found in scale image: {scale_image_path}")
            return None

        # Get bounding box of all black elements
        x, y, w, h = cv2.boundingRect(points)

        # Measure longest dimension (width or height)
        scale_length_pixels = max(w, h)

        # Calculate confidence based on aspect ratio
        # Good scale bars are typically much longer in one dimension
        aspect_ratio = max(w, h) / max(min(w, h), 1)  # Avoid division by zero
        confidence = min(1.0, aspect_ratio / 10.0)  # Higher aspect ratio = higher confidence

        logging.debug(f"Scale bar detected: {scale_length_pixels} pixels, "
                      f"confidence: {confidence:.2f}, dimensions: {w}x{h}")

        if config.get('debug_output', False):
            save_debug_image(scale_image_path, binary, x, y, w, h)

        return scale_length_pixels, confidence

    except (cv2.error, ValueError, IOError) as e:
        logging.error(
            f"Error detecting scale bar in "
            f"{scale_image_path}: {e}"
        )
        return None


def save_debug_image(scale_image_path: str, binary_image: np.ndarray,
                    x: int, y: int, w: int, h: int) -> None:
    """
    Save debug image showing detected scale bar bounding box.

    Args:
        scale_image_path: Original scale image path
        binary_image: Binary threshold image
        x, y, w, h: Bounding box coordinates
    """
    try:
        # Create debug output directory
        debug_dir = os.path.join(os.path.dirname(scale_image_path), '..',
                                'processed', 'scale_debug')
        os.makedirs(debug_dir, exist_ok=True)

        # Draw bounding box on binary image
        debug_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add text with measurements
        text = f"Scale: {max(w, h)}px"
        cv2.putText(debug_image, text, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Save debug image
        base_name = os.path.basename(scale_image_path)
        debug_path = os.path.join(debug_dir, f"debug_{base_name}")
        cv2.imwrite(debug_path, debug_image)
        logging.debug(f"Saved scale debug image: {debug_path}")

    except (cv2.error, IOError, OSError) as e:
        logging.warning(f"Failed to save debug image: {e}")


def calculate_conversion_factor(scale_pixels: int, scale_mm: float) -> float:
    """
    Calculate pixels per millimeter conversion factor.

    Args:
        scale_pixels: Measured scale bar length in pixels
        scale_mm: Real-world scale bar length from CSV

    Returns:
        float: pixels_per_mm conversion factor
    """
    if scale_mm <= 0:
        raise ValueError(f"Invalid scale value: {scale_mm} mm")

    pixels_per_mm = scale_pixels / scale_mm
    logging.debug(f"Conversion factor: {pixels_per_mm:.3f} pixels/mm "
                f"({scale_pixels} pixels = {scale_mm} mm)")
    return pixels_per_mm


def get_calibration_factor(image_path: str, scale_data: Dict,
                          config: Dict) -> Tuple[Optional[float], str, Optional[float]]:
    """
    Get calibration factor using two-option system with a three-way status.

    Args:
        image_path: Path to the artifact image
        scale_data: Dictionary with 'scale_id' and 'scale' from CSV
        config: Configuration dictionary

    Returns:
        tuple[float | None, str, float | None]:
            (pixels_per_mm, method_used, confidence) where method_used is one of:
            - ``"scale_bar"`` — detection succeeded.
            - ``"pixels_no_scale"`` — calibration disabled or no scale provided
              in the metadata; pixel mode is intentional.
            - ``"pixels_detection_failed"`` — scale was provided but the scale
              image was missing or detection returned None; pixel mode is a
              fallback. A WARNING has been logged with the underlying reason.
    """
    calibration_enabled = config.get('scale_calibration', {}).get('enabled', True)
    user_supplied_scale = bool(
        scale_data.get('scale_id') and scale_data.get('scale')
    )

    if not (calibration_enabled and user_supplied_scale):
        logging.debug(
            f"No scale calibration available for {os.path.basename(image_path)}, "
            "measurements will be in pixels"
        )
        return None, "pixels_no_scale", None

    try:
        # Build scale image path
        data_dir = os.path.dirname(os.path.dirname(image_path))
        scale_id = scale_data['scale_id']
        scale_image_path = os.path.join(data_dir, 'scales', scale_id)

        # If file doesn't exist, try adding common extensions
        if not os.path.exists(scale_image_path):
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
                test_path = os.path.join(data_dir, 'scales', scale_id + ext)
                if os.path.exists(test_path):
                    scale_image_path = test_path
                    logging.debug(f"Found scale image with extension: {test_path}")
                    break

        if not os.path.exists(scale_image_path):
            logging.warning(f"Scale image not found: {scale_image_path}")
            return None, "pixels_detection_failed", None

        # Detect and measure scale bar
        result = detect_scale_bar(scale_image_path,
                                config.get('scale_calibration', {}))
        if result is None:
            logging.warning(
                f"Scale bar detection returned no match for {scale_image_path}"
            )
            return None, "pixels_detection_failed", None

        scale_pixels, confidence = result
        scale_mm = float(scale_data['scale'])
        pixels_per_mm = calculate_conversion_factor(scale_pixels, scale_mm)
        logging.debug(
            f"Using scale bar calibration for {os.path.basename(image_path)}"
        )
        return pixels_per_mm, "scale_bar", confidence

    except (ValueError, IOError, OSError) as e:
        logging.warning(f"Scale bar calibration failed: {e}")
        return None, "pixels_detection_failed", None