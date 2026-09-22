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



SCALE_DIRNAME = 'scales'
SCALE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']


def _find_scale_image(image_path: str, scale_id: str):
    """
    Locate a scale image for an artefact, allowing for either layout.

    Images normally sit in ``<project>/images/`` with their scales in
    ``<project>/scales/``. When a folder of images is analysed where it
    stands, ``scales/`` sits beside the images instead. Both are tried,
    with and without a file extension.

    Parameters
    ----------
    image_path : str
        Path to the artefact image being calibrated.
    scale_id : str
        Scale image name from the metadata, extension optional.

    Returns
    -------
    str or None
        Path to the scale image, or None when no candidate exists.
    """
    image_dir = os.path.dirname(image_path)
    roots = [os.path.dirname(image_dir), image_dir]

    for root in roots:
        base = os.path.join(root, SCALE_DIRNAME, scale_id)
        if os.path.exists(base):
            return base
        for ext in SCALE_EXTENSIONS:
            candidate = base + ext
            if os.path.exists(candidate):
                logging.debug(f"Found scale image: {candidate}")
                return candidate
    return None


def detect_scale_bar(
    scale_image_path: str, config: Dict, debug_dir: Optional[str] = None
) -> Optional[Tuple[int, float]]:
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
            logging.error(f"Cannot read the scale image: {scale_image_path}")
            return None

        # Threshold to binary (scale bars are typically black on white)
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

        # Find all non-zero (black) pixels
        points = cv2.findNonZero(binary)
        if points is None:
            logging.warning(f"No black pixels in the scale image: {scale_image_path}")
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

        if config.get('debug_output', False) and debug_dir:
            save_debug_image(scale_image_path, binary, x, y, w, h, debug_dir)

        return scale_length_pixels, confidence

    except (cv2.error, ValueError, IOError) as e:
        logging.error(
            f"Error in the scale bar detection in "
            f"{scale_image_path}: {e}"
        )
        return None


def save_debug_image(scale_image_path: str, binary_image: np.ndarray,
                    x: int, y: int, w: int, h: int, debug_dir: str) -> None:
    """
    Save debug image showing detected scale bar bounding box.

    Written as ``<debug_dir>/<scale image name>.png``: the folder says
    which step, the file name says which scale image.

    Args:
        scale_image_path: Original scale image path
        binary_image: Binary threshold image
        x, y, w, h: Bounding box coordinates
        debug_dir: ``results/scale_debug``
    """
    try:
        os.makedirs(debug_dir, exist_ok=True)

        # Draw bounding box on binary image
        debug_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add text with measurements
        text = f"Scale: {max(w, h)}px"
        cv2.putText(debug_image, text, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Save debug image
        stem = os.path.splitext(os.path.basename(scale_image_path))[0]
        debug_path = os.path.join(debug_dir, f"{stem}.png")
        cv2.imwrite(debug_path, debug_image)
        logging.debug(f"Saved scale debug image: {debug_path}")

    except (cv2.error, IOError, OSError) as e:
        logging.warning(f"Cannot write the debug image: {e}")


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


def get_calibration_factor(
    image_path: str, scale_data: Dict, config: Dict, debug_dir: Optional[str] = None
) -> Tuple[Optional[float], str, Optional[float]]:
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
        scale_image_path = _find_scale_image(
            image_path, scale_data['scale_id']
        )
        if scale_image_path is None:
            logging.warning(
                f"The scale image '{scale_data['scale_id']}' is missing from the "
                f"scales/ directory near {os.path.dirname(image_path)}"
            )
            return None, "pixels_detection_failed", None

        # Detect and measure scale bar
        result = detect_scale_bar(
            scale_image_path, config.get('scale_calibration', {}), debug_dir
        )
        if result is None:
            logging.warning(
                f"No scale bar found in {scale_image_path}"
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
        logging.warning(f"Scale bar calibration error: {e}")
        return None, "pixels_detection_failed", None