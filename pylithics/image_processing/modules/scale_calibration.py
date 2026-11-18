"""
Scale bar calibration module for PyLithics.

This module provides functionality to detect and measure scale bars in images
to calculate pixel-to-millimeter conversion factors.

The scale image holds the bar and, when ``pylithics-pages`` cut it, the
caption that states its length. The measure is the bar alone. With a
``scale_box`` from the metadata the bar is seeded inside that box. The
bar is the largest piece of ink and the solid blocks in line with it:
the blocks of a segmented bar, never a caption, a numeral or a speck.
"""

import cv2
import numpy as np
import logging
import os
from typing import Dict, Optional, Sequence, Tuple

from ..importer import parse_ignore_boxes

SCALE_DIRNAME = 'scales'
SCALE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
# A separate piece is a block of the bar when it is as thick as the seed
# block within this fraction, at least this fraction of its length, solid
# to this density, and no further from the chain than this many seed
# lengths. The blocks of a segmented bar are all one thickness and one
# length, filled, and a missing open block apart. A numeral as tall as the
# bar is thick (a "5" after it) is hollow; a frame is hollow; a speck is
# short; a caption below the bar is not across its axis.
THICKNESS_TOLERANCE = 0.2
SEGMENT_MIN_LENGTH = 0.5
SEGMENT_MIN_DENSITY = 0.8
SEGMENT_MAX_GAP = 1.5
# A bar measured from the box pylithics-pages recorded is the bar: high, not certain.
BOXED_CONFIDENCE = 0.95


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
    scale_image_path: str, config: Dict, debug_dir: Optional[str] = None,
    scale_box: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[Tuple[int, float]]:
    """
    Measure the scale bar in a scale image.

    Handles solid bars, segmented bars and ruled bars with ticks. With a
    ``scale_box`` (from ``meta_data.csv``, in the image's pixels) the bar
    is measured inside that box; without one, see :func:`bar_box`.

    Args:
        scale_image_path: Path to the scale bar image
        config: Configuration dictionary for scale detection
        debug_dir: Folder for the debug image, or None
        scale_box: The bar's box in the image, or None

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

        box = bar_box(binary, scale_box)
        if box is None:
            logging.warning(f"No black pixels in the scale image: {scale_image_path}")
            return None
        if config.get('debug_output', False) and debug_dir:
            save_debug_image(scale_image_path, binary, *box, debug_dir)
        length, confidence = _length_and_confidence(box)
        return length, (BOXED_CONFIDENCE if scale_box is not None else confidence)

    except (cv2.error, ValueError, IOError) as e:
        logging.error(
            f"Error in the scale bar detection in "
            f"{scale_image_path}: {e}"
        )
        return None


def _length_and_confidence(box: Tuple[int, int, int, int]) -> Tuple[int, float]:
    """
    Return the bar's length, its long side, and a confidence from its aspect ratio.

    A bar is much longer than it is thick; ten times as long gives 1.0.
    """
    _, _, w, h = box
    length = max(w, h)
    confidence = min(1.0, length / max(min(w, h), 1) / 10.0)
    logging.debug(f"Scale bar detected: {length} pixels, "
                  f"confidence: {confidence:.2f}, dimensions: {w}x{h}")
    return length, confidence


def bar_box(
    binary: np.ndarray, scale_box: Optional[Tuple[int, int, int, int]] = None
) -> Optional[Tuple[int, int, int, int]]:
    """
    Return the bar's bounding box ``(x, y, w, h)`` in a binary scale image, or None.

    The bar is seeded by the largest connected piece of ink, by pixel
    count, and grown by the solid blocks in line with it: each as thick
    as the seed within ``THICKNESS_TOLERANCE``, at least
    ``SEGMENT_MIN_LENGTH`` of its length, overlapping it across the axis,
    and no further from the chain than ``SEGMENT_MAX_GAP`` seed lengths,
    the space of a missing open block. A caption, a numeral, a speck or
    a frame drawn around the scale fails one of these and is left out.
    With ``scale_box`` the seed is taken inside that box, where the bar
    is; the blocks are still gathered from the whole image, so a bar the
    box cut short is measured whole.
    """
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    if count <= 1:
        return None
    pieces = [tuple(int(v) for v in stats[i, :5]) for i in range(1, count)]
    seed = _seed_piece(pieces, scale_box)
    if scale_box is not None:
        logging.debug("Scale bar seeded inside scale_box %s", scale_box)
    chain = _grow_chain(seed, [p for p in pieces if p is not seed and _is_segment_of(p, seed)])
    if len(chain) < len(pieces):
        logging.debug("Scale bar: %d of %d ink pieces left out as not part of the bar",
                      len(pieces) - len(chain), len(pieces))
    return _union(chain)


def _seed_piece(pieces: Sequence[Sequence[int]], scale_box) -> Sequence[int]:
    """Return the piece with the most ink, among those centred in the box when given."""
    inside = pieces
    if scale_box is not None:
        x0, y0, x1, y1 = scale_box
        inside = [p for p in pieces
                  if x0 <= p[0] + p[2] / 2 <= x1 and y0 <= p[1] + p[3] / 2 <= y1] or pieces
    return max(inside, key=lambda p: p[4])


def _is_segment_of(piece: Sequence[int], seed: Sequence[int]) -> bool:
    """Say whether a piece is a solid block of the seed's thickness, in line with it."""
    horizontal = seed[2] >= seed[3]
    along, across = (2, 3) if horizontal else (3, 2)
    pos = 1 if horizontal else 0
    overlaps = piece[pos] < seed[pos] + seed[across] and seed[pos] < piece[pos] + piece[across]
    as_thick = abs(piece[across] - seed[across]) <= THICKNESS_TOLERANCE * seed[across]
    long_enough = piece[along] >= SEGMENT_MIN_LENGTH * seed[along]
    solid = piece[4] >= SEGMENT_MIN_DENSITY * piece[2] * piece[3]
    return overlaps and as_thick and long_enough and solid


def _grow_chain(seed: Sequence[int], blocks: Sequence[Sequence[int]]) -> list:
    """Return the seed and the blocks reachable from it along the axis, gap by gap."""
    horizontal = seed[2] >= seed[3]
    pos, length = (0, 2) if horizontal else (1, 3)
    reach = SEGMENT_MAX_GAP * seed[length]
    start, end = seed[pos], seed[pos] + seed[length]
    chain = [seed]
    for block in sorted(blocks, key=lambda b: b[pos]):          # rightwards, then leftwards
        if block[pos] >= end and block[pos] - end <= reach:
            chain.append(block)
            end = block[pos] + block[length]
    for block in sorted(blocks, key=lambda b: -(b[pos] + b[length])):
        block_end = block[pos] + block[length]
        if block_end <= start and start - block_end <= reach:
            chain.append(block)
            start = block[pos]
    return chain


def _union(pieces: Sequence[Sequence[int]]) -> Tuple[int, int, int, int]:
    """Return the box ``(x, y, w, h)`` that holds every piece."""
    x0 = min(p[0] for p in pieces)
    y0 = min(p[1] for p in pieces)
    x1 = max(p[0] + p[2] for p in pieces)
    y1 = max(p[1] + p[3] for p in pieces)
    return x0, y0, x1 - x0, y1 - y0


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


def _measure_scale(
    image_path: str, scale_data: Dict, config: Dict, debug_dir: Optional[str]
) -> Optional[Tuple[int, float]]:
    """Find the scale image named in the metadata row and measure its bar, or warn."""
    scale_image_path = _find_scale_image(image_path, scale_data['scale_id'])
    if scale_image_path is None:
        logging.warning(
            f"The scale image '{scale_data['scale_id']}' is missing from the "
            f"scales/ directory near {os.path.dirname(image_path)}"
        )
        return None
    boxes = parse_ignore_boxes(scale_data.get('scale_box', ''), 'scale_box')
    result = detect_scale_bar(
        scale_image_path, config.get('scale_calibration', {}), debug_dir,
        boxes[0] if boxes else None,
    )
    if result is None:
        logging.warning(f"No scale bar found in {scale_image_path}")
    return result


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
        result = _measure_scale(image_path, scale_data, config, debug_dir)
        if result is None:
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