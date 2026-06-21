"""
PyLithics: Symmetry Analysis
=============================

Calculates area-based symmetry metrics for dorsal surfaces
using geometric centroid analysis.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional


def analyze_dorsal_symmetry(
    metrics: List[Dict],
    contours: List[np.ndarray],
    inverted_image: np.ndarray
) -> Dict[str, Optional[float]]:
    """
    Calculate symmetry metrics for the dorsal surface.

    Splits the dorsal contour at its centroid into four quadrants
    and measures area ratios to quantify vertical and horizontal
    symmetry.

    Parameters
    ----------
    metrics : list
        Metric dictionaries with surface classifications.
    contours : list
        Contour arrays corresponding to metrics.
    inverted_image : np.ndarray
        Inverted binary thresholded image.

    Returns
    -------
    dict
        Symmetry results:
        - top_area, bottom_area, left_area, right_area (float)
        - vertical_symmetry, horizontal_symmetry (float 0-1)
        Values are None if analysis cannot be performed.
    """
    empty = {
        "top_area": None, "bottom_area": None,
        "left_area": None, "right_area": None,
    }

    dorsal_metric = next(
        (m for m in metrics
         if m.get("surface_type") == "Dorsal"),
        None
    )
    if not dorsal_metric:
        return empty

    contour = _find_dorsal_contour(
        metrics, contours, dorsal_metric
    )
    if contour is None or len(contour) < 3:
        return empty

    cx = int(dorsal_metric["centroid_x"])
    cy = int(dorsal_metric["centroid_y"])

    if cv2.pointPolygonTest(contour, (cx, cy), False) < 0:
        return empty

    return _calculate_symmetry(contour, inverted_image, cx, cy)


def _find_dorsal_contour(
    metrics: List[Dict],
    contours: List[np.ndarray],
    dorsal_metric: Dict
) -> Optional[np.ndarray]:
    """
    Find the contour matching the dorsal surface metric.

    Parameters
    ----------
    metrics : list
        All metric dictionaries.
    contours : list
        Contour arrays.
    dorsal_metric : dict
        The dorsal surface metric.

    Returns
    -------
    np.ndarray or None
        Dorsal contour, or None if not found.
    """
    parent_label = dorsal_metric["parent"]
    for i, contour in enumerate(contours):
        if i < len(metrics) and metrics[i]["parent"] == parent_label:
            return contour
    return None


def _calculate_symmetry(
    contour: np.ndarray,
    inverted_image: np.ndarray,
    cx: int, cy: int
) -> Dict[str, Optional[float]]:
    """
    Calculate symmetry areas and ratios from a contour.

    Parameters
    ----------
    contour : np.ndarray
        Dorsal surface contour.
    inverted_image : np.ndarray
        Image for mask dimensions.
    cx : int
        Centroid x coordinate.
    cy : int
        Centroid y coordinate.

    Returns
    -------
    dict
        Symmetry areas and ratios.
    """
    mask = np.zeros_like(inverted_image, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)

    top = round(float(np.sum(mask[:cy, :] == 255)), 2)
    bottom = round(float(np.sum(mask[cy:, :] == 255)), 2)
    left = round(float(np.sum(mask[:, :cx] == 255)), 2)
    right = round(float(np.sum(mask[:, cx:] == 255)), 2)

    v_sym = (
        round(1 - abs(top - bottom) / (top + bottom), 2)
        if (top + bottom) > 0 else None
    )
    h_sym = (
        round(1 - abs(left - right) / (left + right), 2)
        if (left + right) > 0 else None
    )

    logging.debug("Symmetry analysis complete for Dorsal surface.")

    return {
        "top_area": top, "bottom_area": bottom,
        "left_area": left, "right_area": right,
        "vertical_symmetry": v_sym,
        "horizontal_symmetry": h_sym,
    }
