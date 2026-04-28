"""
PyLithics: Cortex Detection Module
=================================

This module detects cortex on lithic artifact surfaces. Cortex is distinguished from scars
by its characteristic stippled/dotted texture pattern representing the original weathered
surface of the stone material.

Key Features:
- Detects cortex at child contour level (not parent or nested contours)
- Labels cortex as cortex_1, cortex_2, etc. (distinct from scars)
- Calculates cortex area and percentage of parent surface area
- Maintains archaeological accuracy: cortex ≠ scar

Author: PyLithics Development Team
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Any
from ..config import get_cortex_detection_config


def detect_cortex_in_child_contours(
    metrics: List[Dict[str, Any]],
    inverted_image: np.ndarray
) -> List[Dict[str, Any]]:
    """
    Detect cortex in child contours and relabel them by surface type.

    Args:
        metrics: Contour metrics with surface classification.
        inverted_image: Inverted binary image for texture analysis.

    Returns:
        Updated metrics with cortex areas relabelled and non-cortex children
        renumbered per-surface.
    """
    config = get_cortex_detection_config()
    if not config.get('enabled', True):
        logging.info("Cortex detection is disabled in configuration")
        return metrics

    logging.info("Starting cortex detection in child contours")

    try:
        parents = [m for m in metrics if m["parent"] == m["scar"]]
        children = [m for m in metrics if m["parent"] != m["scar"]]

        parent_areas = {p.get("scar", ""): p.get("area", 0) for p in parents}
        parent_surface = {
            p.get("scar", ""): p.get("surface_type", "Unknown") for p in parents
        }

        cortex_children, non_cortex_children = _classify_cortex_children(
            children, inverted_image, config
        )
        _label_cortex_children(cortex_children, parent_areas)
        renumbered = _renumber_non_cortex_children(
            non_cortex_children, parent_surface
        )

        logging.info(
            f"Cortex detection completed: {len(cortex_children)} cortex areas "
            f"detected, {len(renumbered)} non-cortex children processed"
        )
        return parents + cortex_children + renumbered

    except Exception as e:
        logging.error(f"Error in cortex detection: {e}")
        return metrics


def _classify_cortex_children(
    children: List[Dict[str, Any]],
    inverted_image: np.ndarray,
    config: Dict[str, Any],
) -> "tuple[List[Dict[str, Any]], List[Dict[str, Any]]]":
    """Split children into (cortex, non-cortex) based on texture analysis."""
    cortex, non_cortex = [], []
    for child in children:
        if not child.get("contour"):
            logging.warning(
                f"Child contour {child.get('scar', 'unknown')} "
                f"missing contour data"
            )
            non_cortex.append(child)
            continue

        contour_array = np.array(child["contour"], dtype=np.int32)
        if _detect_cortex_texture(contour_array, inverted_image, config):
            cortex.append(child)
            logging.debug(f"Detected cortex in child: {child.get('scar', 'unknown')}")
        else:
            non_cortex.append(child)
    return cortex, non_cortex


def _label_cortex_children(
    cortex_children: List[Dict[str, Any]],
    parent_areas: Dict[str, float],
) -> None:
    """Relabel cortex children as `cortex N` and add cortex-area metrics."""
    for i, child in enumerate(cortex_children, start=1):
        original = child.get("scar", "unknown")
        parent_area = parent_areas.get(child.get("parent", ""), 0)
        cortex_area = child.get("area", 0)
        percentage = (cortex_area / parent_area * 100) if parent_area > 0 else 0

        child["scar"] = f"cortex {i}"
        child["surface_feature"] = f"cortex {i}"
        child["cortex_area"] = cortex_area
        child["cortex_percentage"] = round(percentage, 2)
        child["is_cortex"] = True

        logging.info(
            f"Relabeled {original} -> {child['scar']} "
            f"(area: {cortex_area}, {percentage:.1f}% of surface)"
        )


def _renumber_non_cortex_children(
    non_cortex_children: List[Dict[str, Any]],
    parent_surface: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Rename Dorsal children to `scar N`, Lateral to `edge N`, drop Platform."""
    by_surface = {"Dorsal": [], "Platform": [], "Lateral": [], "Ventral": []}
    for child in non_cortex_children:
        surface = parent_surface.get(child.get("parent", ""), "Unknown")
        if surface in by_surface:
            by_surface[surface].append(child)

    renamed = []
    for i, child in enumerate(by_surface["Dorsal"], start=1):
        child["scar"] = f"scar {i}"
        child["surface_feature"] = f"scar {i}"
        child["is_cortex"] = False
        renamed.append(child)

    for child in by_surface["Platform"]:
        logging.info(
            f"Excluding platform child (area={child.get('area', 0)}) "
            f"as likely empty space boundary"
        )

    for i, child in enumerate(by_surface["Lateral"], start=1):
        child["scar"] = f"edge {i}"
        child["surface_feature"] = f"edge {i}"
        child["is_cortex"] = False
        renamed.append(child)

    return renamed


def _detect_cortex_texture(
    contour: np.ndarray,
    inverted_image: np.ndarray,
    config: Dict[str, Any]
) -> bool:
    """
    Decide whether a contour region shows cortex-like texture.

    Cortex surfaces are characterized by a stippled pattern with many small
    connected components, high local variance, and high edge density.
    """
    try:
        roi_cropped, mask_cropped = _crop_contour_roi(contour, inverted_image)
        if roi_cropped is None:
            return False

        roi_area = cv2.countNonZero(mask_cropped)
        if roi_area == 0:
            return False

        stippling = _stippling_density(roi_cropped, roi_area)
        variance = _texture_variance(roi_cropped, mask_cropped)
        edge_density = _edge_density(roi_cropped, roi_area)

        cortex_detected = (
            stippling > config.get('stippling_density_threshold', 0.2)
            and variance > config.get('texture_variance_threshold', 100)
            and edge_density > config.get('edge_density_threshold', 0.05)
        )
        logging.info(
            f"Cortex analysis: stippling_density={stippling:.3f}, "
            f"variance={variance:.1f}, edge_density={edge_density:.3f} -> "
            f"cortex={cortex_detected}"
        )
        return cortex_detected

    except Exception as e:
        logging.error(f"Error in cortex texture detection: {e}")
        return False


def _crop_contour_roi(
    contour: np.ndarray, inverted_image: np.ndarray,
):
    """Return (roi, mask) cropped to the contour's bounding box, or (None, None)."""
    mask = np.zeros(inverted_image.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [contour], 255)
    roi = cv2.bitwise_and(inverted_image, mask)

    x, y, w, h = cv2.boundingRect(contour)
    roi_cropped = roi[y:y + h, x:x + w]
    mask_cropped = mask[y:y + h, x:x + w]
    if roi_cropped.size == 0:
        return None, None
    return roi_cropped, mask_cropped


def _stippling_density(roi_cropped: np.ndarray, roi_area: int) -> float:
    """Count small connected components per 1000 pixels of ROI."""
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        roi_cropped, connectivity=8,
    )
    small = sum(
        1 for i in range(1, num_labels)
        if 1 <= stats[i, cv2.CC_STAT_AREA] <= 50
    )
    return small / roi_area * 1000


def _texture_variance(
    roi_cropped: np.ndarray, mask_cropped: np.ndarray,
) -> float:
    """Local variance of Gaussian-blurred ROI within the masked region."""
    if roi_cropped.shape[0] <= 5 or roi_cropped.shape[1] <= 5:
        return 0.0
    blurred = cv2.GaussianBlur(roi_cropped, (5, 5), 0)
    return float(np.var(blurred[mask_cropped > 0])) if np.any(mask_cropped) else 0.0


def _edge_density(roi_cropped: np.ndarray, roi_area: int) -> float:
    """Proportion of Canny-detected edge pixels relative to ROI area."""
    edges = cv2.Canny(roi_cropped, 50, 150)
    return float(np.sum(edges > 0)) / roi_area if roi_area > 0 else 0.0


def calculate_total_cortex_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate aggregate cortex metrics across all surfaces.

    Args:
        metrics (List[Dict]): List of contour metrics with cortex detection

    Returns:
        Dict[str, float]: Aggregate cortex statistics
    """
    cortex_areas = [m.get("cortex_area", 0) for m in metrics if m.get("is_cortex", False)]

    total_cortex_area = sum(cortex_areas)
    cortex_count = len(cortex_areas)
    average_cortex_size = total_cortex_area / cortex_count if cortex_count > 0 else 0

    return {
        "total_cortex_area": total_cortex_area,
        "cortex_count": cortex_count,
        "average_cortex_size": round(average_cortex_size, 2)
    }