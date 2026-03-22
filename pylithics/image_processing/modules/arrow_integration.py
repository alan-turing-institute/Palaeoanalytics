"""
PyLithics: Arrow Integration
=============================

Detects arrows in contours and associates them with scar metrics.
Arrow detection is excluded from cortex regions.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Set, Tuple

from .arrow_detection import analyze_child_contour_for_arrow
from ..config import get_arrow_integration_config


def integrate_arrows(
    sorted_contours: Dict[str, List],
    hierarchy: np.ndarray,
    original_contours: List[np.ndarray],
    metrics: List[Dict],
    image_shape,
    image_dpi: Optional[float] = None
) -> List[Dict]:
    """
    Main orchestrator for arrow integration.

    Parameters
    ----------
    sorted_contours : dict
        {"parents":..., "children":..., "nested_children":...}
    hierarchy : np.ndarray
        Contour hierarchy array.
    original_contours : list
        All detected contours.
    metrics : list
        Metric dictionaries to update with arrow info.
    image_shape : ndarray or tuple
        Source image or image shape.
    image_dpi : float, optional
        Image DPI for scaling detection parameters.

    Returns
    -------
    list
        Updated metrics with arrow information.
    """
    metrics = process_nested_arrows(
        sorted_contours, hierarchy, original_contours,
        metrics, image_shape, image_dpi
    )

    scars_without = [
        m for m in metrics
        if m["parent"] != m["scar"]
        and not m.get('has_arrow', False)
    ]

    if scars_without:
        logging.info(
            f"{len(scars_without)} scars without arrows. "
            f"Running independent detection."
        )
        image = _resolve_image(image_shape)
        metrics = detect_arrows_independently(
            original_contours, metrics, image, image_dpi
        )
    else:
        logging.info("All scars have arrows. Skipping.")

    return metrics


def _resolve_image(image_shape) -> np.ndarray:
    """Resolve image_shape to an actual image array."""
    if hasattr(image_shape, 'shape'):
        return image_shape
    logging.warning(
        "Image shape provided instead of image "
        "for independent arrow detection"
    )
    return np.zeros(image_shape, dtype=np.uint8)


def process_nested_arrows(
    sorted_contours: Dict[str, List],
    hierarchy: np.ndarray,
    original_contours: List[np.ndarray],
    metrics: List[Dict],
    image_shape,
    image_dpi: Optional[float] = None
) -> List[Dict]:
    """
    Detect arrows in nested children and update parent scars.

    Parameters
    ----------
    sorted_contours : dict
        Sorted contour structure.
    hierarchy : np.ndarray
        Contour hierarchy array.
    original_contours : list
        All detected contours.
    metrics : list
        Metric dictionaries to update.
    image_shape : ndarray or tuple
        Source image or shape.
    image_dpi : float, optional
        Image DPI for detection scaling.

    Returns
    -------
    list
        Updated metrics.
    """
    config = get_arrow_integration_config()
    tolerance = config.get('area_match_tolerance', 1.0)

    index_map = _build_contour_index_map(
        sorted_contours, original_contours
    )
    scar_metrics = _build_scar_metrics_map(
        metrics, sorted_contours["children"], tolerance
    )

    if not scar_metrics and sorted_contours.get("nested_children"):
        logging.info("No direct children — skipping nested processing")
        return metrics

    for ni, cnt in enumerate(
        sorted_contours.get("nested_children", [])
    ):
        _process_single_nested(
            ni, cnt, index_map, hierarchy,
            original_contours, scar_metrics,
            image_shape, image_dpi
        )

    return metrics


def _build_contour_index_map(
    sorted_contours: Dict[str, List],
    original_contours: List[np.ndarray]
) -> Dict[str, int]:
    """Map sorted contours to original indices."""
    all_sorted = (
        sorted_contours["parents"]
        + sorted_contours["children"]
        + sorted_contours.get("nested_children", [])
    )
    index_map = {}
    for contour in all_sorted:
        for i, orig in enumerate(original_contours):
            if np.array_equal(contour, orig):
                index_map[str(contour.tobytes())] = i
                break
    return index_map


def _build_scar_metrics_map(
    metrics: List[Dict],
    children: List[np.ndarray],
    tolerance: float
) -> Dict[str, Dict]:
    """Map contour keys to their scar metric entries."""
    scar_map = {}
    for metric in metrics:
        if metric["parent"] == metric["scar"]:
            continue
        for cnt in children:
            area_diff = abs(cv2.contourArea(cnt) - metric["area"])
            if area_diff < tolerance:
                scar_map[str(cnt.tobytes())] = metric
                break
    return scar_map


def _process_single_nested(
    ni: int,
    cnt: np.ndarray,
    index_map: Dict[str, int],
    hierarchy: np.ndarray,
    original_contours: List[np.ndarray],
    scar_metrics: Dict[str, Dict],
    image_shape,
    image_dpi: Optional[float]
) -> None:
    """Process a single nested contour for arrow detection."""
    nested_idx = index_map.get(str(cnt.tobytes()))
    if nested_idx is None or nested_idx >= len(hierarchy):
        logging.warning(
            f"Could not find nested contour {ni} in hierarchy"
        )
        return

    parent_scar = _find_parent_scar(
        nested_idx, hierarchy, original_contours, scar_metrics
    )
    if parent_scar is None:
        logging.debug(f"No parent scar for nested contour {ni}")
        return

    if parent_scar.get('is_cortex', False):
        logging.debug(
            f"Skipping nested contour {ni} — parent is cortex"
        )
        return

    temp_entry = {"scar": f"nested_{ni}"}
    result = analyze_child_contour_for_arrow(
        cnt, temp_entry, image_shape, image_dpi
    )

    if result:
        logging.info(
            f"Arrow in nested contour {ni} "
            f"(parent: {parent_scar['scar']}) "
            f"angle {result.get('compass_angle', '?')}"
        )
        _apply_arrow_result(parent_scar, result)


def _find_parent_scar(
    nested_idx: int,
    hierarchy: np.ndarray,
    original_contours: List[np.ndarray],
    scar_metrics: Dict[str, Dict]
) -> Optional[Dict]:
    """Find the parent scar metric for a nested contour."""
    parent_idx = hierarchy[nested_idx][3]

    if parent_idx < len(original_contours):
        key = str(original_contours[parent_idx].tobytes())
        if key in scar_metrics:
            return scar_metrics[key]

    # Search through hierarchy relationships
    for idx, h in enumerate(hierarchy):
        if idx != parent_idx or h[3] == -1:
            continue
        if idx >= len(original_contours):
            continue
        grandparent = h[3]
        for cidx, ch in enumerate(hierarchy):
            if ch[3] != grandparent or cidx >= len(original_contours):
                continue
            key = str(original_contours[cidx].tobytes())
            if key in scar_metrics:
                return scar_metrics[key]

    return None


def _apply_arrow_result(metric: Dict, result: Dict) -> None:
    """Apply arrow detection result to a metric entry."""
    metric.update({
        "has_arrow": True,
        "arrow_angle_rad": round(result["angle_rad"], 0),
        "arrow_angle_deg": round(result["angle_deg"], 0),
        "arrow_angle": round(result["compass_angle"], 0),
        "arrow_tip": result["arrow_tip"],
        "arrow_back": result["arrow_back"],
    })


def detect_arrows_independently(
    original_contours: List[np.ndarray],
    metrics: List[Dict],
    image: np.ndarray,
    image_dpi: Optional[float] = None
) -> List[Dict]:
    """
    Detect arrows independently of hierarchy and assign to scars.

    Parameters
    ----------
    original_contours : list
        All detected contours.
    metrics : list
        Metric dictionaries.
    image : ndarray
        Source image.
    image_dpi : float, optional
        Image DPI for detection scaling.

    Returns
    -------
    list
        Updated metrics with arrow information.
    """
    config = get_arrow_integration_config()
    tolerance = config.get('area_match_tolerance', 1.0)

    scar_metrics = [
        m for m in metrics if m["parent"] != m["scar"]
    ]

    logging.info("Starting independent arrow detection...")

    parent_indices = _find_metric_contour_indices(
        metrics, original_contours,
        is_parent=True, exclude=set()
    )
    scar_indices = _find_metric_contour_indices(
        metrics, original_contours,
        is_parent=False, exclude=parent_indices
    )

    scar_map, cortex_map = _build_scar_contour_maps(
        scar_indices, original_contours, scar_metrics, tolerance
    )

    candidates = _find_arrow_candidates(
        original_contours, parent_indices, scar_indices,
        image, image_dpi, cortex_map, config
    )

    assigned = _assign_arrows_to_scars(
        candidates, scar_map, metrics
    )

    logging.info(
        f"Independent detection completed. "
        f"Assigned {len(assigned)} arrows."
    )
    return metrics


def _find_metric_contour_indices(
    metrics: List[Dict],
    contours: List[np.ndarray],
    is_parent: bool,
    exclude: Set[int]
) -> Set[int]:
    """Find original contour indices matching metrics."""
    indices = set()
    for m in metrics:
        match = (m["parent"] == m["scar"]) == is_parent
        if not match:
            continue
        for j, cnt in enumerate(contours):
            if j in exclude:
                continue
            if cv2.contourArea(cnt) == m["area"]:
                indices.add(j)
                break
    return indices


def _build_scar_contour_maps(
    scar_indices: Set[int],
    contours: List[np.ndarray],
    scar_metrics: List[Dict],
    tolerance: float
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Build maps from scar labels to their contours."""
    scar_map = {}
    cortex_map = {}
    for idx in scar_indices:
        cnt = contours[idx]
        for m in scar_metrics:
            if abs(cv2.contourArea(cnt) - m["area"]) < tolerance:
                scar_map[m["scar"]] = cnt
                if m.get('is_cortex', False):
                    cortex_map[m["scar"]] = cnt
                break
    return scar_map, cortex_map


def _find_arrow_candidates(
    contours: List[np.ndarray],
    parent_indices: Set[int],
    scar_indices: Set[int],
    image: np.ndarray,
    image_dpi: Optional[float],
    cortex_map: Dict[str, np.ndarray],
    config: Dict
) -> List[Tuple[int, np.ndarray, Dict]]:
    """
    Find contours that could be arrows.

    Parameters
    ----------
    contours : list
        All contours.
    parent_indices : set
        Indices of parent contours.
    scar_indices : set
        Indices of scar contours.
    image : ndarray
        Source image.
    image_dpi : float or None
        Image DPI.
    cortex_map : dict
        Cortex contour map for exclusion.
    config : dict
        Arrow integration configuration.

    Returns
    -------
    list
        List of (index, contour, result) tuples.
    """
    min_area = config.get('min_candidate_area', 1.0)
    min_sol = config.get('min_solidity', 0.4)
    max_sol = config.get('max_solidity', 0.9)

    candidates = []
    for i, cnt in enumerate(contours):
        if i in parent_indices or i in scar_indices:
            continue

        if not _passes_arrow_filters(cnt, min_area, min_sol, max_sol):
            continue

        if _is_within_cortex(cnt, cortex_map):
            continue

        temp_entry = {"scar": f"candidate_{i}"}
        result = analyze_child_contour_for_arrow(
            cnt, temp_entry, image, image_dpi
        )
        if result:
            logging.info(
                f"Arrow found in contour {i}, "
                f"angle {result.get('compass_angle', '?')}"
            )
            candidates.append((i, cnt, result))

    return candidates


def _passes_arrow_filters(
    cnt: np.ndarray,
    min_area: float,
    min_solidity: float,
    max_solidity: float
) -> bool:
    """Check if a contour passes basic arrow shape filters."""
    area = cv2.contourArea(cnt)
    if area < min_area:
        return False

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return False

    solidity = area / hull_area
    return min_solidity <= solidity <= max_solidity


def _is_within_cortex(
    cnt: np.ndarray,
    cortex_map: Dict[str, np.ndarray]
) -> bool:
    """Check if a contour centroid falls within a cortex area."""
    if not cortex_map:
        return False

    M = cv2.moments(cnt)
    if M["m00"] > 0:
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    else:
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w / 2, y + h / 2

    for label, cortex_cnt in cortex_map.items():
        if cv2.pointPolygonTest(cortex_cnt, (cx, cy), False) >= 0:
            logging.debug(
                f"Skipping contour within cortex '{label}'"
            )
            return True
    return False


def _assign_arrows_to_scars(
    candidates: List[Tuple],
    scar_map: Dict[str, np.ndarray],
    metrics: List[Dict]
) -> Set[str]:
    """
    Assign detected arrows to the smallest containing scar.

    Parameters
    ----------
    candidates : list
        Arrow candidate tuples (index, contour, result).
    scar_map : dict
        Maps scar labels to contour arrays.
    metrics : list
        All metric dictionaries.

    Returns
    -------
    set
        Labels of scars that received arrows.
    """
    assigned = set()

    for arrow_idx, arrow_cnt, arrow_result in candidates:
        M = cv2.moments(arrow_cnt)
        if M["m00"] > 0:
            acx = M["m10"] / M["m00"]
            acy = M["m01"] / M["m00"]
        else:
            x, y, w, h = cv2.boundingRect(arrow_cnt)
            acx, acy = x + w / 2, y + h / 2

        best_scar = None
        best_area = float('inf')

        for label, scar_cnt in scar_map.items():
            if cv2.pointPolygonTest(scar_cnt, (acx, acy), False) >= 0:
                area = cv2.contourArea(scar_cnt)
                if area < best_area:
                    best_area = area
                    best_scar = label

        if best_scar is None or best_scar in assigned:
            continue

        metric = next(
            (m for m in metrics if m["scar"] == best_scar),
            None
        )
        if metric is None or metric.get('is_cortex', False):
            continue

        logging.info(
            f"Assigning arrow {arrow_idx} to scar {best_scar}"
        )
        _apply_arrow_result(metric, arrow_result)
        assigned.add(best_scar)

    return assigned
