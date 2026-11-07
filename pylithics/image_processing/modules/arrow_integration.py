"""
Arrow integration for PyLithics.

Finds the arrows among the contours of a drawing and attaches each to
the scar it marks, in three passes. A contour nested inside a scar is
tried against that scar. A scar, or a surface, small enough to be a
mark is tried; when it is an arrow it becomes the arrow of the scar or
surface around it and leaves the metrics, so it is not counted as a
scar. Every other stray contour is tried and given to the smallest scar
that contains it. Marks inside cortex are left alone.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from ..config import get_arrow_integration_config
from .arrow_detection import analyze_child_contour_for_arrow


def integrate_arrows(
    sorted_contours: Dict[str, List],
    hierarchy: np.ndarray,
    original_contours: List[np.ndarray],
    metrics: List[Dict],
    image_shape,
    image_dpi: Optional[float] = None,
    debug_dir: Optional[str] = None,
) -> List[Dict]:
    """
    Detect arrows and attach them to the scar metrics; return the metrics.

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
        Accepted for the callers that pass it; not used by the detector.
    debug_dir : str, optional
        Folder for the per-scar debug files.
    """
    metrics = process_nested_arrows(
        sorted_contours, hierarchy, original_contours,
        metrics, image_shape, image_dpi, debug_dir,
    )
    image = _resolve_image(image_shape)
    metrics = reassign_small_marks(original_contours, metrics, image, debug_dir)

    scars_without = [m for m in metrics if _is_scar(m) and not m.get('has_arrow', False)]
    if scars_without:
        logging.debug(
            f"{len(scars_without)} scars without arrows. Running independent detection."
        )
        metrics = detect_arrows_independently(
            original_contours, metrics, image, image_dpi, debug_dir,
        )
    else:
        logging.debug("All scars have arrows. Skipping.")
    return metrics


def _is_scar(metric: Dict) -> bool:
    """Say whether a metric row is a scar (not a surface)."""
    return metric["parent"] != metric["scar"]


def _resolve_image(image_shape) -> np.ndarray:
    """Return an image array from either an image or a shape tuple."""
    if isinstance(image_shape, np.ndarray):
        return image_shape
    return np.zeros(image_shape[:2], dtype=np.uint8)


### NESTED CONTOURS ###

def process_nested_arrows(
    sorted_contours: Dict[str, List],
    hierarchy: np.ndarray,
    original_contours: List[np.ndarray],
    metrics: List[Dict],
    image_shape,
    image_dpi: Optional[float] = None,
    debug_dir: Optional[str] = None,
) -> List[Dict]:
    """Detect arrows in nested children and update their parent scars."""
    config = get_arrow_integration_config()
    tolerance = config.get('area_match_tolerance', 1.0)

    index_map = _build_contour_index_map(sorted_contours, original_contours)
    scar_metrics = _build_scar_metrics_map(metrics, sorted_contours["children"], tolerance)

    if not scar_metrics and sorted_contours.get("nested_children"):
        logging.debug("No direct children — skipping nested processing")
        return metrics

    skip_counts = {"parent_is_cortex": 0, "no_parent_scar": 0}
    for ni, cnt in enumerate(sorted_contours.get("nested_children", [])):
        status = _process_single_nested(
            ni, cnt, index_map, hierarchy, original_contours, scar_metrics,
            image_shape, image_dpi, debug_dir,
        )
        if status in skip_counts:
            skip_counts[status] += 1

    for status, count in skip_counts.items():
        if count:
            logging.debug(f"Skipped {count} nested contours: {status}")
    return metrics


def _build_contour_index_map(
    sorted_contours: Dict[str, List], original_contours: List[np.ndarray]
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
    metrics: List[Dict], children: List[np.ndarray], tolerance: float
) -> Dict[str, Dict]:
    """Map contour keys to their scar metric entries."""
    scar_map = {}
    for metric in metrics:
        if not _is_scar(metric):
            continue
        for cnt in children:
            if abs(cv2.contourArea(cnt) - metric["area"]) < tolerance:
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
    image_dpi: Optional[float],
    debug_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Process a single nested contour for arrow detection.

    Returns a status string ("no_parent_scar", "parent_is_cortex") when
    the contour is skipped so the caller can aggregate counts, or None
    if the contour was processed.
    """
    nested_idx = index_map.get(str(cnt.tobytes()))
    if nested_idx is None or nested_idx >= len(hierarchy):
        logging.warning(f"Nested contour {ni} not found in the hierarchy")
        return None

    parent_scar = _find_parent_scar(nested_idx, hierarchy, original_contours, scar_metrics)
    if parent_scar is None:
        return "no_parent_scar"
    if parent_scar.get('is_cortex', False):
        return "parent_is_cortex"

    temp_entry = {"scar": f"nested_{ni}", "debug_dir": debug_dir}
    result = analyze_child_contour_for_arrow(cnt, temp_entry, image_shape, image_dpi)
    if result:
        logging.debug(
            f"Arrow in nested contour {ni} (parent: {parent_scar['scar']}) "
            f"angle {result.get('compass_angle', '?')}"
        )
        _apply_arrow_result(parent_scar, result)
    return None


def _find_parent_scar(
    nested_idx: int,
    hierarchy: np.ndarray,
    original_contours: List[np.ndarray],
    scar_metrics: Dict[str, Dict]
) -> Optional[Dict]:
    """Find the parent scar metric for a nested contour, up the hierarchy if needed."""
    parent_idx = hierarchy[nested_idx][3]
    while parent_idx != -1 and parent_idx < len(original_contours):
        key = str(original_contours[parent_idx].tobytes())
        if key in scar_metrics:
            return scar_metrics[key]
        parent_idx = hierarchy[parent_idx][3]
    return None


def _apply_arrow_result(metric: Dict, result: Dict) -> None:
    """Apply arrow detection result to a metric entry."""
    metric.update({
        "has_arrow": True,
        "arrow_angle_rad": round(result["angle_rad"], 2),
        "arrow_angle_deg": round(result["angle_deg"], 0),
        "arrow_angle": round(result["compass_angle"], 0),
        "arrow_tip": result["arrow_tip"],
        "arrow_back": result["arrow_back"],
    })
    if result.get("score") is not None:
        metric["arrow_score"] = result["score"]


### SMALL SCARS AND SURFACES ###

def reassign_small_marks(
    original_contours: List[np.ndarray],
    metrics: List[Dict],
    image: np.ndarray,
    debug_dir: Optional[str] = None,
) -> List[Dict]:
    """
    Turn the small scars and surfaces that are arrows into arrows of their neighbours.

    A row no larger than ``max_candidate_share`` of the main surface is
    tried. When it is an arrow, the smallest other row whose contour
    contains its centre takes the arrow, and the row is dropped (in
    place). A row with no such neighbour, or one already marked, stays.
    """
    config = get_arrow_integration_config()
    share = config.get('max_candidate_share', 0.02)
    tolerance = config.get('area_match_tolerance', 1.0)
    contours = _contours_by_label(metrics, original_contours, tolerance)
    main_area = max((m["area"] for m in metrics if not _is_scar(m)), default=0.0)
    dropped: Set[str] = set()
    for metric in metrics:
        label = metric["scar"]
        if metric.get('is_cortex') or label not in contours or metric["area"] > share * main_area:
            continue
        entry = {"scar": label, "debug_dir": debug_dir}
        result = analyze_child_contour_for_arrow(contours[label], entry, image)
        if not result:
            continue
        target = _smallest_around(
            _centroid(contours[label]), contours, metrics, exclude=dropped | {label},
        )
        if target is None or target.get('has_arrow', False):
            continue
        logging.debug(f"{label} is an arrow of {target['scar']}, not a scar")
        _apply_arrow_result(target, result)
        dropped.add(label)
    metrics[:] = [m for m in metrics if m["scar"] not in dropped]
    return metrics


def _contours_by_label(
    metrics: List[Dict], contours: List[np.ndarray], tolerance: float
) -> Dict[str, np.ndarray]:
    """Map each metric label to its contour, matched by area."""
    areas = [cv2.contourArea(cnt) for cnt in contours]
    found = {}
    for metric in metrics:
        for index, area in enumerate(areas):
            if abs(area - metric["area"]) < tolerance:
                found[metric["scar"]] = contours[index]
                break
    return found


def _centroid(contour: np.ndarray) -> Tuple[float, float]:
    """Return the centre of a contour, from its moments or its box."""
    moments = cv2.moments(contour)
    if moments["m00"] > 0:
        return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
    x, y, w, h = cv2.boundingRect(contour)
    return x + w / 2, y + h / 2


def _smallest_around(
    point: Tuple[float, float], contours: Dict[str, np.ndarray], metrics: List[Dict],
    exclude: Set[str],
) -> Optional[Dict]:
    """Return the smallest metric row whose contour contains the point, scars before surfaces."""
    best, best_key = None, None
    for metric in metrics:
        label = metric["scar"]
        if label in exclude or label not in contours or metric.get('is_cortex'):
            continue
        if cv2.pointPolygonTest(contours[label], point, False) < 0:
            continue
        key = (0 if _is_scar(metric) else 1, metric["area"])
        if best_key is None or key < best_key:
            best, best_key = metric, key
    return best


### STRAY CONTOURS ###

def detect_arrows_independently(
    original_contours: List[np.ndarray],
    metrics: List[Dict],
    image: np.ndarray,
    image_dpi: Optional[float] = None,
    debug_dir: Optional[str] = None,
) -> List[Dict]:
    """Try every contour that is neither a surface nor a scar; give arrows to scars."""
    config = get_arrow_integration_config()
    tolerance = config.get('area_match_tolerance', 1.0)
    scar_metrics = [m for m in metrics if _is_scar(m)]

    parent_indices = _find_metric_contour_indices(
        metrics, original_contours, is_parent=True, exclude=set()
    )
    scar_indices = _find_metric_contour_indices(
        metrics, original_contours, is_parent=False, exclude=parent_indices
    )
    scar_map, cortex_map = _build_scar_contour_maps(
        scar_indices, original_contours, scar_metrics, tolerance
    )
    candidates = _find_arrow_candidates(
        original_contours, parent_indices, scar_indices, image, image_dpi,
        cortex_map, config, debug_dir,
    )
    assigned = _assign_arrows_to_scars(candidates, scar_map, metrics)
    logging.debug(f"Independent detection completed. Assigned {len(assigned)} arrows.")
    return metrics


def _find_metric_contour_indices(
    metrics: List[Dict], contours: List[np.ndarray], is_parent: bool, exclude: Set[int]
) -> Set[int]:
    """Find original contour indices matching metrics."""
    indices = set()
    for m in metrics:
        if (not _is_scar(m)) != is_parent:
            continue
        for j, cnt in enumerate(contours):
            if j in exclude:
                continue
            if cv2.contourArea(cnt) == m["area"]:
                indices.add(j)
                break
    return indices


def _build_scar_contour_maps(
    scar_indices: Set[int], contours: List[np.ndarray], scar_metrics: List[Dict],
    tolerance: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Build maps from scar labels to their contours, and the cortex ones apart."""
    scar_map, cortex_map = {}, {}
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
    config: Dict,
    debug_dir: Optional[str] = None,
) -> List[Tuple[int, np.ndarray, Dict]]:
    """Return (index, contour, result) for each stray contour that is an arrow."""
    min_area = config.get('min_candidate_area', 4.0)
    candidates, cortex_skips = [], 0
    for i, cnt in enumerate(contours):
        if i in parent_indices or i in scar_indices or cv2.contourArea(cnt) < min_area:
            continue
        if _is_within_cortex(cnt, cortex_map):
            cortex_skips += 1
            continue
        temp_entry = {"scar": f"candidate_{i}", "debug_dir": debug_dir}
        result = analyze_child_contour_for_arrow(cnt, temp_entry, image, image_dpi)
        if result:
            logging.debug(f"Arrow found in contour {i}, angle {result.get('compass_angle', '?')}")
            candidates.append((i, cnt, result))
    if cortex_skips:
        logging.debug(f"Skipped {cortex_skips} contours inside cortex regions")
    return candidates


def _is_within_cortex(cnt: np.ndarray, cortex_map: Dict[str, np.ndarray]) -> bool:
    """Check if a contour centroid falls within a cortex area."""
    if not cortex_map:
        return False
    centre = _centroid(cnt)
    return any(
        cv2.pointPolygonTest(cortex_cnt, centre, False) >= 0 for cortex_cnt in cortex_map.values()
    )


def _assign_arrows_to_scars(
    candidates: List[Tuple], scar_map: Dict[str, np.ndarray], metrics: List[Dict]
) -> Set[str]:
    """Assign each arrow to the smallest scar that contains it; return the scars marked."""
    assigned = set()
    for arrow_idx, arrow_cnt, arrow_result in candidates:
        centre = _centroid(arrow_cnt)
        best_scar, best_area = None, float('inf')
        for label, scar_cnt in scar_map.items():
            if cv2.pointPolygonTest(scar_cnt, centre, False) >= 0:
                area = cv2.contourArea(scar_cnt)
                if area < best_area:
                    best_area, best_scar = area, label
        if best_scar is None or best_scar in assigned:
            continue
        metric = next((m for m in metrics if m["scar"] == best_scar), None)
        if metric is None or metric.get('is_cortex', False) or metric.get('has_arrow', False):
            continue
        logging.debug(f"Assigning arrow {arrow_idx} to scar {best_scar}")
        _apply_arrow_result(metric, arrow_result)
        assigned.add(best_scar)
    return assigned
