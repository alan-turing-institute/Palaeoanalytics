"""
PyLithics: Contour Metrics
==========================

Calculates geometric measurements for parent and child contours.
"""

import cv2
import numpy as np
import logging
from scipy.spatial.distance import pdist
from typing import List, Dict, Optional, Tuple


def calculate_contour_metrics(
    sorted_contours: Dict[str, List],
    hierarchy: np.ndarray,
    original_contours: List[np.ndarray],
    image_shape: np.ndarray
) -> List[Dict]:
    """
    Calculate basic geometric metrics for parent and child contours.

    Parameters
    ----------
    sorted_contours : dict
        {"parents":..., "children":..., "nested_children":...}
    hierarchy : np.ndarray
        Contour hierarchy array.
    original_contours : list
        All extracted contours.
    image_shape : np.ndarray
        Source image (used for shape reference).

    Returns
    -------
    list
        Metric dictionaries for all contours.
    """
    index_map = _build_contour_index_map(
        sorted_contours, original_contours
    )

    parent_map = {}
    metrics = _process_parents(
        sorted_contours["parents"], index_map, parent_map
    )
    child_metrics = _process_children(
        sorted_contours["children"], index_map,
        hierarchy, parent_map
    )
    metrics.extend(child_metrics)

    return metrics


def _build_contour_index_map(
    sorted_contours: Dict[str, List],
    original_contours: List[np.ndarray]
) -> Dict[str, int]:
    """
    Map sorted contours to their original indices.

    Parameters
    ----------
    sorted_contours : dict
        Sorted contour structure.
    original_contours : list
        All extracted contours.

    Returns
    -------
    dict
        Mapping from contour bytes key to original index.
    """
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


def _process_parents(
    parents: List[np.ndarray],
    index_map: Dict[str, int],
    parent_map: Dict[int, str]
) -> List[Dict]:
    """
    Calculate metrics for parent contours.

    Parameters
    ----------
    parents : list
        Parent contour arrays.
    index_map : dict
        Contour-to-original-index mapping.
    parent_map : dict
        Populated in place with {index: label}.

    Returns
    -------
    list
        Parent metric dictionaries.
    """
    metrics = []

    for pi, cnt in enumerate(parents):
        idx = index_map.get(str(cnt.tobytes()))
        if idx is None:
            logging.warning(
                f"Could not find parent contour {pi} "
                f"in original contours"
            )
            continue

        lab = f"parent {pi + 1}"
        parent_map[idx] = lab

        cx, cy = _compute_centroid(cnt)
        area = round(cv2.contourArea(cnt), 2)
        peri = round(cv2.arcLength(cnt, True), 2)
        tech_w, tech_h = _compute_technical_dimensions(cnt)
        ml, mw = _compute_max_dimensions(cnt)
        x, y, bbox_w, bbox_h = cv2.boundingRect(cnt)

        aspect = (
            round(tech_h / tech_w, 2)
            if tech_w > 0 else None
        )

        metrics.append({
            "parent": lab, "scar": lab,
            "centroid_x": cx, "centroid_y": cy,
            "technical_width": tech_w,
            "technical_length": tech_h,
            "area": area,
            "aspect_ratio": aspect,
            "bounding_box_x": x, "bounding_box_y": y,
            "bounding_box_width": bbox_w,
            "bounding_box_height": bbox_h,
            "max_length": ml, "max_width": mw,
            "contour": cnt.tolist(),
            "perimeter": peri,
        })

    return metrics


def _process_children(
    children: List[np.ndarray],
    index_map: Dict[str, int],
    hierarchy: np.ndarray,
    parent_map: Dict[int, str]
) -> List[Dict]:
    """
    Calculate metrics for child contours.

    Parameters
    ----------
    children : list
        Child contour arrays.
    index_map : dict
        Contour-to-original-index mapping.
    hierarchy : np.ndarray
        Contour hierarchy array.
    parent_map : dict
        Mapping from original index to parent label.

    Returns
    -------
    list
        Child metric dictionaries.
    """
    metrics = []

    for ci, cnt in enumerate(children):
        idx = index_map.get(str(cnt.tobytes()))
        if idx is None:
            logging.warning(
                f"Could not find child contour {ci} "
                f"in original contours"
            )
            continue

        if idx < len(hierarchy):
            parent_idx = hierarchy[idx][3]
            pl = parent_map.get(parent_idx, "Unknown")
        else:
            logging.warning(
                f"Child contour index {idx} out of bounds "
                f"for hierarchy"
            )
            pl = "Unknown"

        lab = f"child {ci + 1}"
        cx, cy = _compute_centroid(cnt)
        area = round(cv2.contourArea(cnt), 2)
        tech_w, tech_h = _compute_technical_dimensions(cnt)
        ml, mw = _compute_max_dimensions(cnt)
        x, y, bbox_w, bbox_h = cv2.boundingRect(cnt)

        aspect = (
            round(tech_h / tech_w, 2)
            if tech_w > 0 else None
        )

        metrics.append({
            "parent": pl, "scar": lab,
            "centroid_x": cx, "centroid_y": cy,
            "width": tech_w,
            "height": tech_h,
            "area": area,
            "aspect_ratio": aspect,
            "max_length": ml, "max_width": mw,
            "bounding_box_x": x, "bounding_box_y": y,
            "bounding_box_width": bbox_w,
            "bounding_box_height": bbox_h,
            "contour": cnt.tolist(),
        })

    return metrics


def _compute_centroid(
    contour: np.ndarray
) -> Tuple[float, float]:
    """
    Compute centroid of a contour.

    Parameters
    ----------
    contour : np.ndarray
        Contour array.

    Returns
    -------
    tuple
        (cx, cy) rounded to 2 decimal places.
    """
    M = cv2.moments(contour)
    if M["m00"] > 0:
        return (
            round(M["m10"] / M["m00"], 2),
            round(M["m01"] / M["m00"], 2)
        )
    x, y, w, h = cv2.boundingRect(contour)
    return round(x + w / 2, 2), round(y + h / 2, 2)


def _compute_technical_dimensions(
    contour: np.ndarray
) -> Tuple[float, float]:
    """
    Compute Y-axis aligned technical width and height.

    Width is the maximum horizontal span at any Y-level.
    Height is the full Y-axis span.

    Parameters
    ----------
    contour : np.ndarray
        Contour array.

    Returns
    -------
    tuple
        (width, height) rounded to 2 decimal places.
    """
    points = contour.reshape(-1, 2)

    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])
    height = round(float(max_y - min_y), 2)

    max_width = 0.0
    for y in np.unique(points[:, 1]):
        xs = points[points[:, 1] == y, 0]
        width_at_y = float(np.max(xs) - np.min(xs))
        max_width = max(max_width, width_at_y)

    return round(max_width, 2), height


def _compute_max_dimensions(
    contour: np.ndarray
) -> Tuple[float, float]:
    """
    Compute maximum length and perpendicular width.

    Maximum length is the greatest distance between any two
    contour points. Width is the maximum perpendicular distance
    from the length axis.

    Parameters
    ----------
    contour : np.ndarray
        Contour array.

    Returns
    -------
    tuple
        (max_length, max_width) rounded to 2 decimal places.
    """
    # Vectorised replacement for an O(N²) Python loop that called
    # ``np.linalg.norm`` on tiny 2-element arrays ~3,000² times per
    # contour. scipy's ``pdist`` does the same all-pairs distance
    # computation in a single C call. For typical contour sizes the
    # speed-up is ~100x.
    pts = contour[:, 0, :].astype(np.float64)
    n = len(pts)
    if n < 2:
        return 0.0, 0.0

    distances = pdist(pts)
    k = int(np.argmax(distances))
    max_len = float(distances[k])

    # Convert condensed-distance index k back to (i, j). pdist orders
    # pairs as (0,1), (0,2), ..., (0,n-1), (1,2), (1,3), ..., closed-
    # form inversion below; avoids the 2-D squareform memory blow-up
    # on large contours.
    i = int(n - 2 - np.floor(
        np.sqrt(-8 * k + 4 * n * (n - 1) - 7) / 2.0 - 0.5
    ))
    j = int(k + i + 1 - n * (n - 1) // 2 + (n - i) * ((n - i) - 1) // 2)
    p1 = pts[i]
    p2 = pts[j]

    # Vectorised perpendicular-width: project every contour point onto
    # the unit perpendicular of the length axis and take the max
    # absolute projection. Replaces the per-point dot-product generator.
    v = p2 - p1
    perp = np.array([-v[1], v[0]], dtype=np.float64)
    perp_len = np.linalg.norm(perp)
    if perp_len < 1e-12:
        max_wid = 0.0
    else:
        perp /= perp_len
        projections = np.abs((pts - p1) @ perp)
        max_wid = float(projections.max())

    return round(max_len, 2), round(max_wid, 2)


def convert_metrics_to_real_world(
    metrics: List[Dict],
    pixels_per_mm: float
) -> List[Dict]:
    """
    Convert metrics from pixel values to millimeters.

    Parameters
    ----------
    metrics : list
        Metric dictionaries in pixel units.
    pixels_per_mm : float
        Conversion factor from scale calibration.

    Returns
    -------
    list
        Metrics with measurements in millimeters.
    """
    logging.debug(
        f"Converting {len(metrics)} metrics with "
        f"factor: {pixels_per_mm:.3f} pixels/mm"
    )

    if pixels_per_mm <= 0:
        logging.warning(
            f"Invalid conversion factor: {pixels_per_mm}. "
            f"Returning original metrics."
        )
        return metrics

    converted = []
    for metric in metrics:
        converted.append(
            _convert_single_metric(metric, pixels_per_mm)
        )
    return converted


def _convert_single_metric(
    metric: Dict, pixels_per_mm: float
) -> Dict:
    """
    Convert a single metric dictionary to real-world units.

    Parameters
    ----------
    metric : dict
        Single metric dictionary.
    pixels_per_mm : float
        Conversion factor.

    Returns
    -------
    dict
        Converted metric dictionary.
    """
    converted = metric.copy()

    linear_fields = [
        "centroid_x", "centroid_y",
        "technical_width", "technical_length",
        "bounding_box_x", "bounding_box_y",
        "bounding_box_width", "bounding_box_height",
        "max_length", "max_width",
        "perimeter", "distance_to_max_width",
        "convex_hull_width", "convex_hull_height",
    ]

    for field in linear_fields:
        if field in converted and converted[field] is not None:
            converted[field] = round(
                converted[field] / pixels_per_mm, 2
            )

    area_fields = [
        "area", "convex_hull_area", "voronoi_cell_area",
        "top_area", "bottom_area",
        "left_area", "right_area", "cortex_area",
    ]

    for field in area_fields:
        value = converted.get(field)
        if (value is not None
                and isinstance(value, (int, float))
                and value != "NA"):
            converted[field] = round(
                value / (pixels_per_mm ** 2), 2
            )

    return converted
