"""
PyLithics: Visualization and CSV Export
=======================================

Generates labeled output images and exports comprehensive CSV data.
"""

import cv2
import numpy as np
import pandas as pd
import os
import logging
from typing import List, Dict, Optional, Tuple, Any
from ..config import get_arrow_detection_config

# Color constants (BGR format for OpenCV)
COLOR_SURFACE = (153, 60, 94)       # Purple - RGB(94, 60, 153)
COLOR_SCAR = (99, 184, 253)        # Orange - RGB(253, 184, 99)
COLOR_CORTEX = (39, 48, 215)       # Red - RGB(215, 48, 39)
COLOR_PLATFORM = (210, 171, 178)   # Light purple - RGB(178, 171, 210)
COLOR_LATERAL = (193, 205, 128)    # Mint green - RGB(128, 205, 193)
COLOR_ARROW = (219, 191, 145)      # Light blue - RGB(145, 191, 219)
COLOR_UNKNOWN = (128, 128, 128)    # Gray
COLOR_TEXT = (0, 0, 0)             # Black
COLOR_LABEL_BG = (255, 255, 255)   # White

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 1
LABEL_PADDING = 4
CENTROID_RADIUS = 4
CENTROID_RADIUS_SURFACE = 8


def visualize_contours_with_hierarchy(
    contours: List[np.ndarray],
    hierarchy: np.ndarray,
    metrics: List[Dict],
    inverted_image: np.ndarray,
    output_path: str,
    arrow_contours: Optional[List[np.ndarray]] = None,
    config: Optional[Dict] = None
) -> None:
    """
    Visualize contours with hierarchy, labels, and arrows.

    Parameters
    ----------
    contours : list
        Contours (parents + children) in display order.
    hierarchy : ndarray
        Contour hierarchy array.
    metrics : list
        Metric dicts with arrow and classification data.
    inverted_image : ndarray
        Inverted binary image.
    output_path : str
        File path for the labeled image.
    arrow_contours : list, optional
        Grandchild contours containing detected arrows.
    config : dict, optional
        Configuration dictionary.
    """
    original = cv2.bitwise_not(inverted_image)
    labeled = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

    _draw_arrow_contours(labeled, metrics, arrow_contours)
    contour_info = _draw_contours_and_centroids(
        labeled, contours, metrics
    )
    label_positions = _draw_labels(labeled, contour_info)
    _draw_arrow_annotations(labeled, metrics, label_positions, config)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, labeled)
    logging.info("Saved visualized contours with arrows to %s",
                 output_path)


def _get_contour_style(metric: Dict) -> Tuple[Tuple, str]:
    """
    Determine color and label text for a contour.

    Parameters
    ----------
    metric : dict
        Metric dictionary for this contour.

    Returns
    -------
    tuple
        (color, text) for drawing.
    """
    parent = metric["parent"]
    scar = metric["scar"]

    if parent == scar:
        return COLOR_SURFACE, metric.get("surface_type", parent)

    label = scar.lower()
    if "cortex " in label:
        return COLOR_CORTEX, scar
    if "scar " in label:
        return COLOR_SCAR, scar
    if "mark " in label:
        return COLOR_PLATFORM, scar
    if "edge" in label:
        return COLOR_LATERAL, scar
    return COLOR_UNKNOWN, scar


def _draw_arrow_contours(
    labeled: np.ndarray,
    metrics: List[Dict],
    arrow_contours: Optional[List[np.ndarray]]
) -> None:
    """
    Draw grandchild contour outlines for detected arrows.

    Parameters
    ----------
    labeled : ndarray
        BGR image to draw on.
    metrics : list
        Metric dicts with arrow coordinates.
    arrow_contours : list or None
        Grandchild contours to check.
    """
    if not arrow_contours:
        return

    arrow_points = []
    for m in metrics:
        if m.get("has_arrow") and m.get("arrow_back") and m.get("arrow_tip"):
            arrow_points.extend([m["arrow_back"], m["arrow_tip"]])

    for contour in arrow_contours:
        for pt in arrow_points:
            point = (int(pt[0]), int(pt[1]))
            if cv2.pointPolygonTest(contour, point, False) >= 0:
                cv2.drawContours(
                    labeled, [contour], -1, COLOR_ARROW, 2
                )
                break


def _draw_contours_and_centroids(
    labeled: np.ndarray,
    contours: List[np.ndarray],
    metrics: List[Dict]
) -> List[Tuple]:
    """
    Draw contours and centroids, return info for label phase.

    Parameters
    ----------
    labeled : ndarray
        BGR image to draw on.
    contours : list
        Contour arrays.
    metrics : list
        Metric dicts.

    Returns
    -------
    list
        List of (contour, cx, cy, text, index, color) tuples.
    """
    contour_info = []

    for i, cnt in enumerate(contours):
        if i >= len(metrics):
            continue

        color, text = _get_contour_style(metrics[i])
        cv2.drawContours(labeled, [cnt], -1, color, 2)

        cx, cy = _compute_centroid(cnt)
        radius = (CENTROID_RADIUS_SURFACE
                  if color == COLOR_SURFACE
                  else CENTROID_RADIUS)
        cv2.circle(labeled, (cx, cy), radius, color, -1)

        contour_info.append((cnt, cx, cy, text, i, color))

    return contour_info


def _compute_centroid(contour: np.ndarray) -> Tuple[int, int]:
    """
    Compute centroid of a contour.

    Parameters
    ----------
    contour : ndarray
        Contour array.

    Returns
    -------
    tuple
        (cx, cy) centroid coordinates.
    """
    M = cv2.moments(contour)
    if M.get("m00", 0) != 0:
        return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    x, y, w, h = cv2.boundingRect(contour)
    return x + w // 2, y + h // 2


def _find_non_overlapping_position(
    positions: List[Tuple],
    text_size: Tuple[int, int],
    contour: np.ndarray,
    cx: int, cy: int
) -> Tuple[int, int]:
    """
    Find a label position that avoids overlaps.

    Parameters
    ----------
    positions : list
        Existing label positions as (x, y, w, h).
    text_size : tuple
        (width, height) of the text.
    contour : ndarray
        Contour to avoid overlapping.
    cx : int
        Centroid x.
    cy : int
        Centroid y.

    Returns
    -------
    tuple
        (tx, ty) label position.
    """
    tw, th = text_size
    candidates = [
        (cx + 20, cy - 5),
        (cx - tw - 20, cy - 5),
        (cx - tw // 2, cy - 30),
        (cx - tw // 2, cy + 30),
        (cx + 25, cy - 20),
        (cx - tw - 25, cy - 20),
        (cx + 25, cy + 15),
        (cx - tw - 25, cy + 15),
    ]

    for px, py in candidates:
        if _overlaps_labels(px, py, tw, th, positions):
            continue
        dist = cv2.pointPolygonTest(
            contour, (px + tw // 2, py - th // 2), True
        )
        if abs(dist) < 8:
            continue
        return px, py

    return candidates[0]


def _overlaps_labels(
    px: int, py: int, tw: int, th: int,
    positions: List[Tuple]
) -> bool:
    """Check if a proposed label overlaps existing labels."""
    for lx, ly, lw, lh in positions:
        if (px < lx + lw + 8 and px + tw + 8 > lx and
                py < ly + lh + 8 and py + th + 8 > ly):
            return True
    return False


def _draw_labels(
    labeled: np.ndarray,
    contour_info: List[Tuple]
) -> List[Tuple]:
    """
    Draw labels for all contours with overlap avoidance.

    Parameters
    ----------
    labeled : ndarray
        BGR image to draw on.
    contour_info : list
        From _draw_contours_and_centroids.

    Returns
    -------
    list
        Label positions as (x, y, w, h) for arrow annotation.
    """
    label_positions = []

    for cnt, cx, cy, text, i, contour_color in contour_info:
        ts = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)[0]
        tx, ty = _find_non_overlapping_position(
            label_positions, ts, cnt, cx, cy
        )
        label_positions.append((tx, ty, ts[0], ts[1]))

        tx = max(5, min(tx, labeled.shape[1] - ts[0] - 5))
        ty = max(ts[1] + 5, min(ty, labeled.shape[0] - 5))

        border_color = _get_label_border_color(text, contour_color)
        _draw_label_box(labeled, tx, ty, ts, border_color, text)

    return label_positions


def _get_label_border_color(
    text: str, contour_color: Tuple
) -> Tuple:
    """Determine border color based on label type."""
    lower = text.lower()
    if "scar" in lower:
        return COLOR_SCAR
    if lower in ('dorsal', 'ventral', 'platform', 'lateral'):
        return contour_color
    if "edge" in lower or "cortex" in lower:
        return contour_color
    return COLOR_TEXT


def _draw_label_box(
    labeled: np.ndarray,
    tx: int, ty: int,
    text_size: Tuple[int, int],
    border_color: Tuple,
    text: str
) -> None:
    """Draw a label with white background and colored border."""
    pt1 = (tx - LABEL_PADDING, ty - text_size[1] - LABEL_PADDING)
    pt2 = (tx + text_size[0] + LABEL_PADDING, ty + LABEL_PADDING)
    cv2.rectangle(labeled, pt1, pt2, COLOR_LABEL_BG, -1)
    cv2.rectangle(labeled, pt1, pt2, border_color, 2)
    cv2.putText(
        labeled, text, (tx, ty), FONT,
        FONT_SCALE, COLOR_TEXT, FONT_THICKNESS, cv2.LINE_AA
    )


def _draw_arrow_annotations(
    labeled: np.ndarray,
    metrics: List[Dict],
    label_positions: List[Tuple],
    config: Optional[Dict]
) -> None:
    """
    Draw arrow lines and angle labels.

    Parameters
    ----------
    labeled : ndarray
        BGR image to draw on.
    metrics : list
        Metric dicts with arrow data.
    label_positions : list
        Existing label positions for overlap avoidance.
    config : dict or None
        Configuration dictionary.
    """
    arrow_config = get_arrow_detection_config(config)
    show_lines = arrow_config.get('show_arrow_lines', False)

    for m in metrics:
        if not (m.get("has_arrow") and m.get("arrow_back")
                and m.get("arrow_tip")):
            continue

        back = tuple(int(v) for v in m["arrow_back"])
        tip = tuple(int(v) for v in m["arrow_tip"])

        if show_lines:
            cv2.arrowedLine(
                labeled, back, tip,
                color=(0, 0, 255), thickness=2, tipLength=0.2
            )

        angle = m.get("arrow_angle")
        if angle is not None:
            _draw_angle_label(
                labeled, back, tip, angle, label_positions
            )


def _draw_angle_label(
    labeled: np.ndarray,
    back: Tuple[int, int],
    tip: Tuple[int, int],
    angle: float,
    label_positions: List[Tuple]
) -> None:
    """
    Draw an angle label near an arrow shaft.

    Parameters
    ----------
    labeled : ndarray
        BGR image to draw on.
    back : tuple
        Arrow back point (x, y).
    tip : tuple
        Arrow tip point (x, y).
    angle : float
        Arrow angle in degrees.
    label_positions : list
        Existing label positions for overlap avoidance.
    """
    shaft = np.array([tip[0] - back[0], tip[1] - back[1]])
    shaft_len = np.linalg.norm(shaft)
    if shaft_len == 0:
        return

    perp = np.array([shaft[1], -shaft[0]]) / shaft_len
    text = f"{int(angle)} deg"
    (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)

    text_pos = _find_angle_label_position(
        back, shaft, perp, tw, th, label_positions, labeled.shape
    )

    pt1 = (text_pos[0] - LABEL_PADDING,
           text_pos[1] - th - LABEL_PADDING)
    pt2 = (text_pos[0] + tw + LABEL_PADDING,
           text_pos[1] + LABEL_PADDING)
    cv2.rectangle(labeled, pt1, pt2, COLOR_LABEL_BG, -1)
    cv2.rectangle(labeled, pt1, pt2, COLOR_ARROW, 2)
    cv2.putText(
        labeled, text, text_pos, FONT,
        FONT_SCALE, COLOR_TEXT, FONT_THICKNESS, cv2.LINE_AA
    )

    label_positions.append((text_pos[0], text_pos[1], tw, th))


def _find_angle_label_position(
    back: Tuple, shaft: np.ndarray, perp: np.ndarray,
    tw: int, th: int,
    label_positions: List[Tuple],
    image_shape: Tuple
) -> Tuple[int, int]:
    """
    Find a non-overlapping position for an angle label.

    Parameters
    ----------
    back : tuple
        Arrow back point.
    shaft : ndarray
        Shaft vector.
    perp : ndarray
        Perpendicular unit vector.
    tw : int
        Text width.
    th : int
        Text height.
    label_positions : list
        Existing label positions.
    image_shape : tuple
        Image dimensions (h, w, channels).

    Returns
    -------
    tuple
        (x, y) position for the label.
    """
    offset = perp * 40
    h, w = image_shape[:2]

    for frac in [0.2, 0.4, 0.6, 0.8]:
        for mult in [1, -1, 1.5, -1.5]:
            base = (
                int(back[0] + shaft[0] * frac),
                int(back[1] + shaft[1] * frac)
            )
            pos = (
                int(base[0] + offset[0] * mult),
                int(base[1] + offset[1] * mult)
            )

            in_bounds = (
                pos[0] > 10 and pos[0] + tw < w - 10
                and pos[1] > th + 10 and pos[1] < h - 10
            )
            if not in_bounds:
                continue

            if not _overlaps_labels(pos[0], pos[1], tw, th,
                                    label_positions):
                return pos

    # Fallback: clamp to image bounds
    default = (
        int(back[0] + shaft[0] * 0.5 + offset[0]),
        int(back[1] + shaft[1] * 0.5 + offset[1])
    )
    return (
        max(5, min(default[0], w - tw - 5)),
        max(th + 5, min(default[1], h - 5))
    )


# --------------- CSV Export ---------------

def save_measurements_to_csv(
    metrics: List[Dict],
    output_path: str,
    append: bool = False,
    calibration_metadata: Optional[Dict] = None
) -> None:
    """
    Save contour metrics to a CSV file.

    Parameters
    ----------
    metrics : list
        List of metric dictionaries.
    output_path : str
        CSV file path.
    append : bool
        Append to existing file if True.
    calibration_metadata : dict, optional
        Calibration info to include in output.
    """
    total_dorsal_scars = _count_dorsal_scars(metrics)
    updated_data = [
        _build_csv_row(m, metrics, total_dorsal_scars,
                       calibration_metadata)
        for m in metrics
    ]

    all_columns = _build_column_order(metrics, calibration_metadata)
    df = pd.DataFrame(updated_data)

    for col in all_columns:
        if col not in df.columns:
            df[col] = "NA"

    existing_cols = [c for c in all_columns if c in df.columns]
    df = df[existing_cols]
    df.fillna("NA", inplace=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    _write_csv(df, output_path, append)


def _count_dorsal_scars(metrics: List[Dict]) -> int:
    """Count scars on the dorsal surface."""
    return len([
        m for m in metrics
        if (m.get("parent") != m.get("scar")
            and "scar" in m.get("surface_feature", "").lower())
    ])


def _build_csv_row(
    metric: Dict,
    all_metrics: List[Dict],
    total_dorsal_scars: int,
    calibration_metadata: Optional[Dict]
) -> Dict[str, Any]:
    """
    Build a single CSV row from a metric dictionary.

    Parameters
    ----------
    metric : dict
        Individual metric dictionary.
    all_metrics : list
        All metrics for parent lookup.
    total_dorsal_scars : int
        Total dorsal scar count.
    calibration_metadata : dict or None
        Calibration info.

    Returns
    -------
    dict
        CSV row data.
    """
    if metric["parent"] == metric["scar"]:
        surface_type = metric.get("surface_type", "NA")
        surface_feature = surface_type
    else:
        surface_type = next(
            (m["surface_type"] for m in all_metrics
             if m["parent"] == metric["parent"]
             and m["parent"] == m["scar"]),
            "NA"
        )
        surface_feature = metric["scar"]

    has_arrow = metric.get("has_arrow", False)
    arrow_tip = metric.get("arrow_tip", None)
    arrow_back = metric.get("arrow_back", None)

    dorsal_count = (
        total_dorsal_scars
        if (surface_type == "Dorsal"
            and surface_feature == "Dorsal")
        else "NA"
    )

    row = {
        "image_id": metric.get("image_id", "NA"),
        "surface_type": surface_type,
        "surface_feature": surface_feature,
        "total_dorsal_scars": dorsal_count,
        "centroid_x": metric.get("centroid_x", "NA"),
        "centroid_y": metric.get("centroid_y", "NA"),
        "technical_width": metric.get("technical_width", "NA"),
        "technical_length": metric.get("technical_length", "NA"),
        "max_width": metric.get("max_width", "NA"),
        "max_length": metric.get("max_length", "NA"),
        "total_area": metric.get("area", "NA"),
        "aspect_ratio": metric.get("aspect_ratio", "NA"),
        "perimeter": metric.get("perimeter", "NA"),
        "distance_to_max_width": metric.get(
            "distance_to_max_width", "NA"),
        "voronoi_num_cells": metric.get("voronoi_num_cells", "NA"),
        "convex_hull_width": metric.get("convex_hull_width", "NA"),
        "convex_hull_height": metric.get(
            "convex_hull_height", "NA"),
        "convex_hull_area": metric.get("convex_hull_area", "NA"),
        "voronoi_cell_area": metric.get("voronoi_cell_area", "NA"),
        "top_area": metric.get("top_area", "NA"),
        "bottom_area": metric.get("bottom_area", "NA"),
        "left_area": metric.get("left_area", "NA"),
        "right_area": metric.get("right_area", "NA"),
        "vertical_symmetry": metric.get(
            "vertical_symmetry", "NA"),
        "horizontal_symmetry": metric.get(
            "horizontal_symmetry", "NA"),
        "lateral_convexity": metric.get(
            "lateral_convexity", "NA"),
        "is_cortex": metric.get("is_cortex", False),
        "cortex_area": metric.get("cortex_area", "NA"),
        "cortex_percentage": metric.get(
            "cortex_percentage", "NA"),
        "has_arrow": has_arrow,
        "arrow_angle": metric.get("arrow_angle", "NA"),
        "scar_complexity": metric.get("scar_complexity", "NA"),
    }

    if calibration_metadata:
        row["calibration_method"] = calibration_metadata.get(
            "calibration_method", "NA")
        row["pixels_per_mm"] = calibration_metadata.get(
            "pixels_per_mm", "NA")
        row["scale_confidence"] = calibration_metadata.get(
            "scale_confidence", "NA")

    for key in ("triangle_base_length", "triangle_height",
                "shaft_solidity", "tip_solidity"):
        if key in metric:
            row[key] = metric[key]

    return row


def _build_column_order(
    metrics: List[Dict],
    calibration_metadata: Optional[Dict]
) -> List[str]:
    """Build the ordered list of CSV columns."""
    base = [
        "image_id", "surface_type", "surface_feature",
        "total_dorsal_scars", "centroid_x", "centroid_y",
        "technical_width", "technical_length",
        "max_width", "max_length", "total_area",
        "aspect_ratio", "perimeter", "distance_to_max_width",
    ]
    voronoi = [
        "voronoi_num_cells", "convex_hull_width",
        "convex_hull_height", "convex_hull_area",
        "voronoi_cell_area",
    ]
    symmetry = [
        "top_area", "bottom_area", "left_area", "right_area",
        "vertical_symmetry", "horizontal_symmetry",
    ]
    lateral = ["lateral_convexity"]
    cortex = ["is_cortex", "cortex_area", "cortex_percentage"]
    complexity = ["scar_complexity"]
    arrows = ["has_arrow", "arrow_angle"]

    for key in ("triangle_base_length", "triangle_height",
                "shaft_solidity", "tip_solidity"):
        if any(key in m for m in metrics):
            arrows.append(key)

    calibration = []
    if calibration_metadata:
        calibration = [
            "calibration_method", "pixels_per_mm",
            "scale_confidence",
        ]

    return (base + voronoi + symmetry + lateral + cortex
            + complexity + arrows + calibration)


def _write_csv(
    df: pd.DataFrame, output_path: str, append: bool
) -> None:
    """Write or append DataFrame to CSV."""
    if append and os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path)
            existing_cols = existing_df.columns.tolist()

            combined = list(
                set(existing_cols) | set(df.columns)
            )
            for col in combined:
                if col not in df.columns:
                    df[col] = "NA"
                if col not in existing_cols:
                    existing_df[col] = "NA"

            df = df[existing_cols]
            df.to_csv(
                output_path, mode="a",
                header=False, index=False
            )
        except Exception as e:
            logging.warning(
                f"Error aligning columns with existing CSV: {e}"
            )
            df.to_csv(
                output_path, mode="a",
                header=False, index=False
            )
        logging.info("Appended metrics to CSV: %s", output_path)
    else:
        df.to_csv(output_path, index=False)
        logging.info("Saved metrics to CSV: %s", output_path)
