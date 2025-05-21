"""
PyLithics: Image Analysis Module
=================================

This module provides a comprehensive toolkit for the quantitative analysis of lithic artifacts
via image processing. It is designed for researchers in archaeology, lithic analysis, and related
fields, enabling objective, reproducible, and detailed assessment of artifact features.

Key functionalities include:
    - Extracting contours from preprocessed (inverted binary) images and determining their hierarchical
      relationships.
    - Calculating geometric and spatial metrics (e.g., area, perimeter, aspect ratio, bounding box dimensions,
      and symmetry) for both parent and child contours.
    - Classifying parent contours into surface categories (e.g., Dorsal, Ventral, Platform, Lateral) based on
      dimensional tolerances.
    - Visualizing results by overlaying contour hierarchies, Voronoi diagrams, and convex hulls on the original
      images.
    - Exporting computed metrics to CSV files for further analysis.
    - Converting pixel-based measurements into real-world units.

The module includes the following key functions:
    * extract_contours_with_hierarchy(inverted_image, image_id, output_dir)
          - Extracts valid contours (excluding those touching the image border) and computes their hierarchy.
    * sort_contours_by_hierarchy(contours, hierarchy, exclude_nested_flags=None)
          - Organizes contours into parent, child, and nested child categories.
    * calculate_contour_metrics(sorted_contours, hierarchy, original_contours)
          - Computes geometric metrics for contours, including area, perimeter (for parent contours), maximum
            dimensions, and centroid coordinates.
    * hide_nested_child_contours(contours, hierarchy)
          - Flags nested or single-child contours for exclusion.
    * classify_parent_contours(metrics, tolerance=0.1)
          - Classifies parent contours into surface types such as Dorsal, Ventral, Platform, and Lateral.
    * visualize_contours_with_hierarchy(contours, hierarchy, metrics, inverted_image, output_path)
          - Overlays contours, centroids, and labels on the original image.
    * save_measurements_to_csv(metrics, output_path, append=False)
          - Saves computed contour metrics to a CSV file.
    * analyze_dorsal_symmetry(metrics, contours, inverted_image)
          - Performs symmetry analysis on the dorsal surface, calculating areas and symmetry indices.
    * calculate_voronoi_points(metrics, inverted_image, padding_factor=0.02)
          - Generates a Voronoi diagram and convex hull from dorsal surface centroids, and computes related metrics.
    * visualize_voronoi_diagram(voronoi_data, inverted_image, output_path)
          - Visualizes the Voronoi diagram and convex hull overlaid on the original image.
    * process_and_save_contours(inverted_image, conversion_factor, output_dir, image_id)
          - Integrates all steps from contour extraction to CSV export and visualization.
    * convert_metrics_to_real_world(metrics, conversion_factor)
          - Converts pixel-based measurements to real-world units.

Usage Example:
    >>> from pylithics import process_and_save_contours
    >>> process_and_save_contours(inverted_image, 0.01, "/path/to/output", "artifact_001")

All functions are fully documented and include robust logging for debugging and traceability.
"""


import logging
import os
import sys
import cv2
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Voronoi, voronoi_plot_2d
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.geometry.polygon import LinearRing
from shapely.ops import unary_union, voronoi_diagram
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from .arrow_detection import analyze_child_contour_for_arrow
from pylithics.image_processing.utils import filter_contours_by_min_area

def extract_contours_with_hierarchy(inverted_image, image_id, output_dir):
    """
    Extract contours and hierarchy using cv2.RETR_TREE, exclude the image border.
    Uses minimum area from the config file.

    Parameters
    ----------
    inverted_image : ndarray
        Preprocessed binary/grayscale image in which to find contours.
    image_id : str
        Unique identifier for this image (used for debug directory naming).
    output_dir : str
        Base path under which per‐image debug folders will be created.

    Returns
    -------
    valid_contours : list of ndarray
        All contours whose bounding box does not touch the image border.
    valid_hierarchy : ndarray
        Corresponding hierarchy entries for valid_contours.
    """
    # Import the utility function
    from pylithics.image_processing.utils import filter_contours_by_min_area
    from pylithics.image_processing.importer import load_preprocessing_config
    import yaml
    import os


    # Load config to get minimum area
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Get minimum area from config with default fallback
        min_contour_area = config.get('contour_filtering', {}).get('min_area', 300.0)
        logging.info(f"Using minimum contour area: {min_contour_area} pixels for image {image_id}")
    except Exception as e:
        logging.warning(f"Could not load min_area from config, using default value: {e}")
        min_contour_area = 300.0  # Default value that worked previously

    # 1) find all contours + raw hierarchy
    contours, hierarchy = cv2.findContours(
        inverted_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    hierarchy = hierarchy[0] if hierarchy is not None else None

    if not contours:
        logging.warning("No contours found in image %s", image_id)
        return [], None

    # Log the number of raw contours found
    logging.info(f"Found {len(contours)} raw contours in image {image_id}")

    # 2) filter out those touching the border
    height, width = inverted_image.shape
    valid_contours, valid_hierarchy = [], []

    # Create index mapping from old to new indices
    old_to_new_idx = {}
    new_idx = 0

    for old_idx, (cnt, h) in enumerate(zip(contours, hierarchy)):
        x, y, w, h_box = cv2.boundingRect(cnt)
        if x > 0 and y > 0 and x + w < width and y + h_box < height:
            valid_contours.append(cnt)
            valid_hierarchy.append(h.copy())  # Copy to avoid modifying original
            old_to_new_idx[old_idx] = new_idx
            new_idx += 1

    # Update parent indices in the hierarchy to reflect new indexing
    for i, h in enumerate(valid_hierarchy):
        if h[3] != -1:  # If has a parent
            if h[3] in old_to_new_idx:
                valid_hierarchy[i][3] = old_to_new_idx[h[3]]
            else:
                # Parent was filtered out, make this a root
                valid_hierarchy[i][3] = -1

    valid_hierarchy = np.array(valid_hierarchy)

    # Log how many contours remain after border filtering
    logging.info(f"After border filtering: {len(valid_contours)} contours remain in image {image_id}")

    # 3) filter out small contours using the minimum area from config
    valid_contours, valid_hierarchy = filter_contours_by_min_area(
        valid_contours, valid_hierarchy, min_contour_area
    )

    if not valid_contours:
        logging.warning("No valid contours remain after filtering in image %s", image_id)
        return [], None

    logging.info(
        "Extracted %d valid contours: %d parents/%d children total",
        len(valid_contours),
        np.sum(valid_hierarchy[:, 3] == -1),
        np.sum(valid_hierarchy[:, 3] != -1)
    )

    return valid_contours, valid_hierarchy


def sort_contours_by_hierarchy(contours, hierarchy, exclude_nested_flags=None):
    """
    Sort contours into parents, children, and nested children based on hierarchy.
    This version includes robust bounds checking.

    Args:
        contours (list): List of detected contours.
        hierarchy (numpy.ndarray): Hierarchy array corresponding to contours.
        exclude_nested_flags (list): List of booleans where True indicates contours to exclude.

    Returns:
        dict: A dictionary with sorted contours:
            - "parents": List of parent contours.
            - "children": List of child contours.
            - "nested_children": List of nested child contours (if any).
    """
    # Handle empty contours or hierarchy
    if not contours or hierarchy is None or len(hierarchy) == 0:
        return {"parents": [], "children": [], "nested_children": []}

    parents, children, nested = [], [], []

    # Create exclude_nested_flags if not provided
    if exclude_nested_flags is None:
        exclude_nested_flags = [False] * len(contours)

    # Ensure exclude_nested_flags is the right length
    if len(exclude_nested_flags) != len(contours):
        logging.warning("exclude_nested_flags length mismatch: %d vs %d contours. Using defaults.",
                        len(exclude_nested_flags), len(contours))
        exclude_nested_flags = [False] * len(contours)

    # First, identify parent contours (those with no parent)
    for i, h in enumerate(hierarchy):
        if i >= len(exclude_nested_flags) or i >= len(contours):
            continue  # Skip if index is out of bounds

        if exclude_nested_flags[i]:
            continue

        parent_idx = h[3]
        if parent_idx == -1:  # No parent
            parents.append(contours[i])

    # Second, identify direct children (first level)
    for i, h in enumerate(hierarchy):
        if i >= len(exclude_nested_flags) or i >= len(contours):
            continue  # Skip if index is out of bounds

        if exclude_nested_flags[i]:
            continue

        parent_idx = h[3]
        if parent_idx != -1:  # Has a parent
            # Check if the parent is a root contour
            if parent_idx < len(hierarchy) and hierarchy[parent_idx][3] == -1:
                children.append(contours[i])
            else:
                # This is a nested child (child of a child)
                nested.append(contours[i])

    logging.info(
        "Sorted contours: %d parents, %d children, %d nested",
        len(parents), len(children), len(nested)
    )
    return {"parents": parents, "children": children, "nested_children": nested}


def calculate_contour_metrics(sorted_contours, hierarchy, original_contours, image_shape, image_dpi=None):
    """
    Calculate metrics for parent, first-level child, and nested (second-level) child contours.

    Parameters
    ----------
    sorted_contours : dict
        {"parents":…, "children":…, "nested_children":…}
    hierarchy : np.ndarray
        contour hierarchy array
    original_contours : list
        list of all extracted contours
    image_shape : tuple
        shape of the source image (h, w)
    image_dpi : float, optional
        DPI of the image being processed, used for scaling detection parameters

    Returns
    -------
    metrics : list of dict
        consolidated metrics including arrow info for nested children
    """

    metrics = []
    parent_map = {}

    # Create a mapping between filtered contours and original contours
    # This is needed because the hierarchy indices refer to original contours
    contour_index_map = {}  # Maps contour to its index in original_contours

    # Build map of all contours in sorted_contours
    all_sorted_contours = (
        sorted_contours["parents"] +
        sorted_contours["children"] +
        sorted_contours.get("nested_children", [])
    )

    # Create a mapping from contours to their original indices
    for contour in all_sorted_contours:
        for i, orig_cnt in enumerate(original_contours):
            if np.array_equal(contour, orig_cnt):
                contour_key = str(contour.tobytes())  # Use bytes as key
                contour_index_map[contour_key] = i
                break

    # Process parents
    for pi, cnt in enumerate(sorted_contours["parents"]):
        contour_key = str(cnt.tobytes())
        idx = contour_index_map.get(contour_key)
        if idx is None:
            logging.warning(f"Could not find parent contour {pi} in original contours")
            continue

        lab = f"parent {pi+1}"
        parent_map[idx] = lab
        area = round(cv2.contourArea(cnt), 2)
        peri = round(cv2.arcLength(cnt, True), 2)

        # Safely calculate centroid
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = round(M["m10"] / M["m00"], 2)
            cy = round(M["m01"] / M["m00"], 2)
        else:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = round(x + w/2, 2), round(y + h/2, 2)

        x,y,w,h = cv2.boundingRect(cnt)

        # Calculate max length and width
        max_len = max_wid = 0
        p1 = p2 = None
        for i in range(len(cnt)):
            for j in range(i+1, len(cnt)):
                a = cnt[i][0]; b = cnt[j][0]
                d = np.linalg.norm(a - b)
                if d > max_len:
                    max_len, p1, p2 = d, a, b
        if p1 is not None and p2 is not None:
            v = p2 - p1
            perp = np.array([-v[1], v[0]], dtype=float)
            perp /= np.linalg.norm(perp)
            widths = [abs(np.dot(pt[0]-p1, perp)) for pt in cnt]
            max_wid = max(widths)
        ml, mw = round(max_len, 2), round(max_wid, 2)

        metrics.append({
            "parent": lab, "scar": lab,
            "centroid_x": cx, "centroid_y": cy,
            "width": w, "height": h,
            "area": area, "aspect_ratio": round(h/w,2) if w else None,
            "bounding_box_x": x, "bounding_box_y": y,
            "bounding_box_width": w, "bounding_box_height": h,
            "max_length": ml, "max_width": mw,
            "contour": cnt.tolist(),
            "perimeter": peri,
            # arrow defaults
            "has_arrow": False, "arrow_angle_rad": None,
            "arrow_angle_deg": None, "arrow_angle": None
        })

    # Process children/scars with mapping for arrow integration
    scar_metrics = {}  # Map from contour key to scar entry
    scar_entries = {}  # Map from scar label to entry

    for ci, cnt in enumerate(sorted_contours["children"]):
        contour_key = str(cnt.tobytes())
        idx = contour_index_map.get(contour_key)
        if idx is None:
            logging.warning(f"Could not find child contour {ci} in original contours")
            continue

        # Get parent using hierarchy
        if idx < len(hierarchy):
            parent_idx = hierarchy[idx][3]
            pl = parent_map.get(parent_idx, "Unknown")
        else:
            logging.warning(f"Child contour index {idx} out of bounds for hierarchy")
            pl = "Unknown"

        lab = f"scar {ci+1}"
        area = round(cv2.contourArea(cnt), 2)

        # Safe centroid calculation
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = round(M["m10"] / M["m00"], 2)
            cy = round(M["m01"] / M["m00"], 2)
        else:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = round(x + w/2, 2), round(y + h/2, 2)

        x,y,w,h = cv2.boundingRect(cnt)

        # Calculate max length and width
        max_len = max_wid = 0
        p1 = p2 = None
        for i in range(len(cnt)):
            for j in range(i+1, len(cnt)):
                a = cnt[i][0]; b = cnt[j][0]
                d = np.linalg.norm(a - b)
                if d > max_len:
                    max_len, p1, p2 = d, a, b
        if p1 is not None and p2 is not None:
            v = p2 - p1
            perp = np.array([-v[1], v[0]], dtype=float)
            perp /= np.linalg.norm(perp)
            widths = [abs(np.dot(pt[0]-p1, perp)) for pt in cnt]
            max_wid = max(widths)
        ml, mw = round(max_len, 2), round(max_wid, 2)

        entry = {
            "parent": pl, "scar": lab,
            "centroid_x": cx, "centroid_y": cy,
            "width": w, "height": h,
            "area": area, "aspect_ratio": round(h/w,2) if w else None,
            "max_length": ml, "max_width": mw,
            "bounding_box_x": x, "bounding_box_y": y,
            "bounding_box_width": w, "bounding_box_height": h,
            "has_arrow": False, "arrow_angle_rad": None,
            "arrow_angle_deg": None, "arrow_angle": None
        }
        metrics.append(entry)
        scar_metrics[contour_key] = entry  # Store by contour key
        scar_entries[lab] = entry  # Store by label

    # Skip nested children processing if there are no direct children
    if not scar_metrics and sorted_contours.get("nested_children", []):
        logging.info("Skipping nested children processing as there are no direct children (scars)")
        return metrics

    # Process nested children (detect arrows and update parent scars)
    for ni, cnt in enumerate(sorted_contours.get("nested_children", [])):
        contour_key = str(cnt.tobytes())
        nested_idx = contour_index_map.get(contour_key)
        if nested_idx is None or nested_idx >= len(hierarchy):
            logging.warning(f"Could not find nested contour {ni} in original contours or hierarchy")
            continue

        parent_idx = hierarchy[nested_idx][3]  # Get parent contour index

        # Find which scar this belongs to
        parent_scar = None
        parent_contour_key = None

        # Try to find parent contour in original contours
        if parent_idx < len(original_contours):
            parent_contour = original_contours[parent_idx]
            parent_contour_key = str(parent_contour.tobytes())

        # Check if the parent contour is in our scar metrics
        if parent_contour_key in scar_metrics:
            parent_scar = scar_metrics[parent_contour_key]
        else:
            # Find parent through hierarchy relationships if not direct
            found = False
            for idx, h in enumerate(hierarchy):
                if idx == parent_idx and h[3] != -1 and idx < len(original_contours):
                    grandparent_idx = h[3]
                    for cidx, ch in enumerate(hierarchy):
                        if ch[3] == grandparent_idx and cidx < len(original_contours):
                            cidx_contour = original_contours[cidx]
                            cidx_key = str(cidx_contour.tobytes())
                            if cidx_key in scar_metrics:
                                parent_scar = scar_metrics[cidx_key]
                                found = True
                                break
                    if found:
                        break

        if parent_scar is None:
            logging.debug(f"Could not find parent scar for nested contour {ni}")
            continue

        # Get image name without extension
        if hasattr(image_shape, 'filename'):
            image_name = os.path.splitext(image_shape.filename)[0]
        elif isinstance(image_shape, str):
            image_name = os.path.splitext(image_shape)[0]
        else:
            # Default name
            image_name = f"image_{ni}"

        # Create debug directory
        debug_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "image_debug",
                                image_name)
        os.makedirs(debug_dir, exist_ok=True)

        # Create temporary entry for arrow detection
        temp_entry = {
            "scar": f"nested_{ni}",
            "debug_dir": debug_dir
        }

        # Run arrow detection
        logging.debug(f"Running arrow detection on nested contour {ni}")
        result = analyze_child_contour_for_arrow(cnt, temp_entry, image_shape, image_dpi)

        # If arrow detected, update the parent scar's entry
        if result:
            logging.info(f"Arrow detected in nested contour {ni} (parent: {parent_scar['scar']}) with angle {result.get('compass_angle', 'unknown')}")
            parent_scar.update({
                "has_arrow": True,
                "arrow_angle_rad": round(result["angle_rad"], 0),
                "arrow_angle_deg": round(result["angle_deg"], 0),
                "arrow_angle": round(result["compass_angle"], 0),
                "arrow_tip": result["arrow_tip"],
                "arrow_back": result["arrow_back"]
            })
        else:
            logging.debug(f"No arrow detected in nested contour {ni}")

    return metrics

def detect_arrows_independently(original_contours, metrics, image, image_dpi=None):
    """
    Detect arrows in contours independently of hierarchy relationships.
    Searches all contours for potential arrows and associates them with the
    appropriate scar metrics.

    Parameters
    ----------
    original_contours : list
        List of all detected contours.
    metrics : list
        List of metric dictionaries for parents and scars.
    image : ndarray
        Original image for visualization and debug purposes.
    image_dpi : float, optional
        DPI of the image being processed, used for scaling detection parameters.

    Returns
    -------
    metrics : list
        Updated metrics list with arrow information added.
    """
    # Find all scar metrics (they have a parent that's not themselves)
    scar_metrics = [m for m in metrics if m["parent"] != m["scar"]]

    # Track which scars have arrows assigned
    scars_with_arrows = set()

    # For each contour, test if it could be an arrow
    logging.info("Starting independent arrow detection on all contours...")

    # First, find parent contours to exclude them from arrow detection
    parent_contours = set()
    parent_indices = []
    for i, m in enumerate(metrics):
        if m["parent"] == m["scar"]:  # This is a parent contour
            for j, cnt in enumerate(original_contours):
                # Find matching contour for this parent metric
                if cv2.contourArea(cnt) == m["area"]:
                    parent_contours.add(j)
                    parent_indices.append(j)
                    break

    # Now find scar contours to exclude them too
    scar_contours = set()
    scar_indices = []
    for i, m in enumerate(metrics):
        if m["parent"] != m["scar"] and "parent" in m:  # This is a scar
            for j, cnt in enumerate(original_contours):
                if j in parent_contours:
                    continue  # Skip already identified parent contours
                # Find matching contour for this scar metric
                if cv2.contourArea(cnt) == m["area"]:
                    scar_contours.add(j)
                    scar_indices.append(j)
                    break

    # Create debug directory for this image
    debug_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "image_debug",
                        "independent_arrow_detection")
    os.makedirs(debug_dir, exist_ok=True)

    # Prepare scar contours for containment testing
    scar_contour_map = {}  # Maps scar label to its contour
    for idx in scar_indices:
        cnt = original_contours[idx]
        # Find which metric this corresponds to
        for m in scar_metrics:
            if abs(cv2.contourArea(cnt) - m["area"]) < 1.0:  # Allow small rounding differences
                scar_contour_map[m["scar"]] = cnt
                break

    arrow_candidates = []
    # Test all contours that aren't parents or scars
    for i, cnt in enumerate(original_contours):
        if i in parent_contours or i in scar_contours:
            continue  # Skip parents and scars

        # Basic filtering criteria for arrow candidates (must have reasonable area and solidity)
        area = cv2.contourArea(cnt)
        if area < 1.0:  # Skip very tiny contours
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            continue

        solidity = area / hull_area
        if solidity < 0.4 or solidity > 0.9:  # Solidity range for arrow shapes
            continue

        # Create debug entry
        temp_entry = {
            "scar": f"candidate_{i}",
            "debug_dir": os.path.join(debug_dir, f"contour_{i}")
        }
        os.makedirs(temp_entry["debug_dir"], exist_ok=True)

        # Try arrow detection
        result = analyze_child_contour_for_arrow(cnt, temp_entry, image, image_dpi)

        if result:
            logging.info(f"Independent arrow detection found arrow in contour {i} with angle {result.get('compass_angle', 'unknown')}")
            arrow_candidates.append((i, cnt, result))

    # Now assign arrows to scars based on containment
    for arrow_idx, arrow_cnt, arrow_result in arrow_candidates:
        # Calculate arrow centroid
        M = cv2.moments(arrow_cnt)
        if M["m00"] > 0:
            arrow_cx = M["m10"] / M["m00"]
            arrow_cy = M["m01"] / M["m00"]
        else:
            x, y, w, h = cv2.boundingRect(arrow_cnt)
            arrow_cx, arrow_cy = x + w/2, y + h/2

        # Find containing scar
        containing_scar = None
        smallest_area = float('inf')

        for scar_label, scar_cnt in scar_contour_map.items():
            # Check if the arrow centroid is inside this scar
            if cv2.pointPolygonTest(scar_cnt, (arrow_cx, arrow_cy), False) >= 0:
                scar_area = cv2.contourArea(scar_cnt)
                if scar_area < smallest_area:
                    smallest_area = scar_area
                    containing_scar = scar_label

        if containing_scar and containing_scar not in scars_with_arrows:
            # Found a containing scar that doesn't have an arrow yet
            # Update the scar's metrics with arrow information
            for metric in metrics:
                if metric["scar"] == containing_scar:
                    logging.info(f"Assigning arrow from contour {arrow_idx} to scar {containing_scar}")
                    metric.update({
                        "has_arrow": True,
                        "arrow_angle_rad": round(arrow_result["angle_rad"], 0),
                        "arrow_angle_deg": round(arrow_result["angle_deg"], 0),
                        "arrow_angle": round(arrow_result["compass_angle"], 0),
                        "arrow_tip": arrow_result["arrow_tip"],
                        "arrow_back": arrow_result["arrow_back"]
                    })
                    scars_with_arrows.add(containing_scar)
                    break

    logging.info(f"Independent arrow detection completed. Found and assigned {len(scars_with_arrows)} arrows.")
    return metrics


def hide_nested_child_contours(contours, hierarchy):
    """
    Flag only first-level child contours whose parent has exactly one child.
    Do not flag nested (depth ≥2) contours so they can be processed for arrow detection.
    This version includes extensive bounds checking to prevent index errors.
    """
    flags = [False] * len(contours)

    # Safety check for empty contours or hierarchy
    if not contours or hierarchy is None or len(hierarchy) == 0:
        return flags

    # Count direct children for each parent
    child_counts = {}
    for i, h in enumerate(hierarchy):
        if i >= len(contours):
            continue  # Skip if index out of bounds

        p = h[3]
        if p != -1:
            child_counts[p] = child_counts.get(p, 0) + 1

    # Flag single-child first-level scars only
    for i, h in enumerate(hierarchy):
        if i >= len(contours):
            continue  # Skip if index out of bounds

        parent_idx = h[3]
        # Skip if parent index invalid
        if parent_idx == -1 or parent_idx >= len(hierarchy):
            continue

        # only flag if it's a first-level child and its parent has exactly one child
        if parent_idx != -1 and hierarchy[parent_idx][3] == -1 and child_counts.get(parent_idx, 0) == 1:
            if i < len(flags):
                flags[i] = True

    logging.info("Flagged %d single-child contours for exclusion.", sum(flags))
    return flags


def classify_parent_contours(metrics, tolerance=0.1):
    """
    Classify parent contours into surfaces: Dorsal, Ventral, Platform, Lateral.
    Robustly handles cases with fewer than all four surface types.

    Args:
        metrics (list): List of dictionaries containing contour metrics.
        tolerance (float): Dimensional tolerance for surface comparison.

    Returns:
        list: Updated metrics with surface classifications.
    """
    # Extract parent contours only
    parents = [m for m in metrics if m["parent"] == m["scar"]]

    if not parents:
        logging.warning("No parent contours found for classification.")
        return metrics

    # Initialize classification
    for parent in parents:
        parent["surface_type"] = None

    surfaces_identified = []

    # Identify Dorsal Surface (always present if any parents exist)
    try:
        dorsal = max(parents, key=lambda p: p["area"])
        dorsal["surface_type"] = "Dorsal"
        surfaces_identified.append("Dorsal")
    except ValueError:
        logging.error("Unable to identify the dorsal surface due to missing or invalid parent metrics.")
        return metrics

    # If only one parent contour, we're done - it's the dorsal surface
    if len(parents) == 1:
        logging.info("Only one parent contour found, classified as Dorsal surface.")
        return metrics

    # Identify Ventral Surface
    ventral = None
    for parent in parents:
        if parent["surface_type"] is None:
            if (
                abs(parent["height"] - dorsal["height"]) <= tolerance * dorsal["height"]
                and abs(parent["width"] - dorsal["width"]) <= tolerance * dorsal["width"]
                and abs(parent["area"] - dorsal["area"]) <= tolerance * dorsal["area"]
            ):
                parent["surface_type"] = "Ventral"
                ventral = parent
                surfaces_identified.append("Ventral")
                break

    # Identify Platform Surface
    platform = None
    platform_candidates = [
        p for p in parents if p["surface_type"] is None and p["height"] < dorsal["height"] and p["width"] < dorsal["width"]
    ]
    if platform_candidates:
        platform = min(platform_candidates, key=lambda p: p["area"])
        platform["surface_type"] = "Platform"
        surfaces_identified.append("Platform")

    # Identify Lateral Surface - only if platform exists
    if platform is not None:
        for parent in parents:
            if parent["surface_type"] is None:
                if (
                    abs(parent["height"] - dorsal["height"]) <= tolerance * dorsal["height"]
                    and abs(parent["height"] - platform["height"]) > tolerance * platform["height"]
                    and parent["width"] != dorsal["width"]
                ):
                    parent["surface_type"] = "Lateral"
                    surfaces_identified.append("Lateral")
                    break
    # Alternative logic when platform doesn't exist but we need to classify lateral
    elif ventral is not None:
        for parent in parents:
            if parent["surface_type"] is None:
                if (
                    abs(parent["height"] - dorsal["height"]) <= tolerance * dorsal["height"]
                    and abs(parent["width"] - dorsal["width"]) > tolerance * dorsal["width"]
                ):
                    parent["surface_type"] = "Lateral"
                    surfaces_identified.append("Lateral")
                    break

    # Assign default surface type if still None
    for parent in parents:
        if parent["surface_type"] is None:
            parent["surface_type"] = "Unclassified"

    logging.info("Classified parent contours into surfaces: %s.", ", ".join(surfaces_identified))
    return metrics


def visualize_contours_with_hierarchy(contours, hierarchy, metrics, inverted_image, output_path):
    """
    Visualize contours with hierarchy, label them, and overlay detected arrows (if any).

    Args:
        contours (list): List of contours (parents + children) in display order.
        hierarchy (ndarray): Contour hierarchy array.
        metrics (list): List of metric dicts, each may contain 'has_arrow', 'arrow_back', 'arrow_tip', 'arrow_angle'.
        inverted_image (ndarray): Inverted binary image (0=foreground, 255=background).
        output_path (str): File path to write the labeled image.
    """
    # Invert back to white background
    original = cv2.bitwise_not(inverted_image)
    # Make BGR image for color drawing
    labeled = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

    # Draw contours, centroids, and labels
    label_positions = []
    for i, cnt in enumerate(contours):
        if i >= len(metrics):
            continue

        m = metrics[i]
        parent_label = m["parent"]
        scar_label   = m["scar"]

        # Color & text
        if parent_label == scar_label:
            color = (153, 60, 94)   # purple for parents
            text  = m.get("surface_type", parent_label)
        else:
            color = (99, 184, 253)  # orange for children
            text  = scar_label

        # Draw the contour
        cv2.drawContours(labeled, [cnt], -1, color, 2)

        # Compute and draw centroid
        M = cv2.moments(cnt)
        if M.get("m00", 0) != 0:
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
        else:
            x,y,w,h = cv2.boundingRect(cnt)
            cx,cy = x + w//2, y + h//2
        cv2.circle(labeled, (cx, cy), 4, (1, 97, 230), -1)

        # Place label (avoid overlaps)
        ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        tx, ty = cx + 10, cy - 10
        for lx, ly, lw, lh in label_positions:
            if tx < lx+lw and tx+ts[0] > lx and ty < ly+lh and ty+ts[1] > ly:
                ty = ly + lh + 10
        label_positions.append((tx, ty, ts[0], ts[1]))
        cv2.putText(labeled, text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (186,186,186), 2)


    # Draw arrows for all detected arrow features
    for m in metrics:
        if m.get("has_arrow") and m.get("arrow_back") and m.get("arrow_tip"):
            # Convert to integer tuples if needed
            back = tuple(int(v) for v in m["arrow_back"])
            tip = tuple(int(v) for v in m["arrow_tip"])

            # Draw arrowed line (red)
            cv2.arrowedLine(
                labeled,
                back,
                tip,
                color=(0, 0, 255),
                thickness=2,
                tipLength=0.2
            )

            # Remove the yellow dots at back and tip
            # (Comment out or remove these lines)
            # cv2.circle(labeled, back, 4, (0, 255, 255), -1)
            # cv2.circle(labeled, tip, 4, (0, 255, 255), -1)

            # Annotate compass bearing with better positioning
            angle = m.get("arrow_angle", None)
            if angle is not None:
                # Calculate shaft vector
                shaft_vector = np.array([tip[0] - back[0], tip[1] - back[1]])
                shaft_length = np.linalg.norm(shaft_vector)

                if shaft_length > 0:
                    # Calculate perpendicular vector (90 degrees clockwise from shaft)
                    perp_vector = np.array([shaft_vector[1], -shaft_vector[0]]) / shaft_length

                    # Use a larger offset distance (40 pixels) for better separation
                    offset_distance = 40
                    perp_offset = perp_vector * offset_distance

                    # Calculate text position at 1/3 of the shaft from the back, offset perpendicular
                    shaft_fraction = 1/3
                    text_base_pos = (
                        int(back[0] + shaft_vector[0] * shaft_fraction),
                        int(back[1] + shaft_vector[1] * shaft_fraction)
                    )
                    text_pos = (
                        int(text_base_pos[0] + perp_offset[0]),
                        int(text_base_pos[1] + perp_offset[1])
                    )


                    text = f"{int(angle)} deg"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    thickness = 2

                    # Get text size
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                    # Create bigger background rectangle with padding
                    padding = 4
                    text_bg_pt1 = (text_pos[0] - padding, text_pos[1] - text_height - padding)
                    text_bg_pt2 = (text_pos[0] + text_width + padding, text_pos[1] + padding)

                    # Draw white background with black border
                    cv2.rectangle(labeled, text_bg_pt1, text_bg_pt2, (255, 255, 255), -1)  # White fill
                    cv2.rectangle(labeled, text_bg_pt1, text_bg_pt2, (0, 0, 0), 1)        # Black border

                    # Draw text in black
                    cv2.putText(
                        labeled,
                        text,
                        text_pos,
                        font,
                        font_scale,
                        (0, 0, 0),  # Black text
                        thickness,
                        cv2.LINE_AA
                    )

    # Save the labeled image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, labeled)
    logging.info("Saved visualized contours with arrows to %s", output_path)


def save_measurements_to_csv(metrics, output_path, append=False):
    """
    Save contour metrics to a CSV file with updated column structure.
    Includes detailed arrow measurements and coordinates.
    """
    updated_data = []
    for metric in metrics:
        if metric["parent"] == metric["scar"]:
            surface_type = metric.get("surface_type", "NA")
            surface_feature = surface_type
        else:
            parent_surface_type = next(
                (m["surface_type"] for m in metrics if m["parent"] == metric["parent"] and m["parent"] == m["scar"]),
                "NA"
            )
            surface_type = parent_surface_type
            surface_feature = metric["scar"]

        # Process arrow-specific data with proper handling of coordinates
        has_arrow = metric.get("has_arrow", False)
        arrow_tip = metric.get("arrow_tip", None)
        arrow_back = metric.get("arrow_back", None)

        # Extract coordinates safely
        arrow_tip_x = arrow_tip[0] if isinstance(arrow_tip, (list, tuple)) and len(arrow_tip) >= 1 else "NA"
        arrow_tip_y = arrow_tip[1] if isinstance(arrow_tip, (list, tuple)) and len(arrow_tip) >= 2 else "NA"
        arrow_back_x = arrow_back[0] if isinstance(arrow_back, (list, tuple)) and len(arrow_back) >= 1 else "NA"
        arrow_back_y = arrow_back[1] if isinstance(arrow_back, (list, tuple)) and len(arrow_back) >= 2 else "NA"

        data_entry = {
            "image_id": metric.get("image_id", "NA"),
            "surface_type": surface_type,
            "surface_feature": surface_feature,
            "centroid_x": metric.get("centroid_x", "NA"),
            "centroid_y": metric.get("centroid_y", "NA"),
            "width": metric.get("width", "NA"),
            "height": metric.get("height", "NA"),
            "max_width": metric.get("max_width", "NA"),
            "max_length": metric.get("max_length", "NA"),
            "total_area": metric.get("area", "NA"),
            "aspect_ratio": metric.get("aspect_ratio", "NA"),
            "perimeter": metric.get("perimeter", "NA"),
            "voronoi_num_cells": metric.get("voronoi_num_cells", "NA"),
            "convex_hull_width": metric.get("convex_hull_width", "NA"),
            "convex_hull_height": metric.get("convex_hull_height", "NA"),
            "convex_hull_area": metric.get("convex_hull_area", "NA"),
            "voronoi_cell_area": metric.get("voronoi_cell_area", "NA"),
            "top_area": metric.get("top_area", "NA"),
            "bottom_area": metric.get("bottom_area", "NA"),
            "left_area": metric.get("left_area", "NA"),
            "right_area": metric.get("right_area", "NA"),
            "vertical_symmetry": metric.get("vertical_symmetry", "NA"),
            "horizontal_symmetry": metric.get("horizontal_symmetry", "NA"),
            # arrow data with explicit type handling
            "has_arrow": has_arrow,
            # "arrow_angle_rad": metric.get("arrow_angle_rad", "NA"), # angle of the arrow in radians
            # "arrow_angle_deg": metric.get("arrow_angle_deg", "NA"), # same angle as arrow_angle_rad, but converted to degrees for easier human interpretation.
            "arrow_angle": metric.get("arrow_angle", "NA"), # Rob's arrow angles schema
            #"arrow_tip_x": arrow_tip_x, # x pixel coordinates of the arrow's tip point
            #"arrow_tip_y": arrow_tip_y, # y pixel coordinates of the arrow's tip point
            #"arrow_back_x": arrow_back_x, # x pixel coordinates of the arrow's tail point
            #"arrow_back_y": arrow_back_y, # y pixel coordinates of the arrow's tail point
        }

        # Add additional arrow metrics if available
        if "triangle_base_length" in metric:
            data_entry["triangle_base_length"] = metric["triangle_base_length"]
        if "triangle_height" in metric:
            data_entry["triangle_height"] = metric["triangle_height"]
        if "shaft_solidity" in metric:
            data_entry["shaft_solidity"] = metric["shaft_solidity"]
        if "tip_solidity" in metric:
            data_entry["tip_solidity"] = metric["tip_solidity"]

        updated_data.append(data_entry)

    # Define column order for the CSV
    base_columns = [
        "image_id", "surface_type", "surface_feature", "centroid_x", "centroid_y",
        "width", "height", "max_width", "max_length", "total_area", "aspect_ratio",
        "perimeter"
    ]

    voronoi_columns = [
        "voronoi_num_cells", "convex_hull_width", "convex_hull_height", "convex_hull_area",
        "voronoi_cell_area"
    ]

    symmetry_columns = [
        "top_area", "bottom_area", "left_area", "right_area",
        "vertical_symmetry", "horizontal_symmetry"
    ]

    # Expanded arrow columns to include additional metrics
    arrow_columns = [
        "has_arrow",
        # "arrow_angle_rad",
        # "arrow_angle_deg",
        "arrow_angle", # Rob's arrow angle schema
        # "arrow_tip_x",
        # "arrow_tip_y",
        # "arrow_back_x",
        # "arrow_back_y",
    ]

    # Check if any metrics have the additional arrow fields
    if any("triangle_base_length" in m for m in metrics):
        arrow_columns.append("triangle_base_length")
    if any("triangle_height" in m for m in metrics):
        arrow_columns.append("triangle_height")
    if any("shaft_solidity" in m for m in metrics):
        arrow_columns.append("shaft_solidity")
    if any("tip_solidity" in m for m in metrics):
        arrow_columns.append("tip_solidity")

    all_columns = base_columns + voronoi_columns + symmetry_columns + arrow_columns

    # Create DataFrame with all columns, handling any missing columns gracefully
    df = pd.DataFrame(updated_data)

    # Ensure all required columns exist, adding empty ones if needed
    for col in all_columns:
        if col not in df.columns:
            df[col] = "NA"

    # Reorder columns to match expected order
    # Only include columns that exist in the DataFrame
    existing_columns = [col for col in all_columns if col in df.columns]
    df = df[existing_columns]

    # Fill any NaN values
    df.fillna("NA", inplace=True)

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write to CSV
    if append and os.path.exists(output_path):
        # Read existing CSV to get its columns
        try:
            existing_df = pd.read_csv(output_path)
            existing_columns = existing_df.columns.tolist()

            # Align columns with existing file
            combined_columns = list(set(existing_columns) | set(df.columns))
            for col in combined_columns:
                if col not in df.columns:
                    df[col] = "NA"
                if col not in existing_columns:
                    existing_df[col] = "NA"

            # Write with matching columns
            df = df[existing_columns]
            df.to_csv(output_path, mode="a", header=False, index=False)
        except Exception as e:
            # If there's an error reading the existing file, just append
            logging.warning(f"Error aligning columns with existing CSV: {e}")
            df.to_csv(output_path, mode="a", header=False, index=False)

        logging.info("Appended metrics to existing CSV file: %s", output_path)
    else:
        df.to_csv(output_path, index=False)
        logging.info("Saved metrics to new CSV file: %s", output_path)


def analyze_dorsal_symmetry(metrics, contours, inverted_image):
    """
    Perform symmetry analysis for the Dorsal surface and calculate areas for its halves.

    Args:
        metrics (list): List of dictionaries containing contour metrics.
        contours (list): List of valid contours.
        inverted_image (numpy.ndarray): Inverted binary thresholded image.

    Returns:
        dict: Symmetry areas for the Dorsal surface (top, bottom, left, right).
    """
    # Find the Dorsal surface from metrics
    dorsal_metric = next((m for m in metrics if m.get("surface_type") == "Dorsal"), None)
    if not dorsal_metric:
        return {"top_area": None, "bottom_area": None, "left_area": None, "right_area": None}

    # Extract Dorsal contour based on parent label
    dorsal_parent = dorsal_metric["parent"]
    dorsal_contour = next(
        (contour for i, contour in enumerate(contours) if metrics[i]["parent"] == dorsal_parent),
        None
    )

    if dorsal_contour is None or len(dorsal_contour) < 3:
        return {"top_area": None, "bottom_area": None, "left_area": None, "right_area": None}

    # Extract centroid from the metrics
    centroid_x = int(dorsal_metric["centroid_x"])
    centroid_y = int(dorsal_metric["centroid_y"])

    # Verify that the centroid belongs to the selected contour
    if cv2.pointPolygonTest(dorsal_contour, (centroid_x, centroid_y), measureDist=False) < 0:
        return {"top_area": None, "bottom_area": None, "left_area": None, "right_area": None}

    # Create a binary mask for the Dorsal contour
    mask = np.zeros_like(inverted_image, dtype=np.uint8)
    cv2.drawContours(mask, [dorsal_contour], -1, 255, thickness=cv2.FILLED)

    # Split the mask into regions
    top_half = mask[:centroid_y, :]
    bottom_half = mask[centroid_y:, :]
    left_half = mask[:, :centroid_x]
    right_half = mask[:, centroid_x:]

    # Calculate areas (number of non-zero pixels) and convert to floats
    top_area = round(float(np.sum(top_half == 255)), 2)
    bottom_area = round(float(np.sum(bottom_half == 255)), 2)
    left_area = round(float(np.sum(left_half == 255)), 2)
    right_area = round(float(np.sum(right_half == 255)), 2)
    # Vertical symmetry calculation
    vertical_symmetry = (
        round(
            1 - abs(top_area - bottom_area) / (top_area + bottom_area), 2
        ) if (top_area + bottom_area) > 0 else None
    )

    # Horizontal symmetry calculation
    horizontal_symmetry = (
        round(
            1 - abs(left_area - right_area) / (left_area + right_area), 2
        ) if (left_area + right_area) > 0 else None
    )

    # Logging for analysis completion
    logging.info("Symmetry analysis complete for Dorsal surface.")

    return {
        "top_area": top_area,
        "bottom_area": bottom_area,
        "left_area": left_area,
        "right_area": right_area,
        "vertical_symmetry": vertical_symmetry,
        "horizontal_symmetry": horizontal_symmetry
    }


def calculate_voronoi_points(metrics, inverted_image, padding_factor=0.02):
    """
    Calculate Voronoi diagram polygons and convex hull from dorsal surface centroids.
    Also computes metrics for the Voronoi cells (number of cells, area of each cell,
    and number of shared edges) as well as the convex hull metrics (width, height, area).

    This function filters the input metrics to include only those with a dorsal surface
    classification, extracts centroids from both the dorsal parent and its child contours,
    creates a shapely MultiPoint, and then computes a Voronoi diagram clipped to the dorsal
    contour. It also calculates the convex hull of the centroids and a padded bounding box
    based on the dorsal contour bounds.

    Args:
        metrics (list): List of contour metrics containing centroids, contours, and surface types.
        inverted_image (numpy.ndarray): Inverted binary thresholded image.
        padding_factor (float): Percentage of padding to add to the dorsal contour bounding box.
                                  Defaults to 0.02 (i.e., 2%).

    Returns:
        dict: A dictionary containing:
            - 'voronoi_diagram': Shapely GeometryCollection of the Voronoi cells.
            - 'voronoi_cells': List of dictionaries for each Voronoi cell with keys:
                  'polygon' (shapely Polygon), 'area' (float), 'shared_edges' (int),
                  'metric_index' (int) - index of the corresponding metric in the input list.
            - 'voronoi_metrics': Dictionary with overall Voronoi metrics (e.g., 'num_cells').
            - 'convex_hull': Shapely Geometry representing the convex hull of the centroids.
            - 'convex_hull_metrics': Dictionary with keys 'width', 'height', and 'area'.
            - 'points': Shapely MultiPoint object of the dorsal centroids.
            - 'bounding_box': Dictionary with padded bounding box values:
                  'x_min', 'x_max', 'y_min', 'y_max'.
            - 'dorsal_contour': Shapely Polygon for the dorsal contour.
    """
    # Find all metrics related to the dorsal surface (parent and scars)
    dorsal_metrics = []
    dorsal_metric_indices = []

    # First, find the parent dorsal surface
    dorsal_parent = None
    for i, m in enumerate(metrics):
        if m.get("surface_type") == "Dorsal" and m["parent"] == m["scar"]:
            dorsal_parent = m["parent"]
            dorsal_metrics.append(m)
            dorsal_metric_indices.append(i)
            break

    if dorsal_parent is None:
        logging.warning("No Dorsal surface parent found.")
        return None

    # Then, find all scars on the dorsal surface
    for i, m in enumerate(metrics):
        # Skip the parent (already added)
        if m["parent"] == dorsal_parent and m["parent"] != m["scar"]:
            dorsal_metrics.append(m)
            dorsal_metric_indices.append(i)

    if not dorsal_metrics:
        logging.warning("No Dorsal surface metrics available for Voronoi diagram.")
        return None

    # Collect centroids and process the dorsal contour
    centroids = []
    centroid_to_metric_idx = {}  # Maps centroid coordinates to original metric index
    dorsal_contour = None

    for i, dorsal_metric in enumerate(dorsal_metrics):
        # Add centroid and track which metric it came from
        cx = dorsal_metric["centroid_x"]
        cy = dorsal_metric["centroid_y"]
        centroids.append(Point(cx, cy))

        # Store original index in the metrics list
        original_idx = dorsal_metric_indices[i]
        centroid_to_metric_idx[(cx, cy)] = original_idx

        # Extract the dorsal surface contour (only once)
        if dorsal_contour is None and "contour" in dorsal_metric and dorsal_metric["parent"] == dorsal_metric["scar"]:
            raw_contour = dorsal_metric["contour"]
            if isinstance(raw_contour, list) and isinstance(raw_contour[0], list):
                flat_contour = [(point[0][0], point[0][1]) for point in raw_contour]
                dorsal_contour = Polygon(flat_contour)
            else:
                raise ValueError("Contour format is not as expected. Please check the metrics data structure.")

    if dorsal_contour is None:
        logging.warning("No dorsal contour data available for the Dorsal surface.")
        return None

    # Create a Shapely MultiPoint from the centroids
    points = MultiPoint(centroids)

    # Generate Voronoi polygons clipped to the dorsal contour
    vor = voronoi_diagram(points, envelope=dorsal_contour)

    # Calculate the convex hull of the centroids
    convex_hull = points.convex_hull

    # Calculate padded bounding box of the dorsal contour
    x_min, y_min, x_max, y_max = dorsal_contour.bounds
    x_padding = (x_max - x_min) * padding_factor
    y_padding = (y_max - y_min) * padding_factor
    bounding_box = {
        'x_min': x_min - x_padding,
        'x_max': x_max + x_padding,
        'y_min': y_min - y_padding,
        'y_max': y_max + y_padding
    }

    # Process each Voronoi region: clip to the dorsal contour and collect valid polygons
    voronoi_cells = []

    # Convert points to list for indexed access
    point_list = list(points.geoms)

    for i, region in enumerate(vor.geoms):
        if i >= len(point_list):  # Safety check
            continue

        point = point_list[i]
        point_key = (point.x, point.y)

        clipped_region = region.intersection(dorsal_contour)

        if not clipped_region.is_empty:
            if clipped_region.geom_type == 'Polygon':
                cell_polygon = clipped_region
            elif clipped_region.geom_type == 'MultiPolygon':
                cell_polygon = max(clipped_region.geoms, key=lambda p: p.area)
            else:
                continue  # Skip geometries that are not polygons

            # Store the polygon, its area, and original metric index
            metric_idx = centroid_to_metric_idx.get(point_key, -1)
            voronoi_cells.append({
                'polygon': cell_polygon,
                'area': cell_polygon.area,
                'metric_index': metric_idx  # Track which metric this cell belongs to
            })

    num_cells = len(voronoi_cells)

    # Calculate shared edges between Voronoi cells
    shared_edges_counts = [0] * num_cells
    tolerance = 1e-6  # small tolerance for geometric comparisons
    for i in range(num_cells):
        for j in range(i + 1, num_cells):
            inter = voronoi_cells[i]['polygon'].exterior.intersection(voronoi_cells[j]['polygon'].exterior)
            if not inter.is_empty:
                if inter.geom_type == 'LineString':
                    if inter.length > tolerance:
                        shared_edges_counts[i] += 1
                        shared_edges_counts[j] += 1
                elif inter.geom_type == 'MultiLineString':
                    total_length = sum(line.length for line in inter.geoms)
                    if total_length > tolerance:
                        shared_edges_counts[i] += 1
                        shared_edges_counts[j] += 1

    # Assemble the metrics for each Voronoi cell
    voronoi_cell_metrics = []
    for idx, cell in enumerate(voronoi_cells):
        cell_dict = {
            'polygon': cell['polygon'],
            'area': cell['area'],
            'shared_edges': shared_edges_counts[idx],
            'metric_index': cell['metric_index']  # Include the metric index
        }
        voronoi_cell_metrics.append(cell_dict)

    # Calculate convex hull metrics based on its bounding box and area
    ch_minx, ch_miny, ch_maxx, ch_maxy = convex_hull.bounds
    ch_width = ch_maxx - ch_minx
    ch_height = ch_maxy - ch_miny
    convex_hull_metrics = {
        'width': ch_width,
        'height': ch_height,
        'area': convex_hull.area
    }

    # # Voronoi cell debugging. Re-work for logging
    # print("=====================================")
    # # Print the Voronoi debug info
    # print("\n=== VORONOI DIAGRAM DEBUG INFO ===")
    # print(f"Total number of cells: {num_cells}")

    # # Print the point coordinates used to generate the Voronoi diagram
    # print("\nCentroids used to generate Voronoi cells:")
    # point_coords = [(p.x, p.y) for p in points.geoms]
    # for i, coord in enumerate(point_coords):
    #     print(f"  Point {i+1}: ({coord[0]:.2f}, {coord[1]:.2f})")

    # # Print the cell areas with corresponding point coordinates
    # print("\nVoronoi cell areas with their corresponding centroids:")
    # for i, cell in enumerate(voronoi_cell_metrics):
    #     if i < len(point_coords):
    #         print(f"  Cell {i+1}: Area = {cell['area']:.2f}, Centroid = ({point_coords[i][0]:.2f}, {point_coords[i][1]:.2f})")
    #     else:
    #         print(f"  Cell {i+1}: Area = {cell['area']:.2f}, Centroid = Unknown")

    # # Also print the mapping information for verification
    # print("\nMapping between cells and metrics:")
    # for i, cell in enumerate(voronoi_cell_metrics):
    #     metric_idx = cell['metric_index']
    #     if metric_idx != -1 and metric_idx < len(metrics):
    #         metric = metrics[metric_idx]
    #         print(f"  Cell {i+1}: Mapped to metric index {metric_idx}, Feature = {metric.get('scar', 'Unknown')}")
    #     else:
    #         print(f"  Cell {i+1}: No valid metric mapping")

    # print("=====================================")

    result = {
        'voronoi_diagram': vor,
        'voronoi_cells': voronoi_cell_metrics,
        'voronoi_metrics': {
            'num_cells': num_cells
        },
        'convex_hull': convex_hull,
        'convex_hull_metrics': convex_hull_metrics,
        'points': points,
        'bounding_box': bounding_box,
        'dorsal_contour': dorsal_contour
    }
    return result


def visualize_voronoi_diagram(voronoi_data, inverted_image, output_path):
    """
    Visualize the Voronoi diagram and convex hull on the dorsal surface. This function replicates
    the visualization features from the original generate_voronoi_diagram() function:
      - Displays the original (inverted back) image as a background.
      - Plots the Voronoi cells (clipped to the dorsal contour) as colored patches using a color-blind-friendly colormap.
      - Overlays the convex hull of the centroids.
      - Highlights and annotates all centroids (both dorsal surface and child contours) with labels.
      - Dynamically adjusts the plot bounds using the padded bounding box.
      - Does not display cell metrics on the image.

    Args:
        voronoi_data (dict): Dictionary produced by calculate_voronoi_points() containing:
            - 'voronoi_cells': List of dicts (each with 'polygon', 'area', 'shared_edges').
            - 'convex_hull': Shapely Geometry for the convex hull.
            - 'points': Shapely MultiPoint of the centroids.
            - 'bounding_box': Dict with keys 'x_min', 'x_max', 'y_min', 'y_max'.
        inverted_image (numpy.ndarray): Inverted binary thresholded image.
        output_path (str): Path to save the generated Voronoi diagram visualization.

    Returns:
        None
    """
    # Invert the image back to its original form (black illustration on white background)
    original_image = cv2.bitwise_not(inverted_image)
    background_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(background_image)

    # Prepare a list to hold matplotlib polygon patches for the Voronoi cells
    patches = []
    for cell_dict in voronoi_data['voronoi_cells']:
        cell_polygon = cell_dict['polygon']
        if cell_polygon.geom_type == 'Polygon':
            patch = MplPolygon(np.array(cell_polygon.exterior.coords), closed=True)
            patches.append(patch)
        elif cell_polygon.geom_type == 'MultiPolygon':
            largest = max(cell_polygon.geoms, key=lambda p: p.area)
            patch = MplPolygon(np.array(largest.exterior.coords), closed=True)
            patches.append(patch)
        else:
            logging.warning("Skipping Voronoi cell with unsupported geometry type: %s", cell_polygon.geom_type)

    # Use a color-blind-friendly colormap (e.g., 'tab10') and normalization
    colormap = get_cmap('tab10')
    norm = Normalize(vmin=0, vmax=len(patches))
    colors = [colormap(norm(i)) for i in range(len(patches))]

    patch_collection = PatchCollection(
        patches,
        alpha=0.6,
        facecolor=colors,
        edgecolor="white",
        linewidths=2
    )
    ax.add_collection(patch_collection)

    # Overlay the convex hull
    convex_hull = voronoi_data['convex_hull']
    if not convex_hull.is_empty:
        if convex_hull.geom_type == 'Polygon':
            hull_coords = np.array(convex_hull.exterior.coords)
            ax.plot(hull_coords[:, 0], hull_coords[:, 1], color="black", linewidth=2, label="Convex Hull")
        elif convex_hull.geom_type == 'LineString':
            hull_coords = np.array(convex_hull.coords)
            ax.plot(hull_coords[:, 0], hull_coords[:, 1], color="black", linewidth=2, label="Convex Hull")
        elif convex_hull.geom_type == 'Point':
            ax.plot(convex_hull.x, convex_hull.y, 'ko', label="Convex Hull")
        else:
            logging.warning("Convex hull has unsupported geometry type: %s", convex_hull.geom_type)

    # Highlight all centroids from the MultiPoint object and annotate them.
    points = voronoi_data['points']
    centroid_xs = [p.x for p in points.geoms]
    centroid_ys = [p.y for p in points.geoms]
    ax.plot(centroid_xs, centroid_ys, 'ro', markersize=5, label='Dorsal Surface Centroids')
    for i, p in enumerate(points.geoms):
        label = "Surface Center" if i == 0 else f"C{i}"
        ax.text(p.x + 10, p.y + 2, label, color="black", fontsize=12)

    # Adjust plot limits using the padded bounding box
    bbox = voronoi_data['bounding_box']
    ax.set_xlim(bbox['x_min'], bbox['x_max'])
    ax.set_ylim(bbox['y_max'], bbox['y_min'])  # Invert y-axis to match image coordinates

    # Set title, labels, and legend
    ax.set_title("Voronoi Diagram with Convex Hull")
    ax.set_xlabel("Horizontal Distance")
    ax.set_ylabel("Vertical Distance")
    ax.legend()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    logging.info("Saved Voronoi diagram visualization to: %s", output_path)


def process_and_save_contours(inverted_image, conversion_factor, output_dir, image_id, image_dpi=None):
    """
    Process contours, calculate metrics, classify surfaces, analyze symmetry, and append results.

    Parameters
    ----------
    inverted_image : numpy.ndarray
        Inverted binary thresholded image.
    conversion_factor : float
        Conversion factor for pixels to real-world units.
    output_dir : str
        Directory to save processed outputs.
    image_id : str
        Unique identifier for the image being processed.
    image_dpi : float, optional
        DPI of the image being processed, used for scaling detection parameters.
    """

    try:
        # Step 1: Extract contours and hierarchy
        contours, hierarchy = extract_contours_with_hierarchy(inverted_image, image_id, output_dir)
        if not contours:
            logging.warning("No valid contours found for image: %s", image_id)
            return

        # Print hierarchy shape for debugging
        logging.info(f"DEBUG: hierarchy shape = {hierarchy.shape}")

        try:
            # Step 2: Flag nested and single child contours
            exclude_nested_flags = hide_nested_child_contours(contours, hierarchy)
        except Exception as e:
            logging.error(f"Error in hide_nested_child_contours: {e}")
            import traceback
            traceback.print_exc()
            # Create empty flags as fallback
            exclude_nested_flags = [False] * len(contours)

        try:
            # Step 3: Sort contours by hierarchy, excluding flagged nested contours
            sorted_contours = sort_contours_by_hierarchy(contours, hierarchy, exclude_nested_flags)
        except Exception as e:
            logging.error(f"Error in sort_contours_by_hierarchy: {e}")
            import traceback
            traceback.print_exc()
            # Create empty sorted_contours as fallback
            sorted_contours = {"parents": [], "children": [], "nested_children": []}
            if contours:
                sorted_contours["parents"] = [contours[0]]
                if len(contours) > 1:
                    sorted_contours["children"] = contours[1:]

        # Handle special case of zero child contours but with nested children
        if len(sorted_contours["children"]) == 0 and len(sorted_contours.get("nested_children", [])) > 0:
            logging.info(f"Special case for {image_id}: Promoting nested children to direct children")
            sorted_contours["children"] = sorted_contours["nested_children"]
            sorted_contours["nested_children"] = []
            logging.info("Promoted %d nested children to direct children.",
                        len(sorted_contours["children"]))

        try:
            # Step 4: Calculate metrics for all contours
            metrics = calculate_contour_metrics(
                sorted_contours,
                hierarchy,
                contours,
                inverted_image,  # Pass the actual image for visualization
                image_dpi # Pass the DPI for scaled detection
            )
        except Exception as e:
            logging.error(f"Error in calculate_contour_metrics: {e}")
            # Print exact line where error occurred
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logging.error(f"Error at {fname}:{exc_tb.tb_lineno}")

            import traceback
            traceback.print_exc()

            # Minimal metrics as fallback
            metrics = []
            for i, cnt in enumerate(sorted_contours.get("parents", [])):
                # Create minimal parent metrics
                area = cv2.contourArea(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                metrics.append({
                    "parent": f"parent {i+1}",
                    "scar": f"parent {i+1}",
                    "surface_type": "Dorsal" if i == 0 else "Unknown",
                    "centroid_x": x + w/2,
                    "centroid_y": y + h/2,
                    "width": w,
                    "height": h,
                    "area": area,
                    "has_arrow": False
                })

            # If no metrics created, create at least one
            if not metrics and contours:
                cnt = contours[0]
                area = cv2.contourArea(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                metrics.append({
                    "parent": "parent 1",
                    "scar": "parent 1",
                    "surface_type": "Dorsal",
                    "centroid_x": x + w/2,
                    "centroid_y": y + h/2,
                    "width": w,
                    "height": h,
                    "area": area,
                    "has_arrow": False
                })

        try:
            # Step 5: Classify parent contours into surfaces
            metrics = classify_parent_contours(metrics)
        except Exception as e:
            logging.error(f"Error in classify_parent_contours: {e}")
            # Don't halt execution if classification fails

        # Step 6: Add image_id to each metric entry
        for metric in metrics:
            metric["image_id"] = image_id

        try:
            # Step 7: Perform symmetry analysis for the dorsal surface
            symmetry_scores = analyze_dorsal_symmetry(metrics, sorted_contours.get("parents", []), inverted_image)

            # Step 8: Add symmetry scores to the dorsal metrics only
            for metric in metrics:
                if metric.get("surface_type") == "Dorsal":
                    metric.update(symmetry_scores)
        except Exception as e:
            logging.error(f"Error in symmetry analysis: {e}")
            # Continue if symmetry analysis fails

        try:
            # Step 9: Calculate Voronoi diagram and convex hull metrics
            voronoi_data = calculate_voronoi_points(metrics, inverted_image, padding_factor=0.02)

            # Initialize voronoi_cell_area field for all metrics with "NA"
            for metric in metrics:
                metric['voronoi_cell_area'] = "NA"

            if voronoi_data is not None:
                vor_num_cells = voronoi_data['voronoi_metrics']['num_cells']
                ch_width = round(voronoi_data['convex_hull_metrics']['width'], 2)
                ch_height = round(voronoi_data['convex_hull_metrics']['height'], 2)
                ch_area = round(voronoi_data['convex_hull_metrics']['area'], 2)

                # Use the metric_index from each cell to directly update the corresponding metric
                for cell in voronoi_data['voronoi_cells']:
                    metric_idx = cell.get('metric_index', -1)
                    if metric_idx != -1 and metric_idx < len(metrics):
                        # Round the area to 2 decimal places
                        cell_area = round(cell['area'], 2)
                        metrics[metric_idx]['voronoi_cell_area'] = cell_area

                # Append Voronoi and convex hull metrics to all dorsal metrics
                for metric in metrics:
                    if metric.get("surface_type") == "Dorsal":
                        metric['voronoi_num_cells'] = vor_num_cells
                        metric['convex_hull_width'] = ch_width
                        metric['convex_hull_height'] = ch_height
                        metric['convex_hull_area'] = ch_area
        except Exception as e:
            logging.error(f"Error in Voronoi processing: {e}")
            # Continue if Voronoi processing fails


        # New Step: Run independent arrow detection if needed
        try:
            # Check if any arrows were detected through the traditional method
            arrows_detected = any(metric.get('has_arrow', False) for metric in metrics)

            # If no arrows were detected but the image might have them, try independent detection
            if not arrows_detected:
                logging.info(f"No arrows detected through traditional hierarchy. Trying independent detection.")
                metrics = detect_arrows_independently(contours, metrics, inverted_image, image_dpi)
        except Exception as e:
            logging.error(f"Error in independent arrow detection: {e}")
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logging.error(f"Error at {fname}:{exc_tb.tb_lineno}")
            import traceback
            traceback.print_exc()

        # Step 10: Save metrics to CSV
        try:
            combined_csv_path = os.path.join(output_dir, "processed_metrics.csv")
            save_measurements_to_csv(metrics, combined_csv_path, append=True)
        except Exception as e:
            logging.error(f"Error saving CSV: {e}")

        # Step 11: Visualize contours with hierarchy
        try:
            # Collect all contours to visualize
            all_visualization_contours = (
                sorted_contours.get("parents", []) +
                sorted_contours.get("children", []) +
                sorted_contours.get("nested_children", [])
            )

            visualization_path = os.path.join(output_dir, f"{image_id}_labeled.png")
            visualize_contours_with_hierarchy(
                all_visualization_contours,
                hierarchy,
                metrics,
                inverted_image,
                visualization_path
            )
        except Exception as e:
            logging.error(f"Error in visualization: {e}")

        # Step 12: Generate and visualize Voronoi diagram
        try:
            if voronoi_data is not None:
                voronoi_output_path = os.path.join(output_dir, f"{image_id}_voronoi.png")
                visualize_voronoi_diagram(voronoi_data, inverted_image, voronoi_output_path)
        except Exception as e:
            logging.error(f"Error in Voronoi visualization: {e}")

        logging.info("Analysis complete for image: %s", image_id)

    except Exception as e:
        logging.error(f"Error analyzing image {image_id}: {e}")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logging.error(f"Error at {fname}:{exc_tb.tb_lineno}")
        traceback.print_exc()


def convert_metrics_to_real_world(metrics, conversion_factor):
    """
    Convert metrics from pixel values to real-world units.

    Args:
        metrics (list): List of dictionaries containing raw metrics in pixel units.
        conversion_factor (float): Conversion factor for pixels to real-world units.

    Returns:
        list: Converted metrics in real-world units.
    """
    converted_metrics = []

    for metric in metrics:
        converted_metrics.append({
            "parent": metric["parent"],
            "scar": metric["scar"],
            "centroid_x": round(metric["centroid_x"] * conversion_factor, 2),
            "centroid_y": round(metric["centroid_y"] * conversion_factor, 2),
            "width": round(metric["width"] * conversion_factor, 2),
            "height": round(metric["height"] * conversion_factor, 2),
            "area": round(metric["area"] * (conversion_factor ** 2), 2),  # Area scales quadratically
        })
    return converted_metrics
