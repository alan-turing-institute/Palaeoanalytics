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

# new imports
import logging
from .modules.surface_classification import classify_parent_contours
from .modules.symmetry_analysis import analyze_dorsal_symmetry
from .modules.contour_metrics import calculate_contour_metrics, convert_metrics_to_real_world
from .modules.visualization import visualize_contours_with_hierarchy, save_measurements_to_csv
from .modules.voronoi_analysis import calculate_voronoi_points, visualize_voronoi_diagram
from .modules.arrow_integration import integrate_arrows

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
                inverted_image  # Pass the actual image for visualization

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


        # Run secondary independent arrow detection for scars without arrows
        try:
            # Check which scars don't have arrows yet
            scars_without_arrows = [m for m in metrics if m["parent"] != m["scar"] and not m.get('has_arrow', False)]

            if scars_without_arrows:
                logging.info(f"Found {len(scars_without_arrows)} scars without arrows. Running independent detection.")
                scar_labels = [m["scar"] for m in scars_without_arrows]
                logging.info(f"Scars without arrows: {scar_labels}")
                metrics = integrate_arrows(sorted_contours, hierarchy, contours, metrics, inverted_image, image_dpi)
            else:
                logging.info("All scars already have arrows or no scars found. Skipping independent detection.")
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
