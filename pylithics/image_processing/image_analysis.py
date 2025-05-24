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


        # Run secondary independent arrow detection for scars without arrows
        try:
            # Check which scars don't have arrows yet
            scars_without_arrows = [m for m in metrics if m["parent"] != m["scar"] and not m.get('has_arrow', False)]

            if scars_without_arrows:
                logging.info(f"Found {len(scars_without_arrows)} scars without arrows. Running independent detection.")
                scar_labels = [m["scar"] for m in scars_without_arrows]
                logging.info(f"Scars without arrows: {scar_labels}")
                metrics = detect_arrows_independently(contours, metrics, inverted_image, image_dpi)
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
