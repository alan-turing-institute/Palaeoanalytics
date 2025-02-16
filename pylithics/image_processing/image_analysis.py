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


def extract_contours_with_hierarchy(inverted_image, image_id, output_dir):
    """
    Extract contours and hierarchy using cv2.RETR_TREE, excluding the image border.

    Args:
        inverted_image (numpy.ndarray): Inverted binary thresholded image.
        image_id (str): Unique identifier for the image.
        output_dir (str): Directory to save processed outputs.

    Returns:
        tuple: A tuple containing:
            - valid_contours (list): List of detected contours excluding borders.
            - valid_hierarchy (numpy.ndarray): Hierarchy array corresponding to valid contours.
    """
    contours, hierarchy = cv2.findContours(inverted_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0] if hierarchy is not None else None

    if contours is None or len(contours) == 0:
        logging.warning("No contours found in the image.")
        return [], None

    # Exclude the border by filtering out contours touching the edges of the image
    height, width = inverted_image.shape
    valid_contours = []
    valid_hierarchy = []

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if not (x == 0 or y == 0 or x + w == width or y + h == height):
            valid_contours.append(contour)
            valid_hierarchy.append(hierarchy[i])

    parent_count = sum(1 for h in valid_hierarchy if h[3] == -1)  # No parent
    child_count = len(valid_hierarchy) - parent_count

    logging.info("Extracted %d valid contours: %d parent(s) and %d child(ren).", len(valid_contours), parent_count, child_count)
    return valid_contours, np.array(valid_hierarchy) if valid_hierarchy else None


def sort_contours_by_hierarchy(contours, hierarchy, exclude_nested_flags=None):
    """
    Sort contours into parents, children, and nested children based on hierarchy.

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
    # Initialize categories
    parents = []
    children = []
    nested_children = []

    # Apply exclusion flags if provided
    if exclude_nested_flags is None:
        exclude_nested_flags = [False] * len(contours)

    # Traverse the hierarchy to sort contours
    for i, h in enumerate(hierarchy):
        if exclude_nested_flags[i]:
            continue  # Skip excluded contours

        parent_idx = h[3]  # Parent index
        if parent_idx == -1:  # Parent contour (no parent)
            parents.append(contours[i])
        else:
            grandparent_idx = hierarchy[parent_idx][3]  # Check if the parent has a parent
            if grandparent_idx == -1:  # Child contour (parent is a top-level parent)
                children.append(contours[i])
            else:  # Nested child contour (parent is a child contour)
                nested_children.append(contours[i])

    logging.info(
        "Sorted contours: %d parent(s), %d child(ren), %d nested child(ren).",
        len(parents), len(children), len(nested_children)
    )

    return {"parents": parents, "children": children, "nested_children": nested_children}


def calculate_contour_metrics(sorted_contours, hierarchy, original_contours):
    """
    Calculate metrics for contours in a sorted order (parents first, children second),
    ensuring children are correctly grouped with their respective parents.
    """
    def calculate_max_length_and_width(contour):
        # Compute the maximum distance (max_length) and its perpendicular max_width
        max_length = 0
        max_width = 0
        point1, point2 = None, None
        for i in range(len(contour)):
            for j in range(i + 1, len(contour)):
                p1 = contour[i][0]
                p2 = contour[j][0]
                distance = np.linalg.norm(p1 - p2)
                if distance > max_length:
                    max_length = distance
                    point1, point2 = p1, p2
        if point1 is not None and point2 is not None:
            direction_vector = point2 - point1
            norm_vector = np.array([-direction_vector[1], direction_vector[0]])
            norm_vector = norm_vector / np.linalg.norm(norm_vector)
            projections = [abs(np.dot(point[0] - point1, norm_vector)) for point in contour]
            max_width = max(projections)
        return round(max_length, 2), round(max_width, 2)

    metrics = []
    parent_count = 0
    child_count = 0
    parent_index_to_label = {}

    # Process parent contours first
    for parent_contour in sorted_contours["parents"]:
        parent_index = next(i for i, c in enumerate(original_contours) if np.array_equal(c, parent_contour))
        parent_count += 1
        parent_label = f"parent {parent_count}"
        parent_index_to_label[parent_index] = parent_label

        # Calculate metrics for the parent contour
        area = round(cv2.contourArea(parent_contour), 2)
        # --- New: Compute the perimeter (arc length) of the contour ---
        perimeter = round(cv2.arcLength(parent_contour, True), 2)

        moments = cv2.moments(parent_contour)
        centroid_x, centroid_y = (0.0, 0.0)
        if moments["m00"] != 0:
            centroid_x = round(moments["m10"] / moments["m00"], 2)
            centroid_y = round(moments["m01"] / moments["m00"], 2)
        x, y, w, h = cv2.boundingRect(parent_contour)
        max_length, max_width = calculate_max_length_and_width(parent_contour)

        metrics.append({
            "parent": parent_label,
            "scar": parent_label,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "width": w,
            "height": h,
            "area": area,
            "aspect_ratio": round(h / w, 2),
            "bounding_box_x": x,
            "bounding_box_y": y,
            "bounding_box_width": w,
            "bounding_box_height": h,
            "max_length": max_length,
            "max_width": max_width,
            "contour": parent_contour.tolist(),
            "perimeter": perimeter
        })

    # Process child contours
    for child_contour in sorted_contours["children"]:
        child_index = next(i for i, c in enumerate(original_contours) if np.array_equal(c, child_contour))
        parent_index = hierarchy[child_index][3]
        parent_label = parent_index_to_label.get(parent_index, "Unknown")
        child_count += 1
        child_label = f"scar {child_count}"

        area = round(cv2.contourArea(child_contour), 2)
        moments = cv2.moments(child_contour)
        centroid_x, centroid_y = (0.0, 0.0)
        if moments["m00"] != 0:
            centroid_x = round(moments["m10"] / moments["m00"], 2)
            centroid_y = round(moments["m01"] / moments["m00"], 2)
        x, y, w, h = cv2.boundingRect(child_contour)
        max_length, max_width = calculate_max_length_and_width(child_contour)

        metrics.append({
            "parent": parent_label,
            "scar": child_label,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "width": round(w, 2),
            "height": round(h, 2),
            "max_length": round(max_length, 2),
            "max_width": round(max_width, 2),
            "area": area,
            "aspect_ratio": round(h / w, 2),
            "bounding_box_x": x,
            "bounding_box_y": y,
            "bounding_box_width": w,
            "bounding_box_height": h
            # No perimeter for child contours (unless you choose to compute it)
        })

    logging.info(
        "Calculated metrics for %d contours: %d parent(s) and %d child(ren).",
        len(metrics), parent_count, child_count
    )
    return metrics


def hide_nested_child_contours(contours, hierarchy):
    """
    Flag nested contours and single child contours for exclusion.

    Args:
        contours (list): List of detected contours.
        hierarchy (numpy.ndarray): Hierarchy array corresponding to contours.

    Returns:
        list: A list of booleans where True indicates a contour is nested or a single child and should be excluded.
    """
    nested_child_contours = [False] * len(contours)  # Initialize all contours as not nested or single child
    parent_child_count = {}  # Track number of children for each parent

    for i, h in enumerate(hierarchy):
        parent_idx = h[3]  # Parent index
        if parent_idx != -1:  # If the contour has a parent
            parent_child_count[parent_idx] = parent_child_count.get(parent_idx, 0) + 1

    for i, h in enumerate(hierarchy):
        parent_idx = h[3]  # Parent index
        if parent_idx != -1:
            grandparent_idx = hierarchy[parent_idx][3]  # Parent's parent index
            if grandparent_idx != -1:  # If the parent itself has a parent
                # This is a second-level nested contour, mark it as nested
                nested_child_contours[i] = True
            elif parent_child_count[parent_idx] == 1:
                # Mark single child contours for exclusion
                nested_child_contours[i] = True

    logging.info("Flagged %d nested or single child contours for exclusion.", sum(nested_child_contours))
    return nested_child_contours


def classify_parent_contours(metrics, tolerance=0.1):
    """
    Classify parent contours into surfaces: Dorsal, Ventral, Platform, Lateral.

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

    # Identify Dorsal Surface
    try:
        dorsal = max(parents, key=lambda p: p["area"])
        dorsal["surface_type"] = "Dorsal"
        surfaces_identified.append("Dorsal")
    except ValueError:
        logging.error("Unable to identify the dorsal surface due to missing or invalid parent metrics.")
        return metrics

    # Identify Ventral Surface
    for parent in parents:
        if parent["surface_type"] is None:
            if (
                abs(parent["height"] - dorsal["height"]) <= tolerance * dorsal["height"]
                and abs(parent["width"] - dorsal["width"]) <= tolerance * dorsal["width"]
                and abs(parent["area"] - dorsal["area"]) <= tolerance * dorsal["area"]
            ):
                parent["surface_type"] = "Ventral"
                surfaces_identified.append("Ventral")
                break

    # Identify Platform Surface
    platform_candidates = [
        p for p in parents if p["surface_type"] is None and p["height"] < dorsal["height"] and p["width"] < dorsal["width"]
    ]
    if platform_candidates:
        platform = min(platform_candidates, key=lambda p: p["area"])
        platform["surface_type"] = "Platform"
        surfaces_identified.append("Platform")

    # Identify Lateral Surface
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

    # Assign default surface type if still None
    for parent in parents:
        if parent["surface_type"] is None:
            parent["surface_type"] = "Unclassified"

    logging.info("Classified parent contours into surfaces: %s.", ", ".join(surfaces_identified))
    return metrics


def visualize_contours_with_hierarchy(contours, hierarchy, metrics, inverted_image, output_path):
    """
    Visualize contours with hierarchy, label them, and include a red dot at the centroid on the original image background.

    Args:
        contours (list): List of detected contours.
        hierarchy (numpy.ndarray): Contour hierarchy array.
        metrics (list): List of contour metrics containing labels and other information.
        inverted_image (numpy.ndarray): Inverted binary thresholded image.
        output_path (str): Path to save the labeled image.

    Returns:
        None
    """
    # Invert the image back to its original form (black illustration on white background)
    original_image = cv2.bitwise_not(inverted_image)

    # Convert the image to a BGR image for visualization
    labeled_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

    # Track label bounding boxes to avoid overlap
    label_positions = []

    # Iterate over contours and draw them with labels and centroids
    for i, contour in enumerate(contours):
        # Get metrics for the current contour
        contour_metric = metrics[i]
        parent_label = contour_metric["parent"]
        scar_label = contour_metric["scar"]

        # Determine the color and text label for parent and child contours
        if parent_label == scar_label:  # Parent contour
            color = (153, 60, 94)  # Color safe purple
            text_label = f"{contour_metric['surface_type']}"  # e.g., "Dorsal"

        else:  # Child contour
            color = (99, 184, 253)  # Color safe orange
            text_label = scar_label  # e.g., "scar_1"

        # Draw the contour on the image
        cv2.drawContours(labeled_image, [contour], -1, color, 2)

        # Use moments to find the centroid for the current contour
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            centroid_x = int(moments["m10"] / moments["m00"])
            centroid_y = int(moments["m01"] / moments["m00"])
        else:
            # Fallback if centroid can't be calculated
            x, y, w, h = cv2.boundingRect(contour)
            centroid_x = x + w // 2
            centroid_y = y + h // 2

        # Draw a red dot at the centroid
        cv2.circle(labeled_image, (centroid_x, centroid_y), 5, (1, 97, 230), -1) # Color safe red

        # Adjust label position to prevent overlap with the centroid
        label_x = centroid_x + 10  # Offset horizontally
        label_y = centroid_y - 10  # Offset vertically

        # Ensure the label does not overlap with any existing label
        label_width, label_height = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        adjusted = False
        for (existing_x, existing_y, existing_width, existing_height) in label_positions:
            if (
                label_x < existing_x + existing_width and
                label_x + label_width > existing_x and
                label_y < existing_y + existing_height and
                label_y + label_height > existing_y
            ):
                label_y += existing_height + 10  # Shift the label down to avoid overlap
                adjusted = True

        # Store the label position
        label_positions.append((label_x, label_y, label_width, label_height))

        # Draw the label on the image at the adjusted position
        cv2.putText(
            labeled_image,
            text_label,  # Text to display (e.g., "Dorsal", "scar 1")
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,  # Font scale
            (186, 186, 186),  # Text color (Color safe gray)
            2,  # Thickness
            cv2.LINE_AA
        )

    # Save the labeled image to the output path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, labeled_image)
    logging.info("Saved visualized contours with labels to: %s", output_path)


def save_measurements_to_csv(metrics, output_path, append=False):
    """
    Save contour metrics to a CSV file with updated column structure.
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
            "perimeter": metric.get("perimeter", "NA"),  # <-- New perimeter field
            "voronoi_num_cells": metric.get("voronoi_num_cells", "NA"),
            "convex_hull_width": metric.get("convex_hull_width", "NA"),
            "convex_hull_height": metric.get("convex_hull_height", "NA"),
            "convex_hull_area": metric.get("convex_hull_area", "NA"),
            "top_area": metric.get("top_area", "NA"),
            "bottom_area": metric.get("bottom_area", "NA"),
            "left_area": metric.get("left_area", "NA"),
            "right_area": metric.get("right_area", "NA"),
            "vertical_symmetry": metric.get("vertical_symmetry", "NA"),
            "horizontal_symmetry": metric.get("horizontal_symmetry", "NA"),
        }
        updated_data.append(data_entry)

    base_columns = [
        "image_id", "surface_type", "surface_feature", "centroid_x", "centroid_y",
        "width", "height", "max_width", "max_length", "total_area", "aspect_ratio",
        "perimeter",  # <-- New column
        "voronoi_num_cells", "convex_hull_width", "convex_hull_height", "convex_hull_area"
    ]
    symmetry_columns = [
        "top_area", "bottom_area", "left_area", "right_area",
        "vertical_symmetry", "horizontal_symmetry"
    ]
    all_columns = base_columns + symmetry_columns

    df = pd.DataFrame(updated_data, columns=all_columns)
    df.fillna("NA", inplace=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if append and os.path.exists(output_path):
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
                  'polygon' (shapely Polygon), 'area' (float), 'shared_edges' (int)
            - 'voronoi_metrics': Dictionary with overall Voronoi metrics (e.g., 'num_cells').
            - 'convex_hull': Shapely Geometry representing the convex hull of the centroids.
            - 'convex_hull_metrics': Dictionary with keys 'width', 'height', and 'area'.
            - 'points': Shapely MultiPoint object of the dorsal centroids.
            - 'bounding_box': Dictionary with padded bounding box values:
                  'x_min', 'x_max', 'y_min', 'y_max'.
            - 'dorsal_contour': Shapely Polygon for the dorsal contour.
    """
    # Filter metrics for Dorsal surface
    dorsal_metrics = [m for m in metrics if m.get("surface_type") == "Dorsal"]

    if not dorsal_metrics:
        logging.warning("No Dorsal surface metrics available for Voronoi diagram.")
        return None

    # Collect centroids and process the dorsal contour
    centroids = []
    dorsal_contour = None

    for dorsal_metric in dorsal_metrics:
        # Add centroid of the parent dorsal contour
        centroids.append(Point(dorsal_metric["centroid_x"], dorsal_metric["centroid_y"]))

        # Extract the dorsal surface contour (only once)
        if dorsal_contour is None and "contour" in dorsal_metric:
            raw_contour = dorsal_metric["contour"]
            if isinstance(raw_contour, list) and isinstance(raw_contour[0], list):
                flat_contour = [(point[0][0], point[0][1]) for point in raw_contour]
                dorsal_contour = Polygon(flat_contour)
            else:
                raise ValueError("Contour format is not as expected. Please check the metrics data structure.")

        # Add centroids of child contours (scars) linked to this parent contour (surface)
        child_metrics = [m for m in metrics if m["parent"] == dorsal_metric["parent"] and m["parent"] != m["scar"]]
        centroids.extend(Point(child["centroid_x"], child["centroid_y"]) for child in child_metrics)

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
    for region in vor.geoms:
        clipped_region = region.intersection(dorsal_contour)
        if not clipped_region.is_empty:
            if clipped_region.geom_type == 'Polygon':
                cell_polygon = clipped_region
            elif clipped_region.geom_type == 'MultiPolygon':
                cell_polygon = max(clipped_region.geoms, key=lambda p: p.area)
            else:
                continue  # Skip geometries that are not polygons
            voronoi_cells.append(cell_polygon)

    num_cells = len(voronoi_cells)

    # Calculate shared edges between Voronoi cells
    shared_edges_counts = [0] * num_cells
    tolerance = 1e-6  # small tolerance for geometric comparisons
    for i in range(num_cells):
        for j in range(i + 1, num_cells):
            inter = voronoi_cells[i].exterior.intersection(voronoi_cells[j].exterior)
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
            'polygon': cell,
            'area': cell.area,
            'shared_edges': shared_edges_counts[idx]
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


def process_and_save_contours(inverted_image, conversion_factor, output_dir, image_id):
    """
    Process contours, calculate metrics, classify surfaces, analyze symmetry, and append results to a single CSV file.

    Args:
        inverted_image (numpy.ndarray): Inverted binary thresholded image.
        conversion_factor (float): Conversion factor for pixels to real-world units.
        output_dir (str): Directory to save processed outputs.
        image_id (str): Unique identifier for the image being processed.
    """
    # Step 1: Extract contours and hierarchy
    contours, hierarchy = extract_contours_with_hierarchy(inverted_image, image_id, output_dir)
    if not contours:
        logging.warning("No valid contours found for image: %s", image_id)
        return

    # Step 2: Flag nested and single child contours
    exclude_nested_flags = hide_nested_child_contours(contours, hierarchy)

    # Step 3: Sort contours by hierarchy, excluding flagged nested contours
    sorted_contours = sort_contours_by_hierarchy(contours, hierarchy, exclude_nested_flags)

    # Step 4: Calculate metrics for parents and children only
    metrics = calculate_contour_metrics(sorted_contours, hierarchy, contours)

    # Step 5: Classify parent contours into surfaces
    metrics = classify_parent_contours(metrics)

    # Step 6: Add image_id to each metric entry
    for metric in metrics:
        metric["image_id"] = image_id

    # Step 7: Perform symmetry analysis for the dorsal surface
    symmetry_scores = analyze_dorsal_symmetry(metrics, sorted_contours["parents"], inverted_image)

    # Step 8: Add symmetry scores to the dorsal metrics only (as symmetry applies to the dorsal surface)
    for metric in metrics:
        if metric.get("surface_type") == "Dorsal":
            metric.update(symmetry_scores)

    # *** NEW STEP 9: Calculate Voronoi diagram and convex hull metrics ***
    # Note: These are computed using the dorsal surface metrics, so they represent image-level values.
    voronoi_data = calculate_voronoi_points(metrics, inverted_image, padding_factor=0.02)
    if voronoi_data is not None:
        vor_num_cells = voronoi_data['voronoi_metrics']['num_cells']
        ch_width = round(voronoi_data['convex_hull_metrics']['width'], 2)
        ch_height = round(voronoi_data['convex_hull_metrics']['height'], 2)
        ch_area = round(voronoi_data['convex_hull_metrics']['area'], 2)
    else:
        vor_num_cells = "NA"
        ch_width = "NA"
        ch_height = "NA"
        ch_area = "NA"

# Append Voronoi and convex hull metrics only for the Dorsal surface;
# for non-dorsal entries, mark as "NA".
    for metric in metrics:
        if metric.get("surface_type") == "Dorsal":
            metric['voronoi_num_cells'] = vor_num_cells
            metric['convex_hull_width'] = ch_width
            metric['convex_hull_height'] = ch_height
            metric['convex_hull_area'] = ch_area
        else:
            metric['voronoi_num_cells'] = "NA"
            metric['convex_hull_width'] = "NA"
            metric['convex_hull_height'] = "NA"
            metric['convex_hull_area'] = "NA"

    # Step 10: Save metrics to CSV (now with additional voronoi/hull fields)
    combined_csv_path = os.path.join(output_dir, "processed_metrics.csv")
    save_measurements_to_csv(metrics, combined_csv_path, append=True)

    # Step 11: Visualize contours with hierarchy
    visualization_path = os.path.join(output_dir, f"{image_id}_labeled.png")
    visualize_contours_with_hierarchy(
        sorted_contours["parents"] + sorted_contours["children"],
        hierarchy,
        metrics,
        inverted_image,
        visualization_path
    )

    # Step 12: Generate and visualize Voronoi diagram for the Dorsal surface and its children
    if voronoi_data is not None:
        voronoi_output_path = os.path.join(output_dir, f"{image_id}_voronoi.png")
        visualize_voronoi_diagram(voronoi_data, inverted_image, voronoi_output_path)


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
