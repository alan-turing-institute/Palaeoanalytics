"""
PyLithics: Image Analysis Module

This module provides functions for extracting contours and hierarchy
relationships from preprocessed images, calculating metrics, visualizing results,
and saving data to a CSV file.

Usage:
    - extract_contours_with_hierarchy(): Extract contours and their hierarchy.
    - analyze_image_contours_with_hierarchy(): Perform contour analysis and exclude border contours.
    - calculate_contour_metrics(): Compute metrics for parent and child contours.
    - extract_contours_with_hierarchy(): Visualize contours and save labeled images.
    - save_measurements_to_csv(): Save contour metrics to a CSV file.
"""

import cv2
import numpy as np
import logging
import os
import pandas as pd


def extract_contours_with_hierarchy(inverted_image, image_id, conversion_factor, output_dir):
    """
    Extract contours and hierarchy using cv2.RETR_TREE, excluding the image border.

    Args:
        inverted_image (numpy.ndarray): Inverted binary thresholded image.
        image_id (str): Unique identifier for the image.
        conversion_factor (float): Conversion factor for pixels to real-world units.
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
        # If the contour does not touch the border, add it to the valid contours
        if not (x == 0 or y == 0 or x + w == width or y + h == height):
            valid_contours.append(contour)
            valid_hierarchy.append(hierarchy[i])

    logging.info("Extracted %d valid contours (excluding borders).", len(valid_contours))
    return valid_contours, np.array(valid_hierarchy) if valid_hierarchy else None

def calculate_contour_metrics(contours, hierarchy, conversion_factor, nested_child_contours=None):
    """
    Calculate metrics for each contour, excluding nested and single child contours.

    Args:
        contours (list): List of valid contours.
        hierarchy (numpy.ndarray): Hierarchy array corresponding to valid contours.
        conversion_factor (float): Conversion factor for pixels to real-world units.
        nested_child_contours (list): Optional list of booleans indicating nested or single child contours to exclude.

    Returns:
        list: A list of dictionaries containing rounded contour metrics.
    """
    metrics = []
    parent_count = 0
    child_count_map = {}

    if nested_child_contours is None:
        nested_child_contours = [False] * len(contours)  # Default: No exclusions

    for i, (contour, h) in enumerate(zip(contours, hierarchy)):
        if nested_child_contours[i]:
            continue  # Skip nested or single child contours

        # Calculate contour area (converted to real-world units)
        area = round(cv2.contourArea(contour) * (conversion_factor ** 2), 2)

        # Calculate centroid using image moments
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            centroid_x = round((moments["m10"] / moments["m00"]) * conversion_factor, 2)
            centroid_y = round((moments["m01"] / moments["m00"]) * conversion_factor, 2)
        else:
            centroid_x, centroid_y = 0.0, 0.0

        # Calculate bounding box dimensions (converted to real-world units)
        x, y, w, h = cv2.boundingRect(contour)
        width = round(w * conversion_factor, 2)
        height = round(h * conversion_factor, 2)

        if hierarchy[i][3] == -1:  # Parent contour
            parent_count += 1
            label = f"parent {parent_count}"
            metrics.append({
                "parent": label,
                "scar": label,
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                "width": width,
                "height": height,
                "area": area,
            })
            child_count_map[label] = 0
        else:  # Child contour
            parent_label = f"parent {parent_count}"
            child_count_map[parent_label] += 1
            child_label = f"scar {child_count_map[parent_label]}"
            metrics.append({
                "parent": parent_label,
                "scar": child_label,
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                "width": width,
                "height": height,
                "area": area,
            })

    logging.info("Calculated metrics for %d contours with rounded values.", len(metrics))
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
    # Extract parent contours
    parents = [m for m in metrics if m["parent"] == m["scar"]]

    # Initialize classification
    for parent in parents:
        parent["surface_type"] = None

    # Identify Dorsal Surface (A)
    dorsal = max(parents, key=lambda p: p["area"])
    dorsal["surface_type"] = "Dorsal"

    # Identify Ventral Surface (B)
    for parent in parents:
        if parent["surface_type"] is None:  # Skip already classified surfaces
            # Check dimensional similarity to Dorsal
            if (
                abs(parent["height"] - dorsal["height"]) <= tolerance * dorsal["height"]
                and abs(parent["width"] - dorsal["width"]) <= tolerance * dorsal["width"]
                and abs(parent["area"] - dorsal["area"]) <= tolerance * dorsal["area"]
            ):
                parent["surface_type"] = "Ventral"
                break

    # Identify Platform Surface (C)
    platform_candidates = [
        p for p in parents if p["surface_type"] is None and p["height"] < dorsal["height"] and p["width"] < dorsal["width"]
    ]
    if platform_candidates:
        platform = min(platform_candidates, key=lambda p: p["area"])
        platform["surface_type"] = "Platform"

    # Identify Lateral Surface (D)
    for parent in parents:
        if parent["surface_type"] is None:  # Skip already classified surfaces
            if (
                abs(parent["height"] - dorsal["height"]) <= tolerance * dorsal["height"]
                and abs(parent["height"] - platform["height"]) > tolerance * platform["height"]
                and parent["width"] != dorsal["width"]
            ):
                parent["surface_type"] = "Lateral"
                break

    logging.info("Classified parent contours into surfaces: Dorsal, Ventral, Platform, and Lateral.")
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

    Args:
        metrics (list): List of dictionaries containing contour metrics.
        output_path (str): Path to save the CSV file.
        append (bool): Whether to append to an existing file. Defaults to False.

    Returns:
        None
    """
    # Convert metrics to DataFrame
    updated_data = []
    for metric in metrics:
        # If it's a parent, set `scar` as the same as `surface_type`
        if metric["parent"] == metric["scar"]:
            updated_data.append({
                "image_id": metric["image_id"],
                "surface_type": metric["surface_type"],
                "scar": metric["surface_type"],  # Parent uses its surface type as scar
                "centroid_x": metric["centroid_x"],
                "centroid_y": metric["centroid_y"],
                "width": metric["width"],
                "height": metric["height"],
                "area": metric["area"]
            })
        else:
            # For scars, repeat the parent's surface type
            parent_surface_type = next(
                (m["surface_type"] for m in metrics if m["parent"] == metric["parent"] and m["parent"] == m["scar"]),
                "Unknown"
            )
            updated_data.append({
                "image_id": metric["image_id"],
                "surface_type": parent_surface_type,
                "scar": metric["scar"],
                "centroid_x": metric["centroid_x"],
                "centroid_y": metric["centroid_y"],
                "width": metric["width"],
                "height": metric["height"],
                "area": metric["area"]
            })

    # Convert updated data to a DataFrame
    columns = ["image_id", "surface_type", "scar", "centroid_x", "centroid_y", "width", "height", "area"]
    df = pd.DataFrame(updated_data, columns=columns)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write or append data
    if append and os.path.exists(output_path):
        df.to_csv(output_path, mode="a", header=False, index=False)
        logging.info("Appended metrics to existing CSV file: %s", output_path)
    else:
        df.to_csv(output_path, index=False)
        logging.info("Saved metrics to new CSV file: %s", output_path)



def process_and_save_contours(inverted_image, conversion_factor, output_dir, image_id):
    """
    Process contours, calculate metrics, classify surfaces, and append results to a single CSV file.

    Args:
        inverted_image (numpy.ndarray): Inverted binary thresholded image.
        conversion_factor (float): Conversion factor for pixels to real-world units.
        output_dir (str): Directory to save processed outputs.
        image_id (str): Name of the image being processed.
    """
    # Extract contours
    contours, hierarchy = extract_contours_with_hierarchy(
        inverted_image,
        image_id,
        conversion_factor,
        output_dir
    )
    if not contours:
        logging.warning("No valid contours found for image: %s", image_id)
        return

    # Flag nested and single child contours
    nested_child_contours = hide_nested_child_contours(contours, hierarchy)

    # Calculate metrics (excluding nested and single child contours)
    metrics = calculate_contour_metrics(contours, hierarchy, conversion_factor, nested_child_contours)

    # Classify surfaces
    metrics = classify_parent_contours(metrics)

    # Add image_id to each metric entry
    for metric in metrics:
        metric["image_id"] = image_id

    # Save metrics to the combined CSV file
    combined_csv_path = os.path.join(output_dir, "processed_metrics.csv")
    save_measurements_to_csv(metrics, combined_csv_path, append=True)

    # Visualize contours (excluding nested and single child contours)
    filtered_contours = [c for i, c in enumerate(contours) if not nested_child_contours[i]]
    visualization_path = os.path.join(output_dir, f"{image_id}_labeled.png")
    visualize_contours_with_hierarchy(filtered_contours, hierarchy, metrics, inverted_image, visualization_path)
