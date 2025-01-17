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
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

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

    Args:
        sorted_contours (dict): Dictionary with sorted contours:
            - "parents": List of parent contours.
            - "children": List of child contours.
            - "nested_children": List of nested child contours (will be excluded).
        hierarchy (numpy.ndarray): Hierarchy array corresponding to the contours.
        original_contours (list): Original list of all contours.

    Returns:
        list: A list of dictionaries containing raw contour metrics for parents and children.
    """
    metrics = []
    parent_count = 0
    child_count = 0

    # Create a mapping of parent indices to labels
    parent_index_to_label = {}

    # Process parents first
    for parent_contour in sorted_contours["parents"]:
        # Find the index of the parent contour in the original list
        parent_index = next(i for i, c in enumerate(original_contours) if np.array_equal(c, parent_contour))
        parent_count += 1
        parent_label = f"parent {parent_count}"
        parent_index_to_label[parent_index] = parent_label

        # Calculate metrics for the parent contour
        area = round(cv2.contourArea(parent_contour), 2)
        moments = cv2.moments(parent_contour)
        centroid_x, centroid_y = (0.0, 0.0)
        if moments["m00"] != 0:
            centroid_x = round(moments["m10"] / moments["m00"], 2)
            centroid_y = round(moments["m01"] / moments["m00"], 2)

        x, y, w, h = cv2.boundingRect(parent_contour)
        width = round(w, 2)
        height = round(h, 2)
        aspect_ratio = round(height / width, 2)

        metrics.append({
            "parent": parent_label,
            "scar": parent_label,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "width": width,
            "height": height,
            "area": area,
            "aspect_ratio": aspect_ratio
        })

    # Process children next
    for child_contour in sorted_contours["children"]:
        # Find the index of the child contour in the original list
        child_index = next(i for i, c in enumerate(original_contours) if np.array_equal(c, child_contour))
        parent_index = hierarchy[child_index][3]  # Get the parent index from the hierarchy
        parent_label = parent_index_to_label.get(parent_index, "Unknown")  # Get the parent label

        child_count += 1
        child_label = f"scar {child_count}"

        # Calculate metrics for the child contour
        area = round(cv2.contourArea(child_contour), 2)
        moments = cv2.moments(child_contour)
        centroid_x, centroid_y = (0.0, 0.0)
        if moments["m00"] != 0:
            centroid_x = round(moments["m10"] / moments["m00"], 2)
            centroid_y = round(moments["m01"] / moments["m00"], 2)

        x, y, w, h = cv2.boundingRect(child_contour)
        width = round(w, 2)
        height = round(h, 2)
        aspect_ratio = round(height / width, 2)

        metrics.append({
            "parent": parent_label,
            "scar": child_label,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "width": width,
            "height": height,
            "area": area,
            "aspect_ratio": aspect_ratio
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

    Args:
        metrics (list): List of dictionaries containing contour metrics.
        output_path (str): Path to save the CSV file.
        append (bool): Whether to append to an existing file. Defaults to False.

    Returns:
        None
    """
    # Prepare data for the DataFrame
    updated_data = []
    for metric in metrics:
        if metric["parent"] == metric["scar"]:
            # Parent contours: use their own surface_type
            surface_type = metric.get("surface_type", "NA")
            surface_feature = surface_type
        else:
            # Child contours: inherit surface_type from their parent
            parent_surface_type = next(
                (m["surface_type"] for m in metrics if m["parent"] == metric["parent"] and m["parent"] == m["scar"]),
                "NA"
            )
            surface_type = parent_surface_type
            surface_feature = metric["scar"]

        # Prepare the data entry
        data_entry = {
            "image_id": metric["image_id"],
            "surface_type": surface_type,
            "surface_feature": surface_feature,
            "centroid_x": metric.get("centroid_x", "NA"),
            "centroid_y": metric.get("centroid_y", "NA"),
            "width": metric.get("width", "NA"),
            "height": metric.get("height", "NA"),
            "total_area": metric.get("area", "NA"),
            "aspect_ratio": metric.get("aspect_ratio", "NA"),
            "top_area": metric.get("top_area", "NA"),
            "bottom_area": metric.get("bottom_area", "NA"),
            "left_area": metric.get("left_area", "NA"),
            "right_area": metric.get("right_area", "NA"),
            "vertical_symmetry": metric.get("vertical_symmetry", "NA"),
            "horizontal_symmetry": metric.get("horizontal_symmetry", "NA"),
        }
        updated_data.append(data_entry)

    # Define columns dynamically
    base_columns = [
        "image_id", "surface_type", "surface_feature", "centroid_x", "centroid_y",
        "width", "height", "total_area", "aspect_ratio"
    ]
    symmetry_columns = [
        "top_area", "bottom_area", "left_area", "right_area",
        "vertical_symmetry", "horizontal_symmetry"
    ]
    all_columns = base_columns + symmetry_columns

    # Convert updated data to a DataFrame
    df = pd.DataFrame(updated_data, columns=all_columns)

    # Ensure all `NA` replacements are applied
    df.fillna("NA", inplace=True)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write or append data to the CSV
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

def generate_voronoi_diagram(metrics, inverted_image, output_path):
    """
    Generate and visualize a Voronoi diagram for Dorsal surface contours,
    including centroids from associated child contours.

    Args:
        metrics (list): List of contour metrics containing centroids and surface types.
        inverted_image (numpy.ndarray): Inverted binary thresholded image.
        output_path (str): Path to save the Voronoi diagram visualization.

    Returns:
        None
    """
    # Filter metrics for Dorsal surface
    dorsal_metrics = [m for m in metrics if m.get("surface_type") == "Dorsal"]

    if not dorsal_metrics:
        logging.warning("No Dorsal surface metrics available for Voronoi diagram.")
        return

    # Collect centroids for Dorsal parents and their associated children
    centroids = []
    for dorsal_metric in dorsal_metrics:
        # Add centroid of the parent dorsal contour
        centroids.append((dorsal_metric["centroid_x"], dorsal_metric["centroid_y"]))

        # Add centroids of child contours linked to this parent
        child_metrics = [m for m in metrics if m["parent"] == dorsal_metric["parent"] and m["parent"] != m["scar"]]
        centroids.extend((child["centroid_x"], child["centroid_y"]) for child in child_metrics)

    centroids = np.array(centroids)

    # Check if there are enough points for Voronoi
    if len(centroids) < 4:
        logging.warning(
            "Insufficient centroids (%d points) for Voronoi diagram. At least 4 points are required.", len(centroids)
        )
        return

    # Generate Voronoi diagram
    vor = Voronoi(centroids)

    # Create a plot for visualization
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw the original inverted image as the background
    ax.imshow(cv2.cvtColor(cv2.bitwise_not(inverted_image), cv2.COLOR_GRAY2RGB))

    # Plot the Voronoi diagram
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='blue', line_width=2, point_size=5)

    # Highlight centroids
    ax.plot(centroids[:, 0], centroids[:, 1], 'ro', label='Dorsal Centroids')

    # Annotate centroids with labels
    for i, (x, y) in enumerate(centroids):
        ax.text(x + 5, y - 5, f"C{i+1}", color="grey", fontsize=12, fontweight="bold")

    # Set title and legend
    ax.set_title("Voronoi Diagram for Dorsal Surface and Associated Children")
    ax.legend()

    # Save the plot
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

    # Step 8: Add symmetry scores to the metrics
    for metric in metrics:
        if metric.get("surface_type") == "Dorsal":
            metric.update(symmetry_scores)

    # Step 9: Save metrics to CSV
    combined_csv_path = os.path.join(output_dir, "processed_metrics.csv")
    save_measurements_to_csv(metrics, combined_csv_path, append=True)

    # Step 10: Visualize contours with hierarchy
    visualization_path = os.path.join(output_dir, f"{image_id}_labeled.png")
    visualize_contours_with_hierarchy(
        sorted_contours["parents"] + sorted_contours["children"],
        hierarchy,
        metrics,
        inverted_image,
        visualization_path
    )

    # Step 11: Generate and visualize Voronoi diagram for the Dorsal surface and its children
    voronoi_output_path = os.path.join(output_dir, f"{image_id}_voronoi.png")
    generate_voronoi_diagram(metrics, inverted_image, voronoi_output_path)



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
