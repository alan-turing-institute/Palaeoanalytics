"""
PyLithics: Image Analysis Module

This module provides functions for performing contour analysis on
preprocessed images. It accepts thresholded images from the preprocessing
pipeline, finds contours, and calculates measurements such as area, centroid,
maximum height, and width for each contour.

Usage:
    - analyze_image_contours_with_hierarchy(): Perform hierarchical contour analysis on a given image.
    - save_measurements_to_csv(): Save the contour measurements to a CSV file.
"""

import os
import cv2
import logging
import pandas as pd

def extract_contours_with_hierarchy(thresholded_image):
    """
    Extract contours and their hierarchy from the thresholded image.

    :param thresholded_image: Thresholded image data.
    :return: Contours and hierarchy.
    """
    contours, hierarchy = cv2.findContours(
        thresholded_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is not None:
        hierarchy = hierarchy[0]  # Flatten hierarchy array
    return contours, hierarchy

def exclude_invalid_contours(contours, hierarchy, image_shape):
    """
    Exclude invalid contours such as borders or unnecessary inner contours.

    :param contours: List of contours.
    :param hierarchy: Hierarchy array for the contours.
    :param image_shape: Shape of the image (height, width).
    :return: Filtered contours and hierarchy.
    """
    image_height, image_width = image_shape
    valid_contours = []
    valid_hierarchy = []

    for idx, (contour, h) in enumerate(zip(contours, hierarchy)):
        parent_idx = h[3]

        # Exclude contours matching the image borders
        x, y, width, height = cv2.boundingRect(contour)
        if x == 0 and y == 0 and width == image_width and height == image_height:
            continue

        # Exclude inner contours (holes) if the parent contour has no children
        if parent_idx != -1 and hierarchy[parent_idx][2] == -1:
            continue

        valid_contours.append(contour)
        valid_hierarchy.append(h)

    return valid_contours, valid_hierarchy

def calculate_contour_metrics(contours, hierarchy, conversion_factor, image_id):
    """
    Calculate metrics for each contour.

    :param contours: List of valid contours.
    :param hierarchy: Hierarchy of the valid contours.
    :param conversion_factor: Conversion factor for pixels to real-world units.
    :param image_id: Identifier for the image being processed.
    :return: List of dictionaries containing contour metrics.
    """
    metrics = []
    for idx, (contour, h) in enumerate(zip(contours, hierarchy)):
        area = cv2.contourArea(contour) * (conversion_factor ** 2)
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            centroid_x = (moments["m10"] / moments["m00"]) * conversion_factor
            centroid_y = (moments["m01"] / moments["m00"]) * conversion_factor
        else:
            centroid_x, centroid_y = 0, 0

        x, y, width, height = cv2.boundingRect(contour)
        width = width * conversion_factor
        height = height * conversion_factor

        parent_idx = h[3]
        parent_label = f"contour_{parent_idx + 1}" if parent_idx != -1 else None

        metrics.append({
            "image_id": image_id,
            "label": f"contour_{idx + 1}",
            "area": area,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "width": width,
            "height": height,
            "parent_label": parent_label
        })
    return metrics

def visualize_contours(contours, hierarchy, metrics, thresholded_image, output_path, conversion_factor):
    """
    Create a labeled visualization of the contours.

    :param contours: List of contours.
    :param hierarchy: Hierarchy of the contours.
    :param metrics: List of contour metrics (real-world units).
    :param thresholded_image: Thresholded image data.
    :param output_path: Path to save the labeled image.
    :param conversion_factor: Conversion factor to convert real-world units to pixels.
    """
    # Convert the thresholded image to BGR for visualization
    labeled_image = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2BGR)

    for metric, h in zip(metrics, hierarchy):
        # Convert centroid from real-world units to pixel space
        cx = int(metric["centroid_x"] / conversion_factor)
        cy = int(metric["centroid_y"] / conversion_factor)
        label = metric["label"]

        # Define color based on hierarchy level
        color = (255, 0, 0) if h[3] == -1 else (0, 255, 0)  # Blue for parent, Green for child

        # Find the corresponding contour index
        contour_idx = int(label.split('_')[-1]) - 1

        # Draw the contour
        cv2.drawContours(labeled_image, [contours[contour_idx]], -1, color, 2)

        # Draw the centroid as a red dot
        cv2.circle(labeled_image, (cx, cy), 5, (0, 0, 255), -1)

        # Calculate the position for the label (avoid overlapping)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = max(0, cx - text_size[0] // 2)
        text_y = max(15, cy - 10)  # Ensure text is not drawn out of bounds

        # Draw the label near the centroid
        cv2.putText(
            labeled_image,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),  # Red for text
            1,
            cv2.LINE_AA,
        )

    # Save the labeled image to the specified output path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, labeled_image)
    logging.info("Saved labeled visualization to: %s", output_path)


def analyze_image_contours_with_hierarchy(thresholded_image, image_id, conversion_factor, output_dir):
    """
    Orchestrate the process of contour analysis with hierarchy.
    """
    contours, hierarchy = extract_contours_with_hierarchy(thresholded_image)
    if hierarchy is None:
        logging.warning("No contours found in image: %s", image_id)
        return pd.DataFrame()

    contours, hierarchy = exclude_invalid_contours(contours, hierarchy, thresholded_image.shape)
    metrics = calculate_contour_metrics(contours, hierarchy, conversion_factor, image_id)
    output_path = os.path.join(output_dir, f"{image_id}_labeled.png")

    # Pass the conversion_factor to the visualization function
    visualize_contours(contours, hierarchy, metrics, thresholded_image, output_path, conversion_factor)

    return pd.DataFrame(metrics)


def save_measurements_to_csv(df_measurements, output_dir, output_filename="contour_measurements.csv"):
    """
    Save the contour measurements DataFrame to a CSV file, rounding numerical values to two decimals.

    :param df_measurements: DataFrame containing the contour measurements.
    :param output_dir: Directory where the output CSV file will be saved.
    :param output_filename: Name of the output CSV file.
    """
    # Round numeric columns to 2 decimal places
    df_measurements = df_measurements.round(2)

    # Ensure the output directory exists
    output_csv = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)

    # Save the DataFrame to a CSV file
    df_measurements.to_csv(output_csv, index=False)
    logging.info("Saved contour measurements to: %s", output_csv)


def main(data_dir):
    """
    Main function to perform image analysis. This function remains for backward compatibility.

    :param data_dir: The root data directory containing a 'processed' subdirectory.
    """
    logging.info("This version of main is deprecated. Direct calls from preprocessing are recommended.")


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="PyLithics: Image Contour Analysis")
    parser.add_argument('--data_dir', required=True, help="Directory containing the processed images.")

    args = parser.parse_args()
    main(args.data_dir)
