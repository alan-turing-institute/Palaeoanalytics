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


def analyze_image_contours_with_hierarchy(thresholded_image, image_id, conversion_factor, output_dir):
    """
    Perform hierarchical contour finding and analysis on the thresholded image,
    excluding inner contours (holes) for shapes without children and image borders.

    :param thresholded_image: The thresholded image data.
    :param image_id: Identifier for the image being processed.
    :param conversion_factor: Conversion factor for pixels to real-world units.
    :param output_dir: Directory where the processed results will be saved.
    :return: DataFrame containing contour measurements (excluding unnecessary inner contours and edges).
    """
    # Find contours with hierarchy information
    contours, hierarchy = cv2.findContours(
        thresholded_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None:
        logging.warning("No contours found in image: %s", image_id)
        return pd.DataFrame()

    measurements = []
    labeled_image = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2BGR)

    hierarchy = hierarchy[0]  # Flatten hierarchy array
    image_height, image_width = thresholded_image.shape

    for idx, (contour, h) in enumerate(zip(contours, hierarchy)):
        parent_idx = h[3]

        # Exclude contours matching the image borders
        x, y, width, height = cv2.boundingRect(contour)
        if x == 0 and y == 0 and width == image_width and height == image_height:
            logging.info(f"Skipping border contour: contour_{idx + 1}")
            continue

        # Exclude inner contours (holes) if the parent contour has no children
        if parent_idx != -1 and hierarchy[parent_idx][2] == -1:
            logging.info(f"Skipping inner contour: contour_{idx + 1}")
            continue

        # Calculate contour properties
        area = cv2.contourArea(contour) * (conversion_factor ** 2)
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            centroid_x = (moments["m10"] / moments["m00"]) * conversion_factor
            centroid_y = (moments["m01"] / moments["m00"]) * conversion_factor
            cx, cy = int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])
        else:
            centroid_x, centroid_y, cx, cy = 0, 0, 0, 0

        width = width * conversion_factor
        height = height * conversion_factor

        parent_label = f"contour_{parent_idx + 1}" if parent_idx != -1 else None

        # Store measurement data
        measurements.append({
            "image_id": image_id,
            "label": f"contour_{idx + 1}",
            "area": area,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "width": width,
            "height": height,
            "parent_label": parent_label
        })

        # Visualization
        color = (255, 0, 0) if parent_idx == -1 else (0, 255, 0)  # Blue for parent, Green for child
        cv2.drawContours(labeled_image, [contour], -1, color, 2)
        cv2.circle(labeled_image, (cx, cy), 5, (0, 0, 255), -1)  # Red dot for centroid
        cv2.putText(
            labeled_image,
            f"contour_{idx + 1}",
            (cx, max(0, cy - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    # Save labeled image
    output_path = os.path.join(output_dir, f"{image_id}_labeled.png")
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(output_path, labeled_image)
    logging.info("Saved labeled visualization for %s to %s", image_id, output_path)

    return pd.DataFrame(measurements)



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
