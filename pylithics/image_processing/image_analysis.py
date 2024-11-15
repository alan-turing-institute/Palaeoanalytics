"""
PyLithics: Image Analysis Module

This module provides functions for performing contour analysis on
preprocessed images. It accepts thresholded images from the preprocessing
pipeline, finds contours, and calculates measurements such as area, centroid,
maximum height, and width for each contour.

Usage:
    - analyze_image_contours(): Perform contour analysis on a given image.
    - save_measurements_to_csv(): Save the contour measurements to a CSV file.
"""

import os
import cv2
import logging
import pandas as pd


def analyze_image_contours(thresholded_image, image_id, conversion_factor):
    """
    Perform contour analysis on the thresholded image.
    """
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    measurements = []

    for idx, contour in enumerate(contours):
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

        measurements.append({
            "label": f"contour_{idx + 1}",
            "area": area,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "width": width,
            "height": height,
            "image_id": image_id
        })

    return pd.DataFrame(measurements)


def save_measurements_to_csv(df_measurements, output_dir, output_filename="contour_measurements.csv"):
    """
    Save the contour measurements to a CSV file.
    """
    output_csv = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)
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
