"""
PyLithics: Image Analysis Module

This module provides functions for performing contour analysis on
preprocessed images. It reads thresholded images, finds contours,
and calculates measurements such as area, centroid, maximum height,
and width for each contour.

Usage:
    - analyze_contours(): Find contours and calculate measurements.
    - load_processed_images(): Load preprocessed images from a specified directory.
"""

import os
import cv2
import logging
import pandas as pd

def load_processed_images(processed_dir):
    """
    Load preprocessed images from the specified directory.

    :param processed_dir: Directory containing processed images.
    :return: List of image file paths.
    """
    if not os.path.exists(processed_dir):
        logging.error("Processed directory not found: %s", processed_dir)
        return []

    # Get all image file paths in the directory
    image_files = [os.path.join(processed_dir, f) for f in os.listdir(processed_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]

    if not image_files:
        logging.warning("No processed images found in: %s", processed_dir)

    return image_files

def analyze_contours(image_path):
    """
    Analyze contours in the given preprocessed image.

    :param image_path: Path to the thresholded image.
    :return: DataFrame containing contour measurements (area, centroid, height, width).
    """
    # Load the image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        logging.error("Could not load image: %s", image_path)
        return pd.DataFrame()

    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Data to be stored in the DataFrame
    measurements = []

    for i, contour in enumerate(contours):
        # Calculate contour area
        area = cv2.contourArea(contour)

        # Calculate contour centroid
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx, cy = 0, 0

        # Calculate bounding box (for width and height)
        x, y, w, h = cv2.boundingRect(contour)

        # Store the data in a dictionary
        measurements.append({
            "label": f"contour_{i+1}",
            "area": area,
            "centroid_x": cx,
            "centroid_y": cy,
            "width": w,
            "height": h
        })

    # Convert the list of dictionaries to a DataFrame
    df_measurements = pd.DataFrame(measurements)

    return df_measurements

def main(data_dir):
    """
    Main function to perform image analysis on all processed images in a directory.

    :param data_dir: The root data directory containing a 'processed' subdirectory.
    """
    processed_dir = os.path.join(data_dir, 'processed')
    image_files = load_processed_images(processed_dir)

    all_measurements = []

    for image_file in image_files:
        logging.info("Analyzing contours in image: %s", image_file)
        df = analyze_contours(image_file)
        if not df.empty:
            df["image"] = os.path.basename(image_file)  # Add image name to DataFrame
            all_measurements.append(df)

    if all_measurements:
        # Concatenate all measurements into a single DataFrame
        final_df = pd.concat(all_measurements, ignore_index=True)
        # Save the results to a CSV file
        output_csv = os.path.join(data_dir, "contour_measurements.csv")
        final_df.to_csv(output_csv, index=False)
        logging.info("Saved contour measurements to: %s", output_csv)
    else:
        logging.warning("No contours found in any processed images.")

if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="PyLithics: Image Contour Analysis")
    parser.add_argument('--data_dir', required=True, help="Directory containing the processed images.")

    args = parser.parse_args()
    main(args.data_dir)
