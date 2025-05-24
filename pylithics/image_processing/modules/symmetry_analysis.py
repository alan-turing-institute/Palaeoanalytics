import cv2
import numpy as np
import logging

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