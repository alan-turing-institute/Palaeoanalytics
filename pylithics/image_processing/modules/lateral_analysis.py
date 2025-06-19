"""
Lateral Surface Analysis Module for PyLithics
============================================

This module provides convexity analysis for lateral surface properties of lithic artifacts.
It detects the convexity of lateral surfaces using area-based methods.

The module is designed to work specifically with lateral surface classifications and integrates
with the existing PyLithics pipeline for surface type analysis.

Main Functions:
    * analyze_lateral_surface(metrics, parent_contours, inverted_image)
          - Main orchestrator for lateral surface convexity analysis
    * detect_lateral_convexity(contour)
          - Calculate convexity measure using area-based method

All functions include comprehensive error handling and logging for debugging and traceability.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple, Union


def analyze_lateral_surface(metrics: List[Dict[str, Any]],
                          parent_contours: List[np.ndarray],
                          inverted_image: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Main orchestrator function for analyzing lateral surface convexity.

    This function coordinates the lateral surface convexity analysis workflow.
    It operates only on contours classified as "Lateral" surface type.

    Parameters
    ----------
    metrics : list of dict
        List of metric dictionaries containing surface classifications and contour data
    parent_contours : list of ndarray
        List of parent contours corresponding to the metrics
    inverted_image : ndarray
        Inverted binary thresholded image for spatial reference

    Returns
    -------
    dict or None
        Dictionary containing lateral surface analysis results:
        - 'lateral_convexity': float, convexity ratio (0-1)
        - 'distance_to_max_width': float, distance from top to max width center
        Returns None if no lateral surface found or analysis fails

    Raises
    ------
    Exception
        Logs detailed error information if analysis fails but continues processing
        where possible to maximize data recovery

    Notes
    -----
    This function is designed to work with the existing PyLithics surface classification
    system and only processes contours that have been classified as "Lateral" type.
    """
    try:
        # Find lateral surface metrics
        lateral_metric = None
        lateral_contour = None
        lateral_contour_index = -1

        for i, metric in enumerate(metrics):
            if (metric.get("surface_type") == "Lateral" and
                metric["parent"] == metric["scar"]):  # This is a parent contour
                lateral_metric = metric
                lateral_contour_index = i
                break

        if lateral_metric is None:
            logging.info("No Lateral surface found for convexity analysis")
            return None

        # Find the corresponding contour
        if lateral_contour_index < len(parent_contours):
            lateral_contour = parent_contours[lateral_contour_index]
        else:
            # Fallback: find contour by matching area
            for contour in parent_contours:
                contour_area = cv2.contourArea(contour)
                if abs(contour_area - lateral_metric.get("area", 0)) < 1.0:
                    lateral_contour = contour
                    break

        if lateral_contour is None:
            logging.error("Could not find lateral contour for convexity analysis")
            return None

        logging.info("Starting lateral surface convexity analysis")

        # Calculate convexity
        try:
            convexity = detect_lateral_convexity(lateral_contour)
            logging.info(f"Lateral convexity calculated: {convexity}")
        except Exception as e:
            logging.error(f"Error calculating lateral convexity: {e}")
            convexity = None

        # Calculate distance to max width for lateral surface
        distance_to_max_width = None
        try:
            distance_to_max_width = _calculate_lateral_distance_to_max_width(lateral_contour)
            logging.info(f"Distance to max width: {distance_to_max_width}")
        except Exception as e:
            logging.error(f"Error calculating distance to max width: {e}")

        # Compile results
        lateral_results = {
            'lateral_convexity': round(convexity, 2) if convexity is not None else None,
            'distance_to_max_width': round(distance_to_max_width, 2) if distance_to_max_width is not None else None
        }

        logging.info("Lateral surface analysis completed successfully")
        return lateral_results

    except Exception as e:
        logging.error(f"Critical error in lateral surface analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


def detect_lateral_convexity(contour: np.ndarray) -> Optional[float]:
    """
    Calculate the convexity of a lateral surface contour using area-based method.

    This function calculates convexity as the ratio of the contour area to its
    convex hull area. Values closer to 1.0 indicate more convex shapes, while
    values closer to 0 indicate more concave shapes.

    Parameters
    ----------
    contour : ndarray
        Input contour points from cv2.findContours

    Returns
    -------
    float or None
        Convexity ratio between 0 and 1, where:
        - 1.0 = perfectly convex (contour equals its convex hull)
        - < 1.0 = concave (contour has indentations)
        Returns None if calculation fails

    Raises
    ------
    Exception
        If contour is invalid or convex hull calculation fails

    Notes
    -----
    This method uses the standard computer vision approach of comparing
    the original contour area to its convex hull area. The convex hull
    represents the smallest convex polygon that contains all contour points.
    """
    try:
        if contour is None or len(contour) < 3:
            logging.warning("Invalid contour provided for convexity calculation")
            return None

        # Calculate contour area
        contour_area = cv2.contourArea(contour)
        if contour_area <= 0:
            logging.warning("Contour area is zero or negative")
            return None

        # Calculate convex hull and its area
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        if hull_area <= 0:
            logging.warning("Convex hull area is zero or negative")
            return None

        # Calculate convexity ratio
        convexity = contour_area / hull_area

        # Sanity check - convexity should be between 0 and 1
        if convexity > 1.0:
            logging.warning(f"Convexity ratio {convexity} exceeds 1.0, clamping to 1.0")
            convexity = 1.0
        elif convexity < 0:
            logging.warning(f"Convexity ratio {convexity} is negative, setting to 0")
            convexity = 0.0

        return convexity

    except Exception as e:
        logging.error(f"Error calculating lateral convexity: {e}")
        return None


def _integrate_lateral_metrics(metrics: List[Dict[str, Any]],
                             lateral_results: Dict[str, Any]) -> None:
    """
    Integrate lateral surface analysis results into the main metrics list.

    This helper function updates the lateral surface metric dictionary with
    the calculated lateral analysis results. It follows the same pattern as
    other metric integration functions in the PyLithics pipeline.

    Parameters
    ----------
    metrics : list of dict
        Main metrics list to update
    lateral_results : dict
        Dictionary containing lateral analysis results

    Returns
    -------
    None
        Updates metrics list in place

    Notes
    -----
    This function modifies the metrics list in place, adding lateral analysis
    results to the appropriate lateral surface metric entry.
    """
    try:
        # Find the lateral surface metric and update it
        for metric in metrics:
            if (metric.get("surface_type") == "Lateral" and
                metric["parent"] == metric["scar"]):
                metric.update(lateral_results)
                logging.debug("Integrated lateral analysis results into metrics")
                break
        else:
            logging.warning("No lateral surface metric found for integration")

    except Exception as e:
        logging.error(f"Error integrating lateral metrics: {e}")


def _calculate_lateral_distance_to_max_width(cnt: np.ndarray) -> Optional[float]:
    """
    Calculate distance from top to max width center for lateral surface.

    Parameters
    ----------
    cnt : ndarray
        Lateral surface contour to analyze

    Returns
    -------
    float or None
        Distance from top point to center of maximum width, or None if calculation fails
    """
    try:
        # Find max length points to get center of max width
        max_len = 0
        p1 = p2 = None
        for i in range(len(cnt)):
            for j in range(i+1, len(cnt)):
                a = cnt[i][0]; b = cnt[j][0]
                d = np.linalg.norm(a - b)
                if d > max_len:
                    max_len, p1, p2 = d, a, b

        if p1 is None or p2 is None:
            return None

        max_width_center = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

        # Find top point (max Y coordinate)
        contour_points = cnt.reshape(-1, 2)
        max_y = np.max(contour_points[:, 1])
        top_points = contour_points[contour_points[:, 1] == max_y]

        if len(top_points) == 1:
            top_point = top_points[0]
        else:
            median_x_idx = len(top_points) // 2
            sorted_top_points = top_points[np.argsort(top_points[:, 0])]
            top_point = sorted_top_points[median_x_idx]

        # Calculate distance
        dx = max_width_center[0] - top_point[0]
        dy = max_width_center[1] - top_point[1]
        return np.sqrt(dx * dx + dy * dy)

    except Exception as e:
        logging.error(f"Error calculating lateral distance: {e}")
        return None