"""
Arrow Detection Module for PyLithics
============================================

This integrates arrow detection with the pipeline,
provides improved error handling, and configurable parameters.
"""

import os
import cv2
import numpy as np
import math
import logging
from collections import defaultdict
from typing import Optional, Dict, Any, List, Tuple, Union


from .config import get_arrow_detection_config


class ArrowDetector:
    """
    Enhanced arrow detection class with configurable parameters and better error handling.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the arrow detector.

        Parameters
        ----------
        config : dict, optional
            Configuration dictionary. If None, loads from config manager.
        """
        self.config = config or get_arrow_detection_config()
        self.reference_dpi = self.config.get('reference_dpi', 300.0)

        # Reference thresholds (calibrated for 300 DPI)
        self.ref_thresholds = {
            'min_area': 1,
            'min_defect_depth': 2,
            'solidity_bounds': (0.4, 1.0),
            'min_triangle_height': 8,
            'min_significant_defects': 2
        }

        self.debug_enabled = self.config.get('debug_enabled', False)

    def scale_parameters_for_dpi(self, image_dpi: Optional[float]) -> Dict[str, Any]:
        """
        Scale detection parameters based on image DPI.

        Parameters
        ----------
        image_dpi : float, optional
            DPI of the current image being processed

        Returns
        -------
        dict
            Dictionary of scaled parameters
        """
        if image_dpi is None or image_dpi <= 0:
            logging.warning("Invalid DPI value. Using reference thresholds.")
            return self.ref_thresholds.copy()

        # Scale factor relative to reference DPI
        linear_scale = image_dpi / self.reference_dpi
        area_scale = linear_scale * linear_scale

        # Get scale factors from config
        area_safety = self.config.get('min_area_scale_factor', 0.7)
        depth_safety = self.config.get('min_defect_depth_scale_factor', 0.8)
        height_safety = self.config.get('min_triangle_height_scale_factor', 0.8)

        return {
            'min_area': self.ref_thresholds['min_area'] * area_scale * area_safety,
            'min_defect_depth': self.ref_thresholds['min_defect_depth'] * linear_scale * depth_safety,
            'solidity_bounds': self.ref_thresholds['solidity_bounds'],
            'min_triangle_height': self.ref_thresholds['min_triangle_height'] * linear_scale * height_safety,
            'min_significant_defects': self.ref_thresholds['min_significant_defects']
        }

    def analyze_contour_for_arrow(self,
                                 contour: np.ndarray,
                                 entry: Dict[str, Any],
                                 image: np.ndarray,
                                 image_dpi: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Detect an arrow within a given contour and compute its orientation.

        Parameters
        ----------
        contour : ndarray
            Single child contour from cv2.findContours
        entry : dict
            Metrics dictionary for this contour
        image : ndarray
            Original image for overlaying debug visualizations
        image_dpi : float, optional
            DPI of the image being processed

        Returns
        -------
        dict or None
            Arrow properties if valid arrow found, None otherwise
        """
        debug_dir = entry.get('debug_dir') if self.debug_enabled else None
        contour_id = entry.get('scar', 'unknown')

        # Scale parameters based on image DPI
        params = self.scale_parameters_for_dpi(image_dpi)

        debug_log = None
        if debug_dir:
            debug_log = self._setup_debug_logging(debug_dir, contour_id, image_dpi, params)

        try:
            # Step 1: Basic filtering
            if not self._validate_basic_properties(contour, params, debug_log):
                return None

            # Step 2: Find significant defects
            significant_defects = self._find_significant_defects(contour, params['min_defect_depth'])
            if not self._validate_defects(significant_defects, params, debug_log):
                return None

            # Step 3: Triangle analysis
            triangle_data = self._analyze_triangle_structure(
                contour, significant_defects, params, debug_log
            )
            if triangle_data is None:
                return None

            # Step 4: Calculate arrow properties
            arrow_data = self._calculate_arrow_properties(triangle_data, debug_log)

            # Step 5: Generate debug visualizations
            if debug_dir and image is not None:
                self._create_debug_visualizations(
                    contour, triangle_data, arrow_data, image, debug_dir
                )

            if debug_log:
                debug_log.write("Arrow detection succeeded\n")

            return arrow_data

        except Exception as e:
            if debug_log:
                debug_log.write(f"Error in arrow detection: {str(e)}\n")
            logging.error(f"Arrow detection failed for {contour_id}: {e}")
            return None
        finally:
            if debug_log:
                debug_log.close()

    def _setup_debug_logging(self,
                           debug_dir: str,
                           contour_id: str,
                           image_dpi: Optional[float],
                           params: Dict[str, Any]):
        """Setup debug logging for arrow detection."""
        os.makedirs(debug_dir, exist_ok=True)
        debug_log = open(os.path.join(debug_dir, 'arrow_detection_log.txt'), 'w')

        debug_log.write(f"Arrow detection analysis for contour {contour_id}\n")
        if image_dpi:
            debug_log.write(f"Image DPI: {image_dpi}, scaling applied\n")

        debug_log.write("Scaled parameters:\n")
        for key, value in params.items():
            debug_log.write(f"  {key}={value}\n")

        return debug_log

    def _validate_basic_properties(self,
                                 contour: np.ndarray,
                                 params: Dict[str, Any],
                                 debug_log) -> bool:
        """Validate basic contour properties."""
        # Area test
        area = cv2.contourArea(contour)
        min_area = params['min_area']

        if debug_log:
            debug_log.write(f"Area test: {area:.2f} >= {min_area:.2f}\n")

        if area < min_area:
            if debug_log:
                debug_log.write(f"Failed: Area too small\n")
            return False

        # Solidity test
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0

        solidity_bounds = params['solidity_bounds']

        if debug_log:
            debug_log.write(f"Solidity test: {solidity:.3f} in range {solidity_bounds}\n")

        if solidity < solidity_bounds[0] or solidity > solidity_bounds[1]:
            if debug_log:
                debug_log.write(f"Failed: Solidity outside acceptable range\n")
            return False

        if debug_log:
            debug_log.write("Basic property tests: PASSED\n")

        return True

    def _find_significant_defects(self,
                                contour: np.ndarray,
                                min_defect_depth: float) -> Optional[List[Tuple]]:
        """Find significant convexity defects in the contour."""
        try:
            hull_indices = cv2.convexHull(contour, returnPoints=False)
            if hull_indices is None or len(hull_indices) < 3:
                return None

            defects = cv2.convexityDefects(contour, hull_indices)
            if defects is None or defects.shape[0] < 2:
                return None

            significant_defects = []
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                depth = d / 256.0
                if depth > min_defect_depth:
                    start = tuple(contour[s][0])
                    end = tuple(contour[e][0])
                    far = tuple(contour[f][0])
                    significant_defects.append((start, end, far, depth))

            # Sort by depth (deepest first)
            significant_defects.sort(key=lambda x: x[3], reverse=True)
            return significant_defects

        except Exception:
            return None

    def _validate_defects(self,
                        significant_defects: Optional[List[Tuple]],
                        params: Dict[str, Any],
                        debug_log) -> bool:
        """Validate the found defects."""
        min_defects = params['min_significant_defects']

        if not significant_defects or len(significant_defects) < min_defects:
            if debug_log:
                count = len(significant_defects) if significant_defects else 0
                debug_log.write(f"Failed: Not enough defects ({count} < {min_defects})\n")
            return False

        if debug_log:
            depths = [d[3] for d in significant_defects]
            debug_log.write(f"Defect test: PASSED (found {len(significant_defects)} defects)\n")
            debug_log.write(f"Defect depths: {depths}\n")

        return True

    def _analyze_triangle_structure(self,
                                  contour: np.ndarray,
                                  significant_defects: List[Tuple],
                                  params: Dict[str, Any],
                                  debug_log) -> Optional[Dict[str, Any]]:
        """Analyze the triangle structure from defects."""
        # Keep only the two deepest defects
        significant_defects = significant_defects[:2]

        # Identify triangle base
        triangle_base_info = self._identify_triangle_base(significant_defects)
        if triangle_base_info is None:
            if debug_log:
                debug_log.write("Failed: Could not identify triangle base\n")
            return None

        base_p1, base_p2, base_midpoint, base_length = triangle_base_info

        if debug_log:
            debug_log.write(f"Triangle base length: {base_length:.2f} pixels\n")

        # Analyze half-spaces
        halfspace_results = self._analyze_halfspaces(contour, triangle_base_info)
        if halfspace_results is None:
            if debug_log:
                debug_log.write("Failed: Half-space analysis failed\n")
            return None

        shaft_halfspace, tip_halfspace, solidity1, solidity2 = halfspace_results

        if debug_log:
            debug_log.write(f"Half-space solidities: {solidity1:.3f}, {solidity2:.3f}\n")
            debug_log.write(f"Shaft half-space: {shaft_halfspace}, Tip half-space: {tip_halfspace}\n")

        # Find triangle tip
        halfspace_points = self._divide_contour_points(contour, triangle_base_info)
        triangle_tip = self._find_triangle_tip(halfspace_points[tip_halfspace], base_midpoint)

        if triangle_tip is None:
            if debug_log:
                debug_log.write("Failed: Could not find triangle tip\n")
            return None

        # Validate triangle height
        triangle_height = np.sqrt(
            (triangle_tip[0] - base_midpoint[0])**2 +
            (triangle_tip[1] - base_midpoint[1])**2
        )

        min_height = params['min_triangle_height']

        if debug_log:
            debug_log.write(f"Triangle height test: {triangle_height:.2f} >= {min_height:.2f}\n")

        if triangle_height < min_height:
            if debug_log:
                debug_log.write("Failed: Triangle height too small\n")
            return False

        if debug_log:
            debug_log.write("Triangle analysis: PASSED\n")

        return {
            'base_p1': base_p1,
            'base_p2': base_p2,
            'base_midpoint': base_midpoint,
            'triangle_tip': triangle_tip,
            'triangle_height': triangle_height,
            'significant_defects': significant_defects
        }

    def _calculate_arrow_properties(self,
                                  triangle_data: Dict[str, Any],
                                  debug_log) -> Dict[str, Any]:
        """Calculate final arrow properties."""
        base_midpoint = triangle_data['base_midpoint']
        triangle_tip = triangle_data['triangle_tip']

        # Arrow direction: from triangle tip to base midpoint
        arrow_back = triangle_tip
        arrow_tip = base_midpoint

        # Calculate angles
        dx = arrow_tip[0] - arrow_back[0]
        dy = arrow_tip[1] - arrow_back[1]

        angle_rad = math.atan2(dy, dx)
        angle_deg = (math.degrees(angle_rad) + 360) % 360
        compass_angle = (270 + angle_deg) % 360

        if debug_log:
            debug_log.write(f"Arrow direction: from {arrow_back} to {arrow_tip}\n")
            debug_log.write(f"Angles - deg: {angle_deg:.2f}°, compass: {compass_angle:.2f}°\n")

        return {
            'arrow_back': arrow_back,
            'arrow_tip': arrow_tip,
            'angle_rad': round(angle_rad, 1),
            'angle_deg': round(angle_deg, 1),
            'compass_angle': round(compass_angle, 1),
            'triangle_data': triangle_data
        }

    def _identify_triangle_base(self, defects: List[Tuple]) -> Optional[Tuple]:
        """Identify the triangle base from defect pairs."""
        defect_points = [defect[2] for defect in defects]  # 'far' points

        if len(defect_points) < 2:
            return None

        # Get the pair with largest distance as base
        p1, p2 = defect_points[0], defect_points[1]
        distance = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        base_midpoint = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

        return (p1, p2, base_midpoint, distance)

    def _divide_contour_points(self,
                             contour: np.ndarray,
                             triangle_base_info: Tuple) -> Dict[int, List]:
        """Divide contour points into half-spaces."""
        base_p1, base_p2, base_midpoint, _ = triangle_base_info

        base_vector = np.array([base_p2[0] - base_p1[0], base_p2[1] - base_p1[1]])
        normal = np.array([-base_vector[1], base_vector[0]])

        halfspace_points = defaultdict(list)

        for point in contour:
            p = point[0]
            vec = np.array([p[0] - base_midpoint[0], p[1] - base_midpoint[1]])
            side = np.dot(vec, normal)

            if side > 0:
                halfspace_points[1].append(p)
            else:
                halfspace_points[2].append(p)

        return halfspace_points

    def _analyze_halfspaces(self,
                          contour: np.ndarray,
                          triangle_base_info: Tuple) -> Optional[Tuple]:
        """Analyze solidity of each half-space."""
        halfspace_points = self._divide_contour_points(contour, triangle_base_info)

        if len(halfspace_points[1]) < 5 or len(halfspace_points[2]) < 5:
            return None

        try:
            # Calculate solidity for each half-space
            solidities = {}
            for i in [1, 2]:
                points = np.array(halfspace_points[i], dtype=np.int32).reshape(-1, 1, 2)
                hull = cv2.convexHull(points)

                area = cv2.contourArea(points)
                hull_area = cv2.contourArea(hull)

                solidities[i] = area / hull_area if hull_area > 0 else 0

            # More solid half-space is likely the shaft
            if solidities[1] > solidities[2]:
                return (1, 2, solidities[1], solidities[2])  # shaft, tip, sol1, sol2
            else:
                return (2, 1, solidities[1], solidities[2])

        except Exception:
            return None

    def _find_triangle_tip(self,
                         tip_halfspace_points: List,
                         base_midpoint: Tuple) -> Optional[Tuple]:
        """Find the triangle tip (furthest point from base midpoint)."""
        if not tip_halfspace_points:
            return None

        max_dist = -1
        tip_point = None

        for p in tip_halfspace_points:
            dist = np.sqrt((p[0] - base_midpoint[0])**2 + (p[1] - base_midpoint[1])**2)
            if dist > max_dist:
                max_dist = dist
                tip_point = tuple(p)

        return tip_point

    def _create_debug_visualizations(self,
                                   contour: np.ndarray,
                                   triangle_data: Dict[str, Any],
                                   arrow_data: Dict[str, Any],
                                   image: np.ndarray,
                                   debug_dir: str) -> None:
        """Create debug visualization images."""
        vis = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)

        # Draw contour
        cv2.drawContours(vis, [contour], 0, (0, 255, 0), 2)

        # Draw hull
        hull = cv2.convexHull(contour)
        cv2.drawContours(vis, [hull], 0, (0, 0, 255), 1)

        # Draw defects
        for defect in triangle_data['significant_defects']:
            start, end, far, _ = defect
            cv2.circle(vis, far, 5, (255, 0, 0), -1)

        # Draw triangle base
        base_p1, base_p2 = triangle_data['base_p1'], triangle_data['base_p2']
        cv2.line(vis, base_p1, base_p2, (255, 0, 255), 2)

        # Draw triangle tip
        triangle_tip = triangle_data['triangle_tip']
        cv2.circle(vis, triangle_tip, 5, (0, 255, 255), -1)

        # Draw arrow
        arrow_back, arrow_tip = arrow_data['arrow_back'], arrow_data['arrow_tip']
        cv2.arrowedLine(vis, arrow_back, arrow_tip, (0, 255, 0), 2)

        # Draw angle text
        compass_angle = arrow_data['compass_angle']
        text_pos = (arrow_tip[0] + 10, arrow_tip[1] - 10)
        cv2.putText(vis, f"{compass_angle:.1f}°", text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(vis, f"{compass_angle:.1f}°", text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        # Save visualization
        cv2.imwrite(os.path.join(debug_dir, "arrow_debug.png"), vis)


# Backward compatibility functions
def scale_parameters_for_dpi(image_dpi: Optional[float]) -> Dict[str, Any]:
    """Scale detection parameters based on image DPI (backward compatibility)."""
    detector = ArrowDetector()
    return detector.scale_parameters_for_dpi(image_dpi)


def analyze_child_contour_for_arrow(contour: np.ndarray,
                                   entry: Dict[str, Any],
                                   image: np.ndarray,
                                   image_dpi: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """
    Detect an arrow within a given contour (backward compatibility).

    Parameters
    ----------
    contour : ndarray
        Single child contour from cv2.findContours
    entry : dict
        Metrics dictionary for this contour
    image : ndarray
        Original image for overlaying debug visualizations
    image_dpi : float, optional
        DPI of the image being processed

    Returns
    -------
    dict or None
        Arrow properties if valid arrow found, None otherwise
    """
    detector = ArrowDetector()
    return detector.analyze_contour_for_arrow(contour, entry, image, image_dpi)


# Additional utility functions for the pipeline
def create_arrow_detection_pipeline(config: Optional[Dict[str, Any]] = None) -> ArrowDetector:
    """
    Create a configured arrow detection pipeline.

    Parameters
    ----------
    config : dict, optional
        Arrow detection configuration

    Returns
    -------
    ArrowDetector
        Configured arrow detector instance
    """
    return ArrowDetector(config)


def batch_detect_arrows(contours: List[np.ndarray],
                       entries: List[Dict[str, Any]],
                       image: np.ndarray,
                       image_dpi: Optional[float] = None,
                       config: Optional[Dict[str, Any]] = None) -> List[Optional[Dict[str, Any]]]:
    """
    Detect arrows in multiple contours efficiently.

    Parameters
    ----------
    contours : list
        List of contours to analyze
    entries : list
        List of metric dictionaries corresponding to contours
    image : ndarray
        Original image for debug visualizations
    image_dpi : float, optional
        DPI of the image being processed
    config : dict, optional
        Arrow detection configuration

    Returns
    -------
    list
        List of arrow detection results (None for failed detections)
    """
    detector = ArrowDetector(config)
    results = []

    for contour, entry in zip(contours, entries):
        try:
            result = detector.analyze_contour_for_arrow(contour, entry, image, image_dpi)
            results.append(result)
        except Exception as e:
            logging.error(f"Failed to detect arrow for {entry.get('scar', 'unknown')}: {e}")
            results.append(None)

    return results


def validate_arrow_detection_config(config: Dict[str, Any]) -> bool:
    """
    Validate arrow detection configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary to validate

    Returns
    -------
    bool
        True if configuration is valid, False otherwise
    """
    required_keys = ['reference_dpi']
    optional_keys = [
        'min_area_scale_factor', 'min_defect_depth_scale_factor',
        'min_triangle_height_scale_factor', 'debug_enabled'
    ]

    # Check required keys
    for key in required_keys:
        if key not in config:
            logging.error(f"Missing required arrow detection config key: {key}")
            return False

    # Validate types and ranges
    if not isinstance(config['reference_dpi'], (int, float)) or config['reference_dpi'] <= 0:
        logging.error("reference_dpi must be a positive number")
        return False

    # Validate optional scale factors
    for key in ['min_area_scale_factor', 'min_defect_depth_scale_factor', 'min_triangle_height_scale_factor']:
        if key in config:
            value = config[key]
            if not isinstance(value, (int, float)) or not (0 < value <= 1):
                logging.error(f"{key} must be a number between 0 and 1")
                return False

    return True