"""
PyLithics: Cortex Detection Module
=================================

This module detects cortex on lithic artifact surfaces. Cortex is distinguished from scars
by its characteristic stippled/dotted texture pattern representing the original weathered
surface of the stone material.

Key Features:
- Detects cortex at child contour level (not parent or nested contours)
- Labels cortex as cortex_1, cortex_2, etc. (distinct from scars)
- Calculates cortex area and percentage of parent surface area
- Maintains archaeological accuracy: cortex ≠ scar

Author: PyLithics Development Team
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Any
from ..config import get_cortex_detection_config


def detect_cortex_in_child_contours(metrics: List[Dict[str, Any]], inverted_image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detect cortex in child contours based on texture analysis.

    Args:
        metrics (List[Dict]): List of contour metrics with surface classification
        inverted_image (np.ndarray): Inverted binary image for texture analysis

    Returns:
        List[Dict]: Updated metrics with cortex detection and relabeling
    """
    # Load configuration
    config = get_cortex_detection_config()

    # Check if cortex detection is enabled
    if not config.get('enabled', True):
        logging.info("Cortex detection is disabled in configuration")
        return metrics

    logging.info("Starting cortex detection in child contours")

    try:
        # Separate parents, children, and grandchildren
        parents = [m for m in metrics if m["parent"] == m["scar"]]
        children = [m for m in metrics if m["parent"] != m["scar"]]

        # Create mapping of parent labels to surface areas for percentage calculations
        parent_areas = {}
        for parent in parents:
            parent_label = parent.get("scar", "")
            parent_area = parent.get("area", 0)
            parent_areas[parent_label] = parent_area

        # Process each child contour for cortex detection
        cortex_children = []
        non_cortex_children = []

        for child in children:
            if not child.get("contour"):
                logging.warning(f"Child contour {child.get('scar', 'unknown')} missing contour data")
                non_cortex_children.append(child)
                continue

            # Convert contour back to numpy array
            contour_array = np.array(child["contour"], dtype=np.int32)

            # Detect cortex based on texture analysis
            is_cortex = _detect_cortex_texture(contour_array, inverted_image, config)

            if is_cortex:
                cortex_children.append(child)
                logging.debug(f"Detected cortex in child: {child.get('scar', 'unknown')}")
            else:
                non_cortex_children.append(child)

        # Relabel cortex children with cortex_N labeling
        cortex_count = 0
        for cortex_child in cortex_children:
            cortex_count += 1
            original_scar_label = cortex_child.get("scar", "unknown")
            parent_label = cortex_child.get("parent", "")
            parent_area = parent_areas.get(parent_label, 0)
            cortex_area = cortex_child.get("area", 0)

            # Calculate cortex percentage of parent surface
            cortex_percentage = (cortex_area / parent_area * 100) if parent_area > 0 else 0

            # Update labels and add cortex-specific metrics
            cortex_child["scar"] = f"cortex {cortex_count}"
            cortex_child["surface_feature"] = f"cortex {cortex_count}"
            cortex_child["cortex_area"] = cortex_area
            cortex_child["cortex_percentage"] = round(cortex_percentage, 2)
            cortex_child["is_cortex"] = True

            logging.info(f"Relabeled {original_scar_label} -> {cortex_child['scar']} "
                        f"(area: {cortex_area}, {cortex_percentage:.1f}% of surface)")

        # Group remaining non-cortex children by surface type for proper numbering
        surface_children = {"Dorsal": [], "Platform": [], "Lateral": [], "Ventral": []}

        # Create parent surface type mapping
        parent_surface_map = {}
        for parent in parents:
            parent_label = parent.get("scar", "")
            surface_type = parent.get("surface_type", "Unknown")
            parent_surface_map[parent_label] = surface_type

        for child in non_cortex_children:
            parent_label = child.get("parent", "")
            parent_surface = parent_surface_map.get(parent_label, "Unknown")
            if parent_surface in surface_children:
                surface_children[parent_surface].append(child)

        # Renumber remaining children (non-cortex) to maintain proper sequence
        final_non_cortex_children = []

        # Dorsal children → scars (excluding cortex)
        for i, child in enumerate(surface_children["Dorsal"]):
            child["scar"] = f"scar {i+1}"
            child["surface_feature"] = f"scar {i+1}"
            child["is_cortex"] = False
            final_non_cortex_children.append(child)

        # Platform children → marks (excluding cortex) - currently filtered out
        for child in surface_children["Platform"]:
            area = child.get("area", 0)
            logging.info(f"Excluding platform child (area={area}) as likely empty space boundary")
            continue

        # Lateral children → edges (excluding cortex)
        for i, child in enumerate(surface_children["Lateral"]):
            child["scar"] = f"edge_{i+1}"
            child["surface_feature"] = f"edge_{i+1}"
            child["is_cortex"] = False
            final_non_cortex_children.append(child)

        # Combine all results
        final_metrics = parents + cortex_children + final_non_cortex_children

        logging.info(f"Cortex detection completed: {len(cortex_children)} cortex areas detected, "
                    f"{len(final_non_cortex_children)} non-cortex children processed")

        return final_metrics

    except Exception as e:
        logging.error(f"Error in cortex detection: {e}")
        # Return original metrics on error to prevent pipeline failure
        return metrics


def _detect_cortex_texture(contour: np.ndarray, inverted_image: np.ndarray, config: Dict[str, Any]) -> bool:
    """
    Detect cortex based on texture analysis within a contour region.

    Cortex is characterized by:
    - High density of small features (stippled texture)
    - Irregular, non-directional patterns
    - Higher local variance in pixel intensities

    Args:
        contour (np.ndarray): Contour array defining the region
        inverted_image (np.ndarray): Inverted binary image

    Returns:
        bool: True if cortex is detected, False otherwise
    """
    try:
        # Create mask for the contour region
        mask = np.zeros(inverted_image.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)

        # Extract region of interest
        roi = cv2.bitwise_and(inverted_image, mask)

        # Get bounding rectangle for analysis
        x, y, w, h = cv2.boundingRect(contour)
        roi_cropped = roi[y:y+h, x:x+w]
        mask_cropped = mask[y:y+h, x:x+w]

        if roi_cropped.size == 0:
            return False

        # Feature 1: Density of small connected components (stippling detection)
        # Find connected components in the ROI
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi_cropped, connectivity=8)

        # Count small components (excluding background)
        small_components = 0
        total_component_area = 0

        for i in range(1, num_labels):  # Skip background (label 0)
            component_area = stats[i, cv2.CC_STAT_AREA]
            total_component_area += component_area

            # Count small components (stippling characteristic)
            if 1 <= component_area <= 50:  # Small dots/stipples
                small_components += 1

        # Calculate stippling density
        roi_area = cv2.countNonZero(mask_cropped)
        if roi_area == 0:
            return False

        stippling_density = small_components / roi_area * 1000  # Normalized density

        # Feature 2: Texture variance (cortex has more irregular texture)
        # Apply Gaussian blur and calculate local variance
        if roi_cropped.shape[0] > 5 and roi_cropped.shape[1] > 5:
            blurred = cv2.GaussianBlur(roi_cropped, (5, 5), 0)
            variance = np.var(blurred[mask_cropped > 0]) if np.any(mask_cropped) else 0
        else:
            variance = 0

        # Feature 3: Edge density (cortex has many small edges from stippling)
        edges = cv2.Canny(roi_cropped, 50, 150)
        edge_density = np.sum(edges > 0) / roi_area if roi_area > 0 else 0

        # Decision thresholds from configuration
        stippling_threshold = config.get('stippling_density_threshold', 0.2)
        variance_threshold = config.get('texture_variance_threshold', 100)
        edge_threshold = config.get('edge_density_threshold', 0.05)

        cortex_detected = (
            stippling_density > stippling_threshold and  # High density of small components
            variance > variance_threshold and            # High texture variance
            edge_density > edge_threshold                # Significant edge density
        )

        logging.info(f"Cortex analysis: stippling_density={stippling_density:.3f}, "
                     f"variance={variance:.1f}, edge_density={edge_density:.3f} -> "
                     f"cortex={cortex_detected}")

        return cortex_detected

    except Exception as e:
        logging.error(f"Error in cortex texture detection: {e}")
        return False


def calculate_total_cortex_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate aggregate cortex metrics across all surfaces.

    Args:
        metrics (List[Dict]): List of contour metrics with cortex detection

    Returns:
        Dict[str, float]: Aggregate cortex statistics
    """
    cortex_areas = [m.get("cortex_area", 0) for m in metrics if m.get("is_cortex", False)]

    total_cortex_area = sum(cortex_areas)
    cortex_count = len(cortex_areas)
    average_cortex_size = total_cortex_area / cortex_count if cortex_count > 0 else 0

    return {
        "total_cortex_area": total_cortex_area,
        "cortex_count": cortex_count,
        "average_cortex_size": round(average_cortex_size, 2)
    }