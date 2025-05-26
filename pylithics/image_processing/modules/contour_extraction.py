import cv2
import numpy as np
import logging
import os
import yaml

# For the config loading in extract_contours_with_hierarchy
from ..config import get_contour_filtering_config


def extract_contours_with_hierarchy(inverted_image, image_id, output_dir):
    """
    Extract contours and hierarchy using cv2.RETR_TREE, exclude the image border.
    Uses minimum area from the config file.

    Parameters
    ----------
    inverted_image : ndarray
        Preprocessed binary/grayscale image in which to find contours.
    image_id : str
        Unique identifier for this image (used for debug directory naming).
    output_dir : str
        Base path under which per‐image debug folders will be created.

    Returns
    -------
    valid_contours : list of ndarray
        All contours whose bounding box does not touch the image border.
    valid_hierarchy : ndarray
        Corresponding hierarchy entries for valid_contours.
    """

    # Load config to get minimum area
    filtering_config = get_contour_filtering_config()
    min_contour_area = filtering_config['min_area']
    logging.info(f"Using minimum contour area: {min_contour_area} pixels for image {image_id}")

    # 1) find all contours + raw hierarchy
    contours, hierarchy = cv2.findContours(
        inverted_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    hierarchy = hierarchy[0] if hierarchy is not None else None

    if not contours:
        logging.warning("No contours found in image %s", image_id)
        return [], None

    # Log the number of raw contours found
    logging.info(f"Found {len(contours)} raw contours in image {image_id}")

    # 2) filter out those touching the border
    height, width = inverted_image.shape
    valid_contours, valid_hierarchy = [], []

    # Create index mapping from old to new indices
    old_to_new_idx = {}
    new_idx = 0

    for old_idx, (cnt, h) in enumerate(zip(contours, hierarchy)):
        x, y, w, h_box = cv2.boundingRect(cnt)
        if x > 0 and y > 0 and x + w < width and y + h_box < height:
            valid_contours.append(cnt)
            valid_hierarchy.append(h.copy())  # Copy to avoid modifying original
            old_to_new_idx[old_idx] = new_idx
            new_idx += 1

    # Update parent indices in the hierarchy to reflect new indexing
    for i, h in enumerate(valid_hierarchy):
        if h[3] != -1:  # If has a parent
            if h[3] in old_to_new_idx:
                valid_hierarchy[i][3] = old_to_new_idx[h[3]]
            else:
                # Parent was filtered out, make this a root
                valid_hierarchy[i][3] = -1

    valid_hierarchy = np.array(valid_hierarchy)

    # Log how many contours remain after border filtering
    logging.info(f"After border filtering: {len(valid_contours)} contours remain in image {image_id}")

    # 3) filter out small contours using the minimum area from config
    valid_contours, valid_hierarchy = filter_contours_by_min_area(
        valid_contours, valid_hierarchy, min_contour_area
    )

    if not valid_contours:
        logging.warning("No valid contours remain after filtering in image %s", image_id)
        return [], None

    logging.info(
        "Extracted %d valid contours: %d parents/%d children total",
        len(valid_contours),
        np.sum(valid_hierarchy[:, 3] == -1),
        np.sum(valid_hierarchy[:, 3] != -1)
    )

    return valid_contours, valid_hierarchy


def sort_contours_by_hierarchy(contours, hierarchy, exclude_nested_flags=None):
    """
    Sort contours into parents, children, and nested children based on hierarchy.
    This version includes robust bounds checking.

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
    # Handle empty contours or hierarchy
    if not contours or hierarchy is None or len(hierarchy) == 0:
        return {"parents": [], "children": [], "nested_children": []}

    parents, children, nested = [], [], []

    # Create exclude_nested_flags if not provided
    if exclude_nested_flags is None:
        exclude_nested_flags = [False] * len(contours)

    # Ensure exclude_nested_flags is the right length
    if len(exclude_nested_flags) != len(contours):
        logging.warning("exclude_nested_flags length mismatch: %d vs %d contours. Using defaults.",
                        len(exclude_nested_flags), len(contours))
        exclude_nested_flags = [False] * len(contours)

    # First, identify parent contours (those with no parent)
    for i, h in enumerate(hierarchy):
        if i >= len(exclude_nested_flags) or i >= len(contours):
            continue  # Skip if index is out of bounds

        if exclude_nested_flags[i]:
            continue

        parent_idx = h[3]
        if parent_idx == -1:  # No parent
            parents.append(contours[i])

    # Second, identify direct children (first level)
    for i, h in enumerate(hierarchy):
        if i >= len(exclude_nested_flags) or i >= len(contours):
            continue  # Skip if index is out of bounds

        if exclude_nested_flags[i]:
            continue

        parent_idx = h[3]
        if parent_idx != -1:  # Has a parent
            # Check if the parent is a root contour
            if parent_idx < len(hierarchy) and hierarchy[parent_idx][3] == -1:
                children.append(contours[i])
            else:
                # This is a nested child (child of a child)
                nested.append(contours[i])

    logging.info(
        "Sorted contours: %d parents, %d children, %d nested",
        len(parents), len(children), len(nested)
    )
    return {"parents": parents, "children": children, "nested_children": nested}


def sort_contours_by_hierarchy(contours, hierarchy, exclude_nested_flags=None):
    """
    Sort contours into parents, children, and nested children based on hierarchy.
    This version includes robust bounds checking.

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
    # Handle empty contours or hierarchy
    if not contours or hierarchy is None or len(hierarchy) == 0:
        return {"parents": [], "children": [], "nested_children": []}

    parents, children, nested = [], [], []

    # Create exclude_nested_flags if not provided
    if exclude_nested_flags is None:
        exclude_nested_flags = [False] * len(contours)

    # Ensure exclude_nested_flags is the right length
    if len(exclude_nested_flags) != len(contours):
        logging.warning("exclude_nested_flags length mismatch: %d vs %d contours. Using defaults.",
                        len(exclude_nested_flags), len(contours))
        exclude_nested_flags = [False] * len(contours)

    # First, identify parent contours (those with no parent)
    for i, h in enumerate(hierarchy):
        if i >= len(exclude_nested_flags) or i >= len(contours):
            continue  # Skip if index is out of bounds

        if exclude_nested_flags[i]:
            continue

        parent_idx = h[3]
        if parent_idx == -1:  # No parent
            parents.append(contours[i])

    # Second, identify direct children (first level)
    for i, h in enumerate(hierarchy):
        if i >= len(exclude_nested_flags) or i >= len(contours):
            continue  # Skip if index is out of bounds

        if exclude_nested_flags[i]:
            continue

        parent_idx = h[3]
        if parent_idx != -1:  # Has a parent
            # Check if the parent is a root contour
            if parent_idx < len(hierarchy) and hierarchy[parent_idx][3] == -1:
                children.append(contours[i])
            else:
                # This is a nested child (child of a child)
                nested.append(contours[i])

    logging.info(
        "Sorted contours: %d parents, %d children, %d nested",
        len(parents), len(children), len(nested)
    )
    return {"parents": parents, "children": children, "nested_children": nested}

def hide_nested_child_contours(contours, hierarchy):
    """
    Flag only first-level child contours whose parent has exactly one child.
    Do not flag nested (depth ≥2) contours so they can be processed for arrow detection.
    This version includes extensive bounds checking to prevent index errors.
    """
    flags = [False] * len(contours)

    # Safety check for empty contours or hierarchy
    if not contours or hierarchy is None or len(hierarchy) == 0:
        return flags

    # Count direct children for each parent
    child_counts = {}
    for i, h in enumerate(hierarchy):
        if i >= len(contours):
            continue  # Skip if index out of bounds

        p = h[3]
        if p != -1:
            child_counts[p] = child_counts.get(p, 0) + 1

    # Flag single-child first-level scars only
    for i, h in enumerate(hierarchy):
        if i >= len(contours):
            continue  # Skip if index out of bounds

        parent_idx = h[3]
        # Skip if parent index invalid
        if parent_idx == -1 or parent_idx >= len(hierarchy):
            continue

        # only flag if it's a first-level child and its parent has exactly one child
        if parent_idx != -1 and hierarchy[parent_idx][3] == -1 and child_counts.get(parent_idx, 0) == 1:
            if i < len(flags):
                flags[i] = True

    logging.info("Flagged %d single-child contours for exclusion.", sum(flags))
    return flags


def filter_contours_by_min_area(contours, hierarchy, min_area=1.0):
    """
    Filter out contours with area less than the minimum threshold.

    Args:
        contours (list): List of contours to filter
        hierarchy (numpy.ndarray): Hierarchy array corresponding to contours
        min_area (float): Minimum contour area in pixels (default: 1.0)

    Returns:
        tuple: (filtered_contours, filtered_hierarchy)
    """
    if not contours or hierarchy is None:
        return [], None

    filtered_indices = []
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) >= min_area:
            filtered_indices.append(i)

    filtered_contours = [contours[i] for i in filtered_indices]
    filtered_hierarchy = hierarchy[filtered_indices] if len(filtered_indices) > 0 else None

    return filtered_contours, filtered_hierarchy