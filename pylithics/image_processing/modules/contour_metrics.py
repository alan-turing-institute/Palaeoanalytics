import cv2
import numpy as np
import logging
import sys
import os

def calculate_contour_metrics(sorted_contours, hierarchy, original_contours, image_shape):
    """
    Calculate basic geometric metrics for parent and child contours.

    Parameters
    ----------
    sorted_contours : dict
        {"parents":…, "children":…, "nested_children":…}
    hierarchy : np.ndarray
        contour hierarchy array
    original_contours : list
        list of all extracted contours
    image_shape : tuple
        shape of the source image (h, w)

    Returns
    -------
    metrics : list of dict
        basic geometric metrics for contours
    """

    metrics = []
    parent_map = {}

    # Create a mapping between filtered contours and original contours
    # This is needed because the hierarchy indices refer to original contours
    contour_index_map = {}  # Maps contour to its index in original_contours

    # Build map of all contours in sorted_contours
    all_sorted_contours = (
        sorted_contours["parents"] +
        sorted_contours["children"] +
        sorted_contours.get("nested_children", [])
    )

    # Create a mapping from contours to their original indices
    for contour in all_sorted_contours:
        for i, orig_cnt in enumerate(original_contours):
            if np.array_equal(contour, orig_cnt):
                contour_key = str(contour.tobytes())  # Use bytes as key
                contour_index_map[contour_key] = i
                break

    # Process parents
    for pi, cnt in enumerate(sorted_contours["parents"]):
        contour_key = str(cnt.tobytes())
        idx = contour_index_map.get(contour_key)
        if idx is None:
            logging.warning(f"Could not find parent contour {pi} in original contours")
            continue

        lab = f"parent {pi+1}"
        parent_map[idx] = lab
        area = round(cv2.contourArea(cnt), 2)
        peri = round(cv2.arcLength(cnt, True), 2)

        # Safely calculate centroid
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = round(M["m10"] / M["m00"], 2)
            cy = round(M["m01"] / M["m00"], 2)
        else:
            bounding_rect = cv2.boundingRect(cnt)
            cx, cy = round(bounding_rect[0] + bounding_rect[2]/2, 2), round(bounding_rect[1] + bounding_rect[3]/2, 2)

        # Calculate Y-axis aligned width and height
        contour_points = cnt.reshape(-1, 2)

        # Height: full Y-axis span
        min_y = np.min(contour_points[:, 1])
        max_y = np.max(contour_points[:, 1])
        y_axis_height = round(max_y - min_y, 2)

        # Width: maximum horizontal span at any Y-level
        max_width_at_y = 0.0
        unique_y_coords = np.unique(contour_points[:, 1])

        for y in unique_y_coords:
            # Find all points at this Y-coordinate
            points_at_y = contour_points[contour_points[:, 1] == y]

            if len(points_at_y) > 0:
                # Find min and max X-coordinates at this Y-level
                min_x = np.min(points_at_y[:, 0])
                max_x = np.max(points_at_y[:, 0])
                width_at_y = max_x - min_x

                # Update maximum width
                max_width_at_y = max(max_width_at_y, width_at_y)

        y_axis_width = round(max_width_at_y, 2)

        # Calculate max length and width (object-oriented measurements)
        max_len = max_wid = 0
        p1 = p2 = None
        for i in range(len(cnt)):
            for j in range(i+1, len(cnt)):
                a = cnt[i][0]; b = cnt[j][0]
                d = np.linalg.norm(a - b)
                if d > max_len:
                    max_len, p1, p2 = d, a, b

        # Calculate perpendicular width (max_width)
        if p1 is not None and p2 is not None:
            v = p2 - p1
            perp = np.array([-v[1], v[0]], dtype=float)
            perp /= np.linalg.norm(perp)
            widths = [abs(np.dot(pt[0]-p1, perp)) for pt in cnt]
            max_wid = max(widths)

        ml, mw = round(max_len, 2), round(max_wid, 2)

        # Get bounding box for legacy fields
        x, y, bbox_w, bbox_h = cv2.boundingRect(cnt)

        metrics.append({
            "parent": lab, "scar": lab,
            "centroid_x": cx, "centroid_y": cy,
            "technical_width": y_axis_width,
            "technical_length": y_axis_height,
            "area": area, "aspect_ratio": round(y_axis_height/y_axis_width, 2) if y_axis_width > 0 else None,
            "bounding_box_x": x, "bounding_box_y": y,
            "bounding_box_width": bbox_w, "bounding_box_height": bbox_h,
            "max_length": ml, "max_width": mw,
            "contour": cnt.tolist(),
            "perimeter": peri
        })

    # Process children/scars
    for ci, cnt in enumerate(sorted_contours["children"]):
        contour_key = str(cnt.tobytes())
        idx = contour_index_map.get(contour_key)
        if idx is None:
            logging.warning(f"Could not find child contour {ci} in original contours")
            continue

        # Get parent using hierarchy
        if idx < len(hierarchy):
            parent_idx = hierarchy[idx][3]
            pl = parent_map.get(parent_idx, "Unknown")
        else:
            logging.warning(f"Child contour index {idx} out of bounds for hierarchy")
            pl = "Unknown"

        lab = f"child {ci+1}"  # Temporary label - will be updated after surface classification
        area = round(cv2.contourArea(cnt), 2)

        # Safe centroid calculation
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = round(M["m10"] / M["m00"], 2)
            cy = round(M["m01"] / M["m00"], 2)
        else:
            bounding_rect = cv2.boundingRect(cnt)
            cx, cy = round(bounding_rect[0] + bounding_rect[2]/2, 2), round(bounding_rect[1] + bounding_rect[3]/2, 2)

        # Calculate Y-axis aligned width and height for children
        contour_points = cnt.reshape(-1, 2)

        # Height: full Y-axis span
        min_y = np.min(contour_points[:, 1])
        max_y = np.max(contour_points[:, 1])
        y_axis_height = round(max_y - min_y, 2)

        # Width: maximum horizontal span at any Y-level
        max_width_at_y = 0.0
        unique_y_coords = np.unique(contour_points[:, 1])

        for y in unique_y_coords:
            # Find all points at this Y-coordinate
            points_at_y = contour_points[contour_points[:, 1] == y]

            if len(points_at_y) > 0:
                # Find min and max X-coordinates at this Y-level
                min_x = np.min(points_at_y[:, 0])
                max_x = np.max(points_at_y[:, 0])
                width_at_y = max_x - min_x

                # Update maximum width
                max_width_at_y = max(max_width_at_y, width_at_y)

        y_axis_width = round(max_width_at_y, 2)

        # Calculate max length and width for children
        max_len = max_wid = 0
        p1 = p2 = None
        for i in range(len(cnt)):
            for j in range(i+1, len(cnt)):
                a = cnt[i][0]; b = cnt[j][0]
                d = np.linalg.norm(a - b)
                if d > max_len:
                    max_len, p1, p2 = d, a, b

        if p1 is not None and p2 is not None:
            v = p2 - p1
            perp = np.array([-v[1], v[0]], dtype=float)
            perp /= np.linalg.norm(perp)
            widths = [abs(np.dot(pt[0]-p1, perp)) for pt in cnt]
            max_wid = max(widths)

        ml, mw = round(max_len, 2), round(max_wid, 2)

        # Get bounding box for legacy fields
        x, y, bbox_w, bbox_h = cv2.boundingRect(cnt)

        entry = {
            "parent": pl, "scar": lab,
            "centroid_x": cx, "centroid_y": cy,
            "width": y_axis_width,  # Y-axis aligned width
            "height": y_axis_height,  # Y-axis aligned height
            "area": area, "aspect_ratio": round(y_axis_height/y_axis_width, 2) if y_axis_width > 0 else None,
            "max_length": ml, "max_width": mw,
            "bounding_box_x": x, "bounding_box_y": y,
            "bounding_box_width": bbox_w, "bounding_box_height": bbox_h,
            "contour": cnt.tolist()  # Store contour data for visualization
        }
        metrics.append(entry)

    return metrics

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
            "technical_width": round(metric["technical_width"] * conversion_factor, 2),
            "technical_length": round(metric["technical_length"] * conversion_factor, 2),
            "area": round(metric["area"] * (conversion_factor ** 2), 2),  # Area scales quadratically
        })
    return converted_metrics