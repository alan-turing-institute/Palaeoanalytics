import cv2
import numpy as np
import logging
import os
from ..arrow_detection import analyze_child_contour_for_arrow


def process_nested_arrows(sorted_contours, hierarchy, original_contours, metrics, image_shape, image_dpi=None):
    """
    Process nested children contours for arrow detection and update parent scar metrics.
    This contains the logic that was previously in calculate_contour_metrics().

    Parameters
    ----------
    sorted_contours : dict
        {"parents":…, "children":…, "nested_children":…}
    hierarchy : np.ndarray
        contour hierarchy array
    original_contours : list
        list of all extracted contours
    metrics : list of dict
        existing metrics to update with arrow information
    image_shape : tuple
        shape of the source image (h, w)
    image_dpi : float, optional
        DPI of the image being processed, used for scaling detection parameters

    Returns
    -------
    metrics : list of dict
        updated metrics with arrow information
    """
    # Create mapping for scar metrics
    scar_metrics = {}  # Map from contour key to scar entry
    scar_entries = {}  # Map from scar label to entry

    # Build contour index mapping
    contour_index_map = {}
    all_sorted_contours = (
        sorted_contours["parents"] +
        sorted_contours["children"] +
        sorted_contours.get("nested_children", [])
    )

    for contour in all_sorted_contours:
        for i, orig_cnt in enumerate(original_contours):
            if np.array_equal(contour, orig_cnt):
                contour_key = str(contour.tobytes())
                contour_index_map[contour_key] = i
                break

    # Map scar metrics for arrow integration
    for metric in metrics:
        if metric["parent"] != metric["scar"]:  # This is a scar
            # Find the corresponding contour
            for cnt in sorted_contours["children"]:
                if abs(cv2.contourArea(cnt) - metric["area"]) < 1.0:  # Match by area
                    contour_key = str(cnt.tobytes())
                    scar_metrics[contour_key] = metric
                    scar_entries[metric["scar"]] = metric
                    break

    # Skip nested children processing if there are no direct children
    if not scar_metrics and sorted_contours.get("nested_children", []):
        logging.info("Skipping nested children processing as there are no direct children (scars)")
        return metrics

    # Process nested children (detect arrows and update parent scars)
    for ni, cnt in enumerate(sorted_contours.get("nested_children", [])):
        contour_key = str(cnt.tobytes())
        nested_idx = contour_index_map.get(contour_key)
        if nested_idx is None or nested_idx >= len(hierarchy):
            logging.warning(f"Could not find nested contour {ni} in original contours or hierarchy")
            continue

        parent_idx = hierarchy[nested_idx][3]  # Get parent contour index

        # Find which scar this belongs to
        parent_scar = None
        parent_contour_key = None

        # Try to find parent contour in original contours
        if parent_idx < len(original_contours):
            parent_contour = original_contours[parent_idx]
            parent_contour_key = str(parent_contour.tobytes())

        # Check if the parent contour is in our scar metrics
        if parent_contour_key in scar_metrics:
            parent_scar = scar_metrics[parent_contour_key]
        else:
            # Find parent through hierarchy relationships if not direct
            found = False
            for idx, h in enumerate(hierarchy):
                if idx == parent_idx and h[3] != -1 and idx < len(original_contours):
                    grandparent_idx = h[3]
                    for cidx, ch in enumerate(hierarchy):
                        if ch[3] == grandparent_idx and cidx < len(original_contours):
                            cidx_contour = original_contours[cidx]
                            cidx_key = str(cidx_contour.tobytes())
                            if cidx_key in scar_metrics:
                                parent_scar = scar_metrics[cidx_key]
                                found = True
                                break
                    if found:
                        break

        if parent_scar is None:
            logging.debug(f"Could not find parent scar for nested contour {ni}")
            continue

        # Get image name without extension
        if hasattr(image_shape, 'filename'):
            image_name = os.path.splitext(image_shape.filename)[0]
        elif isinstance(image_shape, str):
            image_name = os.path.splitext(image_shape)[0]
        else:
            # Default name
            image_name = f"image_{ni}"

        # Create debug directory
        debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                "image_debug",
                                image_name)
        os.makedirs(debug_dir, exist_ok=True)

        # Create temporary entry for arrow detection
        temp_entry = {
            "scar": f"nested_{ni}",
            "debug_dir": debug_dir
        }

        # Run arrow detection
        logging.debug(f"Running arrow detection on nested contour {ni}")
        result = analyze_child_contour_for_arrow(cnt, temp_entry, image_shape, image_dpi)

        # If arrow detected, update the parent scar's entry
        if result:
            logging.info(f"Arrow detected in nested contour {ni} (parent: {parent_scar['scar']}) with angle {result.get('compass_angle', 'unknown')}")
            parent_scar.update({
                "has_arrow": True,
                "arrow_angle_rad": round(result["angle_rad"], 0),
                "arrow_angle_deg": round(result["angle_deg"], 0),
                "arrow_angle": round(result["compass_angle"], 0),
                "arrow_tip": result["arrow_tip"],
                "arrow_back": result["arrow_back"]
            })
        else:
            logging.debug(f"No arrow detected in nested contour {ni}")

    return metrics


def detect_arrows_independently(original_contours, metrics, image, image_dpi=None):
    """
    Detect arrows in contours independently of hierarchy relationships.
    Searches all contours for potential arrows and associates them with the
    appropriate scar metrics.

    Parameters
    ----------
    original_contours : list
        List of all detected contours.
    metrics : list
        List of metric dictionaries for parents and scars.
    image : ndarray
        Original image for visualization and debug purposes.
    image_dpi : float, optional
        DPI of the image being processed, used for scaling detection parameters.

    Returns
    -------
    metrics : list
        Updated metrics list with arrow information added.
    """
    # Find all scar metrics (they have a parent that's not themselves)
    scar_metrics = [m for m in metrics if m["parent"] != m["scar"]]

    # Track which scars have arrows assigned
    scars_with_arrows = set()

    # For each contour, test if it could be an arrow
    logging.info("Starting independent arrow detection on all contours...")

    # First, find parent contours to exclude them from arrow detection
    parent_contours = set()
    parent_indices = []
    for i, m in enumerate(metrics):
        if m["parent"] == m["scar"]:  # This is a parent contour
            for j, cnt in enumerate(original_contours):
                # Find matching contour for this parent metric
                if cv2.contourArea(cnt) == m["area"]:
                    parent_contours.add(j)
                    parent_indices.append(j)
                    break

    # Now find scar contours to exclude them too
    scar_contours = set()
    scar_indices = []
    for i, m in enumerate(metrics):
        if m["parent"] != m["scar"] and "parent" in m:  # This is a scar
            for j, cnt in enumerate(original_contours):
                if j in parent_contours:
                    continue  # Skip already identified parent contours
                # Find matching contour for this scar metric
                if cv2.contourArea(cnt) == m["area"]:
                    scar_contours.add(j)
                    scar_indices.append(j)
                    break

    # Create debug directory for this image
    debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        "image_debug",
                        "independent_arrow_detection")
    os.makedirs(debug_dir, exist_ok=True)

    # Prepare scar contours for containment testing
    scar_contour_map = {}  # Maps scar label to its contour
    for idx in scar_indices:
        cnt = original_contours[idx]
        # Find which metric this corresponds to
        for m in scar_metrics:
            if abs(cv2.contourArea(cnt) - m["area"]) < 1.0:  # Allow small rounding differences
                scar_contour_map[m["scar"]] = cnt
                break

    arrow_candidates = []
    # Test all contours that aren't parents or scars
    for i, cnt in enumerate(original_contours):
        if i in parent_contours or i in scar_contours:
            continue  # Skip parents and scars

        # Basic filtering criteria for arrow candidates (must have reasonable area and solidity)
        area = cv2.contourArea(cnt)
        if area < 1.0:  # Skip very tiny contours
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            continue

        solidity = area / hull_area
        if solidity < 0.4 or solidity > 0.9:  # Solidity range for arrow shapes
            continue

        # Create debug entry
        temp_entry = {
            "scar": f"candidate_{i}",
            "debug_dir": os.path.join(debug_dir, f"contour_{i}")
        }
        os.makedirs(temp_entry["debug_dir"], exist_ok=True)

        # Try arrow detection
        result = analyze_child_contour_for_arrow(cnt, temp_entry, image, image_dpi)

        if result:
            logging.info(f"Independent arrow detection found arrow in contour {i} with angle {result.get('compass_angle', 'unknown')}")
            arrow_candidates.append((i, cnt, result))

    # Now assign arrows to scars based on containment
    for arrow_idx, arrow_cnt, arrow_result in arrow_candidates:
        # Calculate arrow centroid
        M = cv2.moments(arrow_cnt)
        if M["m00"] > 0:
            arrow_cx = M["m10"] / M["m00"]
            arrow_cy = M["m01"] / M["m00"]
        else:
            x, y, w, h = cv2.boundingRect(arrow_cnt)
            arrow_cx, arrow_cy = x + w/2, y + h/2

        # Find containing scar
        containing_scar = None
        smallest_area = float('inf')

        for scar_label, scar_cnt in scar_contour_map.items():
            # Check if the arrow centroid is inside this scar
            if cv2.pointPolygonTest(scar_cnt, (arrow_cx, arrow_cy), False) >= 0:
                scar_area = cv2.contourArea(scar_cnt)
                if scar_area < smallest_area:
                    smallest_area = scar_area
                    containing_scar = scar_label

        if containing_scar and containing_scar not in scars_with_arrows:
            # Found a containing scar that doesn't have an arrow yet
            # Update the scar's metrics with arrow information
            for metric in metrics:
                if metric["scar"] == containing_scar:
                    logging.info(f"Assigning arrow from contour {arrow_idx} to scar {containing_scar}")
                    metric.update({
                        "has_arrow": True,
                        "arrow_angle_rad": round(arrow_result["angle_rad"], 0),
                        "arrow_angle_deg": round(arrow_result["angle_deg"], 0),
                        "arrow_angle": round(arrow_result["compass_angle"], 0),
                        "arrow_tip": arrow_result["arrow_tip"],
                        "arrow_back": arrow_result["arrow_back"]
                    })
                    scars_with_arrows.add(containing_scar)
                    break

    logging.info(f"Independent arrow detection completed. Found and assigned {len(scars_with_arrows)} arrows.")
    return metrics


def integrate_arrows(sorted_contours, hierarchy, original_contours, metrics, image_shape, image_dpi=None):
    """
    Main orchestrator function for arrow integration.
    Handles both nested children arrow detection and independent arrow detection.

    Parameters
    ----------
    sorted_contours : dict
        {"parents":…, "children":…, "nested_children":…}
    hierarchy : np.ndarray
        contour hierarchy array
    original_contours : list
        list of all extracted contours
    metrics : list of dict
        existing metrics to update with arrow information
    image_shape : tuple or ndarray
        shape of the source image (h, w) or the image itself
    image_dpi : float, optional
        DPI of the image being processed, used for scaling detection parameters

    Returns
    -------
    metrics : list of dict
        updated metrics with arrow information
    """
    # First, try to detect arrows in nested children
    metrics = process_nested_arrows(sorted_contours, hierarchy, original_contours, metrics, image_shape, image_dpi)

    # Then, run independent arrow detection for scars without arrows
    scars_without_arrows = [m for m in metrics if m["parent"] != m["scar"] and not m.get('has_arrow', False)]

    if scars_without_arrows:
        logging.info(f"Found {len(scars_without_arrows)} scars without arrows. Running independent detection.")
        scar_labels = [m["scar"] for m in scars_without_arrows]
        logging.info(f"Scars without arrows: {scar_labels}")

        # For independent detection, we need the actual image, not just the shape
        if hasattr(image_shape, 'shape'):
            # image_shape is actually an image
            image = image_shape
        else:
            # We only have the shape, create a dummy image for independent detection
            # This shouldn't happen in normal usage, but provides a fallback
            logging.warning("Image shape provided instead of image for independent arrow detection")
            image = np.zeros(image_shape, dtype=np.uint8)

        metrics = detect_arrows_independently(original_contours, metrics, image, image_dpi)
    else:
        logging.info("All scars already have arrows or no scars found. Skipping independent detection.")

    return metrics
