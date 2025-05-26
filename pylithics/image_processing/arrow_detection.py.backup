import os
import cv2
import numpy as np
import math
from collections import defaultdict
import logging

# Module‐level constants (reference values calibrated for 300 DPI)
REFERENCE_DPI = 300.0              # reference DPI for threshold calibration
REF_MIN_AREA = 1                 # reference minimum contour area (in pixels)
REF_MIN_DEFECT_DEPTH = 2           # reference minimum depth of convexity defect (in pixels)
REF_SOLIDITY_BOUNDS = (0.4, 1.0)  # reference acceptable solidity range for shape
REF_MIN_TRIANGLE_HEIGHT = 8       # reference minimum height of triangular head
REF_MIN_SIGNIFICANT_DEFECTS = 2    # reference minimum number of significant defects to consider

# These will be set dynamically based on image DPI
MIN_AREA = REF_MIN_AREA
MIN_DEFECT_DEPTH = REF_MIN_DEFECT_DEPTH
SOLIDITY_BOUNDS = REF_SOLIDITY_BOUNDS
MIN_TRIANGLE_HEIGHT = REF_MIN_TRIANGLE_HEIGHT
MIN_SIGNIFICANT_DEFECTS = REF_MIN_SIGNIFICANT_DEFECTS

def scale_parameters_for_dpi(image_dpi):
    """
    Scale detection parameters based on image DPI relative to reference DPI.

    Parameters
    ----------
    image_dpi : float
        DPI of the current image being processed

    Returns
    -------
    dict
        Dictionary of scaled parameters
    """
    if image_dpi is None or image_dpi <= 0:
        logging.warning("Invalid DPI value. Using reference thresholds.")
        return {
            "min_area": REF_MIN_AREA,
            "min_defect_depth": REF_MIN_DEFECT_DEPTH,
            "solidity_bounds": REF_SOLIDITY_BOUNDS,
            "min_triangle_height": REF_MIN_TRIANGLE_HEIGHT,
            "min_significant_defects": REF_MIN_SIGNIFICANT_DEFECTS
        }

    # FIXED SCALING LOGIC: We want thresholds to be lower at lower DPIs
    # For example, a 100 DPI image should have 1/9 the minimum area of a 300 DPI image
    linear_scale = image_dpi / REFERENCE_DPI  # <-- This is the key change (reversed ratio)
    area_scale = linear_scale * linear_scale

    # Scale parameters with safety factors to ensure detection works across DPIs
    return {
        "min_area": REF_MIN_AREA * area_scale * 0.7,  # Add 30% safety margin
        "min_defect_depth": REF_MIN_DEFECT_DEPTH * linear_scale * 0.8,  # Add 20% safety margin
        "solidity_bounds": REF_SOLIDITY_BOUNDS,  # Solidity is a ratio, so no scaling needed
        "min_triangle_height": REF_MIN_TRIANGLE_HEIGHT * linear_scale * 0.8,  # Add 20% safety margin
        "min_significant_defects": REF_MIN_SIGNIFICANT_DEFECTS  # Count-based, no scaling needed
    }

def analyze_child_contour_for_arrow(contour, entry, image, image_dpi=None):
    """
    Detect an arrow within a given contour and compute its orientation.

    Parameters
    ----------
    contour : ndarray
        Single child contour (as returned by cv2.findContours).
    entry : dict
        Metrics dictionary for this contour. May contain 'debug_dir' for
        writing debug images.
    image : ndarray
        Original image (grayscale or BGR) for overlaying debug visualizations.
    image_dpi : float, optional
        DPI of the image being processed. If provided, detection thresholds will be scaled.

    Returns
    -------
    dict or None
        If a valid arrow head is found, returns a dict with arrow properties.
        Returns None if no valid arrow is detected.
    """
    debug_dir = entry.get('debug_dir')
    contour_id = entry.get('scar', 'unknown')

    # Scale parameters based on image DPI
    params = scale_parameters_for_dpi(image_dpi)
    min_area = params["min_area"]
    min_defect_depth = params["min_defect_depth"]
    solidity_bounds = params["solidity_bounds"]
    min_triangle_height = params["min_triangle_height"]
    min_significant_defects = params["min_significant_defects"]

    # Create debug directory if needed
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        debug_log = open(os.path.join(debug_dir, 'arrow_detection_log.txt'), 'w')
        debug_log.write(f"Arrow detection analysis for contour {contour_id}\n")
        if image_dpi:
            debug_log.write(f"Image DPI: {image_dpi}, scaling applied\n")
            debug_log.write(f"Scaled thresholds: min_area={min_area:.2f}, min_defect_depth={min_defect_depth:.2f}, " +
                          f"min_triangle_height={min_triangle_height:.2f}\n")
        # arrow debug info
    if debug_dir:
        debug_log.write(f"Scaled parameters with DPI={image_dpi}:\n")
        debug_log.write(f"  min_area={min_area:.2f}\n")
        debug_log.write(f"  min_defect_depth={min_defect_depth:.2f}\n")
        debug_log.write(f"  solidity_bounds={solidity_bounds}\n")
        debug_log.write(f"  min_triangle_height={min_triangle_height:.2f}\n")
        debug_log.write(f"  min_significant_defects={min_significant_defects}\n")
    else:
        debug_log = None

    # Helper function to log debug info
    def debug(message):
        if debug_log:
            debug_log.write(f"{message}\n")

    try:
        # Step 1: Basic area filtering (using scaled min_area)
        area = cv2.contourArea(contour)
        if area < min_area:
            debug(f"Failed: Area too small ({area:.2f} < {min_area:.2f})")
            return None

        debug(f"Area: {area:.2f} pixels")

        debug(f"Area test: PASSED ({area:.2f} >= {min_area:.2f})")

        # Step 2: Calculate solidity (area / convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0

        debug(f"Solidity: {solidity:.2f}")

        if solidity < solidity_bounds[0] or solidity > solidity_bounds[1]:
            debug(f"Failed: Solidity outside range {solidity_bounds}")
            return None

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0

        debug(f"Solidity: {solidity:.2f}")
        # ADD THIS:
        debug(f"Solidity test: {solidity_bounds[0]:.2f} <= {solidity:.2f} <= {solidity_bounds[1]:.2f}")

        if solidity < solidity_bounds[0] or solidity > solidity_bounds[1]:
            debug(f"Failed: Solidity outside range {solidity_bounds}")
            return None
        # ADD THIS:
        debug(f"Solidity test: PASSED")

        # Step 3: Find significant convexity defects (using scaled min_defect_depth)
        significant_defects = find_significant_defects(contour, min_defect_depth)
        if not significant_defects or len(significant_defects) < min_significant_defects:
            debug(f"Failed: Not enough defects ({len(significant_defects) if significant_defects else 0} < {min_significant_defects})")
            return None

        # **Keep only the two deepest defects** and reject if more/less
        significant_defects = significant_defects[:2]  # already sorted deepest-first
        if len(significant_defects) != min_significant_defects:
            debug(f"Failed: Need exactly {min_significant_defects} defects, got {len(significant_defects)}")
            return None

        debug(f"Using top-two defects at depths {[d[3] for d in significant_defects]}")

        significant_defects = find_significant_defects(contour, min_defect_depth)
        if not significant_defects or len(significant_defects) < min_significant_defects:
            debug(f"Failed: Not enough defects ({len(significant_defects) if significant_defects else 0} < {min_significant_defects})")
            return None
        # ADD THIS:
        debug(f"Defect test: PASSED (found {len(significant_defects)} significant defects)")
        # ADD THIS - print defect depths:
        defect_depths = [defect[3] for defect in significant_defects]
        debug(f"Defect depths: {defect_depths}")

        # Step 4: Identify triangle base from defect pairs
        triangle_base_info = identify_triangle_base(significant_defects)
        if triangle_base_info is None:
            debug("Failed: Could not identify triangle base")
            return None

        triangle_base_p1, triangle_base_p2, base_midpoint, base_length = triangle_base_info
        debug(f"Triangle base length: {base_length:.2f} pixels")

        # Step 5: Analyze half-spaces
        halfspace_results = analyze_halfspaces(contour, triangle_base_info, image.shape if image is not None else None)
        if halfspace_results is None:
            debug("Failed: Half-space analysis failed")
            return None

        # Step 6: Identify which half-space is more likely to be the shaft vs. the tip
        shaft_halfspace, tip_halfspace, solidity1, solidity2 = halfspace_results
        debug(f"Half-space 1 solidity: {solidity1:.3f}")
        debug(f"Half-space 2 solidity: {solidity2:.3f}")
        debug(f"Shaft half-space: {shaft_halfspace} (more solid)")
        debug(f"Tip half-space: {tip_halfspace} (less solid)")

        # Step 7: Find the triangle tip (farthest point in tip half-space)
        halfspace_points = divide_contour_points(contour, triangle_base_info)
        triangle_tip = find_triangle_tip(halfspace_points[tip_halfspace], base_midpoint)
        if triangle_tip is None:
            debug("Failed: Could not find triangle tip")
            return None

        # Step 8: Calculate triangle height
        triangle_height = np.sqrt((triangle_tip[0] - base_midpoint[0])**2 +
                                  (triangle_tip[1] - base_midpoint[1])**2)

        debug(f"Triangle height: {triangle_height:.2f} pixels")

        if triangle_height < min_triangle_height:
            debug(f"Failed: Triangle height too small ({triangle_height:.2f} < {min_triangle_height:.2f})")
            return None

        triangle_height = np.sqrt((triangle_tip[0] - base_midpoint[0])**2 +
                          (triangle_tip[1] - base_midpoint[1])**2)

        debug(f"Triangle height: {triangle_height:.2f} pixels")
        # ADD THIS:
        debug(f"Triangle height test: {triangle_height:.2f} >= {min_triangle_height:.2f}")

        if triangle_height < min_triangle_height:
            debug(f"Failed: Triangle height too small ({triangle_height:.2f} < {min_triangle_height:.2f})")
            return None
        # ADD THIS:
        debug(f"Triangle height test: PASSED")

        # Step 9: Define arrow direction (from tip to base midpoint)
        arrow_tip = base_midpoint  # Base midpoint is the tip of the arrow
        arrow_back = triangle_tip  # Triangle tip is the back of the arrow

        # Step 10: Calculate arrow angles
        dx = arrow_tip[0] - arrow_back[0]
        dy = arrow_tip[1] - arrow_back[1]

        angle_rad = math.atan2(dy, dx)
        angle_deg = (math.degrees(angle_rad) + 360) % 360
        compass_angle = (270 + angle_deg) % 360

        debug(f"Arrow direction: from ({arrow_back}) to ({arrow_tip})")
        debug(f"Angle: {angle_deg:.2f}°, Compass: {compass_angle:.2f}°")

        # Step 11: Create debug visualizations if requested
        if debug_dir and image is not None:
            create_debug_visualizations(
                contour, hull, significant_defects, triangle_base_p1, triangle_base_p2,
                triangle_tip, arrow_back, arrow_tip, base_midpoint,
                compass_angle, image, debug_dir
            )

        # Close debug log
        if debug_log:
            debug_log.write("Arrow detection succeeded\n")
            debug_log.close()

        # Return arrow data in the format expected by PyLithics
        return {
            'arrow_back': arrow_back,
            'arrow_tip': arrow_tip,
            'angle_rad': round(angle_rad, 0),
            'angle_deg': round(angle_deg, 0),
            'compass_angle': round(compass_angle, 0),
        }

    except Exception as e:
        if debug_log:
            debug_log.write(f"Error in arrow detection: {str(e)}\n")
            debug_log.close()
        return None

def find_significant_defects(contour, min_defect_depth=MIN_DEFECT_DEPTH):
    """
    Find significant convexity defects in the contour.

    Parameters
    ----------
    contour : ndarray
        Contour to analyze
    min_defect_depth : float
        Minimum depth threshold for significant defects

    Returns
    -------
    list or None
        List of significant defects as (start_point, end_point, far_point, depth)
        or None if no significant defects found
    """
    try:
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        if hull_indices is None or len(hull_indices) < 3:
            return None

        defects = cv2.convexityDefects(contour, hull_indices)
        if defects is None or defects.shape[0] < 2:
            return None

        # Extract significant defects (depth > threshold)
        significant_defects = []
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            depth = d / 256.0  # Convert to pixels
            if depth > min_defect_depth:
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                significant_defects.append((start, end, far, depth))

        # Sort by depth (deepest first)
        significant_defects.sort(key=lambda x: x[3], reverse=True)
        return significant_defects
    except:
        return None

def identify_triangle_base(defects):
    """
    Identify the likely base of a triangle from defect pairs.

    Parameters
    ----------
    defects : list
        List of significant defects

    Returns
    -------
    tuple or None
        (base_point1, base_point2, midpoint, length) or None if no valid base found
    """
    # Get defect points (the 'far' point of each defect)
    defect_points = [defect[2] for defect in defects]

    # Calculate distances between all pairs of defect points
    defect_pairs = []
    for i in range(len(defect_points)):
        for j in range(i+1, len(defect_points)):
            p1 = defect_points[i]
            p2 = defect_points[j]
            distance = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            defect_pairs.append((p1, p2, distance))

    # Sort pairs by distance (largest first - likely base of triangle)
    defect_pairs.sort(key=lambda x: x[2], reverse=True)

    if not defect_pairs:
        return None

    # Get the pair with the largest distance as the base
    base_p1, base_p2, base_length = defect_pairs[0]

    # Calculate the midpoint of the base
    base_midpoint = ((base_p1[0] + base_p2[0]) // 2,
                    (base_p1[1] + base_p2[1]) // 2)

    return (base_p1, base_p2, base_midpoint, base_length)

def divide_contour_points(contour, triangle_base_info):
    """
    Divide contour points into two half-spaces based on triangle base.

    Parameters
    ----------
    contour : ndarray
        Contour to divide
    triangle_base_info : tuple
        (base_p1, base_p2, base_midpoint, base_length)

    Returns
    -------
    dict
        Dictionary with keys 1, 2 containing points in each half-space
    """
    base_p1, base_p2, base_midpoint, _ = triangle_base_info

    # Define base vector and normal
    base_vector = np.array([base_p2[0] - base_p1[0], base_p2[1] - base_p1[1]])
    normal = np.array([-base_vector[1], base_vector[0]])

    # Divide points into half-spaces
    halfspace_points = defaultdict(list)

    for point in contour:
        p = point[0]
        # Vector from base_midpoint to point
        vec = np.array([p[0] - base_midpoint[0], p[1] - base_midpoint[1]])

        # Determine which side of the line the point is on
        side = np.dot(vec, normal)

        if side > 0:
            halfspace_points[1].append(p)
        else:
            halfspace_points[2].append(p)

    return halfspace_points

def analyze_halfspaces(contour, triangle_base_info, image_shape=None):
    """
    Analyze the solidity of each half-space of the contour.

    Parameters
    ----------
    contour : ndarray
        Contour to analyze
    triangle_base_info : tuple
        (base_p1, base_p2, base_midpoint, base_length)
    image_shape : tuple or None
        (height, width) of the image, used for creating masks

    Returns
    -------
    tuple or None
        (shaft_halfspace, tip_halfspace, solidity1, solidity2) or None if analysis fails
    """
    halfspace_points = divide_contour_points(contour, triangle_base_info)

    # If either half-space has very few points, not a good arrow candidate
    if len(halfspace_points[1]) < 5 or len(halfspace_points[2]) < 5:
        return None

    # For mask-based calculations, we need the image shape
    if image_shape is None:
        # If no image shape, use a simpler method to estimate solidity
        try:
            # Create hull for each half-space
            hull1 = cv2.convexHull(np.array(halfspace_points[1], dtype=np.int32).reshape(-1, 1, 2))
            hull2 = cv2.convexHull(np.array(halfspace_points[2], dtype=np.int32).reshape(-1, 1, 2))

            # Calculate areas
            area1 = cv2.contourArea(np.array(halfspace_points[1], dtype=np.int32).reshape(-1, 1, 2))
            area2 = cv2.contourArea(np.array(halfspace_points[2], dtype=np.int32).reshape(-1, 1, 2))

            hull_area1 = cv2.contourArea(hull1)
            hull_area2 = cv2.contourArea(hull2)

            solidity1 = area1 / hull_area1 if hull_area1 > 0 else 0
            solidity2 = area2 / hull_area2 if hull_area2 > 0 else 0
        except:
            return None
    else:
        # Create masks for each half-space
        h, w = image_shape[:2]
        mask1 = np.zeros((h, w), dtype=np.uint8)
        mask2 = np.zeros((h, w), dtype=np.uint8)

        # Convert point lists to contours
        try:
            contour1 = np.array(halfspace_points[1], dtype=np.int32).reshape(-1, 1, 2)
            contour2 = np.array(halfspace_points[2], dtype=np.int32).reshape(-1, 1, 2)

            # Draw contours on masks
            cv2.drawContours(mask1, [contour1], 0, 255, -1)
            cv2.drawContours(mask2, [contour2], 0, 255, -1)

            # Calculate areas
            area1 = cv2.countNonZero(mask1)
            area2 = cv2.countNonZero(mask2)

            # Calculate hulls
            hull1 = cv2.convexHull(contour1)
            hull2 = cv2.convexHull(contour2)

            # Create hull masks
            hull_mask1 = np.zeros((h, w), dtype=np.uint8)
            hull_mask2 = np.zeros((h, w), dtype=np.uint8)

            cv2.drawContours(hull_mask1, [hull1], 0, 255, -1)
            cv2.drawContours(hull_mask2, [hull2], 0, 255, -1)

            # Calculate hull areas
            hull_area1 = cv2.countNonZero(hull_mask1)
            hull_area2 = cv2.countNonZero(hull_mask2)

            # Calculate solidity
            solidity1 = area1 / hull_area1 if hull_area1 > 0 else 0
            solidity2 = area2 / hull_area2 if hull_area2 > 0 else 0
        except:
            return None

    # Determine which half-space is more likely to be the shaft (more solid)
    if solidity1 > solidity2:
        shaft_halfspace = 1
        tip_halfspace = 2
    else:
        shaft_halfspace = 2
        tip_halfspace = 1

    return (shaft_halfspace, tip_halfspace, solidity1, solidity2)

def find_triangle_tip(tip_halfspace_points, base_midpoint):
    """
    Find the point in the tip half-space furthest from the base midpoint.

    Parameters
    ----------
    tip_halfspace_points : list
        List of points in the tip half-space
    base_midpoint : tuple
        (x, y) coordinates of the base midpoint

    Returns
    -------
    tuple or None
        (x, y) coordinates of the triangle tip or None if not found
    """
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

def create_debug_visualizations(contour, hull, significant_defects, base_p1, base_p2,
                               triangle_tip, arrow_back, arrow_tip, base_midpoint,
                               compass_angle, image, debug_dir):
    """
    Create debug visualizations for arrow detection.

    Parameters
    ----------
    contour : ndarray
        The contour being analyzed
    hull : ndarray
        Convex hull of the contour
    significant_defects : list
        List of significant convexity defects
    base_p1, base_p2 : tuple
        Points forming the triangle base
    triangle_tip : tuple
        Tip of the triangle
    arrow_back, arrow_tip : tuple
        Arrow direction points
    base_midpoint : tuple
        Midpoint of the triangle base
    compass_angle : float
        Arrow angle in compass degrees
    image : ndarray
        Original image for visualization
    debug_dir : str
        Directory to save debug images
    """
    # Ensure debug directory exists
    os.makedirs(debug_dir, exist_ok=True)

    # Create a visualization image
    vis = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)

    # Draw contour and hull
    cv2.drawContours(vis, [contour], 0, (0, 255, 0), 2)
    cv2.drawContours(vis, [hull], 0, (0, 0, 255), 1)

    # Draw convexity defects
    for defect in significant_defects:
        start, end, far, _ = defect
        cv2.circle(vis, far, 5, (255, 0, 0), -1)

    # Draw triangle base and tip
    cv2.line(vis, base_p1, base_p2, (255, 0, 255), 2)
    cv2.circle(vis, triangle_tip, 5, (0, 255, 255), -1)

    # Draw arrow direction
    cv2.arrowedLine(vis, arrow_back, arrow_tip, (0, 255, 0), 2)

    # Draw angle
    text_pos = (arrow_tip[0] + 10, arrow_tip[1] - 10)
    cv2.putText(vis, f"{compass_angle:.1f}°", text_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    cv2.putText(vis, f"{compass_angle:.1f}°", text_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # Save the visualization
    cv2.imwrite(os.path.join(debug_dir, "arrow_debug.png"), vis)
