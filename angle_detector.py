import cv2
import numpy as np
import math
import os
from collections import defaultdict

def detect_arrows(image_path, output_path=None, debug=True, debug_dir="image_debug/"):
    """
    Arrow detection using convex hull analysis and half-space solidity.

    Args:
        image_path: Path to input image
        output_path: Path to save output image
        debug: Whether to generate debug images
        debug_dir: Directory to save debug images
    """
    # Create debug directory if it doesn't exist
    if debug and debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Create a copy for drawing results
    result_image = image.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    if debug:
        cv2.imwrite(f"{debug_dir}01_binary.jpg", binary)

    # Find contours with hierarchy
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Store information about detected arrows
    arrows_data = []

    # Debug image to visualize all detected shapes
    if debug:
        all_contours = image.copy()
        filtered_contours = image.copy()
        hull_analysis = image.copy()
        defect_analysis = image.copy()
        triangle_detection = image.copy()
        final_result = image.copy()
        halfspace_analysis = image.copy()
        solidity_analysis = image.copy()

    # Create a debug log text file
    if debug:
        log_file = open(f"{debug_dir}arrow_detection_log.txt", "w")
        log_file.write(f"Arrow detection analysis for {image_path}\n")
        log_file.write(f"Found {len(contours)} contours\n\n")

    # Draw all contours for debugging
    if debug:
        for i, contour in enumerate(contours):
            # Draw with a random color
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            cv2.drawContours(all_contours, [contour], 0, color, 2)
            # Label the contour
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Draw text with black outline for better visibility
                cv2.putText(all_contours, f"#{i}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Black outline
                cv2.putText(all_contours, f"#{i}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # White text

        cv2.imwrite(f"{debug_dir}02_all_contours.jpg", all_contours)

    # Process each contour
    for i, contour in enumerate(contours):
        # Log contour information
        if debug:
            log_file.write(f"Processing contour #{i}\n")

        # Skip the outermost contour (the container)
        if hierarchy is not None and hierarchy[0][i][3] == -1:
            if debug:
                log_file.write(f"  Skipped: Top-level contour\n\n")
            continue

        # Filter by size
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if debug:
            log_file.write(f"  Area: {area} pixels\n")
            log_file.write(f"  Perimeter: {perimeter:.2f} pixels\n")

        if area < 50 or area > 5000:
            if debug:
                log_file.write(f"  Skipped: Area outside range (50-5000)\n\n")
            continue

        # Get the centroid
        M = cv2.moments(contour)
        if M["m00"] == 0:
            if debug:
                log_file.write(f"  Skipped: Zero moment\n\n")
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        if debug:
            log_file.write(f"  Centroid: ({cx}, {cy})\n")

        # Calculate aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1

        if debug:
            log_file.write(f"  Bounding rect: {w}x{h} pixels\n")
            log_file.write(f"  Aspect ratio: {aspect_ratio:.2f}\n")

        # # Check if it's elongated (arrows typically are)
        # if aspect_ratio < 1.5:
        #     if debug:
        #         log_file.write(f"  Skipped: Aspect ratio too low (< 1.5)\n\n")
        #     continue

        # Calculate solidity (area / convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0

        if debug:
            log_file.write(f"  Hull area: {hull_area} pixels\n")
            log_file.write(f"  Solidity: {solidity:.2f}\n")

            # Draw the contour and its convex hull
            cv2.drawContours(hull_analysis, [contour], 0, (0, 255, 0), 2)  # Green for original contour
            cv2.drawContours(hull_analysis, [hull], 0, (255, 0, 0), 2)  # Blue for convex hull

        # Check solidity range for arrows
        if solidity < 0.4 or solidity > 0.9:
            if debug:
                log_file.write(f"  Skipped: Solidity outside range (0.4-0.9)\n\n")
            continue

        # Find convexity defects to identify the triangular head
        try:
            hull_indices = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, hull_indices)

            # Store significant defects
            significant_defects = []

            if defects is not None:
                for j in range(defects.shape[0]):
                    s, e, f, d = defects[j, 0]
                    if d / 256.0 > 1.0:  # Only count significant defects
                        start = tuple(contour[s][0])
                        end = tuple(contour[e][0])
                        far = tuple(contour[f][0])
                        significant_defects.append((start, end, far, d / 256.0))

                        if debug:
                            # Draw defect points and lines
                            cv2.circle(defect_analysis, far, 5, (0, 0, 255), -1)  # Red for defect points
                            cv2.line(defect_analysis, start, end, (255, 0, 0), 1)  # Blue for convex hull segment

            if debug:
                log_file.write(f"  Significant defects: {len(significant_defects)}\n")
                cv2.drawContours(defect_analysis, [contour], 0, (0, 255, 0), 2)  # Green for contour
                cv2.drawContours(defect_analysis, [hull], 0, (255, 255, 0), 1)  # Yellow for hull

            # If no significant defects, may not be an arrow or the arrow head isn't well-defined
            if len(significant_defects) < 2:
                if debug:
                    log_file.write(f"  Skipped: Not enough significant defects for a triangular head\n\n")
                continue

            # Sort defects by depth (largest first)
            significant_defects.sort(key=lambda x: x[3], reverse=True)

            # Get the defect points (the 'far' point of each defect)
            defect_points = [defect[2] for defect in significant_defects]

            # Calculate distances between all pairs of defect points
            defect_pairs = []
            for a in range(len(defect_points)):
                for b in range(a+1, len(defect_points)):
                    p1 = defect_points[a]
                    p2 = defect_points[b]
                    distance = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                    defect_pairs.append((p1, p2, distance))

            # Sort pairs by distance (largest first as it likely represents the base of the triangle)
            defect_pairs.sort(key=lambda x: x[2], reverse=True)

            if not defect_pairs:
                if debug:
                    log_file.write(f"  Skipped: Could not form defect pairs\n\n")
                continue

            # Get the pair with the largest distance as the base of the triangle
            triangle_base_p1, triangle_base_p2, base_length = defect_pairs[0]

            # Calculate the midpoint of the triangle base
            base_midpoint = ((triangle_base_p1[0] + triangle_base_p2[0]) // 2,
                            (triangle_base_p1[1] + triangle_base_p2[1]) // 2)

            # Draw the triangle base in debug image
            if debug:
                cv2.line(triangle_detection, triangle_base_p1, triangle_base_p2, (255, 0, 255), 2)  # Magenta for triangle base
                cv2.circle(triangle_detection, triangle_base_p1, 5, (255, 0, 0), -1)  # Blue for base point 1
                cv2.circle(triangle_detection, triangle_base_p2, 5, (0, 0, 255), -1)  # Red for base point 2
                cv2.circle(triangle_detection, base_midpoint, 5, (0, 255, 255), -1)  # Yellow for base midpoint
                cv2.drawContours(triangle_detection, [contour], 0, (0, 255, 0), 2)  # Green for contour

            # Define the base line for dividing space
            base_vector = np.array([triangle_base_p2[0] - triangle_base_p1[0],
                                  triangle_base_p2[1] - triangle_base_p1[1]])

            # Create a normal vector to the base (perpendicular)
            normal = np.array([-base_vector[1], base_vector[0]])

            # Dictionaries to store points and masks for each half-space
            halfspace_points = defaultdict(list)
            halfspace_masks = {}

            # Create separate binary masks for each half-space
            h, w = binary.shape
            mask1 = np.zeros((h, w), dtype=np.uint8)
            mask2 = np.zeros((h, w), dtype=np.uint8)

            # Check each contour point and assign to half-spaces
            for point in contour:
                p = point[0]
                # Vector from base_midpoint to point
                vec = np.array([p[0] - base_midpoint[0], p[1] - base_midpoint[1]])

                # Dot product determines which side of the line the point is on
                side = np.dot(vec, normal)

                if side > 0:
                    halfspace_points[1].append(p)
                else:
                    halfspace_points[2].append(p)

            # If either half-space has very few points, this might not be a good arrow candidate
            if len(halfspace_points[1]) < 5 or len(halfspace_points[2]) < 5:
                if debug:
                    log_file.write(f"  Skipped: Unbalanced half-spaces\n\n")
                continue

            # Create contours for each half-space
            if halfspace_points[1] and halfspace_points[2]:
                halfspace1_contour = np.array([halfspace_points[1]], dtype=np.int32)
                halfspace2_contour = np.array([halfspace_points[2]], dtype=np.int32)

                # Draw contours on separate masks
                cv2.drawContours(mask1, [halfspace1_contour], 0, 255, -1)
                cv2.drawContours(mask2, [halfspace2_contour], 0, 255, -1)

                # Calculate the area of each half-space
                area1 = cv2.countNonZero(mask1)
                area2 = cv2.countNonZero(mask2)

                # Calculate convex hulls for each half-space
                if len(halfspace1_contour[0]) >= 3:  # Need at least 3 points for a hull
                    hull1 = cv2.convexHull(halfspace1_contour)
                    hull_mask1 = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(hull_mask1, [hull1], 0, 255, -1)
                    hull_area1 = cv2.countNonZero(hull_mask1)
                    solidity1 = area1 / hull_area1 if hull_area1 > 0 else 0
                else:
                    solidity1 = 0

                if len(halfspace2_contour[0]) >= 3:  # Need at least 3 points for a hull
                    hull2 = cv2.convexHull(halfspace2_contour)
                    hull_mask2 = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(hull_mask2, [hull2], 0, 255, -1)
                    hull_area2 = cv2.countNonZero(hull_mask2)
                    solidity2 = area2 / hull_area2 if hull_area2 > 0 else 0
                else:
                    solidity2 = 0

                if debug:
                    log_file.write(f"  Half-space 1 points: {len(halfspace_points[1])}\n")
                    log_file.write(f"  Half-space 2 points: {len(halfspace_points[2])}\n")
                    log_file.write(f"  Half-space 1 area: {area1}\n")
                    log_file.write(f"  Half-space 2 area: {area2}\n")
                    log_file.write(f"  Half-space 1 solidity: {solidity1:.3f}\n")
                    log_file.write(f"  Half-space 2 solidity: {solidity2:.3f}\n")

                    # Visualize the half-spaces
                    halfspace_vis = np.zeros((h, w, 3), dtype=np.uint8)
                    halfspace_vis[mask1 > 0] = [0, 255, 0]  # Green for half-space 1
                    halfspace_vis[mask2 > 0] = [0, 0, 255]  # Red for half-space 2
                    cv2.line(halfspace_vis, triangle_base_p1, triangle_base_p2, (255, 255, 0), 2)  # Yellow for base line

                    # Overlay on the original image for reference
                    overlay = image.copy()
                    alpha = 0.5
                    mask = (halfspace_vis > 0).any(axis=2)
                    overlay[mask] = cv2.addWeighted(overlay[mask], alpha, halfspace_vis[mask], 1 - alpha, 0)
                    cv2.line(overlay, triangle_base_p1, triangle_base_p2, (255, 255, 0), 2)  # Yellow for base line
                    cv2.imwrite(f"{debug_dir}halfspace_overlay_{i}.jpg", overlay)

                    # Create solidity visualization
                    solidity_vis = image.copy()
                    cv2.putText(solidity_vis, f"Solidity 1: {solidity1:.3f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                    cv2.putText(solidity_vis, f"Solidity 1: {solidity1:.3f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(solidity_vis, f"Solidity 2: {solidity2:.3f}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                    cv2.putText(solidity_vis, f"Solidity 2: {solidity2:.3f}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imwrite(f"{debug_dir}solidity_analysis_{i}.jpg", solidity_vis)

                # Determine which half-space is more likely to be the shaft based on solidity
                # Shaft is typically more solid/filled than the triangular head
                if solidity1 > solidity2:
                    shaft_halfspace = 1
                    tip_halfspace = 2
                else:
                    shaft_halfspace = 2
                    tip_halfspace = 1

                if debug:
                    log_file.write(f"  Shaft half-space: {shaft_halfspace} (more solid)\n")
                    log_file.write(f"  Tip half-space: {tip_halfspace} (less solid)\n")

                # Find the point in the tip half-space furthest from the base midpoint
                max_dist = -1
                triangle_tip = None

                for p in halfspace_points[tip_halfspace]:
                    dist = np.sqrt((p[0] - base_midpoint[0])**2 + (p[1] - base_midpoint[1])**2)
                    if dist > max_dist:
                        max_dist = dist
                        triangle_tip = tuple(p)

                if triangle_tip is None:
                    if debug:
                        log_file.write(f"  Skipped: Could not find triangle tip\n\n")
                    continue

                # Check if triangle height is sufficient
                if max_dist < 30:  # Adjust threshold as needed
                    if debug:
                        log_file.write(f"  Skipped: Triangle height too small (< 30)\n\n")
                    continue

                if debug:
                    cv2.circle(triangle_detection, triangle_tip, 5, (0, 255, 0), -1)  # Green for triangle tip
                    # ... rest of the debug code

                if debug:
                    cv2.circle(triangle_detection, triangle_tip, 5, (0, 255, 0), -1)  # Green for triangle tip
                    cv2.line(triangle_detection, base_midpoint, triangle_tip, (255, 255, 0), 2)  # Yellow for direction vector
                    log_file.write(f"  Triangle base: {triangle_base_p1} to {triangle_base_p2}\n")
                    log_file.write(f"  Triangle tip: {triangle_tip}\n")
                    log_file.write(f"  Triangle base length: {base_length:.2f}\n")
                    log_file.write(f"  Triangle height: {max_dist:.2f}\n")

                # Now we have the triangle's tip and base
                # The arrow direction is from tip to base midpoint (shaft direction)
                arrow_tip = base_midpoint  # The base midpoint is actually the tip of the arrow
                arrow_back = triangle_tip  # The triangle tip is the back of the arrow

                # Calculate the arrow direction vector and angle
                dx = arrow_tip[0] - arrow_back[0]
                dy = arrow_tip[1] - arrow_back[1]

                # Calculate angle in degrees
                angle_rad = math.atan2(dy, dx)
                angle_deg = math.degrees(angle_rad)

                # Convert to compass angle (0° north, 90° east, 180° south, 270° west)
                compass_angle = (270 +  angle_deg) % 360

                if debug:
                    log_file.write(f"  Compass angle: {compass_angle:.1f}\n")

                # Store the arrow data
                arrows_data.append({
                    'contour_id': i,
                    'centroid': (cx, cy),
                    'arrow_back': arrow_back,
                    'arrow_tip': arrow_tip,
                    'angle': compass_angle,
                    'triangle_base': (triangle_base_p1, triangle_base_p2),
                    'triangle_height': max_dist,
                    'triangle_base_length': base_length,
                    'tip_halfspace': tip_halfspace,
                    'shaft_halfspace': shaft_halfspace,
                    'solidity1': solidity1,
                    'solidity2': solidity2,
                    'aspect_ratio': aspect_ratio,
                    'area': area
                })

                # Draw to result image
                cv2.drawContours(result_image, [contour], 0, (255, 255, 0), 1)  # Yellow contour
                cv2.arrowedLine(result_image, arrow_back, arrow_tip, (0, 255, 0), 2)  # Green arrow

                # Draw the angle with black outline for visibility
                text_pos_x = cx + 20
                text_pos_y = cy - 10
                cv2.putText(result_image, f"{compass_angle:.1f}",
                           (text_pos_x, text_pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)  # Black outline
                cv2.putText(result_image, f"{compass_angle:.1f}",
                           (text_pos_x, text_pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)  # Red text

                # Draw to filtered contours image
                cv2.drawContours(filtered_contours, [contour], 0, (0, 255, 255), 2)
                cv2.putText(filtered_contours, f"#{i}", (cx, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Black outline
                cv2.putText(filtered_contours, f"#{i}", (cx, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # White text

                # Draw to final result image
                cv2.drawContours(final_result, [contour], 0, (255, 255, 0), 1)  # Contour
                cv2.drawContours(final_result, [hull], 0, (0, 0, 255), 1)  # Hull in red
                cv2.arrowedLine(final_result, arrow_back, arrow_tip, (0, 255, 0), 2)  # Direction from back to tip

                # Draw triangle points
                cv2.circle(final_result, triangle_base_p1, 4, (255, 0, 0), -1)  # Blue
                cv2.circle(final_result, triangle_base_p2, 4, (0, 0, 255), -1)  # Red
                cv2.circle(final_result, triangle_tip, 4, (0, 255, 0), -1)  # Green

                # Add angle text with black outline
                cv2.putText(final_result, f"{compass_angle:.1f}",
                           (text_pos_x, text_pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)  # Black outline
                cv2.putText(final_result, f"{compass_angle:.1f}",
                           (text_pos_x, text_pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)  # Red text

                if debug:
                    log_file.write(f"  Arrow accepted\n\n")
            else:
                if debug:
                    log_file.write(f"  Skipped: Failed to create half-space contours\n\n")

        except Exception as e:
            if debug:
                log_file.write(f"  Error in convexity analysis: {str(e)}\n\n")
            continue

    # Save debug images
    if debug:
        cv2.imwrite(f"{debug_dir}03_filtered_contours.jpg", filtered_contours)
        cv2.imwrite(f"{debug_dir}04_hull_analysis.jpg", hull_analysis)
        cv2.imwrite(f"{debug_dir}05_defect_analysis.jpg", defect_analysis)
        cv2.imwrite(f"{debug_dir}06_triangle_detection.jpg", triangle_detection)
        cv2.imwrite(f"{debug_dir}08_final_result.jpg", final_result)

        # Close the log file
        log_file.write(f"Total arrows detected: {len(arrows_data)}\n")
        log_file.close()

    print(f"Detected {len(arrows_data)} arrows")

    # Print detailed information about each arrow for debugging
    for idx, arrow in enumerate(arrows_data):
        print(f"Arrow #{idx} (Contour #{arrow['contour_id']}):")
        print(f"  Triangle base length: {arrow.get('triangle_base_length', 'N/A'):.2f}")
        print(f"  Triangle height: {arrow.get('triangle_height', 'N/A'):.2f}")
        print(f"  Half-space 1 solidity: {arrow.get('solidity1', 'N/A'):.3f}")
        print(f"  Half-space 2 solidity: {arrow.get('solidity2', 'N/A'):.3f}")
        print(f"  Shaft half-space: {arrow.get('shaft_halfspace', 'N/A')}")
        print(f"  Angle: {arrow['angle']:.1f}°")

    # Save the result if output path is provided
    if output_path:
        cv2.imwrite(output_path, result_image)

    return result_image, arrows_data

def main(image_path, output_dir="image_debug/"):
    """
    Main function to demonstrate arrow detection

    Args:
        image_path: Path to input image
        output_dir: Directory to save output and debug images
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get base filename without extension
    base_name = os.path.basename(image_path)
    base_name_no_ext = os.path.splitext(base_name)[0]

    # Create a dedicated directory for this image's debug output
    image_debug_dir = os.path.join(output_dir, f"{base_name_no_ext}/")

    # Detect arrows
    result_image, arrows_data = detect_arrows(
        image_path,
        os.path.join(output_dir, f"{base_name_no_ext}_result.jpg"),
        debug=True,
        debug_dir=image_debug_dir
    )

    # Print arrow information
    print(f"\nResults for image: {image_path}")
    print(f"Debug output: {image_debug_dir}")
    print(f"Arrows detected: {len(arrows_data)}")

    # Create a summary CSV
    with open(os.path.join(output_dir, f"{base_name_no_ext}_summary.csv"), "w") as f:
        f.write("ID,X,Y,Angle,BaseLength,TriangleHeight,Solidity1,Solidity2,ShaftHalfspace,Area,AspectRatio\n")
        for i, arrow in enumerate(arrows_data):
            f.write(f"{i},{arrow['centroid'][0]},{arrow['centroid'][1]},{arrow['angle']:.1f},"
                   f"{arrow.get('triangle_base_length', 0):.2f},{arrow.get('triangle_height', 0):.2f},"
                   f"{arrow.get('solidity1', 0):.3f},{arrow.get('solidity2', 0):.3f},{arrow.get('shaft_halfspace', 0)},"
                   f"{arrow['area']:.1f},{arrow['aspect_ratio']:.2f}\n")

    return arrows_data

if __name__ == "__main__":
    import sys

    # Default test case
    if len(sys.argv) <= 1:
        print("Using default test image")
        main("pylithics/data/images/awbari.png")
    else:
        image_path = sys.argv[1]
        output_dir = "image_debug/"

        if len(sys.argv) > 2:
            output_dir = sys.argv[2]

        main(image_path, output_dir)