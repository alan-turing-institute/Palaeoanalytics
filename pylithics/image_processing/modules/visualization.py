import cv2
import numpy as np
import pandas as pd
import os
import logging


def visualize_contours_with_hierarchy(contours, hierarchy, metrics, inverted_image, output_path):
    """
    Visualize contours with hierarchy, label them, and overlay detected arrows.

    Args:
        contours (list): List of contours (parents + children) in display order.
        hierarchy (ndarray): Contour hierarchy array.
        metrics (list): List of metric dicts, each may contain 'has_arrow', 'arrow_back', 'arrow_tip', 'arrow_angle'.
        inverted_image (ndarray): Inverted binary image (0=foreground, 255=background).
        output_path (str): File path to write the labeled image.
    """
    # Invert back to white background
    original = cv2.bitwise_not(inverted_image)
    # Make BGR image for color drawing
    labeled = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

    # Phase 1: Draw ALL contours and centroids first
    contour_info = []  # Store info for label drawing phase
    for i, cnt in enumerate(contours):
        if i >= len(metrics):
            continue

        m = metrics[i]
        parent_label = m["parent"]
        scar_label   = m["scar"]

        # Color & text
        if parent_label == scar_label:
            color = (153, 60, 94)   # purple for parents
            text  = m.get("surface_type", parent_label)
        elif "cortex " in scar_label.lower():
            color = (39, 48, 215)     # RGB(215,48,39) for cortex (distinct from scars) - BGR format
            text  = scar_label
        elif "scar" in scar_label.lower():
            color = (99, 184, 253)    # orange for scars (dorsal surface only)
            text  = scar_label
        elif "mark_" in scar_label.lower():
            color = (210, 171, 178)   # RGB 178,171,210 for platform marks
            text  = scar_label
        elif "edge_" in scar_label.lower():
            color = (193, 205, 128)   # RGB 128,205,193 for lateral edges
            text  = scar_label
        else:
            color = (128, 128, 128)   # gray for unknown/temporary labels
            text  = scar_label

        # Draw the contour
        cv2.drawContours(labeled, [cnt], -1, color, 2)

        # Compute and draw centroid (using same color as contour)
        M = cv2.moments(cnt)
        if M.get("m00", 0) != 0:
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
        else:
            x,y,w,h = cv2.boundingRect(cnt)
            cx,cy = x + w//2, y + h//2
        cv2.circle(labeled, (cx, cy), 4, color, -1)
        
        # Store info for label drawing phase (include contour color for border matching)
        contour_info.append((cnt, cx, cy, text, i, color))

    # Phase 2: Draw ALL labels on top of contours
    label_positions = []
    for cnt, cx, cy, text, i, contour_color in contour_info:

        # Place label with improved positioning to avoid overlaps
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.7  # Consistent font size for ALL labels
        thickness = 1     # Consistent thickness for ALL labels
        ts = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # Try multiple positions: right, left, above, below centroid with increased distances
        potential_positions = [
            (cx + 20, cy - 5),   # Right of centroid (increased distance)
            (cx - ts[0] - 20, cy - 5),  # Left of centroid (increased distance)
            (cx - ts[0]//2, cy - 30),   # Above centroid (increased distance)
            (cx - ts[0]//2, cy + 30),   # Below centroid (increased distance)
            (cx + 25, cy - 20),  # Top-right diagonal
            (cx - ts[0] - 25, cy - 20),  # Top-left diagonal
            (cx + 25, cy + 15),  # Bottom-right diagonal
            (cx - ts[0] - 25, cy + 15)   # Bottom-left diagonal
        ]
        
        # Find best position that avoids overlaps with labels and contour areas
        tx, ty = potential_positions[0]  # Default to right
        for pos_x, pos_y in potential_positions:
            # Check if this position overlaps with existing labels
            overlaps = False
            
            # Check overlap with existing labels
            for lx, ly, lw, lh in label_positions:
                if (pos_x < lx + lw + 8 and pos_x + ts[0] + 8 > lx and 
                    pos_y < ly + lh + 8 and pos_y + ts[1] + 8 > ly):
                    overlaps = True
                    break
            
            # Check if position is too close to contour outline (avoid contour line overlap)
            if not overlaps:
                # Create a small test region around the proposed label position
                test_region = np.array([
                    [pos_x - 5, pos_y - ts[1] - 5],
                    [pos_x + ts[0] + 5, pos_y - ts[1] - 5],
                    [pos_x + ts[0] + 5, pos_y + 5],
                    [pos_x - 5, pos_y + 5]
                ], dtype=np.int32)
                
                # Check if test region intersects with current contour outline
                distance_to_contour = cv2.pointPolygonTest(cnt, (pos_x + ts[0]//2, pos_y - ts[1]//2), True)
                if abs(distance_to_contour) < 8:  # Too close to contour edge
                    overlaps = True
            
            if not overlaps:
                tx, ty = pos_x, pos_y
                break
        
        label_positions.append((tx, ty, ts[0], ts[1]))
        
        # Ensure label stays within image bounds
        tx = max(5, min(tx, labeled.shape[1] - ts[0] - 5))
        ty = max(ts[1] + 5, min(ty, labeled.shape[0] - 5))
        
        # All labels use identical formatting (matching angle labels exactly)
        padding = 4
        text_bg_pt1 = (tx - padding, ty - ts[1] - padding)
        text_bg_pt2 = (tx + ts[0] + padding, ty + padding)
        
        # Clean up edge label formatting (edge_1 -> edge 1)
        if "edge_" in text.lower():
            text = text.replace("edge_", "edge ")
        
        # Determine border color based on label type
        if "scar" in text.lower():
            border_color = (99, 184, 253)      # Orange border for scar labels (matches contour color)
        elif text.lower() in ['dorsal', 'ventral', 'platform', 'lateral']:
            border_color = contour_color        # Use parent contour color for surface labels
        elif "edge " in text.lower():
            border_color = contour_color        # Use lateral contour color for edge labels
        elif "cortex" in text.lower():
            border_color = contour_color        # Use red contour color for cortex labels
        else:
            border_color = (0, 0, 0)           # Black border for all other labels (angles, etc.)
        
        # Draw white background with appropriate border color
        cv2.rectangle(labeled, text_bg_pt1, text_bg_pt2, (255, 255, 255), -1)  # White fill
        cv2.rectangle(labeled, text_bg_pt1, text_bg_pt2, border_color, 1)      # Colored border
        
        # Draw text in black with anti-aliasing (identical to angle labels)
        cv2.putText(labeled, text, (tx, ty), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)


    # Draw arrows for all detected arrow features
    for m in metrics:
        if m.get("has_arrow") and m.get("arrow_back") and m.get("arrow_tip"):
            # Convert to integer tuples if needed
            back = tuple(int(v) for v in m["arrow_back"])
            tip = tuple(int(v) for v in m["arrow_tip"])

            # Draw arrowed line (red)
            cv2.arrowedLine(
                labeled,
                back,
                tip,
                color=(0, 0, 255),
                thickness=2,
                tipLength=0.2
            )

            # Annotate compass bearing with better positioning
            angle = m.get("arrow_angle", None)
            if angle is not None:
                # Calculate shaft vector
                shaft_vector = np.array([tip[0] - back[0], tip[1] - back[1]])
                shaft_length = np.linalg.norm(shaft_vector)

                if shaft_length > 0:
                    # Calculate perpendicular vector (90 degrees clockwise from shaft)
                    perp_vector = np.array([shaft_vector[1], -shaft_vector[0]]) / shaft_length

                    # Use a larger offset distance (40 pixels) for better separation
                    offset_distance = 40
                    perp_offset = perp_vector * offset_distance

                    text = f"{int(angle)}deg"
                    font = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 0.7  # Identical to scar/cortex labels
                    thickness = 1     # Identical to scar/cortex labels

                    # Get text size
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                    # Try multiple positions to avoid overlap with existing labels
                    potential_angle_positions = []
                    
                    # Calculate multiple positions around the arrow shaft
                    for shaft_fraction in [0.2, 0.4, 0.6, 0.8]:  # Different positions along shaft
                        for offset_mult in [1, -1, 1.5, -1.5]:  # Both sides and further out
                            text_base_pos = (
                                int(back[0] + shaft_vector[0] * shaft_fraction),
                                int(back[1] + shaft_vector[1] * shaft_fraction)
                            )
                            current_offset = perp_offset * offset_mult
                            test_pos = (
                                int(text_base_pos[0] + current_offset[0]),
                                int(text_base_pos[1] + current_offset[1])
                            )
                            potential_angle_positions.append(test_pos)
                    
                    # Find best position that doesn't overlap with existing labels
                    text_pos = potential_angle_positions[0]  # Default
                    for candidate_pos in potential_angle_positions:
                        # Check if this position overlaps with existing scar/cortex labels
                        overlaps_with_existing = False
                        for lx, ly, lw, lh in label_positions:
                            if (candidate_pos[0] < lx + lw + 10 and candidate_pos[0] + text_width + 10 > lx and 
                                candidate_pos[1] < ly + lh + 10 and candidate_pos[1] + text_height + 10 > ly):
                                overlaps_with_existing = True
                                break
                        
                        # Check if position is within image bounds
                        if (candidate_pos[0] > 10 and candidate_pos[0] + text_width < labeled.shape[1] - 10 and
                            candidate_pos[1] > text_height + 10 and candidate_pos[1] < labeled.shape[0] - 10):
                            if not overlaps_with_existing:
                                text_pos = candidate_pos
                                break
                    
                    # Ensure final position is within bounds
                    text_pos = (
                        max(5, min(text_pos[0], labeled.shape[1] - text_width - 5)),
                        max(text_height + 5, min(text_pos[1], labeled.shape[0] - 5))
                    )

                    # Create background rectangle with identical formatting to scar/cortex labels
                    padding = 4  # Identical to scar/cortex labels
                    text_bg_pt1 = (text_pos[0] - padding, text_pos[1] - text_height - padding)
                    text_bg_pt2 = (text_pos[0] + text_width + padding, text_pos[1] + padding)

                    # Draw white background with black border (identical to scar/cortex labels)
                    cv2.rectangle(labeled, text_bg_pt1, text_bg_pt2, (255, 255, 255), -1)  # White fill
                    cv2.rectangle(labeled, text_bg_pt1, text_bg_pt2, (0, 0, 0), 1)        # Black border

                    # Draw text in black with anti-aliasing (identical to scar/cortex labels)
                    cv2.putText(
                        labeled,
                        text,
                        text_pos,
                        font,
                        font_scale,
                        (0, 0, 0),  # Black text
                        thickness,
                        cv2.LINE_AA
                    )
                    
                    # Add angle label position to tracking list to prevent future overlaps
                    label_positions.append((text_pos[0], text_pos[1], text_width, text_height))

    # Save the labeled image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, labeled)
    logging.info("Saved visualized contours with arrows to %s", output_path)


def save_measurements_to_csv(metrics, output_path, append=False):
    """
    Save contour metrics to a CSV file.
    """
    updated_data = []
    for metric in metrics:
        if metric["parent"] == metric["scar"]:
            surface_type = metric.get("surface_type", "NA")
            surface_feature = surface_type
        else:
            parent_surface_type = next(
                (m["surface_type"] for m in metrics if m["parent"] == metric["parent"] and m["parent"] == m["scar"]),
                "NA"
            )
            surface_type = parent_surface_type
            surface_feature = metric["scar"]

        # Process arrow-specific data with proper handling of coordinates
        has_arrow = metric.get("has_arrow", False)
        arrow_tip = metric.get("arrow_tip", None)
        arrow_back = metric.get("arrow_back", None)

        # Extract coordinates safely
        arrow_tip_x = arrow_tip[0] if isinstance(arrow_tip, (list, tuple)) and len(arrow_tip) >= 1 else "NA"
        arrow_tip_y = arrow_tip[1] if isinstance(arrow_tip, (list, tuple)) and len(arrow_tip) >= 2 else "NA"
        arrow_back_x = arrow_back[0] if isinstance(arrow_back, (list, tuple)) and len(arrow_back) >= 1 else "NA"
        arrow_back_y = arrow_back[1] if isinstance(arrow_back, (list, tuple)) and len(arrow_back) >= 2 else "NA"

        data_entry = {
            "image_id": metric.get("image_id", "NA"),
            "surface_type": surface_type,
            "surface_feature": surface_feature,
            "centroid_x": metric.get("centroid_x", "NA"),
            "centroid_y": metric.get("centroid_y", "NA"),
            "technical_width": metric.get("technical_width", "NA"),
            "technical_length": metric.get("technical_length", "NA"),
            "max_width": metric.get("max_width", "NA"),
            "max_length": metric.get("max_length", "NA"),
            "total_area": metric.get("area", "NA"),
            "aspect_ratio": metric.get("aspect_ratio", "NA"),
            "perimeter": metric.get("perimeter", "NA"),
            "distance_to_max_width": metric.get("distance_to_max_width", "NA"),
            "voronoi_num_cells": metric.get("voronoi_num_cells", "NA"),
            "convex_hull_width": metric.get("convex_hull_width", "NA"),
            "convex_hull_height": metric.get("convex_hull_height", "NA"),
            "convex_hull_area": metric.get("convex_hull_area", "NA"),
            "voronoi_cell_area": metric.get("voronoi_cell_area", "NA"),
            "top_area": metric.get("top_area", "NA"),
            "bottom_area": metric.get("bottom_area", "NA"),
            "left_area": metric.get("left_area", "NA"),
            "right_area": metric.get("right_area", "NA"),
            "vertical_symmetry": metric.get("vertical_symmetry", "NA"),
            "horizontal_symmetry": metric.get("horizontal_symmetry", "NA"),
            # Lateral surface measurements
            "lateral_convexity": metric.get("lateral_convexity", "NA"),
            # Cortex measurements
            "is_cortex": metric.get("is_cortex", False),
            "cortex_area": metric.get("cortex_area", "NA"),
            "cortex_percentage": metric.get("cortex_percentage", "NA"),
            # arrow data with explicit type handling
            "has_arrow": has_arrow,
            "arrow_angle": metric.get("arrow_angle", "NA"),
        }

        # Add additional arrow metrics if available
        if "triangle_base_length" in metric:
            data_entry["triangle_base_length"] = metric["triangle_base_length"]
        if "triangle_height" in metric:
            data_entry["triangle_height"] = metric["triangle_height"]
        if "shaft_solidity" in metric:
            data_entry["shaft_solidity"] = metric["shaft_solidity"]
        if "tip_solidity" in metric:
            data_entry["tip_solidity"] = metric["tip_solidity"]

        updated_data.append(data_entry)

    # Define column order for the CSV
    base_columns = [
        "image_id", "surface_type", "surface_feature", "centroid_x", "centroid_y",
        "technical_width", "technical_length", "max_width", "max_length", "total_area", "aspect_ratio",
        "perimeter", "distance_to_max_width"
    ]

    voronoi_columns = [
        "voronoi_num_cells", "convex_hull_width", "convex_hull_height", "convex_hull_area",
        "voronoi_cell_area"
    ]

    symmetry_columns = [
        "top_area", "bottom_area", "left_area", "right_area",
        "vertical_symmetry", "horizontal_symmetry"
    ]

    # Lateral surface analysis columns
    lateral_columns = [
        "lateral_convexity"
    ]

    # Cortex analysis columns
    cortex_columns = [
        "is_cortex", "cortex_area", "cortex_percentage"
    ]

    # Arrow columns
    arrow_columns = [
        "has_arrow",
        "arrow_angle",
    ]

    # Check if any metrics have the additional arrow fields
    if any("triangle_base_length" in m for m in metrics):
        arrow_columns.append("triangle_base_length")
    if any("triangle_height" in m for m in metrics):
        arrow_columns.append("triangle_height")
    if any("shaft_solidity" in m for m in metrics):
        arrow_columns.append("shaft_solidity")
    if any("tip_solidity" in m for m in metrics):
        arrow_columns.append("tip_solidity")

    all_columns = base_columns + voronoi_columns + symmetry_columns + lateral_columns + cortex_columns + arrow_columns

    # Create DataFrame with all columns, handling any missing columns gracefully
    df = pd.DataFrame(updated_data)

    # Ensure all required columns exist, adding empty ones if needed
    for col in all_columns:
        if col not in df.columns:
            df[col] = "NA"

    # Reorder columns to match expected order
    # Only include columns that exist in the DataFrame
    existing_columns = [col for col in all_columns if col in df.columns]
    df = df[existing_columns]

    # Fill any NaN values
    df.fillna("NA", inplace=True)

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write to CSV
    if append and os.path.exists(output_path):
        # Read existing CSV to get its columns
        try:
            existing_df = pd.read_csv(output_path)
            existing_columns = existing_df.columns.tolist()

            # Align columns with existing file
            combined_columns = list(set(existing_columns) | set(df.columns))
            for col in combined_columns:
                if col not in df.columns:
                    df[col] = "NA"
                if col not in existing_columns:
                    existing_df[col] = "NA"

            # Write with matching columns
            df = df[existing_columns]
            df.to_csv(output_path, mode="a", header=False, index=False)
        except Exception as e:
            # If there's an error reading the existing file, just append
            logging.warning(f"Error aligning columns with existing CSV: {e}")
            df.to_csv(output_path, mode="a", header=False, index=False)

        logging.info("Appended metrics to existing CSV file: %s", output_path)
    else:
        df.to_csv(output_path, index=False)
        logging.info("Saved metrics to new CSV file: %s", output_path)