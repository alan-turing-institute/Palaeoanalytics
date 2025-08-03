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

    # Draw contours, centroids, and labels
    label_positions = []
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
        elif "scar" in scar_label.lower():
            color = (99, 184, 253)  # orange for scars (dorsal surface only)
            text  = scar_label
        elif "mark_" in scar_label.lower():
            color = (210, 171, 178)   # RGB 178,171,210 for platform marks
            text  = scar_label
        elif "edge_" in scar_label.lower():
            color = (193, 205, 128)   # RGB 128,205,193 for lateral edges
            text  = scar_label
        else:
            color = (128, 128, 128) # gray for unknown/temporary labels
            text  = scar_label

        # Draw the contour
        cv2.drawContours(labeled, [cnt], -1, color, 2)

        # Compute and draw centroid
        M = cv2.moments(cnt)
        if M.get("m00", 0) != 0:
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
        else:
            x,y,w,h = cv2.boundingRect(cnt)
            cx,cy = x + w//2, y + h//2
        cv2.circle(labeled, (cx, cy), 4, (1, 97, 230), -1)

        # Place label (avoid overlaps)
        ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        tx, ty = cx + 10, cy - 10
        for lx, ly, lw, lh in label_positions:
            if tx < lx+lw and tx+ts[0] > lx and ty < ly+lh and ty+ts[1] > ly:
                ty = ly + lh + 10
        label_positions.append((tx, ty, ts[0], ts[1]))
        cv2.putText(labeled, text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (186,186,186), 2)


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

                    # Calculate text position at 1/3 of the shaft from the back, offset perpendicular
                    shaft_fraction = 1/3
                    text_base_pos = (
                        int(back[0] + shaft_vector[0] * shaft_fraction),
                        int(back[1] + shaft_vector[1] * shaft_fraction)
                    )
                    text_pos = (
                        int(text_base_pos[0] + perp_offset[0]),
                        int(text_base_pos[1] + perp_offset[1])
                    )


                    text = f"{int(angle)} deg"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    thickness = 2

                    # Get text size
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                    # Create bigger background rectangle with padding
                    padding = 4
                    text_bg_pt1 = (text_pos[0] - padding, text_pos[1] - text_height - padding)
                    text_bg_pt2 = (text_pos[0] + text_width + padding, text_pos[1] + padding)

                    # Draw white background with black border
                    cv2.rectangle(labeled, text_bg_pt1, text_bg_pt2, (255, 255, 255), -1)  # White fill
                    cv2.rectangle(labeled, text_bg_pt1, text_bg_pt2, (0, 0, 0), 1)        # Black border

                    # Draw text in black
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

    all_columns = base_columns + voronoi_columns + symmetry_columns + lateral_columns + arrow_columns

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