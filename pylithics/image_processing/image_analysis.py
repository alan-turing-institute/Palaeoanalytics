"""
PyLithics: Image Analysis Pipeline
==================================

This module serves as the main orchestrator for the PyLithics image analysis pipeline.
It coordinates various specialized modules to perform comprehensive quantitative analysis
of lithic artifacts via image processing.

The pipeline includes:
    - Contour extraction and hierarchy analysis
    - Geometric metric calculation
    - Surface classification (Dorsal, Ventral, Platform, Lateral)
    - Symmetry analysis
    - Arrow detection and integration
    - Voronoi diagram generation
    - Lateral surface convexity and thickness
    - Visualization and data export

Main Functions:
    * process_and_save_contours(inverted_image, conversion_factor, output_dir, image_id, image_dpi)
          - Main pipeline orchestrator that integrates all analysis steps

Usage Example:
    >>> from pylithics.image_processing.image_analysis import process_and_save_contours
    >>> process_and_save_contours(inverted_image, 0.01, "/path/to/output", "artifact_001", 300)

All functions include robust logging and error handling for debugging and traceability.
"""

import logging
import os
import sys
import traceback
import cv2

# Import specialized modules
from .modules.contour_extraction import (
    extract_contours_with_hierarchy,
    sort_contours_by_hierarchy,
    hide_nested_child_contours
)
from .modules.contour_metrics import calculate_contour_metrics, convert_metrics_to_real_world
from .modules.surface_classification import classify_parent_contours
from .modules.symmetry_analysis import analyze_dorsal_symmetry
from .modules.voronoi_analysis import calculate_voronoi_points, visualize_voronoi_diagram
from .modules.arrow_integration import integrate_arrows
from .modules.visualization import visualize_contours_with_hierarchy, save_measurements_to_csv
from .modules.lateral_analysis import analyze_lateral_surface, _integrate_lateral_metrics

def process_and_save_contours(inverted_image, conversion_factor, output_dir, image_id, image_dpi=None):
    """
    Main pipeline for processing contours and generating comprehensive lithic analysis.

    This function orchestrates the complete analysis workflow including contour extraction,
    metric calculation, surface classification, symmetry analysis, arrow detection,
    Voronoi analysis, and visualization generation.

    Parameters
    ----------
    inverted_image : numpy.ndarray
        Inverted binary thresholded image where foreground is white (255) and
        background is black (0).
    conversion_factor : float
        Conversion factor for transforming pixel measurements to real-world units (mm).
    output_dir : str
        Directory path where all processed outputs will be saved.
    image_id : str
        Unique identifier for the image being processed (used in output filenames).
    image_dpi : float, optional
        DPI of the image being processed. Used for scaling arrow detection parameters.
        If None, default scaling will be applied.

    Returns
    -------
    None
        Results are saved to files in the output directory:
        - CSV file with comprehensive metrics
        - Labeled visualization image
        - Voronoi diagram visualization

    Raises
    ------
    Exception
        Logs detailed error information if any step fails, but continues processing
        where possible to maximize data recovery.

    Notes
    -----
    The function is designed to be robust and will attempt to continue processing
    even if individual steps fail, ensuring maximum data extraction from each image.
    """
    try:
        # Step 1: Extract contours and hierarchy
        logging.info(f"Starting analysis for image: {image_id}")
        contours, hierarchy = extract_contours_with_hierarchy(inverted_image, image_id, output_dir)

        if not contours:
            logging.warning(f"No valid contours found for image: {image_id}")
            return

        logging.info(f"Extracted {len(contours)} valid contours with hierarchy shape: {hierarchy.shape}")

        # Step 2: Flag nested and single child contours for exclusion
        try:
            exclude_nested_flags = hide_nested_child_contours(contours, hierarchy)
        except Exception as e:
            logging.error(f"Error in hide_nested_child_contours: {e}")
            traceback.print_exc()
            exclude_nested_flags = [False] * len(contours)

        # Step 3: Sort contours by hierarchy
        try:
            sorted_contours = sort_contours_by_hierarchy(contours, hierarchy, exclude_nested_flags)
        except Exception as e:
            logging.error(f"Error in sort_contours_by_hierarchy: {e}")
            traceback.print_exc()
            # Fallback: treat first contour as parent, rest as children
            sorted_contours = {"parents": [], "children": [], "nested_children": []}
            if contours:
                sorted_contours["parents"] = [contours[0]]
                if len(contours) > 1:
                    sorted_contours["children"] = contours[1:]

        # Handle special case: promote nested children if no direct children exist
        if len(sorted_contours["children"]) == 0 and len(sorted_contours.get("nested_children", [])) > 0:
            logging.info(f"Promoting {len(sorted_contours['nested_children'])} nested children to direct children for {image_id}")
            sorted_contours["children"] = sorted_contours["nested_children"]
            sorted_contours["nested_children"] = []

        # Step 4: Calculate basic geometric metrics
        try:
            metrics = calculate_contour_metrics(sorted_contours, hierarchy, contours, inverted_image)
        except Exception as e:
            logging.error(f"Error in calculate_contour_metrics: {e}")
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logging.error(f"Error at {fname}:{exc_tb.tb_lineno}")
            traceback.print_exc()

            # Create minimal fallback metrics
            metrics = _create_fallback_metrics(sorted_contours, contours)

        # Step 5: Classify parent contours into surface types
        try:
            metrics = classify_parent_contours(metrics)
        except Exception as e:
            logging.error(f"Error in classify_parent_contours: {e}")

        # Step 6: Add image identifier to all metrics
        for metric in metrics:
            metric["image_id"] = image_id

        # Step 7: Perform symmetry analysis for dorsal surface
        try:
            symmetry_scores = analyze_dorsal_symmetry(metrics, sorted_contours.get("parents", []), inverted_image)

            if symmetry_scores:
                for metric in metrics:
                    if metric.get("surface_type") == "Dorsal":
                        metric.update(symmetry_scores)
                logging.info("Symmetry analysis completed successfully")
            else:
                logging.warning("No symmetry scores returned from analyze_dorsal_symmetry")

        except Exception as e:
            logging.error(f"Error in symmetry analysis: {e}")

        # Step 8: Calculate Voronoi diagram and convex hull metrics
        try:
            voronoi_data = calculate_voronoi_points(metrics, inverted_image, padding_factor=0.02)

            # Initialize voronoi_cell_area field for all metrics
            for metric in metrics:
                metric['voronoi_cell_area'] = "NA"

            if voronoi_data is not None:
                _integrate_voronoi_metrics(metrics, voronoi_data)
                logging.info("Voronoi analysis completed successfully")

        except Exception as e:
            logging.error(f"Error in Voronoi processing: {e}")

        # Step 9: Perform lateral surface analysis
        try:
            lateral_results = analyze_lateral_surface(metrics, sorted_contours.get("parents", []), inverted_image)

            if lateral_results:
                _integrate_lateral_metrics(metrics, lateral_results)
                logging.info("Lateral surface analysis completed successfully")
            else:
                logging.info("No lateral surface found or analysis not applicable")

        except Exception as e:
            logging.error(f"Error in lateral surface analysis: {e}")
            traceback.print_exc()

        # Step 10: Integrate arrow detection
        try:
            scars_without_arrows = [m for m in metrics
                                  if m["parent"] != m["scar"] and not m.get('has_arrow', False)]

            if scars_without_arrows:
                logging.info(f"Running arrow detection for {len(scars_without_arrows)} scars")
                metrics = integrate_arrows(sorted_contours, hierarchy, contours,
                                         metrics, inverted_image, image_dpi)
            else:
                logging.info("All scars have arrows or no scars found. Skipping arrow detection.")

        except Exception as e:
            logging.error(f"Error in arrow detection: {e}")
            traceback.print_exc()

        # Step 11: Save metrics to CSV
        try:
            csv_path = os.path.join(output_dir, "processed_metrics.csv")
            save_measurements_to_csv(metrics, csv_path, append=True)
        except Exception as e:
            logging.error(f"Error saving CSV: {e}")

        # Step 12: Generate labeled visualization
        try:
            all_contours = (sorted_contours.get("parents", []) +
                           sorted_contours.get("children", []) +
                           sorted_contours.get("nested_children", []))

            visualization_path = os.path.join(output_dir, f"{image_id}_labeled.png")
            visualize_contours_with_hierarchy(all_contours, hierarchy, metrics,
                                            inverted_image, visualization_path)
        except Exception as e:
            logging.error(f"Error in visualization: {e}")

        # Step 13: Generate Voronoi diagram visualization
        try:
            if 'voronoi_data' in locals() and voronoi_data is not None:
                voronoi_path = os.path.join(output_dir, f"{image_id}_voronoi.png")
                visualize_voronoi_diagram(voronoi_data, inverted_image, voronoi_path)
        except Exception as e:
            logging.error(f"Error in Voronoi visualization: {e}")

        logging.info(f"Analysis complete for image: {image_id}")

    except Exception as e:
        logging.error(f"Critical error analyzing image {image_id}: {e}")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logging.error(f"Error at {fname}:{exc_tb.tb_lineno}")
        traceback.print_exc()


def _create_fallback_metrics(sorted_contours, contours):
    """
    Create minimal fallback metrics when main metric calculation fails.

    Parameters
    ----------
    sorted_contours : dict
        Dictionary containing sorted contour lists
    contours : list
        List of all contours

    Returns
    -------
    list
        List of minimal metric dictionaries
    """
    metrics = []

    # Create minimal parent metrics
    for i, cnt in enumerate(sorted_contours.get("parents", [])):
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        metrics.append({
            "parent": f"parent {i+1}",
            "scar": f"parent {i+1}",
            "surface_type": "Dorsal" if i == 0 else "Unknown",
            "centroid_x": x + w/2,
            "centroid_y": y + h/2,
            "width": w,
            "height": h,
            "area": area,
            "has_arrow": False
        })

    # Ensure at least one metric exists
    if not metrics and contours:
        cnt = contours[0]
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        metrics.append({
            "parent": "parent 1",
            "scar": "parent 1",
            "surface_type": "Dorsal",
            "centroid_x": x + w/2,
            "centroid_y": y + h/2,
            "width": w,
            "height": h,
            "area": area,
            "has_arrow": False
        })

    return metrics


def _integrate_voronoi_metrics(metrics, voronoi_data):
    """
    Integrate Voronoi analysis results into the metrics.

    Parameters
    ----------
    metrics : list
        List of metric dictionaries to update
    voronoi_data : dict
        Voronoi analysis results
    """
    vor_num_cells = voronoi_data['voronoi_metrics']['num_cells']
    ch_width = round(voronoi_data['convex_hull_metrics']['width'], 2)
    ch_height = round(voronoi_data['convex_hull_metrics']['height'], 2)
    ch_area = round(voronoi_data['convex_hull_metrics']['area'], 2)

    # Update individual cell areas
    for cell in voronoi_data['voronoi_cells']:
        metric_idx = cell.get('metric_index', -1)
        if metric_idx != -1 and metric_idx < len(metrics):
            cell_area = round(cell['area'], 2)
            metrics[metric_idx]['voronoi_cell_area'] = cell_area

    # Add convex hull metrics to dorsal surface
    for metric in metrics:
        if metric.get("surface_type") == "Dorsal":
            metric.update({
                'voronoi_num_cells': vor_num_cells,
                'convex_hull_width': ch_width,
                'convex_hull_height': ch_height,
                'convex_hull_area': ch_area
            })
