"""
PyLithics: Image Analysis Pipeline
==================================

Main orchestrator for the PyLithics image analysis pipeline. Coordinates
specialized modules to perform comprehensive quantitative analysis of
lithic artifacts via image processing.

Main Functions:
    * process_and_save_contours(inverted_image, conversion_factor,
          output_dir, image_id, image_dpi)

Usage Example:
    >>> from pylithics.image_processing.image_analysis import process_and_save_contours
    >>> process_and_save_contours(inverted_image, 0.01, "/path/to/output", "artifact_001", 300)
"""

import logging
import os
import cv2
import numpy

# Errors that any pipeline stage may legitimately raise at runtime:
# cv2.error from OpenCV primitives; ValueError / TypeError / IndexError /
# AttributeError from numpy and shapely geometry; KeyError from malformed
# metric dicts; OSError from pandas CSV writes. Narrower than bare
# `Exception` so that MemoryError / KeyboardInterrupt propagate normally.
_STAGE_ERRORS = (
    cv2.error, ValueError, TypeError, KeyError,
    IndexError, AttributeError, RuntimeError, OSError,
)

from .config import get_config_manager, get_data_export_config
from .modules.contour_extraction import (
    extract_contours_with_hierarchy,
    sort_contours_by_hierarchy,
    hide_nested_child_contours
)
from .modules.contour_metrics import (
    calculate_contour_metrics,
    convert_metrics_to_real_world
)
from .modules.surface_classification import (
    classify_parent_contours,
    classify_child_features
)
from .modules.cortex_detection import (
    detect_cortex_in_child_contours,
    calculate_total_cortex_metrics
)
from .modules.symmetry_analysis import analyze_dorsal_symmetry
from .modules.voronoi_analysis import (
    calculate_voronoi_points,
    visualize_voronoi_diagram
)
from .modules.arrow_integration import integrate_arrows
from .modules.visualization import (
    visualize_contours_with_hierarchy,
    save_measurements_to_csv
)
from .modules.json_export import save_measurements_to_json
from .modules.lateral_analysis import (
    analyze_lateral_surface,
    _integrate_lateral_metrics
)
from .modules.scar_complexity import (
    analyze_scar_complexity,
    _integrate_complexity_results
)


def process_and_save_contours(
    inverted_image,
    conversion_factor,
    output_dir,
    image_id,
    image_dpi=None,
    calibration_method="pixels",
    scale_confidence=None
) -> None:
    """
    Main pipeline for processing contours and generating
    comprehensive lithic analysis.

    Parameters
    ----------
    inverted_image : numpy.ndarray
        Inverted binary thresholded image.
    conversion_factor : float
        Pixel-to-mm conversion factor.
    output_dir : str
        Directory for processed outputs.
    image_id : str
        Unique image identifier.
    image_dpi : float, optional
        Image DPI for scaling arrow detection.
    calibration_method : str, optional
        Conversion method ("scale_bar", "dpi", "pixels").
    scale_confidence : float, optional
        Scale detection confidence (0-1).

    Returns
    -------
    None
        Results saved to output_dir as CSV and images.
    """
    try:
        logging.debug(f"Starting analysis for image: {image_id}")
        image_stem = os.path.splitext(image_id)[0]

        contours, hierarchy, sorted_contours = _extract_and_sort_contours(
            inverted_image, image_id, output_dir
        )
        if contours is None:
            return

        metrics = _calculate_and_classify(
            sorted_contours, hierarchy, contours, inverted_image
        )

        _run_cortex_detection(metrics, inverted_image)
        _run_scar_complexity(metrics)

        for metric in metrics:
            metric["image_id"] = image_id

        _run_symmetry_analysis(metrics, sorted_contours, inverted_image)

        voronoi_data = _run_voronoi_analysis(metrics, inverted_image)
        _run_lateral_analysis(metrics, sorted_contours, inverted_image)
        _run_arrow_detection(
            metrics, sorted_contours, hierarchy,
            contours, inverted_image, image_dpi
        )

        _convert_and_export(
            metrics, conversion_factor, output_dir, image_id,
            calibration_method, scale_confidence
        )

        config_manager = get_config_manager()
        _generate_visualizations(
            metrics, sorted_contours, hierarchy, contours,
            inverted_image, output_dir, image_stem,
            voronoi_data, conversion_factor, config_manager.config
        )

        logging.debug(f"Analysis complete for image: {image_id}")

    # Broad catch is intentional here: this is the outermost safety net that
    # prevents a single bad image from aborting a multi-image batch run.
    except Exception:
        logging.exception(f"Critical error analyzing image {image_id}")


def _extract_and_sort_contours(inverted_image, image_id, output_dir):
    """
    Extract contours, flag nested children, and sort by hierarchy.

    Parameters
    ----------
    inverted_image : numpy.ndarray
        Inverted binary thresholded image.
    image_id : str
        Image identifier for logging.
    output_dir : str
        Output directory path.

    Returns
    -------
    tuple
        (contours, hierarchy, sorted_contours) or (None, None, None)
        if no valid contours found.
    """
    contours, hierarchy = extract_contours_with_hierarchy(
        inverted_image, image_id, output_dir
    )

    if not contours:
        logging.warning(f"No valid contours for image: {image_id}")
        return None, None, None

    logging.debug(
        f"Extracted {len(contours)} contours, "
        f"hierarchy shape: {hierarchy.shape}"
    )

    try:
        exclude_flags = hide_nested_child_contours(contours, hierarchy)
    except _STAGE_ERRORS:
        logging.exception("Error in hide_nested_child_contours")
        exclude_flags = [False] * len(contours)

    try:
        sorted_contours = sort_contours_by_hierarchy(
            contours, hierarchy, exclude_flags
        )
    except _STAGE_ERRORS:
        logging.exception("Error in sort_contours_by_hierarchy")
        sorted_contours = _build_fallback_sorted(contours)

    # Promote nested children if no direct children exist
    nested = sorted_contours.get("nested_children", [])
    if len(sorted_contours["children"]) == 0 and len(nested) > 0:
        logging.debug(
            f"Promoting {len(nested)} nested children to "
            f"direct children for {image_id}"
        )
        sorted_contours["children"] = nested
        sorted_contours["nested_children"] = []

    return contours, hierarchy, sorted_contours


def _build_fallback_sorted(contours):
    """
    Build fallback sorted contours when sorting fails.

    Parameters
    ----------
    contours : list
        List of all contours.

    Returns
    -------
    dict
        Minimal sorted contour structure.
    """
    sorted_contours = {
        "parents": [], "children": [], "nested_children": []
    }
    if contours:
        sorted_contours["parents"] = [contours[0]]
        if len(contours) > 1:
            sorted_contours["children"] = contours[1:]
    return sorted_contours


def _calculate_and_classify(
    sorted_contours, hierarchy, contours, inverted_image
):
    """
    Calculate geometric metrics and classify surfaces.

    Parameters
    ----------
    sorted_contours : dict
        Sorted contour structure.
    hierarchy : numpy.ndarray
        Contour hierarchy from OpenCV.
    contours : list
        All extracted contours.
    inverted_image : numpy.ndarray
        Inverted binary image.

    Returns
    -------
    list
        List of metric dictionaries.
    """
    try:
        metrics = calculate_contour_metrics(
            sorted_contours, hierarchy, contours, inverted_image
        )
    except _STAGE_ERRORS:
        logging.exception("Error in calculate_contour_metrics")
        metrics = _create_fallback_metrics(sorted_contours, contours)

    try:
        metrics = classify_parent_contours(metrics)
    except _STAGE_ERRORS:
        logging.exception("Error in classify_parent_contours")

    try:
        metrics = classify_child_features(metrics)
        logging.debug("Child feature classification completed")
    except _STAGE_ERRORS:
        logging.exception("Error in classify_child_features")

    return metrics


def _run_cortex_detection(metrics, inverted_image) -> None:
    """
    Detect cortex in child contours.

    Parameters
    ----------
    metrics : list
        Metric dictionaries to update in place.
    inverted_image : numpy.ndarray
        Inverted binary image.
    """
    try:
        detect_cortex_in_child_contours(metrics, inverted_image)
        cortex_stats = calculate_total_cortex_metrics(metrics)

        if cortex_stats["cortex_count"] > 0:
            logging.debug(
                f"Cortex detected: {cortex_stats['cortex_count']} areas "
                f"(area: {cortex_stats['total_cortex_area']:.1f})"
            )
        else:
            logging.debug("No cortex detected in this artifact")
    except _STAGE_ERRORS:
        logging.exception("Error in cortex detection")


def _run_scar_complexity(metrics) -> None:
    """
    Perform scar complexity analysis.

    Parameters
    ----------
    metrics : list
        Metric dictionaries to update in place.
    """
    try:
        config_manager = get_config_manager()
        complexity_results = analyze_scar_complexity(
            metrics, config_manager.config
        )
        if complexity_results:
            _integrate_complexity_results(metrics, complexity_results)
            logging.debug(
                f"Scar complexity completed for "
                f"{len(complexity_results)} scars"
            )
        else:
            logging.debug("No scar complexity results generated")
    except _STAGE_ERRORS:
        logging.exception("Error in scar complexity analysis")


def _run_symmetry_analysis(
    metrics, sorted_contours, inverted_image
) -> None:
    """
    Perform symmetry analysis on dorsal surface.

    Parameters
    ----------
    metrics : list
        Metric dictionaries to update in place.
    sorted_contours : dict
        Sorted contour structure.
    inverted_image : numpy.ndarray
        Inverted binary image.
    """
    try:
        all_contours = (
            sorted_contours.get("parents", []) +
            sorted_contours.get("children", []) +
            sorted_contours.get("nested_children", [])
        )

        scores = analyze_dorsal_symmetry(
            metrics, all_contours, inverted_image
        )

        if scores and any(v is not None for v in scores.values()):
            for metric in metrics:
                if metric.get("surface_type") == "Dorsal":
                    metric.update(scores)
            logging.debug("Symmetry analysis completed")
        else:
            logging.warning("No valid symmetry scores returned")
    except _STAGE_ERRORS:
        logging.exception("Error in symmetry analysis")


def _run_voronoi_analysis(metrics, inverted_image):
    """
    Calculate Voronoi diagram and convex hull metrics.

    Parameters
    ----------
    metrics : list
        Metric dictionaries to update in place.
    inverted_image : numpy.ndarray
        Inverted binary image.

    Returns
    -------
    dict or None
        Voronoi analysis data, or None on failure.
    """
    for metric in metrics:
        metric['voronoi_cell_area'] = "NA"

    try:
        voronoi_data = calculate_voronoi_points(
            metrics, inverted_image, padding_factor=0.02
        )
        if voronoi_data is not None:
            _integrate_voronoi_metrics(metrics, voronoi_data)
            logging.debug("Voronoi analysis completed")
        return voronoi_data
    except _STAGE_ERRORS:
        logging.exception("Error in Voronoi processing")
        return None


def _run_lateral_analysis(
    metrics, sorted_contours, inverted_image
) -> None:
    """
    Perform lateral surface convexity analysis.

    Parameters
    ----------
    metrics : list
        Metric dictionaries to update in place.
    sorted_contours : dict
        Sorted contour structure.
    inverted_image : numpy.ndarray
        Inverted binary image.
    """
    try:
        lateral_results = analyze_lateral_surface(
            metrics, sorted_contours.get("parents", []),
            inverted_image
        )
        if lateral_results:
            _integrate_lateral_metrics(metrics, lateral_results)
            logging.debug("Lateral surface analysis completed")
        else:
            logging.debug("No lateral surface found or not applicable")
    except _STAGE_ERRORS:
        logging.exception("Error in lateral surface analysis")


def _run_arrow_detection(
    metrics, sorted_contours, hierarchy,
    contours, inverted_image, image_dpi
) -> None:
    """
    Detect and integrate arrows with scars.

    Parameters
    ----------
    metrics : list
        Metric dictionaries to update in place.
    sorted_contours : dict
        Sorted contour structure.
    hierarchy : numpy.ndarray
        Contour hierarchy.
    contours : list
        All extracted contours.
    inverted_image : numpy.ndarray
        Inverted binary image.
    image_dpi : float or None
        Image DPI for scaling detection parameters.
    """
    try:
        scars_without_arrows = [
            m for m in metrics
            if m["parent"] != m["scar"]
            and not m.get('has_arrow', False)
        ]

        if scars_without_arrows:
            logging.debug(
                f"Running arrow detection for "
                f"{len(scars_without_arrows)} scars"
            )
            integrate_arrows(
                sorted_contours, hierarchy, contours,
                metrics, inverted_image, image_dpi
            )
        else:
            logging.debug("No scars need arrow detection")
    except _STAGE_ERRORS:
        logging.exception("Error in arrow detection")


def _convert_and_export(
    metrics, conversion_factor, output_dir, image_id,
    calibration_method, scale_confidence
) -> None:
    """
    Convert metrics to real-world units and save to CSV.

    Parameters
    ----------
    metrics : list
        Metric dictionaries to export.
    conversion_factor : float
        Pixel-to-mm conversion factor.
    output_dir : str
        Output directory path.
    image_id : str
        Image identifier.
    calibration_method : str
        Calibration method used.
    scale_confidence : float or None
        Scale detection confidence.
    """
    if conversion_factor and conversion_factor != 1.0:
        try:
            convert_metrics_to_real_world(metrics, conversion_factor)
            logging.debug(
                f"Converted to mm using factor: "
                f"{conversion_factor:.3f}"
            )
        except _STAGE_ERRORS:
            logging.exception("Error converting to real-world units")

    calibration_metadata = {
        'calibration_method': calibration_method,
        'pixels_per_mm': (
            conversion_factor
            if conversion_factor and conversion_factor != 1.0
            else None
        ),
        'scale_confidence': scale_confidence
    }

    try:
        csv_path = os.path.join(output_dir, "processed_metrics.csv")
        save_measurements_to_csv(
            metrics, csv_path, append=True,
            calibration_metadata=calibration_metadata
        )
    except _STAGE_ERRORS:
        logging.exception("Error saving CSV")

    if get_data_export_config().get('json_per_lithic', False):
        try:
            image_stem = os.path.splitext(image_id)[0]
            json_path = os.path.join(
                output_dir, "json", f"{image_stem}.json"
            )
            save_measurements_to_json(
                metrics, json_path, calibration_metadata,
            )
        except _STAGE_ERRORS:
            logging.exception("Error saving per-lithic JSON")


def _generate_visualizations(
    metrics, sorted_contours, hierarchy, contours,
    inverted_image, output_dir, image_id,
    voronoi_data, conversion_factor, config
) -> None:
    """
    Generate labeled image and Voronoi diagram visualizations.

    Parameters
    ----------
    metrics : list
        Metric dictionaries with analysis results.
    sorted_contours : dict
        Sorted contour structure.
    hierarchy : numpy.ndarray
        Contour hierarchy.
    contours : list
        All extracted contours.
    inverted_image : numpy.ndarray
        Inverted binary image.
    output_dir : str
        Output directory path.
    image_id : str
        Image identifier.
    voronoi_data : dict or None
        Voronoi analysis data.
    conversion_factor : float
        Pixel-to-mm conversion factor.
    config : dict
        Configuration dictionary.
    """
    try:
        viz_contours = _build_visualization_contours(metrics)
        arrow_contours = sorted_contours.get("nested_children", [])

        viz_path = os.path.join(
            output_dir, f"{image_id}_labeled.png"
        )
        visualize_contours_with_hierarchy(
            viz_contours, hierarchy, metrics,
            inverted_image, viz_path, arrow_contours, config
        )
    except _STAGE_ERRORS:
        logging.exception("Error in visualization")

    try:
        if voronoi_data is not None:
            voronoi_path = os.path.join(
                output_dir, f"{image_id}_voronoi.png"
            )
            visualize_voronoi_diagram(
                voronoi_data, inverted_image,
                voronoi_path, conversion_factor
            )
    except _STAGE_ERRORS:
        logging.exception("Error in Voronoi visualization")


def _build_visualization_contours(metrics):
    """
    Extract contour arrays from metrics for visualization.

    Parameters
    ----------
    metrics : list
        Metric dictionaries containing contour data.

    Returns
    -------
    list
        List of numpy contour arrays.
    """
    viz_contours = []
    for metric in metrics:
        if 'contour' in metric and metric['contour']:
            contour_array = numpy.array(
                metric['contour'], dtype=numpy.int32
            )
            viz_contours.append(contour_array)
    return viz_contours


def _create_fallback_metrics(sorted_contours, contours):
    """
    Create minimal fallback metrics when calculation fails.

    Parameters
    ----------
    sorted_contours : dict
        Dictionary containing sorted contour lists.
    contours : list
        List of all contours.

    Returns
    -------
    list
        List of minimal metric dictionaries.
    """
    metrics = []

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


def _integrate_voronoi_metrics(metrics, voronoi_data) -> None:
    """
    Integrate Voronoi analysis results into the metrics.

    Parameters
    ----------
    metrics : list
        List of metric dictionaries to update.
    voronoi_data : dict
        Voronoi analysis results.
    """
    vor_num_cells = voronoi_data['voronoi_metrics']['num_cells']
    ch_width = round(
        voronoi_data['convex_hull_metrics']['width'], 2
    )
    ch_height = round(
        voronoi_data['convex_hull_metrics']['height'], 2
    )
    ch_area = round(
        voronoi_data['convex_hull_metrics']['area'], 2
    )

    for cell in voronoi_data['voronoi_cells']:
        metric_idx = cell.get('metric_index', -1)
        if metric_idx != -1 and metric_idx < len(metrics):
            metrics[metric_idx]['voronoi_cell_area'] = round(
                cell['area'], 2
            )

    for metric in metrics:
        if metric.get("surface_type") == "Dorsal":
            metric.update({
                'voronoi_num_cells': vor_num_cells,
                'convex_hull_width': ch_width,
                'convex_hull_height': ch_height,
                'convex_hull_area': ch_area
            })
