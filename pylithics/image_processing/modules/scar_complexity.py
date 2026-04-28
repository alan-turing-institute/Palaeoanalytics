"""
Scar Complexity Analysis Module
===============================

Analyzes border-sharing relationships between scars on dorsal surfaces.
Counts how many scars each scar shares a border with using geometric analysis.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
from shapely.geometry import Polygon
from shapely.errors import ShapelyError
from ..config import get_scar_complexity_config


def analyze_scar_complexity(
    metrics: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, int]:
    """
    Count border-sharing relationships between scars on the dorsal surface.

    Parameters
    ----------
    metrics : list of dict
        Contour metrics with surface classifications.
    config : dict, optional
        Scar-complexity configuration. Uses the global config manager
        when None.

    Returns
    -------
    dict
        Map of surface_feature name → number of adjacent scars.
    """
    try:
        scar_config = get_scar_complexity_config(config)
        if not scar_config.get('enabled', True):
            logging.info("Scar complexity analysis is disabled")
            return {}

        dorsal_scars = _find_dorsal_scars(metrics)
        if len(dorsal_scars) < 2:
            logging.info(
                f"Not enough dorsal scars ({len(dorsal_scars)}) for complexity"
            )
            return {
                s.get('surface_feature', 'unknown'): 0 for s in dorsal_scars
            }

        threshold = scar_config.get('distance_threshold', 10.0)
        results = _count_shared_borders(dorsal_scars, threshold)

        logging.info(
            f"Scar complexity analysis completed for {len(results)} scars"
        )
        return results

    except Exception as e:
        logging.error(f"Error in scar complexity analysis: {e}")
        return _create_fallback_complexity_results(metrics)


def _find_dorsal_scars(metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return child scars that belong to the dorsal parent surface."""
    dorsal_parent = next(
        (
            m.get('scar') for m in metrics
            if m.get('surface_type') == 'Dorsal'
            and m.get('parent') == m.get('scar')
        ),
        None,
    )
    if not dorsal_parent:
        logging.info("No dorsal parent surface found")
        return []

    scars = [
        m for m in metrics
        if (m.get('parent') == dorsal_parent
            and m.get('parent') != m.get('scar')
            and 'scar' in m.get('surface_feature', '').lower())
    ]
    logging.info(f"Found {len(scars)} dorsal scars out of {len(metrics)} metrics")
    return scars


def _count_shared_borders(
    scars: List[Dict[str, Any]], distance_threshold: float,
) -> Dict[str, int]:
    """For each scar, count others whose polygon lies within threshold distance."""
    polygons = [_create_polygon_from_contour(s.get('contour')) for s in scars]
    results: Dict[str, int] = {}

    for i, scar in enumerate(scars):
        feature = scar.get('surface_feature', f'scar_{i}')
        poly_i = polygons[i]
        if poly_i is None:
            results[feature] = 0
            continue

        count = 0
        for j, poly_j in enumerate(polygons):
            if j == i or poly_j is None:
                continue
            try:
                if poly_i.distance(poly_j) <= distance_threshold:
                    count += 1
            except ShapelyError as e:
                other = scars[j].get('surface_feature')
                logging.warning(
                    f"Error checking border sharing between "
                    f"{feature} and {other}: {e}"
                )
        results[feature] = count
    return results


def _create_polygon_from_contour(contour: Optional[List]) -> Optional[Polygon]:
    """
    Create a Shapely polygon from a contour.
    
    Parameters
    ----------
    contour : list or None
        Contour points as list of coordinates
        
    Returns
    -------
    Polygon or None
        Shapely polygon object, or None if creation fails
    """
    if contour is None:
        return None
    
    # Handle numpy arrays and lists
    if hasattr(contour, 'size') and contour.size == 0:
        return None
    elif hasattr(contour, '__len__') and len(contour) < 3:
        return None
        
    try:
        # Convert contour to numpy array if needed
        if isinstance(contour, list):
            contour_array = np.array(contour)
        else:
            contour_array = contour
            
        # Ensure we have the right shape
        if len(contour_array.shape) == 3:
            # Remove singleton dimension if present
            contour_array = contour_array.squeeze()
            
        if len(contour_array.shape) != 2 or contour_array.shape[1] != 2:
            logging.warning(f"Invalid contour shape: {contour_array.shape}")
            return None
            
        # Create polygon
        polygon = Polygon(contour_array)
        
        # Validate polygon
        if not polygon.is_valid:
            # Try to fix invalid polygon
            polygon = polygon.buffer(0)
            if not polygon.is_valid:
                logging.warning("Could not create valid polygon from contour")
                return None
                
        return polygon
        
    except (ShapelyError, ValueError) as e:
        logging.warning(f"Error creating polygon from contour: {e}")
        return None


def _create_fallback_complexity_results(metrics: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Create fallback complexity results when analysis fails.
    
    Returns zero complexity for all dorsal scars.
    
    Parameters
    ----------
    metrics : list of dict
        List of contour metrics
        
    Returns
    -------
    dict
        Dictionary mapping surface features to zero complexity
    """
    try:
        dorsal_scars = [
            metric for metric in metrics 
            if (metric.get('surface_type') == 'Dorsal' and 
                'scar' in metric.get('surface_feature', '').lower())
        ]
        
        return {scar.get('surface_feature', 'unknown'): 0 
               for scar in dorsal_scars}
                
    except Exception:
        logging.error("Error creating fallback complexity results")
        return {}


def _integrate_complexity_results(metrics: List[Dict[str, Any]], 
                                complexity_results: Dict[str, int]) -> None:
    """
    Integrate complexity results back into the metrics list.
    
    Parameters
    ----------
    metrics : list of dict
        List of contour metrics to update in-place
    complexity_results : dict
        Dictionary mapping surface features to complexity counts
    """
    for metric in metrics:
        surface_feature = metric.get('surface_feature', '')
        if surface_feature in complexity_results:
            metric['scar_complexity'] = complexity_results[surface_feature]
        else:
            # Set to None for non-dorsal scars or failed analysis
            metric['scar_complexity'] = None