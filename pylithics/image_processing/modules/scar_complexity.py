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


def analyze_scar_complexity(metrics: List[Dict[str, Any]], 
                          config: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    """
    Analyze scar complexity by counting border-sharing relationships.
    
    For each scar on the dorsal surface, counts how many other scars
    share a border with it using geometric boundary detection.
    
    Parameters
    ----------
    metrics : list of dict
        List of contour metrics containing surface classification data
    config : dict
        Configuration parameters for scar complexity analysis
        
    Returns
    -------
    dict
        Dictionary mapping surface_feature names to complexity counts
        
    Examples
    --------
    >>> dorsal_scars = [
    ...     {"surface_type": "Dorsal", "surface_feature": "scar 1", "contour": [...]},
    ...     {"surface_type": "Dorsal", "surface_feature": "scar 2", "contour": [...]}
    ... ]
    >>> config = {"enabled": True}
    >>> result = analyze_scar_complexity(dorsal_scars, config)
    >>> print(result)
    {"scar 1": 1, "scar 2": 1}  # Each scar shares border with 1 other
    """
    try:
        # Get scar complexity configuration using PyLithics pattern
        scar_config = get_scar_complexity_config(config)
        
        # Check if scar complexity analysis is enabled
        if not scar_config.get('enabled', True):
            logging.info("Scar complexity analysis is disabled")
            return {}
        
        # Debug: Log all metrics to understand what we're working with
        logging.info(f"Scar complexity analysis received {len(metrics)} total metrics")
        for i, metric in enumerate(metrics):
            surface_type = metric.get('surface_type', 'unknown')
            surface_feature = metric.get('surface_feature', 'unknown')
            logging.info(f"Metric {i}: surface_type='{surface_type}', surface_feature='{surface_feature}'")
        
        # Extract dorsal scars by finding child contours of dorsal parent
        # First, find the dorsal parent surface
        dorsal_parent = None
        for metric in metrics:
            if (metric.get('surface_type') == 'Dorsal' and 
                metric.get('parent') == metric.get('scar')):  # This is a parent contour
                dorsal_parent = metric.get('scar')
                break
        
        if not dorsal_parent:
            logging.info("No dorsal parent surface found")
            return {}
        
        logging.info(f"Found dorsal parent: {dorsal_parent}")
        
        # Extract scars that belong to the dorsal parent
        dorsal_scars = [
            metric for metric in metrics 
            if (metric.get('parent') == dorsal_parent and 
                metric.get('parent') != metric.get('scar') and  # This is a child contour
                'scar' in metric.get('surface_feature', '').lower())
        ]
        
        logging.info(f"Found {len(dorsal_scars)} dorsal scars out of {len(metrics)} total metrics")
        for scar in dorsal_scars:
            logging.debug(f"Dorsal scar: {scar.get('surface_feature', 'unknown')}")
        
        if len(dorsal_scars) < 2:
            # Need at least 2 scars to have complexity
            logging.info(f"Not enough dorsal scars ({len(dorsal_scars)}) for complexity analysis - need at least 2")
            return {scar.get('surface_feature', 'unknown'): 0 
                   for scar in dorsal_scars}
        
        logging.info(f"Analyzing scar complexity for {len(dorsal_scars)} dorsal scars")
        
        complexity_results = {}
        
        # Compare each scar with every other scar
        for i, scar in enumerate(dorsal_scars):
            border_count = 0
            scar_feature = scar.get('surface_feature', f'scar_{i}')
            
            try:
                # Create polygon from contour
                scar_polygon = _create_polygon_from_contour(scar.get('contour'))
                if scar_polygon is None:
                    complexity_results[scar_feature] = 0
                    continue
                
                # Check against all other scars
                for j, other_scar in enumerate(dorsal_scars):
                    if i != j:  # Don't compare scar to itself
                        try:
                            other_polygon = _create_polygon_from_contour(
                                other_scar.get('contour')
                            )
                            if other_polygon is None:
                                continue
                                
                            # Check if scars are adjacent using configurable distance threshold
                            # Use configurable distance threshold to account for 
                            # OpenCV contours being inside black line boundaries
                            distance_threshold = scar_config.get('distance_threshold', 10.0)
                            distance = scar_polygon.distance(other_polygon)
                            if distance <= distance_threshold:
                                border_count += 1
                                logging.debug(f"{scar_feature} is adjacent to {other_scar.get('surface_feature')} (distance: {distance:.1f}, threshold: {distance_threshold})")
                                
                        except (ShapelyError, Exception) as e:
                            logging.warning(
                                f"Error checking border sharing between "
                                f"{scar_feature} and {other_scar.get('surface_feature')}: {e}"
                            )
                            continue
                
                complexity_results[scar_feature] = border_count
                logging.debug(f"{scar_feature} shares borders with {border_count} scars")
                
            except (ShapelyError, Exception) as e:
                logging.warning(f"Error analyzing {scar_feature}: {e}")
                complexity_results[scar_feature] = 0
        
        logging.info(f"Scar complexity analysis completed for {len(complexity_results)} scars")
        return complexity_results
        
    except Exception as e:
        logging.error(f"Error in scar complexity analysis: {e}")
        # Return fallback results
        return _create_fallback_complexity_results(metrics)


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
        
    except (ShapelyError, ValueError, Exception) as e:
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