"""
PyLithics: Surface Classification
==================================

Classifies parent contours into archaeological surface types
(Dorsal, Ventral, Platform, Lateral) and applies surface-specific
labeling to child contours.
"""

import logging
from typing import List, Dict, Optional

from ..config import get_surface_classification_config


def classify_parent_contours(
    metrics: List[Dict],
    config: Optional[Dict] = None,
    tolerance: Optional[float] = None
) -> List[Dict]:
    """
    Classify parent contours into surface types.

    Parameters
    ----------
    metrics : list
        Metric dictionaries with contour data.
    config : dict, optional
        Full configuration dictionary.
    tolerance : float, optional
        Dimensional tolerance override. If None, uses config.

    Returns
    -------
    list
        Updated metrics with surface classifications.
    """
    sc_config = get_surface_classification_config(config)
    if tolerance is None:
        tolerance = sc_config.get('tolerance', 0.1)

    parents = [m for m in metrics if m["parent"] == m["scar"]]

    if not parents:
        logging.warning("No parent contours for classification.")
        return metrics

    for parent in parents:
        parent["surface_type"] = None

    dorsal = _identify_dorsal(parents)
    if dorsal is None:
        return metrics

    if len(parents) == 1:
        logging.debug("Single parent contour, classified Dorsal.")
        return metrics

    surfaces = ["Dorsal"]
    ventral = _identify_ventral(parents, dorsal, tolerance)
    if ventral:
        surfaces.append("Ventral")

    platform = _identify_platform(parents, dorsal)
    if platform:
        surfaces.append("Platform")

    lateral = _identify_lateral(
        parents, dorsal, ventral, platform, tolerance
    )
    if lateral:
        surfaces.append("Lateral")

    for parent in parents:
        if parent["surface_type"] is None:
            parent["surface_type"] = "Unclassified"

    logging.debug(
        "Classified surfaces: %s.", ", ".join(surfaces)
    )
    return metrics


def _identify_dorsal(parents: List[Dict]) -> Optional[Dict]:
    """Identify the dorsal surface (largest area)."""
    try:
        dorsal = max(parents, key=lambda p: p["area"])
        dorsal["surface_type"] = "Dorsal"
        return dorsal
    except ValueError:
        logging.error("Unable to identify dorsal surface.")
        return None


def _identify_ventral(
    parents: List[Dict],
    dorsal: Dict,
    tolerance: float
) -> Optional[Dict]:
    """Identify the ventral surface (similar to dorsal)."""
    for parent in parents:
        if parent["surface_type"] is not None:
            continue

        d_len = dorsal["technical_length"]
        d_wid = dorsal["technical_width"]
        d_area = dorsal["area"]

        len_match = (
            abs(parent["technical_length"] - d_len)
            <= tolerance * d_len
        )
        wid_match = (
            abs(parent["technical_width"] - d_wid)
            <= tolerance * d_wid
        )
        area_match = (
            abs(parent["area"] - d_area)
            <= tolerance * d_area
        )

        if len_match and wid_match and area_match:
            parent["surface_type"] = "Ventral"
            return parent

    return None


def _identify_platform(
    parents: List[Dict], dorsal: Dict
) -> Optional[Dict]:
    """Identify the platform surface (smallest, shorter)."""
    candidates = [
        p for p in parents
        if (p["surface_type"] is None
            and p["technical_length"] < dorsal["technical_length"]
            and p["technical_width"] < dorsal["technical_width"])
    ]
    if not candidates:
        return None

    platform = min(candidates, key=lambda p: p["area"])
    platform["surface_type"] = "Platform"
    return platform


def _identify_lateral(
    parents: List[Dict],
    dorsal: Dict,
    ventral: Optional[Dict],
    platform: Optional[Dict],
    tolerance: float
) -> Optional[Dict]:
    """Identify the lateral surface."""
    if platform is not None:
        return _lateral_with_platform(
            parents, dorsal, platform, tolerance
        )
    if ventral is not None:
        return _lateral_without_platform(
            parents, dorsal, tolerance
        )
    return None


def _lateral_with_platform(
    parents: List[Dict],
    dorsal: Dict,
    platform: Dict,
    tolerance: float
) -> Optional[Dict]:
    """Find lateral when platform exists."""
    d_len = dorsal["technical_length"]
    p_len = platform["technical_length"]

    for parent in parents:
        if parent["surface_type"] is not None:
            continue

        similar_length = (
            abs(parent["technical_length"] - d_len)
            <= tolerance * d_len
        )
        diff_from_platform = (
            abs(parent["technical_length"] - p_len)
            > tolerance * p_len
        )
        diff_width = (
            parent["technical_width"]
            != dorsal["technical_width"]
        )

        if similar_length and diff_from_platform and diff_width:
            parent["surface_type"] = "Lateral"
            return parent

    return None


def _lateral_without_platform(
    parents: List[Dict],
    dorsal: Dict,
    tolerance: float
) -> Optional[Dict]:
    """Find lateral when no platform but ventral exists."""
    d_len = dorsal["technical_length"]
    d_wid = dorsal["technical_width"]

    for parent in parents:
        if parent["surface_type"] is not None:
            continue

        similar_length = (
            abs(parent["technical_length"] - d_len)
            <= tolerance * d_len
        )
        diff_width = (
            abs(parent["technical_width"] - d_wid)
            > tolerance * d_wid
        )

        if similar_length and diff_width:
            parent["surface_type"] = "Lateral"
            return parent

    return None


def classify_child_features(
    metrics: List[Dict]
) -> List[Dict]:
    """
    Classify child contours based on parent surface type.

    Labeling rules:
    - Dorsal children: "scar 1", "scar 2", etc.
    - Platform children: excluded (empty space boundaries)
    - Lateral children: "edge 1", "edge 2", etc.
    - Ventral children: excluded from output

    Parameters
    ----------
    metrics : list
        Metric dicts with classified parents.

    Returns
    -------
    list
        Updated metrics with classified child features.
    """
    logging.debug("Starting child feature classification")

    try:
        parents = [
            m for m in metrics if m["parent"] == m["scar"]
        ]
        parent_labels = {p["scar"] for p in parents}

        children = [
            m for m in metrics
            if m["parent"] != m["scar"]
            and m["parent"] in parent_labels
        ]
        grandchildren = [
            m for m in metrics
            if m["parent"] != m["scar"]
            and m["parent"] not in parent_labels
        ]

        logging.debug(
            f"Hierarchy: {len(parents)} parents, "
            f"{len(children)} children, "
            f"{len(grandchildren)} grandchildren"
        )

        surface_map = {
            p["parent"]: p.get("surface_type", "Unknown")
            for p in parents
        }

        classified = _classify_children_by_surface(
            children, surface_map
        )

        result = parents + classified + grandchildren

        logging.debug(
            "Child feature classification completed: "
            f"{len(classified)} classified"
        )
        return result

    except Exception as e:
        logging.error(f"Error in child classification: {e}")
        return metrics


def _classify_children_by_surface(
    children: List[Dict],
    surface_map: Dict[str, str]
) -> List[Dict]:
    """
    Apply surface-specific labels to children.

    Parameters
    ----------
    children : list
        Child metric dictionaries.
    surface_map : dict
        Maps parent labels to surface types.

    Returns
    -------
    list
        Classified children (ventral/platform excluded).
    """
    groups: Dict[str, List[Dict]] = {
        "Dorsal": [], "Platform": [],
        "Lateral": [], "Ventral": [],
    }

    for child in children:
        surface = surface_map.get(
            child.get("parent", ""), "Unknown"
        )
        if surface in groups:
            groups[surface].append(child)
        else:
            logging.warning(
                f"Unknown surface '{surface}', "
                f"defaulting to Dorsal"
            )
            groups["Dorsal"].append(child)

    classified = []

    for i, child in enumerate(groups["Dorsal"]):
        child["scar"] = f"scar {i + 1}"
        child["surface_feature"] = f"scar {i + 1}"
        classified.append(child)
        logging.debug(f"Classified dorsal child as scar {i + 1}")

    for child in groups["Platform"]:
        area = child.get("area", 0)
        logging.debug(
            f"Excluding platform child (area={area}) "
            f"as likely empty space boundary"
        )

    for i, child in enumerate(groups["Lateral"]):
        child["scar"] = f"edge {i + 1}"
        child["surface_feature"] = f"edge {i + 1}"
        classified.append(child)
        logging.debug(f"Classified lateral child as edge {i + 1}")

    excluded = len(groups["Ventral"])
    if excluded > 0:
        logging.debug(
            f"Excluded {excluded} ventral children"
        )

    return classified
