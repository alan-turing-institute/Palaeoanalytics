"""
PyLithics: Per-Lithic JSON Export
=================================

Writes one JSON file per lithic image with metrics nested under the
lithic → surfaces → features hierarchy. Mirrors the data in the
combined ``processed_metrics.csv`` but reorganized for downstream
consumers (per-artifact viewers, R / Python notebooks, GIS tools).

See ``.claude/specs/JsonOutput.md`` for the schema.
"""

import json
import logging
import math
import os
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = 1


# Surface-level metric keys taken straight from the metric dict.
_SURFACE_FIELDS = (
    "centroid_x", "centroid_y",
    "technical_width", "technical_length",
    "max_width", "max_length",
    "area", "aspect_ratio",
    "perimeter", "distance_to_max_width",
)

# Feature-level metric keys (children: scars, edges, cortex).
_FEATURE_FIELDS = (
    "centroid_x", "centroid_y",
    "max_width", "max_length",
    "area", "aspect_ratio",
    "perimeter",
    "voronoi_cell_area",
    "scar_complexity",
    "cortex_area", "cortex_percentage",
    "arrow_angle",
)

# Optional arrow-geometry fields (only included when present in the metric).
_OPTIONAL_ARROW_FIELDS = (
    "triangle_base_length", "triangle_height",
    "shaft_solidity", "tip_solidity",
)

# Voronoi sub-block keys, sourced from the parent metric.
_VORONOI_FIELDS = (
    ("voronoi_num_cells",   "num_cells"),
    ("voronoi_cell_area",   "cell_area"),
    ("convex_hull_width",   "convex_hull_width"),
    ("convex_hull_height",  "convex_hull_height"),
    ("convex_hull_area",    "convex_hull_area"),
)

# Symmetry sub-block keys, sourced from the parent metric.
_SYMMETRY_FIELDS = (
    "top_area", "bottom_area",
    "left_area", "right_area",
    "vertical_symmetry", "horizontal_symmetry",
)


def save_measurements_to_json(
    metrics: List[Dict],
    output_path: str,
    calibration_metadata: Optional[Dict] = None,
) -> None:
    """
    Write a per-lithic JSON file with metrics nested by surface and feature.

    Parameters
    ----------
    metrics : list of dict
        All metric dictionaries for a single image.
    output_path : str
        Destination file path. Parent directory is created if missing.
    calibration_metadata : dict, optional
        Calibration info (method, pixels_per_mm, scale_confidence).
    """
    document = _build_document(metrics, calibration_metadata)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2, default=_json_default)
    logging.debug("Saved per-lithic JSON to %s", output_path)


def _build_document(
    metrics: List[Dict],
    calibration_metadata: Optional[Dict],
) -> Dict[str, Any]:
    """Assemble the top-level JSON document for one lithic."""
    image_id = _resolve_image_id(metrics)
    parents = [m for m in metrics if m.get("parent") == m.get("scar")]
    children_by_parent = _group_children_by_parent(metrics)
    total_dorsal_scars = _count_dorsal_scars(metrics)

    surfaces = [
        _build_surface(parent, children_by_parent, total_dorsal_scars)
        for parent in parents
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "image_id": image_id,
        "calibration": _build_calibration_block(calibration_metadata),
        "surfaces": surfaces,
    }


def _resolve_image_id(metrics: List[Dict]) -> Optional[str]:
    """Pull the image_id from any metric that has it; None if absent."""
    for metric in metrics:
        if metric.get("image_id"):
            return metric["image_id"]
    return None


def _build_calibration_block(
    calibration_metadata: Optional[Dict],
) -> Dict[str, Any]:
    """Always emit the same calibration shape; values default to None."""
    metadata = calibration_metadata or {}
    return {
        "method": metadata.get("calibration_method"),
        "pixels_per_mm": _clean(metadata.get("pixels_per_mm")),
        "scale_confidence": _clean(metadata.get("scale_confidence")),
    }


def _group_children_by_parent(
    metrics: List[Dict],
) -> Dict[str, List[Dict]]:
    """Bucket child metrics under their parent surface label."""
    grouped: Dict[str, List[Dict]] = {}
    for metric in metrics:
        if metric.get("parent") == metric.get("scar"):
            continue
        parent_label = metric.get("parent")
        grouped.setdefault(parent_label, []).append(metric)
    return grouped


def _count_dorsal_scars(metrics: List[Dict]) -> int:
    """Match the CSV writer's count of scars on the dorsal surface."""
    return len([
        m for m in metrics
        if (m.get("parent") != m.get("scar")
            and "scar" in str(m.get("surface_feature", "")).lower())
    ])


def _build_surface(
    parent: Dict,
    children_by_parent: Dict[str, List[Dict]],
    total_dorsal_scars: int,
) -> Dict[str, Any]:
    """Build a single surface entry with its features array."""
    surface_type = parent.get("surface_type") or "Unclassified"
    features = [
        _build_feature(child)
        for child in children_by_parent.get(parent.get("scar"), [])
    ]

    surface: Dict[str, Any] = {
        "surface_type": surface_type,
        "surface_feature": parent.get("surface_feature") or surface_type,
    }
    for field in _SURFACE_FIELDS:
        surface[_csv_to_json_key(field)] = _clean(parent.get(field))

    surface["total_dorsal_scars"] = (
        total_dorsal_scars if surface_type == "Dorsal" else None
    )
    surface["voronoi"] = (
        _build_voronoi_block(parent) if surface_type == "Dorsal" else None
    )
    surface["symmetry"] = (
        _build_symmetry_block(parent) if surface_type == "Dorsal" else None
    )
    surface["lateral_convexity"] = (
        _clean(parent.get("lateral_convexity"))
        if surface_type == "Lateral" else None
    )
    surface["features"] = features
    return surface


def _build_voronoi_block(parent: Dict) -> Dict[str, Any]:
    """Voronoi values nested under the dorsal surface."""
    return {
        json_key: _clean(parent.get(metric_key))
        for metric_key, json_key in _VORONOI_FIELDS
    }


def _build_symmetry_block(parent: Dict) -> Dict[str, Any]:
    """Symmetry values nested under the dorsal surface."""
    return {field: _clean(parent.get(field)) for field in _SYMMETRY_FIELDS}


def _build_feature(child: Dict) -> Dict[str, Any]:
    """Build a single child feature entry."""
    feature: Dict[str, Any] = {
        "surface_feature": child.get("scar"),
    }
    for field in _FEATURE_FIELDS:
        feature[_csv_to_json_key(field)] = _clean(child.get(field))

    feature["is_cortex"] = bool(child.get("is_cortex", False))
    feature["has_arrow"] = bool(child.get("has_arrow", False))

    for key in _OPTIONAL_ARROW_FIELDS:
        if key in child:
            feature[key] = _clean(child[key])
    return feature


def _csv_to_json_key(metric_key: str) -> str:
    """Translate metric-dict keys to JSON keys where the names diverge."""
    if metric_key == "area":
        return "total_area"
    return metric_key


def _clean(value: Any) -> Any:
    """
    Coerce values for JSON output.

    - NaN / inf become None (JSON null).
    - Numpy scalars become Python scalars via ``item()``.
    - Everything else is returned unchanged for ``json.dump`` to handle.
    """
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, bool):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            converted = item()
            if isinstance(converted, float) and (
                math.isnan(converted) or math.isinf(converted)
            ):
                return None
            return converted
        except (ValueError, TypeError):
            return value
    return value


def _json_default(value: Any) -> Any:
    """Last-resort serializer for json.dump (tuples, sets, etc.)."""
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)
