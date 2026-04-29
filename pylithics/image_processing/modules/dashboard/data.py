"""
Pure data-shaping helpers for the PyLithics dashboard.

These functions know nothing about Streamlit. They load the canonical CSV,
inventory the per-image artifacts, and produce summary numbers and filtered
slices for the dashboard pages.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd


# Confidence below this is flagged as "low" on the Overview page.
LOW_SCALE_CONFIDENCE_THRESHOLD = 0.8


# Known column / value labels used across the dashboard UI.  Anything
# missing from this map falls through ``humanize()`` (snake_case → "Title
# case"), which is reasonable for fields we haven't curated.
LABELS = {
    "image_id": "Image ID",
    "surface_type": "Surface type",
    "surface_feature": "Surface feature",
    "centroid_x": "Centroid X",
    "centroid_y": "Centroid Y",
    "technical_width": "Technical width",
    "technical_length": "Technical length",
    "max_width": "Max width",
    "max_length": "Max length",
    "total_area": "Total area",
    "aspect_ratio": "Aspect ratio",
    "perimeter": "Perimeter",
    "distance_to_max_width": "Distance to max width",
    "voronoi_num_cells": "Voronoi cells",
    "voronoi_cell_area": "Voronoi cell area",
    "convex_hull_width": "Convex hull width",
    "convex_hull_height": "Convex hull height",
    "convex_hull_area": "Convex hull area",
    "top_area": "Top area",
    "bottom_area": "Bottom area",
    "left_area": "Left area",
    "right_area": "Right area",
    "vertical_symmetry": "Vertical symmetry",
    "horizontal_symmetry": "Horizontal symmetry",
    "lateral_convexity": "Lateral convexity",
    "is_cortex": "Is cortex",
    "cortex_area": "Cortex area",
    "cortex_percentage": "Cortex percentage",
    "scar_complexity": "Scar complexity",
    "has_arrow": "Has arrow",
    "arrow_angle": "Arrow angle",
    "calibration_method": "Calibration method",
    "pixels_per_mm": "Pixels per mm",
    "scale_confidence": "Scale confidence",
    "total_dorsal_scars": "Total dorsal scars",
    "scale_bar": "Scale bar",
    "pixels": "Pixels",
}


def humanize(value: Any) -> Any:
    """Replace underscores with spaces and capitalise; pass non-strings through."""
    if not isinstance(value, str):
        return value
    return value.replace("_", " ").strip().capitalize()


def label(value: Any) -> Any:
    """Look up a curated label, falling back to ``humanize`` for unknowns."""
    if isinstance(value, str) and value in LABELS:
        return LABELS[value]
    return humanize(value)


# Fields measured in linear units (length / position).
_LINEAR_FIELDS = frozenset({
    "centroid_x", "centroid_y",
    "technical_width", "technical_length",
    "max_width", "max_length",
    "perimeter", "distance_to_max_width",
    "convex_hull_width", "convex_hull_height",
})

# Fields measured in area units.
_AREA_FIELDS = frozenset({
    "total_area", "voronoi_cell_area", "convex_hull_area",
    "top_area", "bottom_area", "left_area", "right_area",
    "cortex_area",
})


def unit_suffix(df: pd.DataFrame, field: str) -> str:
    """
    Return a parenthesised unit suffix for ``field`` based on the
    calibration method(s) present in ``df``.

    - All rows ``scale_bar`` calibrated → ``(mm)`` / ``(mm²)``
    - All rows pixel-only             → ``(px)`` / ``(px²)``
    - Mixed or absent                 → ``""`` (no suffix)
    - Unitless field                  → ``""``
    """
    if field not in _LINEAR_FIELDS and field not in _AREA_FIELDS:
        return ""
    if df.empty or "calibration_method" not in df.columns:
        return ""
    methods = set(df["calibration_method"].dropna().unique().tolist())
    if methods == {"scale_bar"}:
        return " (mm²)" if field in _AREA_FIELDS else " (mm)"
    if methods == {"pixels"}:
        return " (px²)" if field in _AREA_FIELDS else " (px)"
    return ""


def label_with_units(df: pd.DataFrame, field: str) -> str:
    """Combine ``label(field)`` with ``unit_suffix(df, field)``."""
    return f"{label(field)}{unit_suffix(df, field)}"


# Surface-type palette informed by the labeled-image colors in
# ``pylithics/image_processing/modules/visualization.py``.  Dorsal uses the
# main surface purple; Platform and Lateral reuse the colors their *children*
# are drawn with (platform marks and lateral edges); Ventral takes the scar
# orange so it visually separates from the purples; Unclassified is gray.
SURFACE_COLORS = {
    "Dorsal": "rgb(94, 60, 153)",
    "Ventral": "rgb(253, 184, 99)",
    "Platform": "rgb(178, 171, 210)",
    "Lateral": "rgb(128, 205, 193)",
    "Unclassified": "rgb(128, 128, 128)",
}


def load_processed(processed_dir: str) -> Dict[str, Any]:
    """
    Load the canonical metrics CSV and inventory the supporting artifacts.

    Parameters
    ----------
    processed_dir : str
        Path to a PyLithics ``processed/`` output directory.

    Returns
    -------
    dict
        Keys:
        - ``metrics``: pandas DataFrame loaded from processed_metrics.csv.
        - ``processed_dir``: absolute path to ``processed/``.
        - ``json_dir``: absolute path to ``processed/json/`` if present, else None.
        - ``run_summary``: dict from ``run_summary.json`` if present, else None.

    Raises
    ------
    FileNotFoundError
        If ``processed_metrics.csv`` is missing.
    """
    processed_path = Path(processed_dir).resolve()
    csv_path = processed_path / "processed_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"processed_metrics.csv not found at {csv_path}. "
            "Run analysis first (provide --meta_file)."
        )

    metrics = pd.read_csv(csv_path)
    json_dir = processed_path / "json"
    return {
        "metrics": metrics,
        "processed_dir": processed_path,
        "json_dir": json_dir if json_dir.exists() else None,
        "run_summary": _load_run_summary(processed_path),
    }


def _load_run_summary(processed_path: Path) -> Optional[Dict[str, Any]]:
    """Read ``run_summary.json`` if present; tolerate older runs without one."""
    summary_path = processed_path / "run_summary.json"
    if not summary_path.exists():
        return None
    try:
        with open(summary_path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def summarize_assemblage(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute the headline numbers for the Overview page.

    Returns a dict with these keys (always present, sensible zeros on empty
    input so dashboards don't have to special-case missing data):

    - ``n_lithics``: distinct ``image_id`` count
    - ``n_calibrated``: lithics with ``calibration_method == 'scale_bar'``
    - ``arrow_detection_rate``: fraction of dorsal scars with ``has_arrow == True``
    - ``cortex_prevalence``: fraction of lithics with at least one cortex feature
    - ``surface_counts``: dict of surface_type -> row count
    - ``calibration_counts``: dict of calibration_method -> row count
    """
    if df.empty:
        return {
            "n_lithics": 0,
            "n_calibrated": 0,
            "arrow_detection_rate": 0.0,
            "cortex_prevalence": 0.0,
            "surface_counts": {},
            "calibration_counts": {},
        }

    parents = _parent_rows(df)

    n_lithics = int(df["image_id"].nunique())
    n_calibrated = int(
        df[df["calibration_method"] == "scale_bar"]["image_id"].nunique()
    )

    dorsal_scars = _dorsal_scar_rows(df)
    if dorsal_scars.empty:
        arrow_rate = 0.0
    else:
        arrow_rate = float(_as_bool(dorsal_scars["has_arrow"]).mean())

    cortex_lithics = _cortex_rows(df)["image_id"].nunique()
    cortex_prevalence = (
        float(cortex_lithics / n_lithics) if n_lithics else 0.0
    )

    surface_counts = (
        parents["surface_type"]
        .value_counts(dropna=False)
        .to_dict()
    )
    calibration_counts = (
        df.drop_duplicates("image_id")["calibration_method"]
        .value_counts(dropna=False)
        .to_dict()
    )

    return {
        "n_lithics": n_lithics,
        "n_calibrated": n_calibrated,
        "arrow_detection_rate": arrow_rate,
        "cortex_prevalence": cortex_prevalence,
        "surface_counts": surface_counts,
        "calibration_counts": calibration_counts,
    }


def overview_counts(
    df: pd.DataFrame,
    run_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, int]:
    """
    Counts powering the Overview page.

    The dashboard renders these as two rows of metric tiles: data quality
    (Row 1) and assemblage volume (Row 2). All values are integers; missing
    columns default to zero so a sparse run still renders.

    The ``failed`` count comes from ``run_summary.json`` when available; a
    missing manifest yields zero with no warning, since older runs predate it.
    """
    summary = run_summary or {}

    # Row 1a — pipeline data quality
    successful = (
        len(summary.get("successful", []))
        if "successful" in summary
        else (df["image_id"].nunique() if not df.empty else 0)
    )
    failed = len(summary.get("failed", []))
    unclassified_surfaces = _count_unclassified_surfaces(df)
    zero_scar_lithics = _count_zero_scar_lithics(df)

    # Row 1b — calibration & DPI
    low_confidence_scales = _count_low_confidence_scales(df)
    pixel_only_lithics = _count_pixel_only_lithics(df)
    mixed_dpi = _count_mixed_dpi(summary)
    missing_dpi = _count_missing_dpi(summary)

    # Row 2 — assemblage counts
    parents = _parent_rows(df)
    n_lithics = int(df["image_id"].nunique()) if not df.empty else 0
    n_surfaces = int(len(parents))
    n_scars = int(_scar_rows(df).shape[0])
    n_scars_with_arrows = int(_scar_rows_with_arrows(df).shape[0])
    n_cortex_regions = int(_cortex_rows(df).shape[0])

    return {
        "successful": int(successful),
        "failed": int(failed),
        "low_confidence_scales": int(low_confidence_scales),
        "unclassified_surfaces": int(unclassified_surfaces),
        "zero_scar_lithics": int(zero_scar_lithics),
        "pixel_only_lithics": int(pixel_only_lithics),
        "mixed_dpi": int(mixed_dpi),
        "missing_dpi": int(missing_dpi),
        "lithics": n_lithics,
        "surfaces": n_surfaces,
        "scars": n_scars,
        "scars_with_arrows": n_scars_with_arrows,
        "cortex_regions": n_cortex_regions,
    }


def filter_metrics(
    df: pd.DataFrame,
    surface_types: Optional[Iterable[str]] = None,
    calibration_methods: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Return rows matching the provided filter values.

    Empty / None filters are treated as "all values pass" so a dashboard
    page can hand its widget state through unchanged.
    """
    if surface_types:
        df = df[df["surface_type"].isin(list(surface_types))]
    if calibration_methods:
        df = df[df["calibration_method"].isin(list(calibration_methods))]
    return df


def parent_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Public alias for ``_parent_rows`` — the dorsal/ventral/etc. parents."""
    return _parent_rows(df)


def dorsal_scars(df: pd.DataFrame) -> pd.DataFrame:
    """Public alias for ``_dorsal_scar_rows``."""
    return _dorsal_scar_rows(df)


def per_image_image_paths(
    processed_dir: Path, image_id: str,
) -> Dict[str, Optional[Path]]:
    """
    Resolve the labeled image, voronoi diagram, and JSON file for ``image_id``.

    Each value is the absolute path or None if that artifact doesn't exist.
    """
    stem = os.path.splitext(image_id)[0]
    candidates = {
        "labeled": processed_dir / f"{stem}_labeled.png",
        "voronoi": processed_dir / f"{stem}_voronoi.png",
        "json": processed_dir / "json" / f"{stem}.json",
    }
    return {k: (p if p.exists() else None) for k, p in candidates.items()}


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _parent_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where the surface_feature equals the surface_type (the parent)."""
    if df.empty:
        return df
    return df[df["surface_feature"] == df["surface_type"]]


def _dorsal_scar_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Child rows on the Dorsal surface whose surface_feature starts 'scar'."""
    if df.empty:
        return df
    feature = df["surface_feature"].fillna("").str.lower()
    return df[
        (df["surface_type"] == "Dorsal")
        & feature.str.startswith("scar ")
    ]


def _cortex_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows flagged ``is_cortex == True``."""
    if df.empty or "is_cortex" not in df.columns:
        return df.iloc[0:0]
    return df[_as_bool(df["is_cortex"])]


def _scar_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Child rows whose ``surface_feature`` starts with 'scar '."""
    if df.empty:
        return df
    feature = df["surface_feature"].fillna("").str.lower()
    return df[
        (df["surface_feature"] != df["surface_type"])
        & feature.str.startswith("scar ")
    ]


def _scar_rows_with_arrows(df: pd.DataFrame) -> pd.DataFrame:
    """Scar children that also have ``has_arrow == True``."""
    scars = _scar_rows(df)
    if scars.empty or "has_arrow" not in scars.columns:
        return scars.iloc[0:0]
    return scars[_as_bool(scars["has_arrow"])]


def _count_low_confidence_scales(df: pd.DataFrame) -> int:
    """Distinct image_ids whose scale_confidence is below the threshold."""
    if df.empty or "scale_confidence" not in df.columns:
        return 0
    confidence = pd.to_numeric(df["scale_confidence"], errors="coerce")
    flagged = df[confidence.notna() & (confidence < LOW_SCALE_CONFIDENCE_THRESHOLD)]
    return int(flagged["image_id"].nunique())


def _count_unclassified_surfaces(df: pd.DataFrame) -> int:
    """Parent rows whose ``surface_type`` is 'Unclassified'."""
    if df.empty:
        return 0
    parents = _parent_rows(df)
    return int((parents["surface_type"] == "Unclassified").sum())


def _count_zero_scar_lithics(df: pd.DataFrame) -> int:
    """Lithics whose dorsal parent reports zero ``total_dorsal_scars``."""
    if df.empty or "total_dorsal_scars" not in df.columns:
        return 0
    parents = _parent_rows(df)
    dorsal_parents = parents[parents["surface_type"] == "Dorsal"]
    if dorsal_parents.empty:
        return 0
    counts = pd.to_numeric(
        dorsal_parents["total_dorsal_scars"], errors="coerce",
    )
    flagged = dorsal_parents[counts.fillna(-1) == 0]
    return int(flagged["image_id"].nunique())


def _count_pixel_only_lithics(df: pd.DataFrame) -> int:
    """Distinct lithics whose ``calibration_method`` is 'pixels'."""
    if df.empty or "calibration_method" not in df.columns:
        return 0
    pixels_only = df[df["calibration_method"] == "pixels"]
    return int(pixels_only["image_id"].nunique())


def _count_mixed_dpi(run_summary: Dict[str, Any]) -> int:
    """
    Number of distinct non-null DPI values across successful images.

    Returns 0 when fewer than two distinct values are found (the assemblage
    is consistent — nothing to flag). Returns N ≥ 2 when multiple DPIs are
    present, surfacing it as a quality concern.
    """
    successful = run_summary.get("successful") or []
    distinct = {
        entry.get("dpi") for entry in successful
        if isinstance(entry, dict) and entry.get("dpi") is not None
    }
    return len(distinct) if len(distinct) > 1 else 0


def _count_missing_dpi(run_summary: Dict[str, Any]) -> int:
    """Successful images whose DPI could not be extracted from metadata."""
    successful = run_summary.get("successful") or []
    return sum(
        1 for entry in successful
        if isinstance(entry, dict) and entry.get("dpi") is None
    )


def _as_bool(series: pd.Series) -> pd.Series:
    """Coerce mixed-type bool columns ('True'/'False'/True/False) to bool."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().eq("true")
