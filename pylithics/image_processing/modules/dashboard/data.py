"""
Pure data-shaping helpers for the PyLithics Explorer dashboard.

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
# case"), which is reasonable for fields that are not curated here.
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
    "scar_count": "Scar count",
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
    Return a parenthesised unit suffix for ``field``.

    The suffix depends on the calibration method(s) present in ``df``:

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
        Path to a PyLithics ``results/`` output directory.

    Returns
    -------
    dict
        Keys:
        - ``metrics``: pandas DataFrame loaded from processed_metrics.csv.
        - ``processed_dir``: absolute path to ``results/``.
        - ``json_dir``: absolute path to ``results/json/`` if present,
          else None.
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
            f"There is no processed_metrics.csv at {csv_path}. "
            "Do the analysis first."
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
    Count the values shown on the Overview page.

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
    """Return the parent rows (dorsal, ventral, etc.); alias for ``_parent_rows``."""
    return _parent_rows(df)


def dorsal_scars(df: pd.DataFrame) -> pd.DataFrame:
    """Return the dorsal scar rows; alias for ``_dorsal_scar_rows``."""
    return _dorsal_scar_rows(df)


def scar_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-lithic aggregate: dorsal area, count, coverage, mean / sd scar size."""
    parents_d = df[
        (df["surface_type"] == "Dorsal")
        & (df["surface_feature"] == "Dorsal")
    ]
    scars = dorsal_scars(df)
    if parents_d.empty or scars.empty:
        return pd.DataFrame()

    parent_view = (
        parents_d[["image_id", "total_area", "scar_count"]]
        .rename(columns={
            "total_area": "dorsal_area",
            "scar_count": "num_scars",
        })
    )
    scar_agg = (
        scars.groupby("image_id")["total_area"]
        .agg(total_scar_area="sum", mean_scar_area="mean", sd_scar_area="std")
        .reset_index()
    )
    out = parent_view.merge(scar_agg, on="image_id", how="left").dropna(
        subset=["dorsal_area", "num_scars"],
    )
    out["coverage_pct"] = (
        out["total_scar_area"] / out["dorsal_area"]
    ).clip(0, 1) * 100
    out["cv"] = out["sd_scar_area"] / out["mean_scar_area"]
    out["calibration_method"] = parents_d.set_index("image_id").reindex(
        out["image_id"]
    )["calibration_method"].values
    return out


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
    """Return rows where the surface_feature equals the surface_type (the parent)."""
    if df.empty:
        return df
    return df[df["surface_feature"] == df["surface_type"]]


def _dorsal_scar_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return child rows on the Dorsal surface whose surface_feature starts 'scar'."""
    if df.empty:
        return df
    feature = df["surface_feature"].fillna("").str.lower()
    return df[
        (df["surface_type"] == "Dorsal")
        & feature.str.startswith("scar ")
    ]


def _cortex_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows flagged ``is_cortex == True``."""
    if df.empty or "is_cortex" not in df.columns:
        return df.iloc[0:0]
    return df[_as_bool(df["is_cortex"])]


def _scar_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return child rows whose ``surface_feature`` starts with 'scar '."""
    if df.empty:
        return df
    feature = df["surface_feature"].fillna("").str.lower()
    return df[
        (df["surface_feature"] != df["surface_type"])
        & feature.str.startswith("scar ")
    ]


def _scar_rows_with_arrows(df: pd.DataFrame) -> pd.DataFrame:
    """Return scar children that also have ``has_arrow == True``."""
    scars = _scar_rows(df)
    if scars.empty or "has_arrow" not in scars.columns:
        return scars.iloc[0:0]
    return scars[_as_bool(scars["has_arrow"])]


def _count_low_confidence_scales(df: pd.DataFrame) -> int:
    """Count distinct image_ids whose scale_confidence is below the threshold."""
    if df.empty or "scale_confidence" not in df.columns:
        return 0
    confidence = pd.to_numeric(df["scale_confidence"], errors="coerce")
    flagged = df[confidence.notna() & (confidence < LOW_SCALE_CONFIDENCE_THRESHOLD)]
    return int(flagged["image_id"].nunique())


def _count_unclassified_surfaces(df: pd.DataFrame) -> int:
    """Count parent rows whose ``surface_type`` is 'Unclassified'."""
    if df.empty:
        return 0
    parents = _parent_rows(df)
    return int((parents["surface_type"] == "Unclassified").sum())


def _count_zero_scar_lithics(df: pd.DataFrame) -> int:
    """Count lithics whose dorsal parent reports zero ``scar_count``."""
    if df.empty or "scar_count" not in df.columns:
        return 0
    parents = _parent_rows(df)
    dorsal_parents = parents[parents["surface_type"] == "Dorsal"]
    if dorsal_parents.empty:
        return 0
    counts = pd.to_numeric(
        dorsal_parents["scar_count"], errors="coerce",
    )
    flagged = dorsal_parents[counts.fillna(-1) == 0]
    return int(flagged["image_id"].nunique())


def _count_pixel_only_lithics(df: pd.DataFrame) -> int:
    """Count distinct lithics whose ``calibration_method`` is 'pixels'."""
    if df.empty or "calibration_method" not in df.columns:
        return 0
    pixels_only = df[df["calibration_method"] == "pixels"]
    return int(pixels_only["image_id"].nunique())


def _count_mixed_dpi(run_summary: Dict[str, Any]) -> int:
    """
    Count the distinct non-null DPI values across successful images.

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
    """Count successful images whose DPI could not be extracted from metadata."""
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


# ---------------------------------------------------------------------------
# Review — what to examine, why, and what to do
# ---------------------------------------------------------------------------

REVIEW_COLUMNS = ["check", "image_id", "item", "value", "reason", "what_to_do"]

# One entry for each Overview check, in the order of the tiles. The Overview
# help text, the Review page, the report and review.csv all use this text.
CHECKS = {
    "failed": {
        "title": "Failed",
        "meaning": "Images listed in meta_data.csv that gave no metrics.",
        "steps": [
            "Open pylithics.log in the results folder and find the image.",
            "Correct the image, or its row in meta_data.csv.",
            "Run the analysis again.",
        ],
    },
    "unclassified_surfaces": {
        "title": "Unclassified surfaces",
        "meaning": (
            "PyLithics cannot name these outlines as a dorsal, ventral, platform or "
            "lateral surface from their size. A ventral surface must be within 10 "
            "percent of the dorsal surface in length, width and area (the tolerance "
            "setting)."
        ),
        "steps": [
            "Open the labelled image of the lithic. Find the outline with the "
            "label Unclassified.",
            "If the outline is a view of the lithic, its size in the drawing does "
            "not agree with the dorsal view. You cannot correct this in PyLithics. "
            "Use the result as it is.",
            "If the outline is not part of the lithic, for example text, a number "
            "or a mark, remove it from the image. Then run the analysis again.",
        ],
    },
    "zero_scar_lithics": {
        "title": "Zero-scar lithics",
        "meaning": "Lithics whose dorsal face holds no scar.",
        "steps": [
            "Open the labelled image of the lithic.",
            "If the dorsal face is plain in the drawing, the result is correct.",
            "If scar lines are broken or missing, see Troubleshooting for the "
            "threshold settings.",
        ],
    },
    "low_confidence_scales": {
        "title": "Low-confidence scales",
        "meaning": (
            f"Scale bars with a confidence below {LOW_SCALE_CONFIDENCE_THRESHOLD:g}. "
            "Without a scale_box the confidence is the length of the bar divided by "
            "its thickness, divided by 10. A short, thick bar gets a low value even "
            "when it is measured correctly. A bar measured from a scale_box gets 0.95."
        ),
        "steps": [
            "Open the scale image named in scale_id.",
            "Make sure that the measured length is the bar alone, with no caption "
            "and no frame.",
            "If it is, the value stands. To record the box of the bar, cut the "
            "pages again with pylithics-pages.",
        ],
    },
    "pixel_only_lithics": {
        "title": "Pixel-only",
        "meaning": "Lithics measured in pixels, not millimetres.",
        "steps": [
            "Open meta_data.csv and find the row of the image.",
            "Type the printed length of the bar, in mm, in the scale column.",
            "Make sure that the scale image named in scale_id is in scales/.",
            "Run the analysis again.",
        ],
    },
    "mixed_dpi": {
        "title": "Mixed DPI",
        "meaning": (
            "Images with a DPI that is different from most of the batch. You cannot "
            "compare pixel measurements from images with different DPI values."
        ),
        "steps": [
            "Scan the listed images again at the DPI of the batch.",
            "As an alternative, calibrate each image with a scale bar.",
        ],
    },
    "missing_dpi": {
        "title": "Missing DPI",
        "meaning": (
            "Images whose files have no DPI tag. PyLithics uses the default settings "
            "for them. Thus the sizes that use the DPI are less accurate."
        ),
        "steps": [
            "Save the image again with its DPI.",
            "As an alternative, calibrate the image with a scale bar.",
        ],
    },
}


def review_items(
    df: pd.DataFrame,
    run_summary: Optional[Dict[str, Any]] = None,
    tolerance: float = 0.1,
) -> pd.DataFrame:
    """
    Return one row for each item that an Overview check marks for review.

    Columns are ``REVIEW_COLUMNS``. ``reason`` is written from the row's own
    numbers; ``what_to_do`` is the check's procedure. ``tolerance`` is the
    surface rule's, to say why a surface is unclassified. A clean run gives
    an empty frame with the columns.
    """
    summary = run_summary or {}
    rows = (
        _failed_items(summary)
        + _unclassified_items(df, tolerance)
        + _zero_scar_items(df)
        + _low_confidence_items(df)
        + _pixel_only_items(df)
        + _dpi_items(summary)
    )
    for row in rows:
        row["what_to_do"] = " ".join(
            f"{n}. {step}" for n, step in enumerate(CHECKS[row["check"]]["steps"], 1)
        )
        row["check"] = CHECKS[row["check"]]["title"]
    return pd.DataFrame(rows, columns=REVIEW_COLUMNS)


def _item(check: str, image_id: str, item: str, value: str, reason: str) -> Dict[str, str]:
    """Build one review row; ``what_to_do`` is added by :func:`review_items`."""
    return {"check": check, "image_id": image_id, "item": item,
            "value": value, "reason": reason}


def _failed_items(summary: Dict[str, Any]) -> list:
    """Return the images that gave no metrics, with the reason the run recorded."""
    return [
        _item("failed", entry.get("image_id", ""), "image", "",
              f"{entry.get('reason', 'Processing failed')}. The error is in pylithics.log.")
        for entry in summary.get("failed") or [] if isinstance(entry, dict)
    ]


def _unclassified_items(df: pd.DataFrame, tolerance: float) -> list:
    """Return each unclassified surface, with how far it is from the dorsal's size."""
    parents = _parent_rows(df)
    if parents.empty:
        return []
    items = []
    for _, row in parents[parents["surface_type"] == "Unclassified"].iterrows():
        same = parents[parents["image_id"] == row["image_id"]]
        dorsal = same[same["surface_type"] == "Dorsal"]
        where = ""
        if {"centroid_x", "centroid_y"} <= set(row.index):
            where = f" at ({row['centroid_x']:.0f}, {row['centroid_y']:.0f})"
        value = ", ".join(f"{k.split('_')[-1]} {_num(row.get(k))}" for k in _SIZE_KEYS)
        reason = (_size_reason(row, dorsal.iloc[0], tolerance) if not dorsal.empty
                  else "The image has no dorsal surface to compare it with.")
        taken = [t for t in ("Ventral", "Platform", "Lateral")
                 if t in set(same["surface_type"])]
        if taken:
            reason += f" The image already has: {', '.join(taken)}."
        items.append(_item("unclassified_surfaces", row["image_id"],
                           f"Unclassified{where}", value, reason))
    return items


_SIZE_KEYS = ("technical_length", "technical_width", "total_area")


def _size_reason(row: pd.Series, dorsal: pd.Series, tolerance: float) -> str:
    """Say which sizes differ from the dorsal's by more than the tolerance, and by how much."""
    parts = []
    for key in _SIZE_KEYS:
        a, b = pd.to_numeric(row.get(key), errors="coerce"), pd.to_numeric(
            dorsal.get(key), errors="coerce")
        if pd.isna(a) or pd.isna(b) or max(a, b) == 0:
            continue
        apart = abs(a - b) / max(a, b)
        if apart > tolerance:
            name = key.split("_")[-1]
            parts.append(f"{name} {_num(a)} against the dorsal's {_num(b)}, {apart:.1%} apart")
    if not parts:
        return (f"It is within {tolerance:.0%} of the dorsal in length, width and area. "
                "Another outline already has that surface type.")
    return "; ".join(parts).capitalize() + f". The tolerance is {tolerance:.0%}."


def _zero_scar_items(df: pd.DataFrame) -> list:
    """Return the lithics whose dorsal face reports no scar."""
    if df.empty or "scar_count" not in df.columns:
        return []
    parents = _parent_rows(df)
    dorsal = parents[parents["surface_type"] == "Dorsal"]
    zero = dorsal[pd.to_numeric(dorsal["scar_count"], errors="coerce").fillna(-1) == 0]
    return [_item("zero_scar_lithics", image_id, "Dorsal", "0",
                  "The dorsal face holds no scar contour.")
            for image_id in zero["image_id"].unique()]


def _low_confidence_items(df: pd.DataFrame) -> list:
    """Return the images whose scale confidence is below the threshold."""
    if df.empty or "scale_confidence" not in df.columns:
        return []
    confidence = pd.to_numeric(df["scale_confidence"], errors="coerce")
    low = df[confidence.notna() & (confidence < LOW_SCALE_CONFIDENCE_THRESHOLD)]
    items = []
    for image_id, rows in low.groupby("image_id", sort=True):
        value = float(pd.to_numeric(rows["scale_confidence"]).min())
        items.append(_item(
            "low_confidence_scales", image_id, "scale bar", f"{value:.2f}",
            f"The bar is {value * 10:.1f} times as long as it is thick.",
        ))
    return items


def _pixel_only_items(df: pd.DataFrame) -> list:
    """Return the images measured in pixels."""
    if df.empty or "calibration_method" not in df.columns:
        return []
    pixels = df[df["calibration_method"] == "pixels"]
    return [_item("pixel_only_lithics", image_id, "image", "pixels",
                  "PyLithics did not use a scale. The row has no scale value, or the "
                  "scale image is missing or has no bar.")
            for image_id in pixels["image_id"].unique()]


def _dpi_items(summary: Dict[str, Any]) -> list:
    """Return the images whose DPI is missing, or differs from most of the batch."""
    entries = [e for e in summary.get("successful") or [] if isinstance(e, dict)]
    known = [e.get("dpi") for e in entries if e.get("dpi") is not None]
    usual = pd.Series(known).mode().iloc[0] if known else None
    items = []
    for entry in entries:
        dpi = entry.get("dpi")
        if dpi is None:
            items.append(_item("missing_dpi", entry.get("image_id", ""), "image", "",
                               "The file carries no DPI tag."))
        elif len(set(known)) > 1 and dpi != usual:
            items.append(_item("mixed_dpi", entry.get("image_id", ""), "image",
                               f"{_num(dpi)} dpi", f"Most images are {_num(usual)} dpi."))
    return items


def _num(value: Any) -> str:
    """Write a number without needless decimals: 820, 0.77, 494454."""
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return ""
    return f"{number:.0f}" if float(number).is_integer() or abs(number) >= 100 else f"{number:.2f}"
