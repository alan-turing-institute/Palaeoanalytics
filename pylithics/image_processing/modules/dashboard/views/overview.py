"""Overview page — about-the-import view."""

import streamlit as st

from pylithics.image_processing.modules.dashboard.data import (
    LOW_SCALE_CONFIDENCE_THRESHOLD,
    overview_counts,
)


def render(bundle: dict) -> None:
    df = bundle["metrics"]

    st.header("Overview")
    if df.empty and bundle.get("run_summary") is None:
        st.info("No metrics found in processed_metrics.csv.")
        return

    counts = overview_counts(df, run_summary=bundle.get("run_summary"))

    st.subheader("Pipeline")
    _render_pipeline_row(counts)

    st.subheader("Calibration & DPI")
    _render_calibration_row(counts)

    st.subheader("Assemblage")
    _render_counts_row(counts)


# ---------------------------------------------------------------------------
# Row 1a — pipeline data quality
# ---------------------------------------------------------------------------


def _render_pipeline_row(counts: dict) -> None:
    """Four tiles surfacing pipeline-level problems."""
    cols = st.columns(4)

    with cols[0]:
        _good_only_metric(
            "Successful",
            counts["successful"],
            help_text=(
                "Lithics whose pipeline run wrote at least one row to "
                "processed_metrics.csv."
            ),
            zero_is_good=False,
        )

    with cols[1]:
        _good_only_metric(
            "Failed",
            counts["failed"],
            help_text=(
                "Images listed in metadata that did not produce metrics. "
                "See pylithics.log for the underlying error."
            ),
        )

    with cols[2]:
        _good_only_metric(
            "Unclassified surfaces",
            counts["unclassified_surfaces"],
            help_text=(
                "Parent contours that PyLithics could not assign to "
                "Dorsal, Ventral, Platform, or Lateral."
            ),
        )

    with cols[3]:
        _good_only_metric(
            "Zero-scar lithics",
            counts["zero_scar_lithics"],
            help_text=(
                "Lithics whose dorsal surface has no detected scars. Often "
                "indicates a thresholding or contour-detection issue."
            ),
        )


# ---------------------------------------------------------------------------
# Row 1b — calibration and DPI quality
# ---------------------------------------------------------------------------


def _render_calibration_row(counts: dict) -> None:
    """Four tiles surfacing scale/DPI issues across the batch."""
    cols = st.columns(4)

    with cols[0]:
        _good_only_metric(
            "Low-confidence scales",
            counts["low_confidence_scales"],
            help_text=(
                "Lithics whose scale-bar detection produced a confidence "
                f"below {LOW_SCALE_CONFIDENCE_THRESHOLD:g}. Review these "
                "before trusting their measurements."
            ),
        )

    with cols[1]:
        _good_only_metric(
            "Pixel-only",
            counts["pixel_only_lithics"],
            help_text=(
                "Lithics whose measurements are in pixels, not millimetres. "
                "If you intended to use scale-bar calibration, these likely "
                "had no usable scale image."
            ),
        )

    with cols[2]:
        _good_only_metric(
            "Mixed DPI",
            counts["mixed_dpi"],
            help_text=(
                "Number of distinct DPI values across the batch. Anything "
                "above 1 indicates inconsistent scanning. Drop scans of the "
                "outliers and rescan if pixel-only mode is in use."
            ),
        )

    with cols[3]:
        _good_only_metric(
            "Missing DPI",
            counts["missing_dpi"],
            help_text=(
                "Images whose source files contained no DPI metadata. "
                "PyLithics processes them with default settings, but "
                "DPI-aware scaling is degraded for these lithics."
            ),
        )


# ---------------------------------------------------------------------------
# Row 2 — assemblage volume
# ---------------------------------------------------------------------------


def _render_counts_row(counts: dict) -> None:
    """Five tiles describing the volume of features in the assemblage."""
    cols = st.columns(5)

    cols[0].metric("Lithics", counts["lithics"])
    cols[1].metric("Surfaces", counts["surfaces"])
    cols[2].metric("Scars", counts["scars"])
    cols[3].metric(
        "Scars with arrows",
        counts["scars_with_arrows"],
        help="Scars on dorsal surfaces with a detected directional arrow.",
    )
    cols[4].metric(
        "Cortex regions",
        counts["cortex_regions"],
        help="Child contours flagged as cortex by texture analysis.",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _good_only_metric(
    label: str,
    value: int,
    help_text: str = "",
    zero_is_good: bool = True,
) -> None:
    """
    Render a metric tile with an empty-state-friendly subline.

    When ``zero_is_good`` is True (default), a value of 0 shows an "All clear"
    caption underneath; non-zero values prompt the user to review the
    underlying records.
    """
    st.metric(label, value, help=help_text)
    if zero_is_good:
        if value == 0:
            st.markdown(
                "<p style='color:#2e7d32; font-size:0.875rem; "
                "margin-top:-0.5rem;'>✓ All clear</p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<p style='color:#c62828; font-size:0.875rem; "
                "margin-top:-0.5rem;'>⚠ Review</p>",
                unsafe_allow_html=True,
            )
    else:
        # Successful tile — no underline message; the number speaks for itself.
        pass
