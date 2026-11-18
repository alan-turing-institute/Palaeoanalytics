"""Overview page — about-the-import view."""

import streamlit as st

from pylithics.image_processing.modules.dashboard.data import (
    CHECKS,
    overview_counts,
)


def render(bundle: dict) -> None:
    """Render the Overview page."""
    df = bundle["metrics"]

    st.header("Overview")
    if df.empty and bundle.get("run_summary") is None:
        st.info("processed_metrics.csv has no metrics.")
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
    """Render four tiles that surface pipeline-level problems."""
    cols = st.columns(4)

    with cols[0]:
        _good_only_metric(
            "Successful",
            counts["successful"],
            help_text=(
                "Lithics with one or more rows in processed_metrics.csv."
            ),
            zero_is_good=False,
        )

    with cols[1]:
        _good_only_metric(
            "Failed",
            counts["failed"],
            help_text=CHECKS["failed"]["meaning"] + " See the Review page.",
        )

    with cols[2]:
        _good_only_metric(
            "Unclassified surfaces",
            counts["unclassified_surfaces"],
            help_text=CHECKS["unclassified_surfaces"]["meaning"] + " See the Review page.",
        )

    with cols[3]:
        _good_only_metric(
            "Zero-scar lithics",
            counts["zero_scar_lithics"],
            help_text=CHECKS["zero_scar_lithics"]["meaning"] + " See the Review page.",
        )


# ---------------------------------------------------------------------------
# Row 1b — calibration and DPI quality
# ---------------------------------------------------------------------------


def _render_calibration_row(counts: dict) -> None:
    """Render four tiles that surface scale/DPI issues across the batch."""
    cols = st.columns(4)

    with cols[0]:
        _good_only_metric(
            "Low-confidence scales",
            counts["low_confidence_scales"],
            help_text=CHECKS["low_confidence_scales"]["meaning"] + " See the Review page.",
        )

    with cols[1]:
        _good_only_metric(
            "Pixel-only",
            counts["pixel_only_lithics"],
            help_text=CHECKS["pixel_only_lithics"]["meaning"] + " See the Review page.",
        )

    with cols[2]:
        _good_only_metric(
            "Mixed DPI",
            counts["mixed_dpi"],
            help_text=CHECKS["mixed_dpi"]["meaning"] + " See the Review page.",
        )

    with cols[3]:
        _good_only_metric(
            "Missing DPI",
            counts["missing_dpi"],
            help_text=CHECKS["missing_dpi"]["meaning"] + " See the Review page.",
        )


# ---------------------------------------------------------------------------
# Row 2 — assemblage volume
# ---------------------------------------------------------------------------


def _render_counts_row(counts: dict) -> None:
    """Render five tiles that describe the volume of features in the assemblage."""
    cols = st.columns(5)

    cols[0].metric("Lithics", counts["lithics"])
    cols[1].metric("Surfaces", counts["surfaces"])
    cols[2].metric("Scars", counts["scars"])
    cols[3].metric(
        "Scars with arrows",
        counts["scars_with_arrows"],
        help="Dorsal scars with a flaking arrow.",
    )
    cols[4].metric(
        "Cortex regions",
        counts["cortex_regions"],
        help="Scars that are cortex, from the stippling or from a letter such as C.",
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
