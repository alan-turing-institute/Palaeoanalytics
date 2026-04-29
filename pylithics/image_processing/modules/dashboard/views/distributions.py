"""Distributions page — histograms and scatter plots for assemblage metrics."""

import plotly.express as px
import streamlit as st

from pylithics.image_processing.modules.dashboard.data import (
    SURFACE_COLORS,
    dorsal_scars,
    filter_metrics,
    label,
    label_with_units,
    parent_rows,
)


def render(bundle: dict) -> None:
    df = bundle["metrics"]

    st.header("Distributions")
    if df.empty:
        st.info("No metrics found in processed_metrics.csv.")
        return

    surface_options = sorted(df["surface_type"].dropna().unique().tolist())
    calibration_options = sorted(
        df["calibration_method"].dropna().unique().tolist()
    )

    col1, col2 = st.columns(2)
    selected_surfaces = col1.multiselect(
        "Surface types", surface_options,
        default=surface_options, format_func=label,
    )
    selected_calibrations = col2.multiselect(
        "Calibration methods", calibration_options,
        default=calibration_options, format_func=label,
    )

    filtered = filter_metrics(
        df,
        surface_types=selected_surfaces,
        calibration_methods=selected_calibrations,
    )
    if filtered.empty:
        st.warning("No rows match the current filter.")
        return

    parents = parent_rows(filtered)

    st.subheader("Surface dimensions")
    _histogram_grid(parents)

    st.subheader("Length × width by surface type")
    _length_width_scatter(parents)

    st.subheader("Scars per dorsal surface")
    _scars_per_dorsal(filtered)

    st.subheader("Scar complexity")
    _scar_complexity_distribution(filtered)

    st.subheader("Symmetry")
    _symmetry_scatter(parents)

    st.subheader("Voronoi cells per dorsal")
    _voronoi_distribution(parents)


def _histogram_grid(parents):
    if parents.empty:
        st.info("No parent surface rows in filtered data.")
        return
    fields = ("technical_length", "technical_width", "total_area", "aspect_ratio")
    cols = st.columns(2)
    for i, field in enumerate(fields):
        if field not in parents.columns:
            continue
        series = parents[field].dropna()
        if series.empty:
            cols[i % 2].info(f"No values for {label(field)}")
            continue
        fig = px.histogram(
            parents, x=field, nbins=20,
            labels={field: label_with_units(parents, field)},
        )
        fig.update_layout(yaxis_title="Count")
        _force_integer_yaxis(fig, len(series))
        _suppress_si_suffix(fig.update_xaxes)
        cols[i % 2].plotly_chart(fig, use_container_width=True)


def _length_width_scatter(parents):
    needed = {"technical_length", "technical_width", "surface_type"}
    if not needed.issubset(parents.columns) or parents.empty:
        st.info("Not enough data for the length × width scatter.")
        return
    points = parents.dropna(subset=["technical_length", "technical_width"])
    fig = px.scatter(
        points,
        x="technical_length", y="technical_width",
        color="surface_type",
        color_discrete_map=SURFACE_COLORS,
        hover_data=["image_id", "surface_feature"],
        labels={
            "technical_length": label_with_units(parents, "technical_length"),
            "technical_width": label_with_units(parents, "technical_width"),
            "surface_type": label("surface_type"),
            "image_id": label("image_id"),
            "surface_feature": label("surface_feature"),
        },
    )
    _suppress_si_suffix(fig.update_xaxes)
    _suppress_si_suffix(fig.update_yaxes)
    st.plotly_chart(fig, use_container_width=True)


def _scars_per_dorsal(df):
    dorsal = df[
        (df["surface_type"] == "Dorsal")
        & (df["surface_feature"] == "Dorsal")
    ]
    if "total_dorsal_scars" not in dorsal.columns or dorsal.empty:
        st.info("No dorsal scar counts in the filtered data.")
        return
    counts = dorsal["total_dorsal_scars"].dropna().astype(float)
    if counts.empty:
        st.info("No dorsal scar counts in the filtered data.")
        return
    fig = px.histogram(counts, nbins=15)
    fig.update_layout(
        showlegend=False,
        xaxis_title="Scars per dorsal surface",
        yaxis_title="Count",
    )
    _force_integer_yaxis(fig, len(counts))
    st.plotly_chart(fig, use_container_width=True)


def _scar_complexity_distribution(df):
    scars = dorsal_scars(df)
    if scars.empty or "scar_complexity" not in scars.columns:
        st.info("No scar complexity data in the filtered selection.")
        return
    values = scars["scar_complexity"].dropna().astype(float)
    if values.empty:
        st.info("No scar complexity values in the filtered selection.")
        return
    mean = values.mean()
    fig = px.histogram(values, nbins=15)
    fig.update_layout(
        showlegend=False,
        xaxis_title=label("scar_complexity"),
        yaxis_title="Count",
    )
    fig.add_vline(
        x=mean, line_dash="dash",
        annotation_text=f"mean: {mean:.1f}",
        annotation_position="top right",
    )
    _force_integer_yaxis(fig, len(values))
    st.plotly_chart(fig, use_container_width=True)


def _symmetry_scatter(parents):
    needed = {"vertical_symmetry", "horizontal_symmetry", "image_id"}
    if not needed.issubset(parents.columns):
        st.info("Symmetry columns missing from filtered data.")
        return
    points = parents.dropna(subset=["vertical_symmetry", "horizontal_symmetry"])
    if points.empty:
        st.info("No symmetry values in the filtered selection.")
        return
    fig = px.scatter(
        points,
        x="vertical_symmetry", y="horizontal_symmetry",
        hover_data=["image_id"],
        labels={
            "vertical_symmetry": label("vertical_symmetry"),
            "horizontal_symmetry": label("horizontal_symmetry"),
            "image_id": label("image_id"),
        },
    )
    fig.add_shape(
        type="rect",
        x0=0.95, x1=1.0, y0=0.95, y1=1.0,
        fillcolor="lightgreen", opacity=0.25, line_width=0,
        layer="below",
    )
    fig.update_xaxes(range=[0, 1.05])
    fig.update_yaxes(range=[0, 1.05])
    st.plotly_chart(fig, use_container_width=True)


def _voronoi_distribution(parents):
    if "voronoi_num_cells" not in parents.columns:
        st.info("No Voronoi columns in filtered data.")
        return
    values = parents["voronoi_num_cells"].dropna().astype(float)
    if values.empty:
        st.info("No Voronoi cell counts in the filtered selection.")
        return
    fig = px.histogram(values, nbins=15)
    fig.update_layout(
        showlegend=False,
        xaxis_title=label("voronoi_num_cells"),
        yaxis_title="Count",
    )
    _force_integer_yaxis(fig, len(values))
    st.plotly_chart(fig, use_container_width=True)


def _force_integer_yaxis(fig, max_count: int) -> None:
    """
    Force the y-axis to show only integer ticks.

    For small ranges (≤ 20) we use ``dtick=1`` to label every integer.
    For larger ranges Plotly's auto-tick logic is fine — we just suppress
    the decimal point with ``tickformat='d'``.
    """
    if max_count <= 20:
        fig.update_yaxes(tick0=0, dtick=1, tickformat="d")
    else:
        fig.update_yaxes(tickformat="d")


def _suppress_si_suffix(update_axis) -> None:
    """
    Stop Plotly from rendering large values with SI suffixes (1M, 500k, …).

    ``,d`` formats integers with thousands separators. Pass either
    ``fig.update_xaxes`` or ``fig.update_yaxes`` as the callable.
    """
    update_axis(tickformat=",d")
