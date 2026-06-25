"""Distributions page — assemblage metrics organised into thematic tabs."""

import plotly.express as px
import plotly.graph_objects as go
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

    filtered = _apply_top_filters(df)
    if filtered.empty:
        st.warning("No rows match the current filter.")
        return

    parents = parent_rows(filtered)

    # size_tab, sym_tab, scar_tab, spatial_tab, cortex_tab = st.tabs([
    #     "Size & shape", "Symmetry", "Scars", "Spatial", "Cortex",
    # ])
    size_tab, sym_tab, scar_tab, spatial_tab = st.tabs([
        "Size & shape", "Symmetry", "Scars", "Spatial",
    ])

    with size_tab:
        _render_size_tab(parents)
    with sym_tab:
        _render_symmetry_tab(parents)
    with scar_tab:
        _render_scars_tab(filtered)
    with spatial_tab:
        _render_spatial_tab(parents)
    # with cortex_tab:
    #     _render_cortex_tab(filtered)


# ---------------------------------------------------------------------------
# Top-level filters (applied to every tab)
# ---------------------------------------------------------------------------


def _apply_top_filters(df):
    surface_options = sorted(df["surface_type"].dropna().unique().tolist())
    calibration_options = sorted(
        df["calibration_method"].dropna().unique().tolist()
    )

    # Default to Dorsal only when present; users can add other surfaces.
    default_surfaces = (
        ["Dorsal"] if "Dorsal" in surface_options else surface_options
    )

    col1, col2 = st.columns(2)
    selected_surfaces = col1.multiselect(
        "Surface types", surface_options,
        default=default_surfaces, format_func=label,
    )
    selected_calibrations = col2.multiselect(
        "Calibration methods", calibration_options,
        default=calibration_options, format_func=label,
    )

    return filter_metrics(
        df,
        surface_types=selected_surfaces,
        calibration_methods=selected_calibrations,
    )


# ---------------------------------------------------------------------------
# Tab 1 — Size & shape
# ---------------------------------------------------------------------------


def _render_size_tab(parents) -> None:
    if parents.empty:
        st.info("No parent surface rows in filtered data.")
        return

    st.subheader("Length × width")
    cols = st.columns(2)
    with cols[0]:
        _dim_scatter(
            parents, "technical_length", "technical_width",
            title="Technical length × technical width",
        )
    with cols[1]:
        _dim_scatter(
            parents, "max_length", "max_width",
            title="Max length × max width",
        )

    st.subheader("Perimeter and aspect ratio")
    cols = st.columns(2)
    with cols[0]:
        _perimeter_lollipop(parents)
    with cols[1]:
        _aspect_ratio_raincloud(parents)


def _aspect_ratio_raincloud(parents):
    """Horizontal raincloud (violin + box + jittered points) per surface type."""
    if "aspect_ratio" not in parents.columns:
        st.info("`aspect_ratio` column missing from filtered data.")
        return
    points = parents.dropna(subset=["aspect_ratio", "surface_type"])
    if points.empty:
        st.info("No aspect ratio values in the filtered selection.")
        return

    canonical_order = ("Dorsal", "Ventral", "Platform", "Lateral")
    present = list(points["surface_type"].unique())
    surface_order = [s for s in canonical_order if s in present]
    surface_order += [s for s in present if s not in canonical_order]

    fig = go.Figure()
    for surface_type in surface_order:
        group = points[points["surface_type"] == surface_type]
        base_rgb = SURFACE_COLORS.get(surface_type, "rgb(128, 128, 128)")
        cloud, _, box_fill, box_line = _hue_variants(base_rgb)

        n = len(group)
        if n <= 20:
            dot_size, dot_alpha, jitter = 7, 0.85, 0.6
        elif n <= 100:
            dot_size, dot_alpha, jitter = 5, 0.55, 0.5
        else:
            dot_size, dot_alpha, jitter = 4, 0.35, 0.4
        dot_color = _with_alpha(base_rgb, dot_alpha)

        has_id = "image_id" in group.columns
        customdata = (
            group[["image_id"]].to_numpy() if has_id else None
        )
        if has_id:
            hover = (
                "Lithic: %{customdata[0]}<br>"
                f"Surface type: {surface_type}<br>"
                "Aspect ratio: %{x:.2f}<extra></extra>"
            )
        else:
            hover = (
                f"Surface type: {surface_type}<br>"
                "Aspect ratio: %{x:.2f}<extra></extra>"
            )
        fig.add_trace(go.Violin(
            x=group["aspect_ratio"],
            y=[surface_type] * len(group),
            name=surface_type,
            orientation="h",
            fillcolor=cloud,
            line=dict(color=box_line, width=0.6),
            box_visible=True,
            box_fillcolor=box_fill,
            box_line_color=box_line,
            meanline_visible=False,
            points="all",
            jitter=jitter,
            pointpos=0,
            marker=dict(size=dot_size, color=dot_color, line=dict(width=0)),
            opacity=1,
            scalemode="width",
            spanmode="hard",
            width=0.55,
            customdata=customdata,
            hovertemplate=hover,
        ))

    fig.add_vline(
        x=1.0, line_dash="dash", line_color="#9aa0a6", line_width=1,
        annotation_text="H = W", annotation_position="top",
        annotation_font_color="#5f6368",
    )

    n_rows = max(1, len(surface_order))
    fig.update_layout(
        title="Aspect Ratio Distributions of Lithic Artefacts",
        xaxis_title="Aspect Ratio (Height / Width)",
        yaxis_title="Group",
        violinmode="overlay",
        showlegend=False,
        plot_bgcolor="white",
        height=180 + 110 * n_rows,
        margin=dict(l=70, r=20, t=60, b=50),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#ececec", zeroline=False,
                     showline=True, linecolor="#cfcfcf", mirror=False)
    fig.update_yaxes(showgrid=False, categoryorder="array",
                     categoryarray=list(reversed(surface_order)),
                     showline=True, linecolor="#cfcfcf", mirror=False)

    st.plotly_chart(fig, use_container_width=True)
    st.caption(_summary_caption(points["aspect_ratio"]))


def _parse_rgb(rgb_str: str) -> tuple:
    inner = rgb_str.strip()
    if inner.startswith("rgb"):
        inner = inner[inner.find("(") + 1:inner.rfind(")")]
        return tuple(int(v.strip()) for v in inner.split(","))
    h = inner.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _with_alpha(rgb_str: str, alpha: float) -> str:
    r, g, b = _parse_rgb(rgb_str)
    return f"rgba({r}, {g}, {b}, {alpha})"


def _hue_variants(rgb_str: str):
    """
    Derive raincloud layer colours from a single base hue.

    Returns ``(cloud, dots, box_fill, box_line)`` where each is a CSS colour
    string. Cloud is light/transparent, dots medium, box fill darker, box
    line near-black — matches the tonal hierarchy in the colour spec.
    """
    r, g, b = _parse_rgb(rgb_str)

    def shade(scale: float) -> tuple:
        return (int(r * scale), int(g * scale), int(b * scale))

    cloud = f"rgba({r}, {g}, {b}, 0.25)"
    dots = f"rgba({r}, {g}, {b}, 0.55)"
    bf_r, bf_g, bf_b = shade(0.65)
    box_fill = f"rgba({bf_r}, {bf_g}, {bf_b}, 0.85)"
    bl_r, bl_g, bl_b = shade(0.30)
    box_line = f"rgb({bl_r}, {bl_g}, {bl_b})"
    return cloud, dots, box_fill, box_line


def _perimeter_lollipop(parents):
    """Sorted lollipop: one stem + dot per parent surface, sorted by perimeter."""
    if "perimeter" not in parents.columns:
        st.info("`perimeter` column missing from filtered data.")
        return
    points = parents.dropna(subset=["perimeter", "surface_type"]).copy()
    if points.empty:
        st.info("No perimeter values in the filtered selection.")
        return

    points = points.sort_values("perimeter").reset_index(drop=True)
    points["_label"] = (
        points["image_id"].astype(str) + " (" + points["surface_type"].astype(str) + ")"
    )

    # Stems: a single Scatter trace, NaN-separated segments from baseline to value.
    stem_x, stem_y = [], []
    for label_val, perim in zip(points["_label"], points["perimeter"]):
        stem_x.extend([label_val, label_val, None])
        stem_y.extend([0, perim, None])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=stem_x, y=stem_y, mode="lines",
        line=dict(color="lightgray", width=1.5),
        hoverinfo="skip", showlegend=False,
    ))

    # Markers, one trace per surface type so legend / colours work.
    for surface_type, group in points.groupby("surface_type", sort=False):
        fig.add_trace(go.Scatter(
            x=group["_label"], y=group["perimeter"],
            mode="markers",
            marker=dict(size=10, color=SURFACE_COLORS.get(surface_type)),
            name=surface_type,
            customdata=group[["image_id", "surface_feature"]].to_numpy(),
            hovertemplate=(
                "Lithic: %{customdata[0]}<br>"
                "Surface: %{customdata[1]}<br>"
                f"{label_with_units(parents, 'perimeter')}: "
                "%{y}<extra></extra>"
            ),
        ))

    multi_surface = points["surface_type"].nunique() > 1
    fig.update_layout(
        title="Perimeter (sorted)",
        xaxis_title="",
        yaxis_title=label_with_units(parents, "perimeter"),
        showlegend=multi_surface,
        xaxis=dict(
            categoryorder="array",
            categoryarray=points["_label"].tolist(),
            tickangle=-45,
        ),
    )
    _suppress_si_suffix(fig.update_yaxes)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(_summary_caption(points["perimeter"]))


def _dim_scatter(parents, x_field, y_field, title=None):
    needed = {x_field, y_field, "surface_type"}
    if not needed.issubset(parents.columns):
        st.info(f"Columns {x_field} / {y_field} missing from filtered data.")
        return
    points = parents.dropna(subset=[x_field, y_field])
    if points.empty:
        st.info(f"No {label(x_field)} × {label(y_field)} values.")
        return
    fig = px.scatter(
        points,
        x=x_field, y=y_field,
        color="surface_type",
        color_discrete_map=SURFACE_COLORS,
        custom_data=["image_id", "surface_feature"],
        title=title,
        labels={
            x_field: label_with_units(parents, x_field),
            y_field: label_with_units(parents, y_field),
            "surface_type": label("surface_type"),
        },
    )
    fig.update_traces(
        hovertemplate=(
            "Lithic: %{customdata[0]}<br>"
            "Surface: %{customdata[1]}<br>"
            f"{label_with_units(parents, x_field)}: %{{x}}<br>"
            f"{label_with_units(parents, y_field)}: %{{y}}<extra></extra>"
        )
    )
    _suppress_si_suffix(fig.update_xaxes)
    _suppress_si_suffix(fig.update_yaxes)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2 — Surface symmetry
# ---------------------------------------------------------------------------


def _render_symmetry_tab(parents) -> None:
    dorsal = parents[parents["surface_type"] == "Dorsal"]
    if dorsal.empty:
        st.info("No dorsal surfaces in the filtered data — symmetry is dorsal-only.")
        return

    cols = st.columns(2)
    with cols[0]:
        _asymmetry_direction_scatter(dorsal)
    with cols[1]:
        _symmetry_ecdf(dorsal)


def _asymmetry_direction_scatter(dorsal):
    """Signed asymmetry scatter: which way each artefact leans."""
    needed = {"top_area", "bottom_area", "left_area", "right_area", "image_id"}
    if not needed.issubset(dorsal.columns):
        st.info("Quadrant area columns missing from filtered data.")
        return

    points = dorsal.dropna(
        subset=["top_area", "bottom_area", "left_area", "right_area"],
    ).copy()
    tb_sum = points["top_area"] + points["bottom_area"]
    lr_sum = points["left_area"] + points["right_area"]
    points = points[(tb_sum > 0) & (lr_sum > 0)].copy()
    if points.empty:
        st.info("No quadrant-area values in the filtered selection.")
        return

    points["vertical_bias"] = (
        (points["top_area"] - points["bottom_area"])
        / (points["top_area"] + points["bottom_area"])
    )
    points["horizontal_bias"] = (
        (points["right_area"] - points["left_area"])
        / (points["left_area"] + points["right_area"])
    )

    data_max = max(
        float(points["vertical_bias"].abs().max()),
        float(points["horizontal_bias"].abs().max()),
    )
    axis_max = max(0.1, data_max * 1.2)

    n = len(points)
    if n <= 20:
        size, alpha = 9, 0.85
    elif n <= 100:
        size, alpha = 6, 0.6
    else:
        size, alpha = 4, 0.4
    base_rgb = SURFACE_COLORS.get("Dorsal", "rgb(94, 60, 153)")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=points["horizontal_bias"], y=points["vertical_bias"],
        mode="markers",
        marker=dict(
            size=size, color=_with_alpha(base_rgb, alpha),
            line=dict(width=0),
        ),
        customdata=points[["image_id"]].to_numpy(),
        hovertemplate=(
            "Lithic: %{customdata[0]}<br>"
            "Horizontal bias: %{x:.3f}  (+ = right)<br>"
            "Vertical bias: %{y:.3f}  (+ = top)<extra></extra>"
        ),
        showlegend=False,
    ))

    fig.add_hline(y=0, line_color="#9aa0a6", line_width=1)
    fig.add_vline(x=0, line_color="#9aa0a6", line_width=1)

    pos = axis_max * 0.92
    fig.update_layout(
        title="Asymmetry direction (Dorsal surfaces)",
        xaxis_title="← left-leaning   |   right-leaning →",
        yaxis_title="← bottom-heavy   |   top-heavy →",
        plot_bgcolor="white",
        annotations=[
            dict(x=pos, y=pos, text="right-leaning<br>top-heavy",
                 showarrow=False, font=dict(color="#bdbdbd", size=10),
                 align="center"),
            dict(x=-pos, y=pos, text="left-leaning<br>top-heavy",
                 showarrow=False, font=dict(color="#bdbdbd", size=10),
                 align="center"),
            dict(x=-pos, y=-pos, text="left-leaning<br>bottom-heavy",
                 showarrow=False, font=dict(color="#bdbdbd", size=10),
                 align="center"),
            dict(x=pos, y=-pos, text="right-leaning<br>bottom-heavy",
                 showarrow=False, font=dict(color="#bdbdbd", size=10),
                 align="center"),
        ],
        height=480,
        margin=dict(l=60, r=20, t=60, b=80),
    )
    fig.update_xaxes(
        range=[-axis_max, axis_max], zeroline=False,
        showgrid=True, gridcolor="#ececec",
        showline=True, linecolor="#cfcfcf",
    )
    fig.update_yaxes(
        range=[-axis_max, axis_max], zeroline=False,
        showgrid=True, gridcolor="#ececec",
        showline=True, linecolor="#cfcfcf",
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Perfect symmetry: (0, 0) · "
        f"Axis range: ±{axis_max:.2f} · "
        f"N: {n} dorsal surfaces"
    )
    with st.expander("About this plot"):
        st.write(
            "Directional asymmetry of dorsal outlines relative to their "
            "centroid. The x-axis shows left–right bias and the y-axis "
            "shows top–bottom bias, calculated from the relative "
            "distribution of dorsal area around the centroid. Points "
            "near the origin are more symmetric; increasing distance "
            "from the origin indicates stronger asymmetry."
        )


def _symmetry_ecdf(dorsal):
    """Paired ECDFs of vertical and horizontal symmetry on a single axis."""
    needed = {"vertical_symmetry", "horizontal_symmetry"}
    if not needed.issubset(dorsal.columns):
        st.info("Symmetry columns missing from filtered data.")
        return

    has_id = "image_id" in dorsal.columns
    keep = ["image_id"] if has_id else []
    v_df = (
        dorsal[["vertical_symmetry", *keep]]
        .dropna(subset=["vertical_symmetry"])
        .sort_values("vertical_symmetry")
        .reset_index(drop=True)
    )
    h_df = (
        dorsal[["horizontal_symmetry", *keep]]
        .dropna(subset=["horizontal_symmetry"])
        .sort_values("horizontal_symmetry")
        .reset_index(drop=True)
    )
    if v_df.empty and h_df.empty:
        st.info("No symmetry values in the filtered selection.")
        return

    n_max = max(len(v_df), len(h_df))
    if n_max <= 30:
        marker_size, marker_alpha = 7, 0.9
    elif n_max <= 200:
        marker_size, marker_alpha = 5, 0.7
    else:
        marker_size, marker_alpha = 3, 0.5

    fig = go.Figure()
    series = [
        ("vertical_symmetry", v_df, "rgb(31, 119, 180)"),
        ("horizontal_symmetry", h_df, "rgb(214, 39, 40)"),
    ]
    for field, frame, color in series:
        if frame.empty:
            continue
        cumulative = (frame.index + 1) / len(frame)
        marker_color = _with_alpha(color, marker_alpha)
        if has_id:
            customdata = frame[["image_id"]].to_numpy()
            hovertemplate = (
                "Lithic: %{customdata[0]}<br>"
                f"{label(field)}: %{{x:.3f}}<br>"
                "Cumulative: %{y:.0%}<extra></extra>"
            )
        else:
            customdata = None
            hovertemplate = (
                f"{label(field)}: %{{x:.3f}}<br>"
                "Cumulative: %{y:.0%}<extra></extra>"
            )
        fig.add_trace(go.Scatter(
            x=frame[field].values, y=cumulative,
            mode="lines+markers", name=label(field),
            line=dict(color=color, width=2),
            marker=dict(
                size=marker_size, color=marker_color,
                line=dict(width=0),
            ),
            customdata=customdata,
            hovertemplate=hovertemplate,
        ))

    fig.add_vline(
        x=1.0, line_dash="dash", line_color="#9aa0a6", line_width=1,
        annotation_text="perfect", annotation_position="top",
        annotation_font_color="#5f6368",
    )

    x_floor = min(
        float(v_df["vertical_symmetry"].min()) if not v_df.empty else 1.0,
        float(h_df["horizontal_symmetry"].min()) if not h_df.empty else 1.0,
    )
    fig.update_layout(
        title="Cumulative Proportion of Dorsal Surfaces by Symmetry Score",
        xaxis_title="Symmetry score (1 = perfect)",
        yaxis_title="Cumulative proportion",
        plot_bgcolor="white",
        legend=dict(
            orientation="h", yanchor="bottom",
            y=-0.22, xanchor="left", x=0,
        ),
        height=480,
        margin=dict(l=60, r=20, t=60, b=80),
    )
    fig.update_xaxes(
        range=[max(0, x_floor - 0.02), 1.0],
        showgrid=True, gridcolor="#ececec",
        showline=True, linecolor="#cfcfcf",
    )
    fig.update_yaxes(
        range=[0, 1.0], tickformat=".0%",
        showgrid=True, gridcolor="#ececec",
        showline=True, linecolor="#cfcfcf",
    )

    st.plotly_chart(fig, use_container_width=True)
    parts = []
    if not v_df.empty:
        v_median = v_df["vertical_symmetry"].median()
        parts.append(f"{label('vertical_symmetry')} median: {v_median:.3f}")
    if not h_df.empty:
        h_median = h_df["horizontal_symmetry"].median()
        parts.append(f"{label('horizontal_symmetry')} median: {h_median:.3f}")
    n = max(len(v_df), len(h_df))
    parts.append(f"N: {n} dorsal surfaces")
    st.caption(" · ".join(parts))
    with st.expander("About this plot"):
        st.write(
            "This plot summarizes the distribution of symmetry scores "
            "across all dorsal surfaces in the current selection. For "
            "any symmetry value on the x-axis, the y-axis shows the "
            "proportion of specimens with scores at or below that "
            "value. Curves further to the right indicate greater "
            "overall symmetry, while steeper curves indicate less "
            "variation among specimens. Blue represents vertical "
            "symmetry (top–bottom balance) and red represents "
            "horizontal symmetry (left–right balance). Scores range "
            "from 0 (maximally asymmetric) to 1 (perfectly symmetric)."
        )


# ---------------------------------------------------------------------------
# Tab 3 — Scars
# ---------------------------------------------------------------------------


def _render_scars_tab(df) -> None:
    if df.empty:
        st.info("No data in the filtered selection.")
        return

    lithics = _build_scar_summary(df)
    scars = dorsal_scars(df)

    st.subheader("Scarring relationships")
    cols = st.columns(2)
    with cols[0]:
        _scars_count_vs_area(lithics)
    with cols[1]:
        _scars_coverage_vs_area(lithics)

    st.subheader("Scar complexity")
    cols = st.columns(2)
    with cols[0]:
        _scar_complexity_histogram(scars)
    with cols[1]:
        _per_lithic_complexity_strip(scars)

    st.subheader("Scar size & shape")
    cols = st.columns(2)
    with cols[0]:
        _scar_size_ecdf(scars)
    with cols[1]:
        _scar_aspect_ecdf(scars)

    st.subheader("Scar-size variability")
    _scar_cv_vs_count(lithics)


def _build_scar_summary(df):
    """Per-lithic aggregate: dorsal area, count, coverage, mean / sd scar size."""
    import pandas as pd
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


def _linear_fit_xy(x, y):
    """Return ``(xs, ys)`` for a 1-degree fit line, or ``(None, None)``."""
    import numpy as np
    if len(x) < 2:
        return None, None
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(float(x.min()), float(x.max()), 50)
    return xs, slope * xs + intercept


def _flag_outliers(x, y, n_sd: float = 2.0):
    """Boolean mask of points whose residual exceeds ``n_sd`` standard deviations."""
    import numpy as np
    if len(x) < 3:
        return np.zeros(len(x), dtype=bool)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    sd = float(np.std(resid))
    if sd == 0:
        return np.zeros(len(x), dtype=bool)
    return np.abs(resid) > n_sd * sd


def _scars_count_vs_area(lithics):
    """Scatter: # scars vs. dorsal area, with linear fit and outlier halos."""
    if lithics.empty:
        st.info("No scar counts in the filtered selection.")
        return
    pts = lithics.dropna(subset=["dorsal_area", "num_scars"])
    if pts.empty:
        st.info("No scar counts in the filtered selection.")
        return
    x = pts["dorsal_area"].astype(float).values
    y = pts["num_scars"].astype(float).values
    outliers = _flag_outliers(x, y)
    base = SURFACE_COLORS.get("Dorsal", "rgb(94, 60, 153)")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(size=8, color=_with_alpha(base, 0.65),
                    line=dict(width=1, color="white")),
        customdata=pts[["image_id"]].values,
        hovertemplate=(
            "Lithic: %{customdata[0]}<br>"
            "Dorsal surface area: %{x:,.1f}<br>"
            "Scars: %{y}<extra></extra>"
        ),
        name="Lithics",
    ))
    if outliers.any():
        fig.add_trace(go.Scatter(
            x=x[outliers], y=y[outliers], mode="markers",
            marker=dict(size=14, color="rgba(0,0,0,0)",
                        line=dict(width=2, color="#c62828")),
            customdata=pts.loc[outliers, ["image_id"]].values,
            hovertemplate=(
                "Lithic: %{customdata[0]} (outlier)<br>"
                "Dorsal surface area: %{x:,.1f}<br>"
                "Scars: %{y}<extra></extra>"
            ),
            name="> 2 SD from trend",
        ))
    xs, ys = _linear_fit_xy(x, y)
    if xs is not None:
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color=_with_alpha(base, 0.9), dash="dash"),
            hoverinfo="skip", name="Linear fit",
        ))

    fig.update_layout(
        xaxis_title=label_with_units(lithics, "dorsal_area")
                    if "dorsal_area" in lithics.columns
                    else "Dorsal surface area",
        yaxis_title="Number of scars",
        margin=dict(l=60, r=20, t=40, b=80),
        showlegend=False,
    )
    _suppress_si_suffix(fig.update_xaxes)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"N: {len(pts)} lithics · "
        f"Outliers (> 2 SD): {int(outliers.sum())}"
    )
    with st.expander("About this plot"):
        st.write(
            "Each dot is one lithic. The dashed line is the linear fit; "
            "lithics circled in red sit more than two residual standard "
            "deviations from that trend — they have unusually many or few "
            "scars for their dorsal surface area."
        )


def _scars_coverage_vs_area(lithics):
    """Scatter: scar-coverage % vs. dorsal surface area."""
    if lithics.empty:
        st.info("No coverage data in the filtered selection.")
        return
    pts = lithics.dropna(subset=["dorsal_area", "coverage_pct"])
    if pts.empty:
        st.info("No coverage data in the filtered selection.")
        return
    x = pts["dorsal_area"].astype(float).values
    y = pts["coverage_pct"].astype(float).values
    base = SURFACE_COLORS.get("Dorsal", "rgb(94, 60, 153)")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(size=8, color=_with_alpha(base, 0.65),
                    line=dict(width=1, color="white")),
        customdata=pts[["image_id"]].values,
        hovertemplate=(
            "Lithic: %{customdata[0]}<br>"
            "Dorsal surface area: %{x:,.1f}<br>"
            "Coverage: %{y:.1f}%<extra></extra>"
        ),
    ))
    median_cov = float(pts["coverage_pct"].median())
    fig.add_hline(
        y=median_cov, line_dash="dot", line_color="#666",
        annotation_text=f"median: {median_cov:.0f}%",
        annotation_position="top left",
    )

    fig.update_layout(
        xaxis_title=label_with_units(lithics, "dorsal_area")
                    if "dorsal_area" in lithics.columns
                    else "Dorsal surface area",
        yaxis_title="Scar coverage (% of dorsal area)",
        margin=dict(l=60, r=20, t=40, b=80),
        showlegend=False,
    )
    fig.update_yaxes(range=[0, 105])
    _suppress_si_suffix(fig.update_xaxes)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"N: {len(pts)} lithics · "
        f"Median coverage: {median_cov:.1f}%"
    )
    with st.expander("About this plot"):
        st.write(
            "Coverage is the fraction of the dorsal surface taken up by "
            "all scar polygons combined. Inter-scar ridges, the flake "
            "margin, and small contour gaps between adjacent scars are "
            "not counted as scar area, so coverage rarely reaches 100 % "
            "even on fully-worked dorsals. For cortex specifically, see "
            "the Cortex tab."
        )


def _scar_complexity_histogram(scars):
    """Population-level histogram of scar complexity (existing chart, refactored)."""
    if scars.empty or "scar_complexity" not in scars.columns:
        st.info("No scar complexity data in the filtered selection.")
        return
    values = scars["scar_complexity"].dropna().astype(float)
    if values.empty:
        st.info("No scar complexity values in the filtered selection.")
        return
    mean = float(values.mean())
    fig = px.histogram(values)
    fig.update_traces(
        hovertemplate=(
            "Scar complexity: %{x}<br>"
            "Number of scars: %{y}<extra></extra>"
        ),
        selector=dict(type="histogram"),
    )
    fig.update_layout(
        showlegend=False,
        xaxis_title="Scar complexity (neighbours touched)",
        yaxis_title="Number of scars",
        margin=dict(l=60, r=20, t=40, b=80),
    )
    fig.add_vline(
        x=mean, line_dash="dash",
        annotation_text=f"mean: {mean:.1f}",
        annotation_position="top right",
    )
    _align_integer_bins(fig, values)
    _force_integer_yaxis(fig, len(values))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"{_summary_caption(values)} scars")
    with st.expander("About this plot"):
        st.write(
            "Complexity counts how many topological connections each scar "
            "has to its neighbours on the surface. Higher values indicate "
            "scars embedded in denser knapping networks."
        )


def _per_lithic_complexity_strip(scars):
    """Per-lithic sized-circle dot plot: circle size = count of scars at
    that (lithic, complexity) cell. Lithics ordered by median complexity.
    """
    if scars.empty or "scar_complexity" not in scars.columns:
        st.info("No scar complexity data in the filtered selection.")
        return
    s = scars.dropna(subset=["scar_complexity"]).copy()
    if s.empty:
        st.info("No scar complexity values in the filtered selection.")
        return

    counts = s.groupby("image_id").size()
    multi = counts[counts >= 2].index
    s = s[s["image_id"].isin(multi)]
    if s.empty:
        st.info("No lithics with two or more scored scars.")
        return

    medians = s.groupby("image_id")["scar_complexity"].median().sort_values()
    order = [str(i) for i in medians.index.tolist()]
    s["image_id"] = s["image_id"].astype(str)

    grid = (
        s.groupby(["image_id", "scar_complexity"])
        .size().reset_index(name="count")
    )

    palette = px.colors.qualitative.Bold
    lithic_color = {
        lithic: _with_alpha(palette[i % len(palette)], 0.8)
        for i, lithic in enumerate(order)
    }
    colors = [lithic_color[lid] for lid in grid["image_id"]]

    max_count = int(grid["count"].max())
    min_px, max_px = 6, 28
    if max_count > 1:
        sizes = min_px + (grid["count"] - 1) * (
            (max_px - min_px) / (max_count - 1)
        )
    else:
        sizes = [min_px] * len(grid)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=grid["image_id"],
        y=grid["scar_complexity"],
        mode="markers",
        marker=dict(
            size=sizes,
            color=colors,
            line=dict(width=1, color="white"),
        ),
        customdata=grid[["image_id", "count"]].values,
        hovertemplate=(
            "Lithic: %{customdata[0]}<br>"
            "Scar complexity: %{y}<br>"
            "Number of scars: %{customdata[1]}<extra></extra>"
        ),
    ))
    fig.update_layout(
        xaxis_title="Lithic (ordered by median complexity)",
        yaxis_title="Scar complexity (neighbours touched)",
        margin=dict(l=60, r=20, t=40, b=80),
        showlegend=False,
    )
    fig.update_xaxes(
        categoryorder="array", categoryarray=order, showticklabels=False,
    )
    fig.update_yaxes(tickformat="d", dtick=1)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"{len(s)} scars across {len(multi)} lithics · "
        "circle size = number of scars at that complexity on each lithic"
    )
    with st.expander("About this plot"):
        st.write(
            "Each column is one lithic. Each circle marks a complexity "
            "value that occurs on that lithic, and the circle's size is "
            "the number of scars on that lithic with that exact "
            "complexity. A big circle at complexity 2 means the lithic "
            "has many scars that each touch two other scars."
        )
        st.write(
            "Lithics are ordered left-to-right by their median scar "
            "complexity. Tall vertical spreads of circles mean a lithic "
            "mixes simple and densely-connected scars; tight clusters "
            "mean its scars all sit at similar connectivity levels."
        )


def _scar_size_ecdf(scars):
    """ECDF of per-scar total_area on a log x-axis."""
    if scars.empty or "total_area" not in scars.columns:
        st.info("No scar-size data in the filtered selection.")
        return
    values = scars[["image_id", "surface_feature", "total_area"]].dropna()
    values = values[values["total_area"] > 0]
    if values.empty:
        st.info("No scar-size values in the filtered selection.")
        return
    s = values.sort_values("total_area").reset_index(drop=True)
    s["cumulative"] = (s.index + 1) / len(s)
    base = SURFACE_COLORS.get("Dorsal", "rgb(94, 60, 153)")
    size = 6 if len(s) <= 200 else (4 if len(s) <= 1000 else 3)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s["total_area"], y=s["cumulative"],
        mode="lines+markers",
        line=dict(color=_with_alpha(base, 0.9)),
        marker=dict(size=size, color=_with_alpha(base, 0.7)),
        customdata=s[["image_id", "surface_feature"]].values,
        hovertemplate=(
            "Lithic: %{customdata[0]}<br>"
            "Scar: %{customdata[1]}<br>"
            f"{label_with_units(scars, 'total_area')}: %{{x:,.1f}}<br>"
            "Larger than %{y:.0%} of all scars<extra></extra>"
        ),
    ))
    fig.update_layout(
        xaxis_title=label_with_units(scars, "total_area"),
        yaxis_title="Cumulative proportion",
        margin=dict(l=60, r=20, t=40, b=80),
        showlegend=False,
    )
    fig.update_xaxes(type="log")
    fig.update_yaxes(range=[0, 1.0], tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Median: {_fmt_number(s['total_area'].median())} · "
        f"N: {len(s)} scars"
    )
    with st.expander("About this plot"):
        st.write(
            "ECDF of individual scar areas on a log-scaled x-axis. The y "
            "value at any x reads as “the share of scars at or below this "
            "area”. Log scaling spreads the small scars (which dominate "
            "most assemblages) so the curve isn't squashed against zero."
        )


def _scar_aspect_ecdf(scars):
    """ECDF of per-scar aspect_ratio."""
    if scars.empty or "aspect_ratio" not in scars.columns:
        st.info("No scar aspect-ratio data in the filtered selection.")
        return
    values = scars[["image_id", "surface_feature", "aspect_ratio"]].dropna()
    values = values[values["aspect_ratio"] > 0]
    if values.empty:
        st.info("No scar aspect-ratio values in the filtered selection.")
        return
    s = values.sort_values("aspect_ratio").reset_index(drop=True)
    s["cumulative"] = (s.index + 1) / len(s)
    size = 6 if len(s) <= 200 else (4 if len(s) <= 1000 else 3)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s["aspect_ratio"], y=s["cumulative"],
        mode="lines+markers",
        line=dict(color="rgb(214, 39, 40)"),
        marker=dict(size=size, color="rgba(214, 39, 40, 0.7)"),
        customdata=s[["image_id", "surface_feature"]].values,
        hovertemplate=(
            "Lithic: %{customdata[0]}<br>"
            "Scar: %{customdata[1]}<br>"
            "Aspect ratio: %{x:.2f}<br>"
            "More elongated than %{y:.0%} of all scars<extra></extra>"
        ),
    ))
    fig.update_layout(
        xaxis_title=label("aspect_ratio"),
        yaxis_title="Cumulative proportion",
        margin=dict(l=60, r=20, t=40, b=80),
        showlegend=False,
    )
    fig.update_yaxes(range=[0, 1.0], tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Median: {s['aspect_ratio'].median():.2f} · "
        f"N: {len(s)} scars"
    )
    with st.expander("About this plot"):
        st.write(
            "ECDF of per-scar aspect ratio (longer / shorter side). A "
            "value of 1 is square-ish; larger values are more elongated. "
            "A steep early rise means most scars are roughly equant; "
            "long right tails reveal assemblages with a minority of very "
            "elongated removals."
        )


def _scar_cv_vs_count(lithics):
    """Scatter: within-lithic scar-size variability (CV) vs. # scars."""
    if lithics.empty:
        st.info("No scar-size variability data in the filtered selection.")
        return
    pts = lithics.dropna(subset=["num_scars", "cv"])
    pts = pts[pts["num_scars"] >= 2]
    if pts.empty:
        st.info("No lithics with two or more scars in the filtered selection.")
        return
    base = SURFACE_COLORS.get("Dorsal", "rgb(94, 60, 153)")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pts["num_scars"], y=pts["cv"], mode="markers",
        marker=dict(size=8, color=_with_alpha(base, 0.65),
                    line=dict(width=1, color="white")),
        customdata=pts[["image_id"]].values,
        hovertemplate=(
            "Lithic: %{customdata[0]}<br>"
            "Scars: %{x}<br>"
            "Size CV: %{y:.2f}<extra></extra>"
        ),
    ))
    median_cv = float(pts["cv"].median())
    fig.add_hline(
        y=median_cv, line_dash="dot", line_color="#666",
        annotation_text=f"median CV: {median_cv:.2f}",
        annotation_position="top left",
    )
    fig.update_layout(
        xaxis_title="Number of scars",
        yaxis_title="Scar-size coefficient of variation (sd / mean)",
        margin=dict(l=60, r=20, t=40, b=80),
        showlegend=False,
    )
    fig.update_xaxes(tickformat="d")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"N: {len(pts)} lithics with ≥ 2 scars · "
        f"Median CV: {median_cv:.2f}"
    )
    with st.expander("About this plot"):
        st.write(
            "Coefficient of variation captures how uniform a lithic's "
            "scars are in size. CV near 0 means the scars are nearly all "
            "the same size; CV above ~0.7 means the lithic mixes very "
            "small and very large scars on the same surface. Plotting "
            "against count separates monotonous reduction (low CV) from "
            "mixed-strategy reduction (high CV)."
        )


# ---------------------------------------------------------------------------
# Tab 4 — Spatial organization
# ---------------------------------------------------------------------------


def _render_spatial_tab(parents) -> None:
    dorsal = parents[parents["surface_type"] == "Dorsal"]
    if dorsal.empty:
        st.info("No dorsal surfaces in the filtered data.")
        return

    cols = st.columns(2)
    with cols[0]:
        st.subheader("Voronoi cells per dorsal surface")
        _voronoi_distribution(dorsal)
    with cols[1]:
        st.subheader("Scar-Centroid Dispersion (Hull Area / Dorsal Area)")
        _hull_utilization_distribution(dorsal)

    st.subheader("Convex hull area vs dorsal area")
    _hull_vs_dorsal_scatter(dorsal)


def _voronoi_distribution(dorsal):
    if "voronoi_num_cells" not in dorsal.columns:
        st.info("No Voronoi columns in filtered data.")
        return
    values = dorsal["voronoi_num_cells"].dropna().astype(float)
    if values.empty:
        st.info("No Voronoi cell counts in the filtered selection.")
        return
    fig = px.histogram(values)
    fig.update_traces(
        hovertemplate=(
            f"{label('voronoi_num_cells')}: %{{x}}<br>"
            "Number of dorsal surfaces: %{y}<extra></extra>"
        ),
        selector=dict(type="histogram"),
    )
    fig.update_layout(
        showlegend=False,
        xaxis_title=label("voronoi_num_cells"),
        yaxis_title="Number of dorsal surfaces",
        height=320,
        margin=dict(l=50, r=20, t=20, b=60),
        bargap=0.15,
    )
    _align_integer_bins(fig, values)
    _force_integer_yaxis(fig, len(values))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"{_summary_caption(values)} dorsal surfaces")


def _hull_vs_dorsal_scatter(dorsal):
    needed = {"convex_hull_area", "total_area", "image_id"}
    if not needed.issubset(dorsal.columns):
        st.info("Convex-hull columns missing from filtered data.")
        return
    points = dorsal.dropna(subset=["convex_hull_area", "total_area"])
    if points.empty:
        st.info("No hull / area values in the filtered selection.")
        return
    fig = px.scatter(
        points,
        x="total_area", y="convex_hull_area",
        custom_data=["image_id"],
        labels={
            "total_area": label_with_units(points, "total_area"),
            "convex_hull_area": label_with_units(points, "convex_hull_area"),
        },
    )
    fig.update_traces(
        hovertemplate=(
            "Lithic: %{customdata[0]}<br>"
            f"{label_with_units(points, 'total_area')}: %{{x}}<br>"
            f"{label_with_units(points, 'convex_hull_area')}: %{{y}}"
            "<extra></extra>"
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=60, r=20, t=20, b=60),
    )
    _suppress_si_suffix(fig.update_xaxes)
    _suppress_si_suffix(fig.update_yaxes)
    st.plotly_chart(fig, use_container_width=True)


def _hull_utilization_distribution(dorsal):
    """Sorted lollipop: one stick per lithic, height = hull/dorsal ratio."""
    needed = {"convex_hull_area", "total_area", "image_id"}
    if not needed.issubset(dorsal.columns):
        st.info("Convex-hull columns missing from filtered data.")
        return
    points = dorsal.dropna(
        subset=["convex_hull_area", "total_area", "image_id"],
    ).copy()
    points = points[points["total_area"] > 0]
    if points.empty:
        st.info("No hull / area values in the filtered selection.")
        return

    points["hull_ratio"] = (
        points["convex_hull_area"] / points["total_area"]
    )
    if "calibration_method" in points.columns:
        points["_area_unit"] = points["calibration_method"].apply(
            lambda m: "mm²" if m == "scale_bar" else "px²"
        )
    else:
        points["_area_unit"] = "px²"
    points = points.sort_values("hull_ratio").reset_index(drop=True)
    points["rank"] = points.index
    base = SURFACE_COLORS.get("Dorsal", "rgb(94, 60, 153)")

    stick_x, stick_y = [], []
    for _, row in points.iterrows():
        stick_x.extend([row["rank"], row["rank"], None])
        stick_y.extend([0.0, row["hull_ratio"], None])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=stick_x, y=stick_y, mode="lines",
        line=dict(color=_with_alpha(base, 0.5), width=2),
        hoverinfo="skip",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=points["rank"], y=points["hull_ratio"],
        mode="markers",
        marker=dict(
            size=10,
            color=_with_alpha(base, 0.9),
            line=dict(width=1, color="white"),
        ),
        customdata=points[
            ["image_id", "convex_hull_area", "total_area", "_area_unit"]
        ].values,
        hovertemplate=(
            "Lithic: %{customdata[0]}<br><br>"
            "Scar dispersion: %{y:.2f}<br><br>"
            "Scar-centroid hull area: %{customdata[1]:,.0f} %{customdata[3]}<br>"
            "Dorsal area: %{customdata[2]:,.0f} %{customdata[3]}<br><br>"
            "Centroid hull area = %{y:.0%} of dorsal area"
            "<extra></extra>"
        ),
        showlegend=False,
    ))
    fig.add_hline(
        y=1.0, line_dash="dash", line_color="#666",
        annotation_text="centroid hull = dorsal area (1.0)",
        annotation_position="top left",
    )
    fig.update_layout(
        xaxis_title="Lithic (sorted by centroid-hull ratio)",
        yaxis_title="Centroid-hull area / dorsal area",
        margin=dict(l=60, r=20, t=20, b=60),
        height=320,
    )
    fig.update_xaxes(showticklabels=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"{_summary_caption(points['hull_ratio'])} dorsal surfaces"
    )
    with st.expander("About this plot"):
        st.write(
            "Scar dispersion quantifies the spatial extent of dorsal "
            "scar-centroid distribution. For each lithic, the area of "
            "the convex hull enclosing scar centroids is expressed as a "
            "proportion of dorsal surface area. Higher ratios indicate "
            "a broader distribution of scar centres across the dorsal "
            "surface; lower ratios indicate a more localized "
            "concentration. Lithics are sorted from lowest to highest "
            "ratio. The dashed line at 1.0 denotes the theoretical "
            "upper limit."
        )


# ---------------------------------------------------------------------------
# Tab 5 — Cortex
# ---------------------------------------------------------------------------


def _render_cortex_tab(df) -> None:
    cortex = df[df["is_cortex"].astype(str).str.strip().str.lower() == "true"]
    n_lithics = df["image_id"].nunique() if not df.empty else 0
    n_cortex_lithics = cortex["image_id"].nunique() if not cortex.empty else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Lithics with cortex", n_cortex_lithics)
    col2.metric("Total cortex regions", len(cortex))
    if n_lithics:
        col3.metric(
            "Cortex prevalence",
            f"{n_cortex_lithics / n_lithics * 100:.1f}%",
        )

    if cortex.empty:
        st.info("No cortex regions detected in the filtered data.")
        return

    st.subheader("Cortex percentage of dorsal surface")
    _cortex_percentage_distribution(cortex)

    st.subheader("Cortex region size")
    _cortex_area_distribution(cortex)


def _cortex_percentage_distribution(cortex):
    if "cortex_percentage" not in cortex.columns:
        st.info("`cortex_percentage` column missing.")
        return
    values = cortex["cortex_percentage"].dropna().astype(float)
    if values.empty:
        st.info("No cortex-percentage values.")
        return
    fig = px.histogram(values, nbins=15, marginal="rug")
    fig.update_traces(
        hovertemplate=(
            "Cortex percentage: %{x:.1f}%<br>"
            "Number of cortex regions: %{y}<extra></extra>"
        ),
        selector=dict(type="histogram"),
    )
    fig.update_layout(
        showlegend=False,
        xaxis_title="Cortex percentage of parent surface (%)",
        yaxis_title="Count",
    )
    _force_integer_yaxis(fig, len(values))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(_summary_caption(values))


def _cortex_area_distribution(cortex):
    if "cortex_area" not in cortex.columns:
        st.info("`cortex_area` column missing.")
        return
    values = cortex["cortex_area"].dropna()
    if values.empty:
        st.info("No cortex-area values.")
        return
    fig = px.histogram(cortex, x="cortex_area", nbins=15, marginal="rug",
                       labels={"cortex_area": label_with_units(cortex, "cortex_area")})
    fig.update_traces(
        hovertemplate=(
            f"{label_with_units(cortex, 'cortex_area')}: %{{x:,.1f}}<br>"
            "Number of cortex regions: %{y}<extra></extra>"
        ),
        selector=dict(type="histogram"),
    )
    fig.update_layout(yaxis_title="Count")
    _force_integer_yaxis(fig, len(values))
    _suppress_si_suffix(fig.update_xaxes)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(_summary_caption(values))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _force_integer_yaxis(fig, max_count: int) -> None:
    """Force integer-only y-axis ticks for count histograms."""
    if max_count <= 20:
        fig.update_yaxes(tick0=0, dtick=1, tickformat="d")
    else:
        fig.update_yaxes(tickformat="d")


def _suppress_si_suffix(update_axis) -> None:
    """Stop Plotly rendering large values with SI suffixes (1M, 500k, …)."""
    update_axis(tickformat=",d")


def _align_integer_bins(fig, values) -> None:
    """
    Force histogram bins of width 1 centered on integer values, so
    integer-spaced tick labels sit directly under bar centers.

    Applies to the histogram trace only (not the marginal rug strip).
    """
    if values.empty:
        return
    start = float(values.min()) - 0.5
    end = float(values.max()) + 0.5
    fig.update_traces(
        xbins=dict(start=start, end=end, size=1),
        selector=dict(type="histogram"),
    )
    fig.update_xaxes(tick0=int(values.min()), dtick=1, tickformat="d")


def _summary_caption(values) -> str:
    """One-line median / IQR / N summary printed underneath a histogram."""
    n = len(values)
    if n == 0:
        return ""
    median = values.median()
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    return (
        f"Median: {_fmt_number(median)} · "
        f"IQR: {_fmt_number(q1)}–{_fmt_number(q3)} · "
        f"N: {n}"
    )


def _fmt_number(value) -> str:
    """Format a number with thousands separators; integer if whole."""
    try:
        v = float(value)
    except (ValueError, TypeError):
        return str(value)
    if v == int(v):
        return f"{int(v):,}"
    return f"{v:,.2f}"
