"""Distributions page — assemblage metrics organised into thematic tabs."""

import contextlib
import threading

import plotly.express as px
import plotly.graph_objects as go
import streamlit as _streamlit

from pylithics.image_processing.modules.dashboard.data import (
    SURFACE_COLORS,
    dorsal_scars,
    filter_metrics,
    label,
    label_with_units,
    parent_rows,
    scar_summary,
)

_target = threading.local()


class _Surface:
    """
    Streamlit, or a recorder while the dashboard report is built in this thread.

    The report shows the charts of this page. It runs the same functions
    with a recorder in place of Streamlit, so the page and the report draw
    one set of figures from one set of data.
    """

    def __getattr__(self, name):
        return getattr(getattr(_target, "recorder", None) or _streamlit, name)


st = _Surface()


@contextlib.contextmanager
def recording(recorder):
    """Send this page's output to ``recorder`` in this thread, for the report."""
    _target.recorder = recorder
    try:
        yield recorder
    finally:
        _target.recorder = None


def render(bundle: dict) -> None:
    """Render the Distributions page."""
    df = bundle["metrics"]

    st.header("Distributions")
    if df.empty:
        st.info("processed_metrics.csv has no metrics.")
        return

    filtered = _apply_top_filters(df)
    if filtered.empty:
        st.warning("No rows agree with the filter.")
        return

    tabs = st.tabs([name for name, _ in TABS])
    for tab, (_, draw) in zip(tabs, TABS):
        with tab:
            draw(filtered)


def render_all(df) -> None:
    """Draw every tab in turn, unfiltered, with a heading for each: for the report."""
    for name, draw in TABS:
        st.header(name)
        draw(df)


# ---------------------------------------------------------------------------
# Top-level filters (applied to every tab)
# ---------------------------------------------------------------------------


TABS = [
    ("Size & shape", lambda df: _render_size_tab(parent_rows(df))),
    ("Symmetry", lambda df: _render_symmetry_tab(parent_rows(df))),
    ("Scars", lambda df: _render_scars_tab(df)),
    ("Spatial", lambda df: _render_spatial_tab(parent_rows(df))),
]


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
        st.info("The selection has no surface rows.")
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
    """Draw a horizontal raincloud (violin + box + jittered points) per surface type."""
    if "aspect_ratio" not in parents.columns:
        st.info("The data has no aspect_ratio column.")
        return
    points = parents.dropna(subset=["aspect_ratio", "surface_type"])
    if points.empty:
        st.info("The selection has no aspect ratio values.")
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
    line near-black, following the tonal hierarchy of the palette.
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
    """Draw a lollipop chart: one stem + dot per parent surface, sorted by perimeter."""
    if "perimeter" not in parents.columns:
        st.info("The data has no perimeter column.")
        return
    points = parents.dropna(subset=["perimeter", "surface_type"]).copy()
    if points.empty:
        st.info("The selection has no perimeter values.")
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
        st.info(f"The data has no {x_field} column or no {y_field} column.")
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
        st.info("The selection has no dorsal surfaces. "
                "Symmetry is for dorsal surfaces only.")
        return

    cols = st.columns(2)
    with cols[0]:
        _asymmetry_direction_scatter(dorsal)
    with cols[1]:
        _symmetry_ecdf(dorsal)


def _asymmetry_direction_scatter(dorsal):
    """Draw a signed asymmetry scatter that shows which way each artefact leans."""
    needed = {"top_area", "bottom_area", "left_area", "right_area", "image_id"}
    if not needed.issubset(dorsal.columns):
        st.info("The data has no quadrant area columns.")
        return

    points = dorsal.dropna(
        subset=["top_area", "bottom_area", "left_area", "right_area"],
    ).copy()
    tb_sum = points["top_area"] + points["bottom_area"]
    lr_sum = points["left_area"] + points["right_area"]
    points = points[(tb_sum > 0) & (lr_sum > 0)].copy()
    if points.empty:
        st.info("The selection has no quadrant-area values.")
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
    with st.expander("About this chart"):
        st.write(
            "Each point is one dorsal surface. The horizontal axis shows the "
            "left–right balance of the dorsal area around its centroid. The "
            "vertical axis shows the top–bottom balance. A point near (0, 0) is "
            "symmetrical. A point far from (0, 0) is less symmetrical."
        )


def _symmetry_ecdf(dorsal):
    """Draw paired ECDFs of vertical and horizontal symmetry on a single axis."""
    needed = {"vertical_symmetry", "horizontal_symmetry"}
    if not needed.issubset(dorsal.columns):
        st.info("The data has no symmetry columns.")
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
        st.info("The selection has no symmetry values.")
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
    with st.expander("About this chart"):
        st.write(
            "Each curve shows the symmetry scores of the dorsal surfaces in the "
            "selection. For a score on the horizontal axis, the curve gives the "
            "proportion of surfaces with that score or a lower score. A score "
            "of 1 is fully symmetrical, and a score of 0 is not symmetrical. A "
            "curve to the right shows more symmetrical surfaces. A steep curve "
            "shows surfaces with similar scores. The blue curve is the vertical "
            "symmetry (top–bottom). The red curve is the horizontal symmetry "
            "(left–right)."
        )


# ---------------------------------------------------------------------------
# Tab 3 — Scars
# ---------------------------------------------------------------------------


def _render_scars_tab(df) -> None:
    if df.empty:
        st.info("The selection has no data.")
        return

    lithics = scar_summary(df)
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


def _linear_fit_xy(x, y):
    """Return ``(xs, ys)`` for a 1-degree fit line, or ``(None, None)``."""
    import numpy as np
    if len(x) < 2:
        return None, None
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(float(x.min()), float(x.max()), 50)
    return xs, slope * xs + intercept


def _flag_outliers(x, y, n_sd: float = 2.0):
    """Return a mask of points whose residual exceeds ``n_sd`` standard deviations."""
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
    """Draw a scatter of # scars vs. dorsal area, with linear fit and outlier halos."""
    if lithics.empty:
        st.info("The selection has no scar counts.")
        return
    pts = lithics.dropna(subset=["dorsal_area", "num_scars"])
    if pts.empty:
        st.info("The selection has no scar counts.")
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
    with st.expander("About this chart"):
        st.write(
            "Each point is one lithic. The dashed line is the linear fit. A red "
            "circle marks a lithic more than two standard deviations from the "
            "line. That lithic has more scars or fewer scars than the line "
            "gives for its dorsal area."
        )


def _scars_coverage_vs_area(lithics):
    """Draw a scatter of scar-coverage % vs. dorsal surface area."""
    if lithics.empty:
        st.info("The selection has no coverage data.")
        return
    pts = lithics.dropna(subset=["dorsal_area", "coverage_pct"])
    if pts.empty:
        st.info("The selection has no coverage data.")
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
    with st.expander("About this chart"):
        st.write(
            "Coverage is the percentage of the dorsal surface that the scars "
            "occupy. The ridges between scars, the edge of the flake and small "
            "gaps between scars are not scar area. Thus coverage is almost "
            "always less than 100 percent."
        )


def _scar_complexity_histogram(scars):
    """Draw the population-level histogram of scar complexity."""
    if scars.empty or "scar_complexity" not in scars.columns:
        st.info("The selection has no scar complexity data.")
        return
    values = scars["scar_complexity"].dropna().astype(float)
    if values.empty:
        st.info("The selection has no scar complexity values.")
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
    with st.expander("About this chart"):
        st.write(
            "Complexity is the number of other scars that a scar touches on the "
            "same surface. A high value shows a scar in a dense group of "
            "removals."
        )


def _per_lithic_complexity_strip(scars):
    """
    Draw a per-lithic sized-circle dot plot of scar complexity.

    Circle size is the count of scars at that (lithic, complexity) cell.
    Lithics are ordered by median complexity.
    """
    if scars.empty or "scar_complexity" not in scars.columns:
        st.info("The selection has no scar complexity data.")
        return
    s = scars.dropna(subset=["scar_complexity"]).copy()
    if s.empty:
        st.info("The selection has no scar complexity values.")
        return

    counts = s.groupby("image_id").size()
    multi = counts[counts >= 2].index
    s = s[s["image_id"].isin(multi)]
    if s.empty:
        st.info("The selection has no lithic with two or more scored scars.")
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
    with st.expander("About this chart"):
        st.write(
            "Each column is one lithic. Each circle shows one complexity value "
            "on that lithic. The size of the circle is the number of scars with "
            "that value. For example, a large circle at 2 shows many scars that "
            "each touch two other scars."
        )
        st.write(
            "The lithics are in sequence from left to right by their median "
            "complexity. A tall column shows a lithic with simple scars and "
            "connected scars. A short column shows a lithic with scars of "
            "similar complexity."
        )


def _scar_size_ecdf(scars):
    """Draw the ECDF of per-scar total_area on a log x-axis."""
    if scars.empty or "total_area" not in scars.columns:
        st.info("The selection has no scar-size data.")
        return
    values = scars[["image_id", "surface_feature", "total_area"]].dropna()
    values = values[values["total_area"] > 0]
    if values.empty:
        st.info("The selection has no scar-size values.")
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
    with st.expander("About this chart"):
        st.write(
            "The curve shows the area of each scar. For an area on the "
            "horizontal axis, the curve gives the proportion of scars with that "
            "area or a smaller area. The horizontal axis has a logarithmic "
            "scale. This scale makes the many small scars easy to see."
        )


def _scar_aspect_ecdf(scars):
    """Draw the ECDF of per-scar aspect_ratio."""
    if scars.empty or "aspect_ratio" not in scars.columns:
        st.info("The selection has no scar aspect-ratio data.")
        return
    values = scars[["image_id", "surface_feature", "aspect_ratio"]].dropna()
    values = values[values["aspect_ratio"] > 0]
    if values.empty:
        st.info("The selection has no scar aspect-ratio values.")
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
    with st.expander("About this chart"):
        st.write(
            "The curve shows the aspect ratio of each scar: the long side "
            "divided by the short side. A value of 1 is a square shape. A "
            "larger value is a longer, narrower shape. A steep curve near 1 "
            "shows that most scars are almost square. A long tail to the right "
            "shows some very long, narrow scars."
        )


def _scar_cv_vs_count(lithics):
    """Draw a scatter of within-lithic scar-size variability (CV) vs. # scars."""
    if lithics.empty:
        st.info("The selection has no scar-size variability data.")
        return
    pts = lithics.dropna(subset=["num_scars", "cv"])
    pts = pts[pts["num_scars"] >= 2]
    if pts.empty:
        st.info("The selection has no lithic with two or more scars.")
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
    with st.expander("About this chart"):
        st.write(
            "The coefficient of variation (CV) shows the difference in size "
            "between the scars of one lithic. A CV near 0 shows scars of almost "
            "the same size. A CV of more than 0.7 shows very small scars and "
            "very large scars on the same surface. The chart compares the CV "
            "with the number of scars."
        )


# ---------------------------------------------------------------------------
# Tab 4 — Spatial organization
# ---------------------------------------------------------------------------


def _render_spatial_tab(parents) -> None:
    dorsal = parents[parents["surface_type"] == "Dorsal"]
    if dorsal.empty:
        st.info("The selection has no dorsal surfaces.")
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
        st.info("The data has no Voronoi columns.")
        return
    values = dorsal["voronoi_num_cells"].dropna().astype(float)
    if values.empty:
        st.info("The selection has no Voronoi cell counts.")
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
        st.info("The data has no convex hull columns.")
        return
    points = dorsal.dropna(subset=["convex_hull_area", "total_area"])
    if points.empty:
        st.info("The selection has no hull / area values.")
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
    """Draw a lollipop chart: one stick per lithic, height = hull/dorsal ratio."""
    needed = {"convex_hull_area", "total_area", "image_id"}
    if not needed.issubset(dorsal.columns):
        st.info("The data has no convex hull columns.")
        return
    points = dorsal.dropna(
        subset=["convex_hull_area", "total_area", "image_id"],
    ).copy()
    points = points[points["total_area"] > 0]
    if points.empty:
        st.info("The selection has no hull / area values.")
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
    with st.expander("About this chart"):
        st.write(
            "For each lithic, PyLithics draws the convex hull around the scar "
            "centroids. The chart shows the hull area divided by the dorsal "
            "area. A high value shows scar centres across all of the dorsal "
            "surface. A low value shows scar centres in one area. The lithics "
            "are in sequence from the lowest value to the highest value. The "
            "dashed line at 1.0 is the maximum value."
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
        st.info("The selection has no cortex regions.")
        return

    st.subheader("Cortex percentage of dorsal surface")
    _cortex_percentage_distribution(cortex)

    st.subheader("Cortex region size")
    _cortex_area_distribution(cortex)


def _cortex_percentage_distribution(cortex):
    if "cortex_percentage" not in cortex.columns:
        st.info("The data has no cortex_percentage column.")
        return
    values = cortex["cortex_percentage"].dropna().astype(float)
    if values.empty:
        st.info("The data has no cortex percentage values.")
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
        st.info("The data has no cortex_area column.")
        return
    values = cortex["cortex_area"].dropna()
    if values.empty:
        st.info("The data has no cortex area values.")
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
    Force histogram bins of width 1 centered on integer values.

    Integer-spaced tick labels then sit directly under bar centers.

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
    """Return a one-line median / IQR / N summary to print underneath a histogram."""
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
