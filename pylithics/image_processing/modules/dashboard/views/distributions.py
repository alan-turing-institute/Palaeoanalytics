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

    size_tab, sym_tab, scar_tab, spatial_tab, cortex_tab = st.tabs([
        "Size & shape", "Symmetry", "Scars", "Spatial", "Cortex",
    ])

    with size_tab:
        _render_size_tab(parents)
    with sym_tab:
        _render_symmetry_tab(parents)
    with scar_tab:
        _render_scars_tab(filtered)
    with spatial_tab:
        _render_spatial_tab(parents)
    with cortex_tab:
        _render_cortex_tab(filtered)


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
            title="Technical length × width",
        )
    with cols[1]:
        _dim_scatter(
            parents, "max_length", "max_width",
            title="Max length × width",
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
            hovertemplate=(
                f"{surface_type}<br>Aspect ratio: %{{x:.2f}}<extra></extra>"
            ),
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
                "%{customdata[0]} (%{customdata[1]})<br>"
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
        hover_data=["image_id", "surface_feature"],
        title=title,
        labels={
            x_field: label_with_units(parents, x_field),
            y_field: label_with_units(parents, y_field),
            "surface_type": label("surface_type"),
            "image_id": label("image_id"),
            "surface_feature": label("surface_feature"),
        },
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
            "%{customdata[0]}<br>"
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
        st.markdown(
            "**What it shows.** Each dot is one dorsal surface. Its "
            "position tells you *how much* the artefact deviates from a "
            "perfectly symmetric outline, and *which way* it leans.\n\n"
            "**How to read it.** The centre of the plot represents perfect "
            "symmetry. Distance from the centre = how lopsided an artefact "
            "is. Direction from the centre = which way the extra mass sits "
            "(top, bottom, left, or right). The corner labels name each "
            "combination.\n\n"
            "**What to look for.** A tight cloud near the centre means a "
            "balanced assemblage. A cloud sitting *off* the centre suggests "
            "a systematic bias — possibly a knapping or drawing convention. "
            "An elongated cloud points to anisotropy: the artefacts vary "
            "along one axis more than the other. Two separate clouds may "
            "indicate a real subgroup (different reduction stages or raw "
            "materials).\n\n"
            "**How it's calculated.** Each dorsal contour is split at its "
            "centroid into four quadrants (top, bottom, left, right). The "
            "horizontal bias is `(right − left) / (right + left)`; the "
            "vertical bias is `(top − bottom) / (top + bottom)`. Both range "
            "from −1 to +1; zero means perfectly balanced on that axis."
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
                "%{customdata[0]}<br>"
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
        title="Cumulative distribution of symmetry (Dorsal surfaces)",
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
        st.markdown(
            "**What it shows.** Every dorsal surface in the filtered "
            "selection, sorted from least to most symmetric. Each dot is one "
            "artefact — hover to see its image_id.\n\n"
            "**How to read it.** Read *up* from any symmetry score on the "
            "x-axis to find what fraction of the assemblage scores at or "
            "below that value. The point where a curve crosses y = 50% is "
            "the **median**. A curve hugging the right edge means most "
            "artefacts are highly symmetric; a long left tail means some "
            "asymmetric outliers are pulling the distribution down.\n\n"
            "**The two lines.** Blue = vertical symmetry (top vs. bottom). "
            "Red = horizontal symmetry (left vs. right). Comparing them "
            "shows which axis is more consistent across the assemblage. If "
            "the lines diverge, one axis varies more than the other.\n\n"
            "**Why an ECDF instead of a histogram.** Histograms depend on "
            "bin choice and squash detail when scores cluster near 1.0 (as "
            "symmetry scores tend to). An ECDF shows every artefact, makes "
            "no binning choice, and lets you read percentiles directly off "
            "the y-axis — even at small N.\n\n"
            "**How the score is calculated.** "
            "`vertical_symmetry = 1 − |top − bottom| / (top + bottom)`, and "
            "the horizontal version is the same with left/right. Both run "
            "from 0 (maximally lopsided) to 1 (perfectly balanced)."
        )


# ---------------------------------------------------------------------------
# Tab 3 — Scars
# ---------------------------------------------------------------------------


def _render_scars_tab(df) -> None:
    if df.empty:
        st.info("No data in the filtered selection.")
        return

    st.subheader("Scars per dorsal surface")
    _scars_per_dorsal(df)

    st.subheader("Scar complexity")
    _scar_complexity_distribution(df)

    st.subheader("Scar size")
    _scar_size_distribution(df)

    st.subheader("Arrow detection rate per lithic")
    _arrow_rate_per_image(df)


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
    fig = px.histogram(counts, marginal="rug")
    fig.update_layout(
        showlegend=False,
        xaxis_title="Scars per dorsal surface",
        yaxis_title="Count",
    )
    _align_integer_bins(fig, counts)
    _force_integer_yaxis(fig, len(counts))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(_summary_caption(counts))


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
    fig = px.histogram(values, marginal="rug")
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
    _align_integer_bins(fig, values)
    _force_integer_yaxis(fig, len(values))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(_summary_caption(values))


def _scar_size_distribution(df):
    scars = dorsal_scars(df)
    if scars.empty or "total_area" not in scars.columns:
        st.info("No scar-size data in the filtered selection.")
        return
    values = scars["total_area"].dropna()
    if values.empty:
        st.info("No scar-size values in the filtered selection.")
        return
    fig = px.histogram(scars, x="total_area", nbins=20, marginal="rug",
                       labels={"total_area": label_with_units(scars, "total_area")})
    fig.update_layout(yaxis_title="Count")
    _force_integer_yaxis(fig, len(values))
    _suppress_si_suffix(fig.update_xaxes)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(_summary_caption(values))


def _arrow_rate_per_image(df):
    scars = dorsal_scars(df)
    if scars.empty or "has_arrow" not in scars.columns:
        st.info("No scar / arrow data in the filtered selection.")
        return

    import pandas as pd
    scars = scars.copy()
    # Coerce mixed bool / 'True'/'False' strings into a real bool series.
    scars["_arrow"] = scars["has_arrow"].astype(str).str.strip().str.lower().eq("true")

    rates = (
        scars.groupby("image_id")["_arrow"]
        .mean()
        .mul(100)
        .round(1)
        .reset_index()
        .rename(columns={"_arrow": "Arrow rate (%)"})
    )
    if rates.empty:
        st.info("No per-image arrow rates available.")
        return
    rates = rates.sort_values("Arrow rate (%)", ascending=False)
    fig = px.bar(
        rates, x="image_id", y="Arrow rate (%)",
        labels={"image_id": label("image_id")},
    )
    fig.update_layout(yaxis_title="Arrow rate (%)")
    fig.update_yaxes(range=[0, 105])
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4 — Spatial organization
# ---------------------------------------------------------------------------


def _render_spatial_tab(parents) -> None:
    dorsal = parents[parents["surface_type"] == "Dorsal"]
    if dorsal.empty:
        st.info("No dorsal surfaces in the filtered data.")
        return

    st.subheader("Voronoi cells per dorsal")
    _voronoi_distribution(dorsal)

    st.subheader("Convex hull area vs dorsal area")
    _hull_vs_dorsal_scatter(dorsal)

    st.subheader("Hull utilization (hull / dorsal area)")
    _hull_utilization_distribution(dorsal)


def _voronoi_distribution(dorsal):
    if "voronoi_num_cells" not in dorsal.columns:
        st.info("No Voronoi columns in filtered data.")
        return
    values = dorsal["voronoi_num_cells"].dropna().astype(float)
    if values.empty:
        st.info("No Voronoi cell counts in the filtered selection.")
        return
    fig = px.histogram(values, marginal="rug")
    fig.update_layout(
        showlegend=False,
        xaxis_title=label("voronoi_num_cells"),
        yaxis_title="Count",
    )
    _align_integer_bins(fig, values)
    _force_integer_yaxis(fig, len(values))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(_summary_caption(values))


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
        hover_data=["image_id"],
        labels={
            "total_area": label_with_units(points, "total_area"),
            "convex_hull_area": label_with_units(points, "convex_hull_area"),
            "image_id": label("image_id"),
        },
    )
    _suppress_si_suffix(fig.update_xaxes)
    _suppress_si_suffix(fig.update_yaxes)
    st.plotly_chart(fig, use_container_width=True)


def _hull_utilization_distribution(dorsal):
    needed = {"convex_hull_area", "total_area"}
    if not needed.issubset(dorsal.columns):
        st.info("Convex-hull columns missing from filtered data.")
        return
    points = dorsal.dropna(subset=["convex_hull_area", "total_area"])
    points = points[points["total_area"] > 0]
    if points.empty:
        st.info("No hull / area values in the filtered selection.")
        return
    ratio = points["convex_hull_area"] / points["total_area"]
    fig = px.histogram(ratio, nbins=15, marginal="rug")
    fig.update_layout(
        showlegend=False,
        xaxis_title="Hull / dorsal area",
        yaxis_title="Count",
    )
    _force_integer_yaxis(fig, len(ratio))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(_summary_caption(ratio))


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
