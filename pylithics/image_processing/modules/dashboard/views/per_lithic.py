"""Per-Lithic Detail page — drill-down into a single image's results."""

import json

import streamlit as st

from pylithics.image_processing.modules.dashboard.data import (
    label,
    per_image_image_paths,
)


def render(bundle: dict) -> None:
    df = bundle["metrics"]

    st.header("Per-Lithic Detail")
    if df.empty:
        st.info("No metrics found in processed_metrics.csv.")
        return

    image_ids = sorted(df["image_id"].dropna().unique().tolist())
    if not image_ids:
        st.info("No image_ids found.")
        return

    image_id = st.selectbox("Select a lithic", image_ids)
    rows = df[df["image_id"] == image_id]

    paths = per_image_image_paths(bundle["processed_dir"], image_id)

    left, right = st.columns(2)
    with left:
        st.subheader("Labeled image")
        if paths["labeled"]:
            st.image(str(paths["labeled"]), use_container_width=True)
        else:
            st.info(f"No labeled image found for {image_id}.")
    with right:
        st.subheader("Voronoi diagram")
        if paths["voronoi"]:
            st.image(str(paths["voronoi"]), use_container_width=True)
        else:
            st.info(f"No Voronoi diagram found for {image_id}.")

    st.subheader("Metric rows")
    display_rows = rows.rename(columns={c: label(c) for c in rows.columns})
    st.dataframe(display_rows, use_container_width=True)

    if paths["json"]:
        st.subheader("Per-lithic JSON")
        with open(paths["json"]) as f:
            doc = json.load(f)
        st.json(doc, expanded=False)
