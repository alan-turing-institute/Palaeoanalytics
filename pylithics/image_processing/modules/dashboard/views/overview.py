"""Overview page for the PyLithics dashboard."""

import pandas as pd
import plotly.express as px
import streamlit as st

from pylithics.image_processing.modules.dashboard.data import (
    SURFACE_COLORS,
    label,
    summarize_assemblage,
)


def render(bundle: dict) -> None:
    df = bundle["metrics"]
    summary = summarize_assemblage(df)

    st.header("Overview")
    if df.empty:
        st.info("No metrics found in processed_metrics.csv.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Lithics processed", summary["n_lithics"])
    col2.metric(
        "Scale-bar calibrated",
        summary["n_calibrated"],
        help="Lithics with a successful scale-bar calibration",
    )
    col3.metric(
        "Arrow detection rate",
        f"{summary['arrow_detection_rate'] * 100:.1f}%",
        help="Fraction of dorsal scars with a detected arrow",
    )
    col4.metric(
        "Cortex prevalence",
        f"{summary['cortex_prevalence'] * 100:.1f}%",
        help="Fraction of lithics with at least one cortex feature",
    )

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.subheader("Surface types")
        if summary["surface_counts"]:
            surface_df = pd.DataFrame(
                {
                    "Surface type": [
                        label(k) for k in summary["surface_counts"].keys()
                    ],
                    "Count": list(summary["surface_counts"].values()),
                }
            )
            fig = px.bar(
                surface_df, x="Surface type", y="Count",
                color="Surface type",
                color_discrete_map=SURFACE_COLORS,
            )
            fig.update_layout(showlegend=False)
            max_count = int(surface_df["Count"].max())
            if max_count <= 20:
                fig.update_yaxes(tick0=0, dtick=1, tickformat="d")
            else:
                fig.update_yaxes(tickformat="d")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No parent surfaces found.")

    with right:
        st.subheader("Calibration method")
        if summary["calibration_counts"]:
            calib_df = pd.DataFrame(
                {
                    "Method": [
                        label(k) for k in summary["calibration_counts"].keys()
                    ],
                    "Lithics": list(summary["calibration_counts"].values()),
                }
            )
            fig = px.pie(
                calib_df, names="Method", values="Lithics", hole=0.4,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No calibration metadata found.")
