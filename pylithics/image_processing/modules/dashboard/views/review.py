"""Review page: what to examine, why, and what to do."""

import streamlit as st

from pylithics.image_processing.modules.dashboard.data import CHECKS, review_items


def render(bundle: dict) -> None:
    """Render one section for each check that marks something, then the clear checks."""
    st.header("Review")
    items = review_items(bundle["metrics"], bundle.get("run_summary"), _tolerance())
    clear = []
    for check in CHECKS.values():
        rows = items[items["check"] == check["title"]]
        if rows.empty:
            clear.append(check["title"])
            continue
        st.subheader(f"{check['title']}: {len(rows)}")
        st.markdown(check["meaning"])
        st.markdown("**What to do**\n\n" + "\n".join(
            f"{n}. {step}" for n, step in enumerate(check["steps"], 1)))
        st.dataframe(rows[["image_id", "item", "value", "reason"]],
                     hide_index=True, use_container_width=True)
    if clear:
        st.success(f"Clear: {', '.join(clear)}.")
    if items.empty:
        st.info("This run has no items to examine.")


def _tolerance() -> float:
    """Return the surface rule's tolerance from the configuration, or its default."""
    try:
        from pylithics.image_processing.config import get_surface_classification_config
        return float(get_surface_classification_config().get("tolerance", 0.1))
    except Exception:   # the configuration is not needed to show the page
        return 0.1
