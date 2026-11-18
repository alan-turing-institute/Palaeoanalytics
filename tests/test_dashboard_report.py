"""Tests for the dashboard's Review rows and its CSV, HTML and PDF export."""

import csv
import io

import pandas as pd
import pytest

from pylithics.image_processing.modules.dashboard import report
from pylithics.image_processing.modules.dashboard.data import (
    CHECKS,
    REVIEW_COLUMNS,
    overview_counts,
    review_items,
)


def _parent(image_id, surface, length, width, area, **extra):
    row = {"image_id": image_id, "surface_type": surface, "surface_feature": surface,
           "technical_length": length, "technical_width": width, "total_area": area,
           "centroid_x": 100.0, "centroid_y": 50.0, "is_cortex": False, "has_arrow": False,
           "calibration_method": "scale_bar", "scale_confidence": 0.95,
           "aspect_ratio": length / width, "perimeter": 2 * (length + width)}
    row.update(extra)
    return row


def _scar(image_id, number, area):
    return {"image_id": image_id, "surface_type": "Dorsal",
            "surface_feature": f"scar {number}", "total_area": area,
            "scar_complexity": 2, "is_cortex": False, "has_arrow": False,
            "calibration_method": "scale_bar"}


@pytest.fixture
def run():
    """Three lithics: one clean, one with an unclassified face and a weak scale, one in pixels."""
    rows = [
        _parent("clean.png", "Dorsal", 80, 60, 4800, scar_count=2),
        _scar("clean.png", 1, 400), _scar("clean.png", 2, 300),
        _parent("clean.png", "Ventral", 80, 60, 4750),
        _parent("odd.png", "Dorsal", 820, 858, 572254, scar_count=1, scale_confidence=0.77),
        _scar("odd.png", 1, 9000),
        _parent("odd.png", "Unclassified", 817, 771, 494454, scale_confidence=0.77),
        _parent("odd.png", "Platform", 122, 379, 33568, scale_confidence=0.77),
        _parent("pixels.png", "Dorsal", 60, 40, 2400, scar_count=0,
                calibration_method="pixels", scale_confidence=None),
    ]
    summary = {
        "successful": [{"image_id": "clean.png", "dpi": 300},
                       {"image_id": "odd.png", "dpi": 300},
                       {"image_id": "pixels.png", "dpi": 150},
                       {"image_id": "untagged.png", "dpi": None}],
        "failed": [{"image_id": "broken.png", "reason": "Processing failed"}],
    }
    df = pd.DataFrame(rows)
    return {"metrics": df, "run_summary": summary, "processed_dir": "results"}


def _by_check(items):
    return {title: rows for title, rows in items.groupby("check")}


@pytest.mark.unit
class TestReviewItems:

    def test_one_row_for_each_item_of_each_check(self, run):
        items = review_items(run["metrics"], run["run_summary"])
        assert list(items.columns) == REVIEW_COLUMNS
        counts = items["check"].value_counts().to_dict()
        assert counts == {"Failed": 1, "Unclassified surfaces": 1, "Zero-scar lithics": 1,
                          "Low-confidence scales": 1, "Pixel-only": 1, "Mixed DPI": 1,
                          "Missing DPI": 1}

    def test_the_unclassified_reason_gives_the_sizes_against_the_dorsal(self, run):
        row = _by_check(review_items(run["metrics"], run["run_summary"]))[
            "Unclassified surfaces"].iloc[0]
        assert "Width 771 against the dorsal's 858, 10.1% apart" in row["reason"]
        assert "13.6% apart" in row["reason"] and "The tolerance is 10%" in row["reason"]
        assert "already has: Platform" in row["reason"]
        assert row["value"] == "length 817, width 771, area 494454"

    def test_the_other_reasons_are_written_from_the_row(self, run):
        checks = _by_check(review_items(run["metrics"], run["run_summary"]))
        assert checks["Low-confidence scales"].iloc[0]["reason"] == (
            "The bar is 7.7 times as long as it is thick.")
        assert checks["Mixed DPI"].iloc[0]["image_id"] == "pixels.png"
        assert checks["Mixed DPI"].iloc[0]["reason"] == "Most images are 300 dpi."
        assert checks["Missing DPI"].iloc[0]["image_id"] == "untagged.png"
        assert checks["Failed"].iloc[0]["image_id"] == "broken.png"

    def test_what_to_do_is_the_check_s_numbered_procedure(self, run):
        row = _by_check(review_items(run["metrics"], run["run_summary"]))["Failed"].iloc[0]
        assert row["what_to_do"].startswith("1. " + CHECKS["failed"]["steps"][0])
        assert "3. Run the analysis again." in row["what_to_do"]

    def test_a_clean_run_gives_an_empty_frame_with_the_columns(self, run):
        clean = run["metrics"][run["metrics"]["image_id"] == "clean.png"]
        items = review_items(clean, {"successful": [{"image_id": "clean.png", "dpi": 300}]})
        assert items.empty and list(items.columns) == REVIEW_COLUMNS


@pytest.mark.unit
class TestExport:

    def test_csv_round_trips_and_has_a_header_when_clean(self, run, tmp_path):
        items = review_items(run["metrics"], run["run_summary"])
        rows = list(csv.DictReader(io.StringIO(report.report_csv(items))))
        assert len(rows) == len(items) and rows[0]["check"] == "Failed"
        path = report.write_review_csv(items.iloc[0:0], str(tmp_path))
        assert open(path).read().strip() == ",".join(REVIEW_COLUMNS)

    def test_html_holds_every_page_and_loads_nothing_from_the_network(self, run):
        items = review_items(run["metrics"], run["run_summary"])
        counts = overview_counts(run["metrics"], run["run_summary"])
        page = report.report_html(run, counts, items)
        for heading in ("Overview", "Distributions", "Review", "Size &amp; shape",
                        "Scars", "Unclassified surfaces", "odd.png", "broken.png"):
            assert heading in page
        assert "<script src=" not in page and "plotly" in page.lower()

    def test_pdf_has_the_overview_the_charts_and_the_review(self, run):
        items = review_items(run["metrics"], run["run_summary"])
        counts = overview_counts(run["metrics"], run["run_summary"])
        data = report.report_pdf(run, counts, items)
        assert data.startswith(b"%PDF")
        figures = sum(1 for kind, _ in report.chart_entries(run["metrics"]) if kind == "figure")
        checks = items["check"].nunique()
        assert data.count(b"/Type /Page\n") + data.count(b"/Type /Page ") >= (
            1 + figures + checks + 1)

    def test_charts_are_the_distributions_page_s_own(self, run):
        entries = report.chart_entries(run["metrics"])
        sections = [value for kind, value in entries if kind == "section"]
        assert sections == ["Size & shape", "Symmetry", "Scars", "Spatial"]
        assert any(kind == "figure" for kind, _ in entries)


@pytest.mark.unit
class TestDashboardColours:
    """The report shows the dashboard's colours, not Streamlit's placeholders."""

    def test_placeholders_become_the_streamlit_palette(self):
        import plotly.graph_objects as go
        fig = go.Figure(go.Histogram(x=[1, 2, 2], marker={"color": "#000001"}))
        fig.add_trace(go.Scatter(x=[1], y=[1], marker={"color": "rgb(94, 60, 153)"}))
        themed = report._themed(fig)
        assert themed.data[0].marker.color == report.STREAMLIT_COLORWAY[0]
        assert themed.data[1].marker.color == "rgb(94, 60, 153)"      # a page colour stays
        assert list(themed.layout.colorway) == report.STREAMLIT_COLORWAY

    def test_the_html_holds_no_placeholder(self, run):
        import re
        items = review_items(run["metrics"], run["run_summary"])
        counts = overview_counts(run["metrics"], run["run_summary"])
        page = report.report_html(run, counts, items).split("</script>", 1)[-1]
        assert not re.findall(r'"#0000(?:0[1-9]|10)"', page)

    def test_pdf_colours_follow_the_trace_or_the_palette(self):
        import plotly.graph_objects as go
        own = go.Scatter(x=[1], y=[1], marker={"color": "rgba(94, 60, 153, 0.5)"})
        plain = go.Scatter(x=[1], y=[1])
        assert report._trace_colour(own, 0) == (94 / 255, 60 / 255, 153 / 255, 0.5)
        assert report._trace_colour(plain, 2) == report.STREAMLIT_COLORWAY[2]


@pytest.mark.unit
class TestExportNames:
    """The export files carry the name of the run."""

    def test_the_project_folder_names_the_files(self, tmp_path):
        results = tmp_path / "pylithics_test_lithics" / "results"
        results.mkdir(parents=True)
        assert report.export_filename(results, "html") == "pylithics_test_lithics_report.html"
        assert report.export_filename(str(results), "pdf") == "pylithics_test_lithics_report.pdf"
        assert report.export_filename(results, "csv") == "pylithics_test_lithics_report.csv"

    def test_a_results_folder_with_its_own_name_names_the_files(self, tmp_path):
        run = tmp_path / "tanzania_run_2025"
        run.mkdir()
        assert report.export_filename(run, "html") == "tanzania_run_2025_report.html"


@pytest.mark.unit
def test_the_size_rules_are_a_link_on_the_page_and_plain_in_the_csv(run):
    from pylithics.image_processing.modules.dashboard.data import link_docs
    step = CHECKS["unclassified_surfaces"]["steps"][1]
    assert "](https://" in link_docs(step) and "#views-and-their-sizes" in link_docs(step)
    items = review_items(run["metrics"], run["run_summary"])
    assert "https://" not in report.report_csv(items)
    counts = overview_counts(run["metrics"], run["run_summary"])
    assert 'href="https://alan-turing-institute.github.io/Palaeoanalytics/user-guide/' \
        'image-requirements/#views-and-their-sizes"' in report.report_html(run, counts, items)
