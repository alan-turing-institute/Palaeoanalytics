"""Tests for the dashboard data layer."""

import json
import os

import pandas as pd
import pytest

from pylithics.image_processing.modules.dashboard.data import (
    dorsal_scars,
    filter_metrics,
    load_processed,
    overview_counts,
    parent_rows,
    per_image_image_paths,
    summarize_assemblage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def _sample_rows():
    """Two lithics: one fully populated, one minimal pixel-only."""
    return [
        # Full Dorsal parent for awbari.png
        {
            "image_id": "awbari.png",
            "surface_type": "Dorsal", "surface_feature": "Dorsal",
            "total_dorsal_scars": 2,
            "centroid_x": 100.0, "centroid_y": 100.0,
            "technical_width": 60.0, "technical_length": 80.0,
            "max_width": 65.0, "max_length": 85.0,
            "total_area": 4800.0, "aspect_ratio": 1.33, "perimeter": 280.0,
            "voronoi_num_cells": 3, "voronoi_cell_area": 1500.0,
            "convex_hull_area": 4500.0,
            "vertical_symmetry": 0.97, "horizontal_symmetry": 0.99,
            "is_cortex": False,
            "has_arrow": False,
            "calibration_method": "scale_bar",
            "pixels_per_mm": 25.2, "scale_confidence": 1.0,
        },
        # Awbari Dorsal scars
        {
            "image_id": "awbari.png",
            "surface_type": "Dorsal", "surface_feature": "scar 1",
            "max_width": 20.0, "max_length": 25.0, "total_area": 400.0,
            "scar_complexity": 2, "is_cortex": False, "has_arrow": True,
            "arrow_angle": 90.0,
            "calibration_method": "scale_bar",
        },
        {
            "image_id": "awbari.png",
            "surface_type": "Dorsal", "surface_feature": "scar 2",
            "max_width": 18.0, "max_length": 22.0, "total_area": 360.0,
            "scar_complexity": 3, "is_cortex": False, "has_arrow": False,
            "calibration_method": "scale_bar",
        },
        # A cortex feature on awbari
        {
            "image_id": "awbari.png",
            "surface_type": "Dorsal", "surface_feature": "cortex 1",
            "max_width": 12.0, "max_length": 15.0, "total_area": 180.0,
            "is_cortex": True, "cortex_area": 180.0, "cortex_percentage": 3.75,
            "has_arrow": False, "calibration_method": "scale_bar",
        },
        # Ventral parent for awbari
        {
            "image_id": "awbari.png",
            "surface_type": "Ventral", "surface_feature": "Ventral",
            "centroid_x": 95.0, "centroid_y": 100.0,
            "technical_width": 60.0, "technical_length": 80.0,
            "total_area": 4750.0,
            "is_cortex": False, "has_arrow": False,
            "calibration_method": "scale_bar",
        },
        # Pixel-only second lithic — Dorsal parent only
        {
            "image_id": "replica.png",
            "surface_type": "Dorsal", "surface_feature": "Dorsal",
            "total_dorsal_scars": 0,
            "centroid_x": 50.0, "centroid_y": 50.0,
            "technical_width": 40.0, "technical_length": 60.0,
            "total_area": 2400.0,
            "is_cortex": False, "has_arrow": False,
            "calibration_method": "pixels",
        },
    ]


@pytest.fixture
def sample_df():
    return pd.DataFrame(_sample_rows())


# ---------------------------------------------------------------------------
# load_processed
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoadProcessed:

    def test_loads_csv_and_inventories_artifacts(self, tmp_path):
        processed = tmp_path / "processed"
        processed.mkdir()
        _write_csv(processed / "processed_metrics.csv", _sample_rows())

        json_dir = processed / "json"
        json_dir.mkdir()
        (json_dir / "awbari.json").write_text(json.dumps({"image_id": "awbari.png"}))

        bundle = load_processed(str(processed))

        assert isinstance(bundle["metrics"], pd.DataFrame)
        assert len(bundle["metrics"]) == len(_sample_rows())
        assert bundle["processed_dir"] == processed.resolve()
        assert bundle["json_dir"] == json_dir.resolve()

    def test_json_dir_none_when_absent(self, tmp_path):
        processed = tmp_path / "processed"
        processed.mkdir()
        _write_csv(processed / "processed_metrics.csv", _sample_rows())

        bundle = load_processed(str(processed))

        assert bundle["json_dir"] is None

    def test_missing_csv_raises_filenotfound(self, tmp_path):
        processed = tmp_path / "processed"
        processed.mkdir()

        with pytest.raises(FileNotFoundError):
            load_processed(str(processed))


# ---------------------------------------------------------------------------
# summarize_assemblage
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSummarizeAssemblage:

    def test_counts_match_fixture(self, sample_df):
        summary = summarize_assemblage(sample_df)

        assert summary["n_lithics"] == 2
        assert summary["n_calibrated"] == 1  # only awbari has scale_bar
        # Two dorsal scars, one with arrow → 0.5
        assert summary["arrow_detection_rate"] == pytest.approx(0.5)
        # 1 of 2 lithics has a cortex feature
        assert summary["cortex_prevalence"] == pytest.approx(0.5)

    def test_surface_counts_only_count_parents(self, sample_df):
        summary = summarize_assemblage(sample_df)
        # awbari: Dorsal+Ventral, replica: Dorsal -> Dorsal=2, Ventral=1
        assert summary["surface_counts"]["Dorsal"] == 2
        assert summary["surface_counts"]["Ventral"] == 1
        # Children should not appear
        assert "scar 1" not in summary["surface_counts"]

    def test_calibration_counts_one_per_image(self, sample_df):
        summary = summarize_assemblage(sample_df)
        assert summary["calibration_counts"]["scale_bar"] == 1
        assert summary["calibration_counts"]["pixels"] == 1

    def test_empty_dataframe_returns_zero_summary(self):
        summary = summarize_assemblage(pd.DataFrame())

        assert summary == {
            "n_lithics": 0,
            "n_calibrated": 0,
            "arrow_detection_rate": 0.0,
            "cortex_prevalence": 0.0,
            "surface_counts": {},
            "calibration_counts": {},
        }

    def test_string_booleans_round_trip(self):
        """CSV-loaded bools sometimes come back as 'True'/'False' strings."""
        df = pd.DataFrame([
            {
                "image_id": "x.png",
                "surface_type": "Dorsal", "surface_feature": "scar 1",
                "is_cortex": "False", "has_arrow": "True",
                "calibration_method": "scale_bar",
            },
        ])
        summary = summarize_assemblage(df)
        assert summary["arrow_detection_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# filter_metrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFilterMetrics:

    def test_no_filters_returns_full_dataframe(self, sample_df):
        result = filter_metrics(sample_df)
        assert len(result) == len(sample_df)

    def test_filter_by_surface_type(self, sample_df):
        result = filter_metrics(sample_df, surface_types=["Dorsal"])
        assert (result["surface_type"] == "Dorsal").all()

    def test_filter_by_calibration_method(self, sample_df):
        result = filter_metrics(sample_df, calibration_methods=["pixels"])
        assert (result["calibration_method"] == "pixels").all()
        assert result["image_id"].unique().tolist() == ["replica.png"]

    def test_filters_combine(self, sample_df):
        result = filter_metrics(
            sample_df,
            surface_types=["Dorsal"],
            calibration_methods=["pixels"],
        )
        assert len(result) == 1
        assert result.iloc[0]["image_id"] == "replica.png"

    def test_empty_filter_lists_treated_as_no_filter(self, sample_df):
        # An empty list should not exclude everything; pages pass widget state
        # through unchanged and may end up with [] when no rows are checked.
        result = filter_metrics(sample_df, surface_types=[])
        assert len(result) == len(sample_df)


# ---------------------------------------------------------------------------
# parent / dorsal helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_parent_rows_picks_only_parents(sample_df):
    parents = parent_rows(sample_df)
    # Awbari Dorsal + Ventral, Replica Dorsal -> 3 parents
    assert len(parents) == 3
    assert (parents["surface_feature"] == parents["surface_type"]).all()


@pytest.mark.unit
def test_dorsal_scars_excludes_cortex_and_parents(sample_df):
    scars = dorsal_scars(sample_df)
    assert sorted(scars["surface_feature"].tolist()) == ["scar 1", "scar 2"]


# ---------------------------------------------------------------------------
# per_image_image_paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_per_image_image_paths_resolves_existing_files(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "awbari_labeled.png").write_bytes(b"\x89PNG")
    (processed / "awbari_voronoi.png").write_bytes(b"\x89PNG")
    (processed / "json").mkdir()
    (processed / "json" / "awbari.json").write_text("{}")

    paths = per_image_image_paths(processed, "awbari.png")

    assert paths["labeled"] == processed / "awbari_labeled.png"
    assert paths["voronoi"] == processed / "awbari_voronoi.png"
    assert paths["json"] == processed / "json" / "awbari.json"


# ---------------------------------------------------------------------------
# overview_counts
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOverviewCounts:
    """Counts powering the Overview page."""

    def test_assemblage_counts_match_fixture(self, sample_df):
        counts = overview_counts(sample_df)

        assert counts["lithics"] == 2
        # Awbari has Dorsal + Ventral parent rows; replica has Dorsal only
        assert counts["surfaces"] == 3
        # Two dorsal scars on awbari; replica has none
        assert counts["scars"] == 2
        # One of the two awbari scars has has_arrow=True
        assert counts["scars_with_arrows"] == 1
        # One cortex feature in the fixture
        assert counts["cortex_regions"] == 1

    def test_data_quality_counts_default_to_zero(self, sample_df):
        counts = overview_counts(sample_df)

        assert counts["failed"] == 0  # No manifest -> 0
        # The sample fixture has scale_confidence implicitly absent;
        # we expect no low-confidence flags.
        assert counts["low_confidence_scales"] == 0
        # No Unclassified rows in the fixture
        assert counts["unclassified_surfaces"] == 0
        # Both lithics in fixture have at least 0 dorsal scars; replica is 0
        assert counts["zero_scar_lithics"] == 1

    def test_failed_count_uses_manifest(self, sample_df):
        manifest = {
            "successful": [
                {"image_id": "awbari.png", "dpi": 300},
                {"image_id": "replica.png", "dpi": 300},
            ],
            "failed": [
                {"image_id": "broken.png", "reason": "Processing failed"},
                {"image_id": "weird.png", "reason": "Processing failed"},
            ],
        }
        counts = overview_counts(sample_df, run_summary=manifest)
        assert counts["failed"] == 2
        # 'successful' uses manifest length when provided
        assert counts["successful"] == 2

    def test_pixel_only_lithics_count(self, sample_df):
        # Fixture has two lithics: awbari (scale_bar) and replica (pixels)
        counts = overview_counts(sample_df)
        assert counts["pixel_only_lithics"] == 1

    def test_mixed_dpi_zero_when_consistent(self):
        manifest = {
            "successful": [
                {"image_id": "a.png", "dpi": 300},
                {"image_id": "b.png", "dpi": 300},
            ],
            "failed": [],
        }
        counts = overview_counts(pd.DataFrame(), run_summary=manifest)
        assert counts["mixed_dpi"] == 0

    def test_mixed_dpi_reports_distinct_count(self):
        manifest = {
            "successful": [
                {"image_id": "a.png", "dpi": 300},
                {"image_id": "b.png", "dpi": 600},
                {"image_id": "c.png", "dpi": 300},
            ],
            "failed": [],
        }
        counts = overview_counts(pd.DataFrame(), run_summary=manifest)
        assert counts["mixed_dpi"] == 2

    def test_missing_dpi_count(self):
        manifest = {
            "successful": [
                {"image_id": "a.png", "dpi": 300},
                {"image_id": "b.png", "dpi": None},
                {"image_id": "c.png", "dpi": None},
            ],
            "failed": [],
        }
        counts = overview_counts(pd.DataFrame(), run_summary=manifest)
        assert counts["missing_dpi"] == 2

    def test_dpi_counts_zero_without_manifest(self, sample_df):
        counts = overview_counts(sample_df)
        assert counts["mixed_dpi"] == 0
        assert counts["missing_dpi"] == 0

    def test_low_confidence_threshold(self):
        df = pd.DataFrame([
            {
                "image_id": "good.png",
                "surface_type": "Dorsal", "surface_feature": "Dorsal",
                "scale_confidence": 0.95,
                "is_cortex": False, "has_arrow": False,
            },
            {
                "image_id": "weak.png",
                "surface_type": "Dorsal", "surface_feature": "Dorsal",
                "scale_confidence": 0.6,
                "is_cortex": False, "has_arrow": False,
            },
            {
                "image_id": "missing.png",
                "surface_type": "Dorsal", "surface_feature": "Dorsal",
                "scale_confidence": None,
                "is_cortex": False, "has_arrow": False,
            },
        ])
        counts = overview_counts(df)
        assert counts["low_confidence_scales"] == 1

    def test_unclassified_surface_counted(self):
        df = pd.DataFrame([
            {
                "image_id": "x.png",
                "surface_type": "Unclassified",
                "surface_feature": "Unclassified",
                "is_cortex": False, "has_arrow": False,
            },
        ])
        counts = overview_counts(df)
        assert counts["unclassified_surfaces"] == 1

    def test_empty_dataframe_returns_zeros(self):
        counts = overview_counts(pd.DataFrame())
        for key in ("successful", "failed", "low_confidence_scales",
                    "unclassified_surfaces", "zero_scar_lithics",
                    "pixel_only_lithics", "mixed_dpi", "missing_dpi",
                    "lithics", "surfaces", "scars",
                    "scars_with_arrows", "cortex_regions"):
            assert counts[key] == 0


@pytest.mark.unit
def test_load_processed_includes_run_summary(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    _write_csv(processed / "processed_metrics.csv", _sample_rows())
    summary = {
        "schema_version": 1,
        "successful": ["awbari.png"],
        "failed": [],
    }
    (processed / "run_summary.json").write_text(json.dumps(summary))

    bundle = load_processed(str(processed))

    assert bundle["run_summary"] == summary


@pytest.mark.unit
def test_load_processed_run_summary_none_when_absent(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    _write_csv(processed / "processed_metrics.csv", _sample_rows())

    bundle = load_processed(str(processed))

    assert bundle["run_summary"] is None


@pytest.mark.unit
def test_per_image_image_paths_returns_none_for_missing(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()

    paths = per_image_image_paths(processed, "missing.png")

    assert paths == {"labeled": None, "voronoi": None, "json": None}
