"""Tests for visualization and CSV export."""

import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import pytest

from pylithics.image_processing.modules.visualization import (
    save_measurements_to_csv,
    visualize_contours_with_hierarchy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _square_contour(origin=(20, 20), side=60):
    x, y = origin
    pts = np.array(
        [[x, y], [x + side, y], [x + side, y + side], [x, y + side]],
        dtype=np.int32,
    )
    return pts.reshape(-1, 1, 2)


def _parent_metric(**overrides):
    base = {
        "image_id": "test_image",
        "parent": "parent 1",
        "scar": "parent 1",
        "surface_type": "Dorsal",
        "surface_feature": "Dorsal",
        "centroid_x": 50.0,
        "centroid_y": 50.0,
        "technical_width": 60.0,
        "technical_length": 60.0,
        "area": 3600.0,
        "has_arrow": False,
    }
    base.update(overrides)
    return base


def _scar_metric(name="scar 1", **overrides):
    base = {
        "image_id": "test_image",
        "parent": "parent 1",
        "scar": name,
        "surface_type": "Dorsal",
        "surface_feature": name,
        "centroid_x": 40.0,
        "centroid_y": 40.0,
        "width": 20.0,
        "height": 20.0,
        "area": 400.0,
        "has_arrow": False,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# visualize_contours_with_hierarchy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVisualizeContoursWithHierarchy:
    """Rendering of labeled contour visualizations."""

    def _run_and_load(self, contours, hierarchy, metrics, image_shape=(100, 100)):
        inverted = np.zeros(image_shape, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "out.png")
            visualize_contours_with_hierarchy(
                contours, hierarchy, metrics, inverted, output_path
            )
            assert os.path.exists(output_path)
            image = cv2.imread(output_path)
        return image

    def test_writes_valid_bgr_png_matching_input_dimensions(self):
        parent = _square_contour(origin=(20, 20), side=60)
        child = _square_contour(origin=(30, 30), side=20)
        hierarchy = np.array([[-1, -1, 1, -1], [-1, -1, -1, 0]])
        metrics = [_parent_metric(), _scar_metric()]

        rendered = self._run_and_load(
            [parent, child], hierarchy, metrics, image_shape=(120, 140)
        )

        assert rendered is not None
        # OpenCV returns (H, W, 3) for BGR images
        assert rendered.shape == (120, 140, 3)

    def test_arrow_rendering_adds_visible_pixels(self):
        """A metric with has_arrow=True should put arrow-colored pixels on the canvas."""
        contour = _square_contour(origin=(20, 20), side=60)
        hierarchy = np.array([[-1, -1, -1, -1]])

        metric_with_arrow = _scar_metric(
            has_arrow=True,
            arrow_back=(30, 30),
            arrow_tip=(60, 60),
            arrow_angle=45.0,
        )

        rendered_with = self._run_and_load(
            [contour], hierarchy, [metric_with_arrow]
        )
        rendered_without = self._run_and_load(
            [contour], hierarchy, [_scar_metric()]
        )

        # The image with arrow annotations should have strictly more
        # non-white pixels than the one without.
        nonwhite_with = np.sum(np.any(rendered_with < 250, axis=-1))
        nonwhite_without = np.sum(np.any(rendered_without < 250, axis=-1))
        assert nonwhite_with > nonwhite_without

    def test_creates_missing_output_directory(self):
        contour = _square_contour()
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = os.path.join(temp_dir, "a", "b", "out.png")
            visualize_contours_with_hierarchy(
                [contour],
                np.array([[-1, -1, -1, -1]]),
                [_parent_metric()],
                np.zeros((100, 100), dtype=np.uint8),
                nested_path,
            )
            assert os.path.exists(nested_path)

    def test_handles_empty_contour_list(self):
        """With no contours the function must still write a valid image."""
        inverted = np.zeros((80, 80), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "empty.png")
            visualize_contours_with_hierarchy(
                [], np.array([]), [], inverted, output_path
            )
            rendered = cv2.imread(output_path)

        assert rendered is not None
        assert rendered.shape == (80, 80, 3)


# ---------------------------------------------------------------------------
# save_measurements_to_csv
# ---------------------------------------------------------------------------


EXPECTED_BASE_COLUMNS = [
    "image_id", "surface_type", "surface_feature",
    "scar_count", "centroid_x", "centroid_y",
    "technical_width", "technical_length",
    "max_width", "max_length", "total_area",
    "aspect_ratio", "perimeter", "distance_to_max_width",
]


@pytest.mark.unit
class TestSaveMeasurementsToCSV:
    """CSV export for metric dictionaries."""

    def _save_and_read(self, metrics, **kwargs):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "out.csv")
            save_measurements_to_csv(metrics, output_path, **kwargs)
            return pd.read_csv(output_path)

    def test_round_trips_basic_metric(self):
        df = self._save_and_read([_parent_metric(centroid_x=50.0, centroid_y=60.0)])

        assert len(df) == 1
        row = df.iloc[0]
        assert row["image_id"] == "test_image"
        assert row["surface_type"] == "Dorsal"
        assert row["surface_feature"] == "Dorsal"
        assert row["centroid_x"] == 50.0
        assert row["centroid_y"] == 60.0
        assert row["total_area"] == 3600.0

    def test_column_order_matches_specification(self):
        df = self._save_and_read([_parent_metric()])
        # The first 14 columns are the documented base set, in order
        assert list(df.columns[: len(EXPECTED_BASE_COLUMNS)]) == EXPECTED_BASE_COLUMNS

    def test_missing_optional_fields_default_to_na(self):
        # Build a minimal metric lacking most optional fields
        minimal = {
            "image_id": "m",
            "parent": "parent 1",
            "scar": "parent 1",
            "surface_type": "Dorsal",
            "area": 1000.0,
        }
        df = self._save_and_read([minimal])
        row = df.iloc[0]

        # Unset numeric fields round-trip to NaN (pandas reading "NA")
        assert pd.isna(row["arrow_angle"])
        assert pd.isna(row["voronoi_num_cells"])
        # has_arrow defaults to False
        assert row["has_arrow"] == False  # noqa: E712

    def test_handles_multiple_surfaces(self):
        metrics = [
            _parent_metric(parent="parent 1", scar="parent 1", surface_type="Dorsal"),
            _scar_metric(name="scar 1"),
            _parent_metric(
                parent="parent 2", scar="parent 2",
                surface_type="Ventral", surface_feature="Ventral",
            ),
        ]
        df = self._save_and_read(metrics)

        assert len(df) == 3
        assert set(df["surface_type"].unique()) == {"Dorsal", "Ventral"}

    def test_appends_without_duplicating_header(self):
        first = [_parent_metric(image_id="img_a")]
        second = [_parent_metric(image_id="img_b")]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "out.csv")
            save_measurements_to_csv(first, output_path)
            save_measurements_to_csv(second, output_path, append=True)
            df = pd.read_csv(output_path)

        assert len(df) == 2
        assert df["image_id"].tolist() == ["img_a", "img_b"]

    def test_counts_dorsal_scars_on_parent_row(self):
        """Parent row's `scar_count` should equal the number of scar children."""
        metrics = [
            _parent_metric(),
            _scar_metric(name="scar 1"),
            _scar_metric(name="scar 2"),
            _scar_metric(name="scar 3"),
        ]
        df = self._save_and_read(metrics)

        parent_row = df[df["surface_feature"] == "Dorsal"].iloc[0]
        assert int(parent_row["scar_count"]) == 3

        # Scar rows use "NA" which pandas reads as NaN
        scar_rows = df[df["surface_feature"].str.startswith("scar")]
        assert all(pd.isna(scar_rows["scar_count"]))

    def test_calibration_metadata_appends_columns(self):
        df = self._save_and_read(
            [_parent_metric()],
            calibration_metadata={
                "calibration_method": "scale_bar",
                "pixels_per_mm": 4.0,
                "scale_confidence": 0.95,
            },
        )
        row = df.iloc[0]
        assert row["calibration_method"] == "scale_bar"
        assert row["pixels_per_mm"] == 4.0
        assert row["scale_confidence"] == 0.95

    def test_empty_metrics_writes_header_only(self):
        df = self._save_and_read([])
        assert len(df) == 0
        # Header must still include the base column schema
        for col in EXPECTED_BASE_COLUMNS:
            assert col in df.columns


# ---------------------------------------------------------------------------
# Integration: visualization + CSV export together
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_full_pipeline_produces_image_and_csv():
    """Parent + scars render an image and a CSV with matching row count."""
    parent = _square_contour(origin=(30, 30), side=80)
    scar_a = _square_contour(origin=(40, 40), side=15)
    scar_b = _square_contour(origin=(70, 70), side=15)

    hierarchy = np.array([
        [-1, -1, 1, -1],
        [2, -1, -1, 0],
        [-1, 1, -1, 0],
    ])
    metrics = [
        _parent_metric(),
        _scar_metric(
            name="scar 1",
            has_arrow=True,
            arrow_back=(45, 45),
            arrow_tip=(50, 50),
            arrow_angle=45.0,
        ),
        _scar_metric(name="scar 2"),
    ]
    inverted = np.zeros((150, 150), dtype=np.uint8)

    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = os.path.join(temp_dir, "out.png")
        csv_path = os.path.join(temp_dir, "out.csv")

        visualize_contours_with_hierarchy(
            [parent, scar_a, scar_b], hierarchy, metrics, inverted, image_path
        )
        save_measurements_to_csv(metrics, csv_path)

        rendered = cv2.imread(image_path)
        df = pd.read_csv(csv_path)

    assert rendered is not None
    assert rendered.shape == (150, 150, 3)
    assert len(df) == 3
    assert (df["surface_type"] == "Dorsal").all()
    assert df["has_arrow"].sum() == 1


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


@pytest.mark.performance
def test_csv_save_handles_one_thousand_rows_under_five_seconds():
    import time

    metrics = []
    for img_idx in range(50):
        metrics.append(_parent_metric(
            image_id=f"img_{img_idx:03d}",
            parent=f"parent {img_idx}", scar=f"parent {img_idx}",
        ))
        for scar_idx in range(19):
            metrics.append(_scar_metric(
                name=f"scar {scar_idx + 1}",
                image_id=f"img_{img_idx:03d}",
                parent=f"parent {img_idx}",
            ))

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "bulk.csv")
        start = time.time()
        save_measurements_to_csv(metrics, output_path)
        elapsed = time.time() - start

        df = pd.read_csv(output_path)

    assert len(df) == 1000
    assert elapsed < 5.0, f"save took {elapsed:.2f}s (limit 5s)"
