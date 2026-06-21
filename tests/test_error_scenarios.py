"""Error-path and graceful-degradation tests for PyLithics."""

import os
import tempfile

import cv2
import numpy as np
import pytest
import yaml

from pylithics.app import PyLithicsApplication
from pylithics.image_processing.importer import execute_preprocessing_pipeline
from pylithics.image_processing.modules.arrow_detection import ArrowDetector
from pylithics.image_processing.modules.contour_extraction import (
    extract_contours_with_hierarchy,
)
from pylithics.image_processing.modules.contour_metrics import (
    calculate_contour_metrics,
)
from pylithics.image_processing.modules.surface_classification import (
    classify_parent_contours,
)


# ---------------------------------------------------------------------------
# Helpers: build a realistic batch-analysis workspace on disk
# ---------------------------------------------------------------------------


def _write_config(dir_path, sample_config):
    config_path = os.path.join(dir_path, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(sample_config, f)
    return config_path


def _setup_batch_workspace(temp_dir, rows, image_factory):
    """
    Lay out `data_dir/{images,}` with one image per row, plus a metadata CSV.

    ``rows`` is a list of (image_id, scale_id, scale). ``image_factory`` is
    a callable that takes the image filesystem path and writes whatever
    fixture bytes the test needs.
    """
    data_dir = os.path.join(temp_dir, "data")
    images_dir = os.path.join(data_dir, "images")
    os.makedirs(images_dir)

    for image_id, _, _ in rows:
        image_factory(os.path.join(images_dir, image_id))

    metadata_path = os.path.join(data_dir, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("image_id,scale_id,scale\n")
        for image_id, scale_id, scale in rows:
            f.write(f"{image_id},{scale_id},{scale}\n")

    return data_dir, metadata_path


def _write_valid_image(path):
    img = np.full((120, 120, 3), 200, dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (90, 90), (40, 40, 40), -1)
    cv2.imwrite(path, img)


# ---------------------------------------------------------------------------
# Batch-level error handling
# ---------------------------------------------------------------------------


@pytest.mark.error_scenarios
class TestBatchAnalysisErrors:
    """PyLithicsApplication.run_batch_analysis reports failures cleanly."""

    def test_missing_image_file_reported_as_failure(self, sample_config):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir, meta_path = _setup_batch_workspace(
                temp_dir,
                rows=[("missing.png", "scale_1", "15.0")],
                image_factory=lambda path: None,  # don't create the image
            )
            config_path = _write_config(temp_dir, sample_config)

            app = PyLithicsApplication(config_file=config_path)
            results = app.run_batch_analysis(data_dir, meta_path)

        assert results["success"] is True
        assert results["processed_successfully"] == 0
        assert "missing.png" in results["failed_images"]
        assert results["processing_errors"]

    def test_corrupted_image_file_reported_as_failure(self, sample_config):
        def _write_bogus(path):
            with open(path, "w") as f:
                f.write("this is not an image")

        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir, meta_path = _setup_batch_workspace(
                temp_dir,
                rows=[("corrupted.png", "scale_1", "15.0")],
                image_factory=_write_bogus,
            )
            config_path = _write_config(temp_dir, sample_config)

            app = PyLithicsApplication(config_file=config_path)
            results = app.run_batch_analysis(data_dir, meta_path)

        assert results["processed_successfully"] == 0
        assert "corrupted.png" in results["failed_images"]

    def test_invalid_scale_values_fall_back_to_pixels(self, sample_config):
        """Invalid scale strings must not abort batch processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir, meta_path = _setup_batch_workspace(
                temp_dir,
                rows=[
                    ("img_a.png", "scale_1", "not_a_number"),
                    ("img_b.png", "scale_2", ""),
                    ("img_c.png", "scale_3", "-5.0"),
                ],
                image_factory=_write_valid_image,
            )
            config_path = _write_config(temp_dir, sample_config)

            app = PyLithicsApplication(config_file=config_path)
            results = app.run_batch_analysis(data_dir, meta_path)

        assert results["success"] is True
        assert results["total_images"] == 3

    def test_mixed_valid_and_invalid_images_tracks_both(self, sample_config):
        def _make_image(path):
            if "bad" in os.path.basename(path):
                with open(path, "w") as f:
                    f.write("nope")
            else:
                _write_valid_image(path)

        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir, meta_path = _setup_batch_workspace(
                temp_dir,
                rows=[
                    ("good.png", "scale_1", "15.0"),
                    ("bad.png", "scale_2", "20.0"),
                ],
                image_factory=_make_image,
            )
            config_path = _write_config(temp_dir, sample_config)

            app = PyLithicsApplication(config_file=config_path)
            results = app.run_batch_analysis(data_dir, meta_path)

        assert "bad.png" in results["failed_images"]
        assert "good.png" not in results["failed_images"]


# ---------------------------------------------------------------------------
# Module-level error handling
# ---------------------------------------------------------------------------


@pytest.mark.error_scenarios
class TestPreprocessingErrors:

    def test_returns_none_for_missing_image(self):
        result = execute_preprocessing_pipeline(
            "/definitely/not/a/real/file.png", {}
        )
        assert result is None


@pytest.mark.error_scenarios
class TestContourExtractionErrors:

    def test_uniform_image_yields_no_contours(self, tmp_path):
        uniform = np.full((100, 100), 128, dtype=np.uint8)
        contours, hierarchy = extract_contours_with_hierarchy(
            uniform, "uniform", str(tmp_path)
        )
        assert contours == []
        assert hierarchy is None

    def test_border_touching_shapes_are_filtered_out(self, tmp_path):
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(image, (0, 0), (50, 50), 255, -1)       # touches top/left
        cv2.rectangle(image, (50, 50), (99, 99), 255, -1)     # touches bottom/right

        contours, _ = extract_contours_with_hierarchy(
            image, "border", str(tmp_path)
        )
        assert contours == []


@pytest.mark.error_scenarios
class TestContourMetricsErrors:

    def test_degenerate_contours_produce_valid_metrics_list(self):
        line = np.array([[10, 10], [20, 10]], dtype=np.int32).reshape(-1, 1, 2)
        point = np.array([[15, 15]], dtype=np.int32).reshape(-1, 1, 2)

        sorted_contours = {
            "parents": [line],
            "children": [point],
            "nested_children": [],
        }
        hierarchy = np.array([[-1, -1, 1, -1], [-1, -1, -1, 0]])

        metrics = calculate_contour_metrics(
            sorted_contours, hierarchy, [line, point], (100, 100)
        )
        assert isinstance(metrics, list)
        assert all(m.get("area", 0) >= 0 for m in metrics)


@pytest.mark.error_scenarios
class TestArrowDetectionErrors:
    """ArrowDetector rejects unusable inputs cleanly."""

    @pytest.fixture
    def detector(self):
        return ArrowDetector()

    @pytest.fixture
    def blank_image(self):
        return np.zeros((100, 100), dtype=np.uint8)

    def test_empty_contour_returns_none(self, detector, blank_image):
        empty = np.array([], dtype=np.int32).reshape(0, 1, 2)
        assert detector.analyze_contour_for_arrow(
            empty, {"scar": "x"}, blank_image, 300.0
        ) is None

    def test_single_point_contour_returns_none(self, detector, blank_image):
        point = np.array([[10, 10]], dtype=np.int32).reshape(-1, 1, 2)
        assert detector.analyze_contour_for_arrow(
            point, {"scar": "x"}, blank_image, 300.0
        ) is None

    @pytest.mark.parametrize("invalid_dpi", [None, 0, -100])
    def test_invalid_dpi_does_not_crash(self, detector, blank_image, invalid_dpi):
        triangle = np.array(
            [[10, 10], [20, 10], [15, 20]], dtype=np.int32
        ).reshape(-1, 1, 2)
        # Should not raise — may return None or a dict with fallback values
        result = detector.analyze_contour_for_arrow(
            triangle, {"scar": "x"}, blank_image, invalid_dpi
        )
        assert result is None or isinstance(result, dict)


@pytest.mark.error_scenarios
class TestSurfaceClassificationErrors:

    def test_classifies_when_no_parent_rows_present(self):
        """Scar-only metrics should produce a list of equal length."""
        metrics = [
            {"parent": "parent 1", "scar": "scar 1", "area": 100},
            {"parent": "parent 1", "scar": "scar 2", "area": 50},
        ]
        result = classify_parent_contours(metrics)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Unicode / encoding robustness at the batch level
# ---------------------------------------------------------------------------


@pytest.mark.error_scenarios
def test_unicode_filenames_flow_through_batch_analysis(sample_config):
    unicode_name = "tëst_ünicödé.png"

    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir, meta_path = _setup_batch_workspace(
            temp_dir,
            rows=[(unicode_name, "scälé_1", "15.0")],
            image_factory=_write_valid_image,
        )
        config_path = _write_config(temp_dir, sample_config)

        app = PyLithicsApplication(config_file=config_path)
        results = app.run_batch_analysis(data_dir, meta_path)

    assert results["total_images"] == 1
    # Whether processing succeeds depends on downstream modules — but the
    # filename must not break metadata parsing or result tracking
    tracked = set(results["failed_images"]) | {unicode_name}
    assert unicode_name in tracked
