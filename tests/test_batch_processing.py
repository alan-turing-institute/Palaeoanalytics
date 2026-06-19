"""Tests for multi-image batch processing behavior."""

import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

from pylithics.app import PyLithicsApplication
from pylithics.image_processing.config import clear_config_cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_bgr_with_dpi(image_bgr, path, dpi=300):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(path, dpi=(dpi, dpi))


def _valid_image(variation=0):
    img = np.full((300, 400, 3), 240, dtype=np.uint8)
    x = 50 + variation * 5
    cv2.rectangle(img, (x, 50), (x + 200, 200), (80, 70, 60), thickness=-1)
    cv2.circle(img, (x + 60, 100), 20, (50, 45, 40), thickness=-1)
    return img


def _write_metadata(metadata_path, rows):
    with open(metadata_path, "w") as f:
        f.write("image_id,scale_id,scale\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")


def _make_batch(temp_dir, rows, make_valid):
    """
    Build a data_dir/metadata workspace.

    `rows` is a list of (image_id, scale_id, scale). `make_valid(image_id)`
    returns True to save a real image at that path, False to write a bogus
    text file (simulating a corrupt image).
    """
    data_dir = os.path.join(temp_dir, "data")
    images_dir = os.path.join(data_dir, "images")
    os.makedirs(images_dir)

    for i, (image_id, _, _) in enumerate(rows):
        path = os.path.join(images_dir, image_id)
        if make_valid(image_id):
            _save_bgr_with_dpi(_valid_image(variation=i), path)
        else:
            with open(path, "w") as f:
                f.write("not an image")

    metadata_path = os.path.join(data_dir, "metadata.csv")
    _write_metadata(metadata_path, rows)
    return data_dir, metadata_path


def _write_config(temp_dir, sample_config):
    path = os.path.join(temp_dir, "config.yaml")
    with open(path, "w") as f:
        yaml.dump(sample_config, f)
    return path


# ---------------------------------------------------------------------------
# Batch result structure
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBatchResultAggregation:
    """run_batch_analysis returns a result dict that tracks outcomes."""

    def test_all_valid_batch_reports_full_success(self, sample_config):
        rows = [
            ("a.png", "scale_1", "15.0"),
            ("b.png", "scale_2", "16.0"),
            ("c.png", "scale_3", "17.0"),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir, meta = _make_batch(temp_dir, rows, make_valid=lambda _id: True)
            config_path = _write_config(temp_dir, sample_config)

            app = PyLithicsApplication(config_file=config_path)
            result = app.run_batch_analysis(data_dir, meta)

        assert set(result.keys()) >= {
            "success", "total_images", "processed_successfully",
            "failed_images", "processing_errors",
        }
        assert result["success"] is True
        assert result["total_images"] == 3
        assert result["processed_successfully"] == 3
        assert result["failed_images"] == []
        assert result["processing_errors"] == []

    def test_all_invalid_batch_reports_every_row_as_failed(self, sample_config):
        rows = [(f"{name}.png", "s", "15") for name in ("a", "b", "c")]
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir, meta = _make_batch(temp_dir, rows, make_valid=lambda _id: False)
            config_path = _write_config(temp_dir, sample_config)

            app = PyLithicsApplication(config_file=config_path)
            result = app.run_batch_analysis(data_dir, meta)

        assert result["total_images"] == 3
        assert result["processed_successfully"] == 0
        assert len(result["failed_images"]) == 3
        assert len(result["processing_errors"]) == 3

    def test_mixed_batch_tracks_successes_and_failures(self, sample_config):
        rows = [
            ("good_a.png", "s", "15.0"),
            ("bad_a.png", "s", "16.0"),
            ("good_b.png", "s", "17.0"),
            ("bad_b.png", "s", "18.0"),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir, meta = _make_batch(
                temp_dir, rows, make_valid=lambda name: name.startswith("good"),
            )
            config_path = _write_config(temp_dir, sample_config)

            app = PyLithicsApplication(config_file=config_path)
            result = app.run_batch_analysis(data_dir, meta)

        assert result["processed_successfully"] == 2
        assert sorted(result["failed_images"]) == ["bad_a.png", "bad_b.png"]

    def test_empty_metadata_fails_validation(self, sample_config):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data")
            os.makedirs(os.path.join(data_dir, "images"))

            metadata_path = os.path.join(data_dir, "metadata.csv")
            _write_metadata(metadata_path, rows=[])

            config_path = _write_config(temp_dir, sample_config)
            app = PyLithicsApplication(config_file=config_path)
            result = app.run_batch_analysis(data_dir, metadata_path)

        assert result["success"] is False


# ---------------------------------------------------------------------------
# Metadata edge cases
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBatchMetadataHandling:
    """Invalid scale values must not take down the whole batch."""

    @pytest.mark.parametrize("scale_value", ["not_a_number", "", "-5.0"])
    def test_invalid_scale_values_do_not_abort_batch(
        self, sample_config, scale_value
    ):
        rows = [
            ("ok.png", "scale_1", "15.0"),
            ("weird.png", "scale_2", scale_value),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir, meta = _make_batch(
                temp_dir, rows, make_valid=lambda _id: True,
            )
            config_path = _write_config(temp_dir, sample_config)

            app = PyLithicsApplication(config_file=config_path)
            result = app.run_batch_analysis(data_dir, meta)

        assert result["success"] is True
        assert result["total_images"] == 2
        # Processing should not crash on either row
        assert result["processed_successfully"] >= 1


# ---------------------------------------------------------------------------
# Output consolidation
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBatchOutputs:
    """CSV and visualizations should accumulate across the batch."""

    def test_csv_contains_one_row_group_per_successful_image(self, sample_config):
        rows = [(f"img_{i}.png", "s", "15.0") for i in range(4)]
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir, meta = _make_batch(
                temp_dir, rows, make_valid=lambda _id: True,
            )
            config_path = _write_config(temp_dir, sample_config)

            app = PyLithicsApplication(config_file=config_path)
            result = app.run_batch_analysis(data_dir, meta)

            csv_path = Path(data_dir) / "processed" / "processed_metrics.csv"
            df = pd.read_csv(csv_path)

            assert result["processed_successfully"] == 4
            # Exactly one consolidated CSV
            assert len(list(csv_path.parent.glob("processed_metrics.csv"))) == 1
            # Every processed image should show up
            assert df["image_id"].nunique() == 4
            for col in ("image_id", "surface_type", "surface_feature", "total_area"):
                assert col in df.columns

    def test_each_successful_image_gets_a_labeled_visualization(
        self, sample_config
    ):
        rows = [(f"viz_{i}.png", "s", "15.0") for i in range(3)]
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir, meta = _make_batch(
                temp_dir, rows, make_valid=lambda _id: True,
            )
            config_path = _write_config(temp_dir, sample_config)

            app = PyLithicsApplication(config_file=config_path)
            result = app.run_batch_analysis(data_dir, meta)

            processed = Path(data_dir) / "processed"
            labeled = list(processed.glob("*_labeled.png"))

            assert result["processed_successfully"] == 3
            assert len(labeled) >= 3
            # Each file must be a valid image
            for path in labeled:
                assert cv2.imread(str(path)) is not None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_batch_writes_expected_lines_to_log_file(sample_config):
    clear_config_cache()

    rows = [
        ("good.png", "s", "15.0"),
        ("bad.png", "s", "16.0"),
    ]
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir, meta = _make_batch(
            temp_dir, rows,
            make_valid=lambda name: name.startswith("good"),
        )

        config = dict(sample_config)
        config["logging"] = {
            "level": "INFO",
            "log_to_file": True,
            "log_file": os.path.join(temp_dir, "batch.log"),
        }
        config_path = _write_config(temp_dir, config)

        app = PyLithicsApplication(config_file=config_path)
        app.run_batch_analysis(data_dir, meta)

        log_path = Path(temp_dir) / "batch.log"
        log_content = log_path.read_text()

    assert "Starting batch processing" in log_content
    assert "images processed" in log_content


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


@pytest.mark.performance
def test_three_image_batch_completes_under_sixty_seconds(sample_config):
    rows = [(f"perf_{i}.png", "s", "15.0") for i in range(3)]
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir, meta = _make_batch(
            temp_dir, rows, make_valid=lambda _id: True,
        )
        config_path = _write_config(temp_dir, sample_config)
        app = PyLithicsApplication(config_file=config_path)

        start = time.time()
        result = app.run_batch_analysis(data_dir, meta)
        elapsed = time.time() - start

    assert result["processed_successfully"] == 3
    assert elapsed < 60.0, f"3-image batch took {elapsed:.1f}s (limit 60s)"
