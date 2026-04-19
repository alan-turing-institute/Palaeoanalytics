"""End-to-end tests for the PyLithics batch analysis pipeline."""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

from pylithics.app import PyLithicsApplication, main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_image_with_dpi(image_bgr, path, dpi=300):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(path, dpi=(dpi, dpi))


def _artifact_image(variation=0):
    """A light background with a filled rectangle plus one circular scar."""
    image = np.full((300, 400, 3), 240, dtype=np.uint8)
    x0 = 50 + variation * 20
    cv2.rectangle(image, (x0, 50), (x0 + 300, 250), (80, 70, 60), thickness=-1)
    cv2.circle(image, (x0 + 80, 120), 25, (50, 45, 40), thickness=-1)
    return image


def _make_workspace(temp_dir, rows, image_factory=_artifact_image):
    """Build a `data/{images,}` layout and metadata CSV, return their paths."""
    data_dir = os.path.join(temp_dir, "data")
    images_dir = os.path.join(data_dir, "images")
    os.makedirs(images_dir)

    for i, (image_id, _, _) in enumerate(rows):
        _save_image_with_dpi(image_factory(variation=i),
                             os.path.join(images_dir, image_id))

    metadata_path = os.path.join(data_dir, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("image_id,scale_id,scale\n")
        for image_id, scale_id, scale in rows:
            f.write(f"{image_id},{scale_id},{scale}\n")

    return data_dir, metadata_path


def _write_config(temp_dir, sample_config):
    path = os.path.join(temp_dir, "config.yaml")
    with open(path, "w") as f:
        yaml.dump(sample_config, f)
    return path


# ---------------------------------------------------------------------------
# End-to-end batch analysis
# ---------------------------------------------------------------------------


@pytest.mark.functional
class TestEndToEndBatchAnalysis:
    """A full batch run must produce CSV and labeled images on success."""

    def test_single_image_produces_csv_and_labeled_image(self, sample_config):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir, meta_path = _make_workspace(
                temp_dir,
                rows=[("artifact.png", "scale_1", "15.0")],
            )
            config_path = _write_config(temp_dir, sample_config)

            app = PyLithicsApplication(config_file=config_path)
            result = app.run_batch_analysis(data_dir, meta_path)

            processed_dir = Path(data_dir) / "processed"
            csv_files = list(processed_dir.glob("processed_metrics.csv"))
            viz_files = list(processed_dir.glob("*_labeled.png"))

            assert result["success"] is True
            assert result["processed_successfully"] == 1
            assert result["failed_images"] == []
            assert len(csv_files) == 1
            assert len(viz_files) >= 1

            df = pd.read_csv(csv_files[0])
            assert len(df) >= 1
            assert df["image_id"].iloc[0] == "artifact.png"

    def test_multi_image_batch_reports_accurate_counts(self, sample_config):
        rows = [
            ("a.png", "scale_1", "15.0"),
            ("b.png", "scale_2", "17.0"),
            ("c.png", "scale_3", "19.0"),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir, meta_path = _make_workspace(temp_dir, rows)
            config_path = _write_config(temp_dir, sample_config)

            app = PyLithicsApplication(config_file=config_path)
            result = app.run_batch_analysis(data_dir, meta_path)

            df = pd.read_csv(Path(data_dir) / "processed" / "processed_metrics.csv")

            assert result["total_images"] == 3
            assert result["processed_successfully"] + len(result["failed_images"]) == 3
            # Every processed image should appear in the CSV
            assert df["image_id"].nunique() == result["processed_successfully"]


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


@pytest.mark.functional
def test_cli_main_runs_successfully(sample_config):
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir, meta_path = _make_workspace(
            temp_dir, rows=[("cli.png", "scale_1", "20.0")],
        )
        config_path = _write_config(temp_dir, sample_config)

        argv = [
            "pylithics",
            "--data_dir", data_dir,
            "--meta_file", meta_path,
            "--config_file", config_path,
            "--log_level", "INFO",
        ]
        with patch("sys.argv", argv):
            exit_code = main()

        processed_dir = Path(data_dir) / "processed"
        assert exit_code == 0
        assert (processed_dir / "processed_metrics.csv").exists()


# ---------------------------------------------------------------------------
# CSV and visualization output validation
# ---------------------------------------------------------------------------


@pytest.mark.functional
class TestOutputValidation:
    """Outputs must have the expected schema, types, and readability."""

    def _run(self, temp_dir, sample_config):
        data_dir, meta_path = _make_workspace(
            temp_dir, rows=[("out.png", "scale_1", "20.0")],
        )
        config_path = _write_config(temp_dir, sample_config)
        app = PyLithicsApplication(config_file=config_path)
        result = app.run_batch_analysis(data_dir, meta_path)
        return Path(data_dir), result

    def test_csv_has_expected_schema_and_numeric_types(self, sample_config):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir, _ = self._run(temp_dir, sample_config)
            df = pd.read_csv(data_dir / "processed" / "processed_metrics.csv")

        required = {
            "image_id", "surface_type", "surface_feature",
            "centroid_x", "centroid_y", "total_area",
        }
        assert required.issubset(df.columns)

        # Numeric columns must parse as numbers, not strings
        for col in ("total_area", "centroid_x", "centroid_y"):
            assert pd.api.types.is_numeric_dtype(df[col])
            assert (df[col] >= 0).all()

    def test_labeled_image_is_readable_png(self, sample_config):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir, _ = self._run(temp_dir, sample_config)
            labeled = list((data_dir / "processed").glob("*_labeled.png"))

            assert labeled, "pipeline did not produce a labeled image"
            image = cv2.imread(str(labeled[0]))

        assert image is not None
        assert image.ndim == 3


# ---------------------------------------------------------------------------
# Mixed valid/invalid inputs
# ---------------------------------------------------------------------------


@pytest.mark.functional
def test_batch_continues_when_some_images_are_invalid(sample_config):
    def _factory(variation):
        if variation == 1:
            # Zero-byte "image" forces a failure on the second row
            return None
        return _artifact_image(variation)

    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = os.path.join(temp_dir, "data")
        images_dir = os.path.join(data_dir, "images")
        os.makedirs(images_dir)

        _save_image_with_dpi(_artifact_image(0),
                             os.path.join(images_dir, "good.png"))
        # Write a bogus file as the second entry
        with open(os.path.join(images_dir, "bad.png"), "w") as f:
            f.write("not an image")

        metadata_path = os.path.join(data_dir, "metadata.csv")
        with open(metadata_path, "w") as f:
            f.write("image_id,scale_id,scale\n"
                    "good.png,scale_1,15.0\n"
                    "bad.png,scale_2,17.0\n")

        config_path = _write_config(temp_dir, sample_config)
        app = PyLithicsApplication(config_file=config_path)
        result = app.run_batch_analysis(data_dir, metadata_path)

    assert "bad.png" in result["failed_images"]
    assert result["processed_successfully"] == 1


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.functional
class TestInputValidation:

    def test_validate_inputs_rejects_missing_metadata(self, sample_config):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data")
            os.makedirs(os.path.join(data_dir, "images"))
            missing_meta = os.path.join(temp_dir, "missing.csv")

            app = PyLithicsApplication()
            assert app.validate_inputs(data_dir, missing_meta) is False

    def test_validate_inputs_rejects_missing_images_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_data = os.path.join(temp_dir, "missing_data")
            metadata = os.path.join(temp_dir, "metadata.csv")
            with open(metadata, "w") as f:
                f.write("image_id,scale_id,scale\ntest.png,s,15\n")

            app = PyLithicsApplication()
            assert app.validate_inputs(missing_data, metadata) is False


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


@pytest.mark.performance
def test_single_image_pipeline_completes_under_thirty_seconds(sample_config):
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir, meta_path = _make_workspace(
            temp_dir, rows=[("perf.png", "scale_1", "20.0")],
        )
        config_path = _write_config(temp_dir, sample_config)
        app = PyLithicsApplication(config_file=config_path)

        start = time.time()
        result = app.run_batch_analysis(data_dir, meta_path)
        elapsed = time.time() - start

    assert result["success"] is True
    assert elapsed < 30.0, f"pipeline took {elapsed:.1f}s (limit 30s)"
