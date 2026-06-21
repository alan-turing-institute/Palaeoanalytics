"""Tests for image preprocessing primitives and the pipeline."""

import os
import tempfile
import time
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from PIL import Image

from pylithics.image_processing.importer import (
    apply_contrast_normalization,
    apply_grayscale_conversion,
    execute_preprocessing_pipeline,
    invert_image,
    morphological_closing,
    perform_thresholding,
    preprocess_images,
    read_image_from_path,
    verify_image_dpi_and_scale,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_bgr_with_dpi(image_bgr, path, dpi=300):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(path, dpi=(dpi, dpi))


# ---------------------------------------------------------------------------
# read_image_from_path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReadImageFromPath:

    def test_returns_array_for_valid_image(self, test_image_with_dpi):
        image = read_image_from_path(test_image_with_dpi)
        assert image is not None
        assert image.ndim == 3

    def test_returns_none_for_missing_file(self):
        assert read_image_from_path("/nonexistent/path.png") is None

    def test_returns_none_for_invalid_content(self, tmp_path):
        bogus = tmp_path / "not_an_image.png"
        bogus.write_text("nope")
        assert read_image_from_path(str(bogus)) is None


# ---------------------------------------------------------------------------
# apply_grayscale_conversion
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestApplyGrayscaleConversion:

    @pytest.mark.parametrize("method", ["standard", "clahe"])
    def test_produces_single_channel_output(self, method):
        image = np.random.randint(0, 256, (30, 30, 3), dtype=np.uint8)
        result = apply_grayscale_conversion(
            image, {"grayscale_conversion": {"enabled": True, "method": method}}
        )
        assert result is not None
        assert result.ndim == 2
        assert result.shape == image.shape[:2]

    def test_disabled_returns_input_unchanged(self):
        image = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
        result = apply_grayscale_conversion(
            image, {"grayscale_conversion": {"enabled": False}}
        )
        assert result is image

    def test_invalid_method_returns_none_and_logs(self):
        image = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
        with patch("pylithics.image_processing.importer.logging") as mock_log:
            result = apply_grayscale_conversion(
                image, {"grayscale_conversion": {"enabled": True, "method": "bogus"}}
            )
        assert result is None
        mock_log.error.assert_called()

    def test_defaults_used_when_config_missing(self):
        # Missing grayscale_conversion section should fall back to enabled+standard
        image = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
        result = apply_grayscale_conversion(image, {})
        assert result is not None
        assert result.ndim == 2


# ---------------------------------------------------------------------------
# apply_contrast_normalization
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestApplyContrastNormalization:

    def test_minmax_stretches_to_full_range(self):
        # Input span is 25..225; min-max should map to 0..255
        gray = np.array(
            [[25, 100, 150], [75, 125, 175], [50, 200, 225]],
            dtype=np.uint8,
        )
        result = apply_contrast_normalization(
            gray,
            {"normalization": {
                "enabled": True, "method": "minmax", "clip_values": [0, 255],
            }},
        )
        assert result is not None
        assert result.min() == 0
        assert result.max() == 255

    def test_zscore_produces_zero_mean_unit_variance(self):
        rng = np.random.default_rng(42)
        gray = rng.integers(50, 200, (80, 80), dtype=np.uint8)

        result = apply_contrast_normalization(
            gray, {"normalization": {"enabled": True, "method": "zscore"}}
        )
        assert result is not None
        assert result.mean() == pytest.approx(0.0, abs=1e-6)
        assert result.std() == pytest.approx(1.0, abs=1e-6)

    def test_disabled_returns_input_unchanged(self):
        gray = np.random.randint(0, 256, (40, 40), dtype=np.uint8)
        result = apply_contrast_normalization(
            gray, {"normalization": {"enabled": False, "method": "minmax"}}
        )
        assert np.array_equal(result, gray)

    def test_invalid_method_returns_none_and_logs(self):
        gray = np.random.randint(0, 256, (20, 20), dtype=np.uint8)
        with patch("pylithics.image_processing.importer.logging") as mock_log:
            result = apply_contrast_normalization(
                gray, {"normalization": {"enabled": True, "method": "bogus"}}
            )
        assert result is None
        mock_log.error.assert_called()


# ---------------------------------------------------------------------------
# perform_thresholding
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPerformThresholding:

    @pytest.mark.parametrize("method", ["simple", "otsu", "adaptive", "default"])
    def test_output_is_binary_with_matching_shape(self, method):
        # Bimodal input so otsu has something to split on
        gray = np.concatenate([
            np.full((40, 40), 80, dtype=np.uint8),
            np.full((40, 40), 180, dtype=np.uint8),
        ])
        config = {
            "thresholding": {
                "method": method, "threshold_value": 127, "max_value": 255,
            },
        }
        result = perform_thresholding(gray, config)

        assert result is not None
        assert result.shape == gray.shape
        assert set(np.unique(result)).issubset({0, 255})

    def test_simple_threshold_splits_on_threshold_value(self):
        # Ramp from 0..255; threshold 127 should put ~half on each side
        gray = np.tile(np.arange(256, dtype=np.uint8), (1, 1))
        config = {
            "thresholding": {
                "method": "simple", "threshold_value": 127, "max_value": 255,
            },
        }
        result = perform_thresholding(gray, config)
        # cv2.GaussianBlur with kernel=5 softens a bit, so compare roughly
        assert result is not None
        bright = int(np.sum(result == 255))
        dark = int(np.sum(result == 0))
        assert abs(bright - dark) < 30  # roughly balanced

    def test_invalid_method_returns_none_and_logs(self):
        gray = np.random.randint(0, 256, (30, 30), dtype=np.uint8)
        with patch("pylithics.image_processing.importer.logging") as mock_log:
            result = perform_thresholding(
                gray, {"thresholding": {"method": "bogus"}}
            )
        assert result is None
        mock_log.error.assert_called()


# ---------------------------------------------------------------------------
# invert_image
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInvertImage:

    def test_binary_image_is_inverted(self):
        arr = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        assert np.array_equal(invert_image(arr), np.array([[255, 0], [0, 255]]))

    def test_grayscale_inversion_is_255_minus_input(self):
        arr = np.array([[0, 100, 255], [50, 150, 200]], dtype=np.uint8)
        assert np.array_equal(invert_image(arr), 255 - arr)


# ---------------------------------------------------------------------------
# morphological_closing
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMorphologicalClosing:

    def test_fills_gaps_in_binary_shape(self):
        # 5x5 image with a single-pixel hole in the middle
        image = np.full((5, 5), 255, dtype=np.uint8)
        image[2, 2] = 0

        config = {"morphological_closing": {"enabled": True, "kernel_size": 3}}
        closed = morphological_closing(image, config)

        assert closed[2, 2] == 255  # hole filled

    @pytest.mark.parametrize("kernel_size", [1, 3, 5, 7])
    def test_preserves_shape_for_various_kernel_sizes(self, kernel_size):
        image = np.random.randint(0, 2, (40, 40), dtype=np.uint8) * 255
        config = {
            "morphological_closing": {"enabled": True, "kernel_size": kernel_size},
        }
        closed = morphological_closing(image, config)
        assert closed.shape == image.shape


# ---------------------------------------------------------------------------
# verify_image_dpi_and_scale
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVerifyImageDpiAndScale:

    def test_returns_pixels_per_mm_for_valid_image(self, test_image_with_dpi):
        # Fixture is saved with 300 DPI; 300 / 25.4 ≈ 11.811
        result = verify_image_dpi_and_scale(test_image_with_dpi, 15.0)
        assert result == pytest.approx(300 / 25.4, abs=1e-4)

    def test_returns_none_when_dpi_missing(self, tmp_path):
        no_dpi = tmp_path / "no_dpi.png"
        cv2.imwrite(str(no_dpi), np.full((50, 50, 3), 128, dtype=np.uint8))
        assert verify_image_dpi_and_scale(str(no_dpi), 15.0) is None

    def test_returns_none_for_missing_file(self):
        assert verify_image_dpi_and_scale("/nonexistent.png", 15.0) is None


# ---------------------------------------------------------------------------
# execute_preprocessing_pipeline
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExecutePreprocessingPipeline:

    def test_returns_binary_image_for_valid_input(
        self, test_image_with_dpi, sample_config
    ):
        result = execute_preprocessing_pipeline(test_image_with_dpi, sample_config)
        assert result is not None
        assert result.ndim == 2
        assert set(np.unique(result)).issubset({0, 255})

    def test_returns_none_for_missing_image(self, sample_config):
        assert execute_preprocessing_pipeline(
            "/nonexistent.png", sample_config
        ) is None

    def test_short_circuits_on_stage_failure(self, test_image_with_dpi):
        """If grayscale conversion returns None, the pipeline stops early."""
        with patch(
            "pylithics.image_processing.importer.apply_grayscale_conversion",
            return_value=None,
        ):
            result = execute_preprocessing_pipeline(test_image_with_dpi, {})
        assert result is None


# ---------------------------------------------------------------------------
# preprocess_images (batch helper)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPreprocessImages:
    """preprocess_images reads metadata and batches the preprocessing calls."""

    def _write_metadata(self, path, rows):
        with open(path, "w") as f:
            f.write("image_id,scale_id,scale\n")
            for row in rows:
                f.write(",".join(row) + "\n")

    def test_processes_all_listed_images(
        self, test_image_directory, sample_metadata_file, sample_config
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            import shutil
            data_dir = os.path.join(temp_dir, "data")
            shutil.copytree(os.path.dirname(test_image_directory), data_dir)

            # preprocess_images loads config.yaml from cwd — patch the loader
            with patch(
                "pylithics.image_processing.importer.load_preprocessing_config",
                return_value=sample_config,
            ):
                result = preprocess_images(
                    data_dir, sample_metadata_file, show_thresholded_images=False
                )

            # Three metadata entries; only one image file actually exists
            # in the fixture (test_image_1.png), so we get one entry back.
            assert "test_image_1.png" in result
            processed_image, conversion = result["test_image_1.png"]
            assert processed_image.ndim == 2
            assert conversion > 0

    def test_returns_empty_when_config_fails_to_load(
        self, test_image_directory, sample_metadata_file
    ):
        with patch(
            "pylithics.image_processing.importer.load_preprocessing_config",
            return_value=None,
        ):
            result = preprocess_images(
                os.path.dirname(test_image_directory),
                sample_metadata_file,
                show_thresholded_images=False,
            )
        assert result == {}


# ---------------------------------------------------------------------------
# Edge cases: uniform images produce well-defined outputs
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPipelineEdgeCases:

    @pytest.mark.parametrize("fill_value", [0, 128, 255])
    def test_uniform_image_yields_uniform_binary_output(
        self, fill_value, sample_config
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "uniform.png")
            _save_bgr_with_dpi(
                np.full((80, 80, 3), fill_value, dtype=np.uint8), path
            )
            result = execute_preprocessing_pipeline(path, sample_config)

        assert result is not None
        assert set(np.unique(result)).issubset({0, 255})


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


@pytest.mark.performance
def test_pipeline_on_800x600_image_completes_under_five_seconds(sample_config):
    image = np.full((600, 800, 3), 220, dtype=np.uint8)
    cv2.rectangle(image, (200, 150), (600, 450), (60, 60, 60), -1)

    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "perf.png")
        _save_bgr_with_dpi(image, path)

        start = time.time()
        result = execute_preprocessing_pipeline(path, sample_config)
        elapsed = time.time() - start

    assert result is not None
    assert elapsed < 5.0, f"pipeline took {elapsed:.2f}s (limit 5s)"
