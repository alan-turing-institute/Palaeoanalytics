"""Tests for the scale calibration module."""

import os
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from pylithics.image_processing.modules.scale_calibration import (
    calculate_conversion_factor,
    detect_scale_bar,
    get_calibration_factor,
)


SCALE_EXAMPLES_DIR = os.path.join(
    os.path.dirname(__file__), "..", ".claude", "visual_examples",
    "features", "scale_calibration",
)

SCALE_EXAMPLE_FILES = [
    "sc_001.png",
    "sc_004.png",
    "sc_005.png",
    "sc_006.png",
    "sc_007.png",
    "sc_008.png",
]


def _scale_path(filename):
    return os.path.normpath(os.path.join(SCALE_EXAMPLES_DIR, filename))


def _make_horizontal_bar_image(tmp_path, length_px=200, thickness_px=10):
    """Create a synthetic horizontal scale bar image and return its path."""
    image = np.full((50, length_px + 40), 255, dtype=np.uint8)
    cv2.rectangle(
        image,
        (20, 20),
        (20 + length_px, 20 + thickness_px),
        color=0,
        thickness=-1,
    )
    path = str(tmp_path / "synthetic_bar.png")
    cv2.imwrite(path, image)
    return path


class TestDetectScaleBar:
    """Tests for detect_scale_bar."""

    def test_returns_none_for_missing_file(self, tmp_path):
        result = detect_scale_bar(str(tmp_path / "missing.png"), {})
        assert result is None

    def test_returns_none_for_blank_image(self, tmp_path):
        blank = np.full((50, 50), 255, dtype=np.uint8)
        path = str(tmp_path / "blank.png")
        cv2.imwrite(path, blank)
        assert detect_scale_bar(path, {}) is None

    def test_measures_synthetic_horizontal_bar(self, tmp_path):
        path = _make_horizontal_bar_image(tmp_path, length_px=200)
        result = detect_scale_bar(path, {})

        assert result is not None
        scale_pixels, confidence = result
        # Bounding box should equal the bar length (allowing 1px rounding)
        assert abs(scale_pixels - 200) <= 1
        assert 0.0 <= confidence <= 1.0
        # A 200x10 bar has aspect ratio 20 -> capped at 1.0
        assert confidence == pytest.approx(1.0)

    def test_low_confidence_for_square_shape(self, tmp_path):
        square = np.full((40, 40), 255, dtype=np.uint8)
        cv2.rectangle(square, (10, 10), (30, 30), 0, -1)
        path = str(tmp_path / "square.png")
        cv2.imwrite(path, square)

        result = detect_scale_bar(path, {})
        assert result is not None
        _, confidence = result
        assert confidence < 0.5

    @pytest.mark.parametrize("scale_file", SCALE_EXAMPLE_FILES)
    def test_real_world_scale_examples(self, scale_file):
        """Each provided real-world scale image must yield a measurement."""
        path = _scale_path(scale_file)
        if not os.path.exists(path):
            pytest.skip(f"Reference image missing: {scale_file}")

        result = detect_scale_bar(path, {})
        assert result is not None, f"detection failed for {scale_file}"
        scale_pixels, confidence = result
        assert scale_pixels > 0
        assert 0.0 <= confidence <= 1.0


class TestCalculateConversionFactor:
    """Tests for calculate_conversion_factor."""

    def test_basic_division(self):
        assert calculate_conversion_factor(100, 10.0) == pytest.approx(10.0)

    def test_fractional_result(self):
        assert calculate_conversion_factor(150, 50.0) == pytest.approx(3.0)

    @pytest.mark.parametrize("invalid_mm", [0, -1, -50.5])
    def test_rejects_non_positive_scale(self, invalid_mm):
        with pytest.raises(ValueError):
            calculate_conversion_factor(100, invalid_mm)


class TestGetCalibrationFactor:
    """Tests for get_calibration_factor."""

    def _make_layout(self, tmp_path, scale_filename, scale_image_path):
        """Lay out a data dir with images/ and scales/ siblings."""
        images_dir = tmp_path / "images"
        scales_dir = tmp_path / "scales"
        images_dir.mkdir()
        scales_dir.mkdir()

        scale_target = scales_dir / scale_filename
        # Copy the synthetic bar into the scales/ directory
        cv2.imwrite(str(scale_target), cv2.imread(scale_image_path))

        artifact_path = images_dir / "artifact.png"
        cv2.imwrite(str(artifact_path), np.full((10, 10), 255, dtype=np.uint8))
        return str(artifact_path)

    def test_falls_back_to_pixels_when_disabled(self):
        config = {"scale_calibration": {"enabled": False}}
        scale_data = {"scale_id": "sc.png", "scale": "50"}
        factor, method, confidence = get_calibration_factor(
            "/any/path/img.png", scale_data, config
        )
        assert factor is None
        assert method == "pixels"
        assert confidence is None

    def test_falls_back_to_pixels_when_no_scale_data(self):
        factor, method, confidence = get_calibration_factor(
            "/any/path/img.png", {}, {}
        )
        assert factor is None
        assert method == "pixels"
        assert confidence is None

    def test_falls_back_when_scale_image_missing(self, tmp_path):
        artifact = tmp_path / "images" / "artifact.png"
        artifact.parent.mkdir()
        cv2.imwrite(str(artifact), np.full((10, 10), 255, dtype=np.uint8))

        scale_data = {"scale_id": "missing.png", "scale": "50"}
        factor, method, _ = get_calibration_factor(str(artifact), scale_data, {})
        assert factor is None
        assert method == "pixels"

    def test_successful_scale_bar_calibration(self, tmp_path):
        synthetic = _make_horizontal_bar_image(tmp_path, length_px=200)
        artifact = self._make_layout(tmp_path, "sc.png", synthetic)

        scale_data = {"scale_id": "sc.png", "scale": "50"}
        factor, method, confidence = get_calibration_factor(
            artifact, scale_data, {}
        )

        assert method == "scale_bar"
        assert factor == pytest.approx(200 / 50.0, abs=0.1)
        assert 0.0 <= confidence <= 1.0

    def test_resolves_scale_id_without_extension(self, tmp_path):
        synthetic = _make_horizontal_bar_image(tmp_path, length_px=100)
        artifact = self._make_layout(tmp_path, "sc.png", synthetic)

        # scale_id provided WITHOUT extension; the function should add .png
        scale_data = {"scale_id": "sc", "scale": "25"}
        factor, method, _ = get_calibration_factor(artifact, scale_data, {})

        assert method == "scale_bar"
        assert factor == pytest.approx(100 / 25.0, abs=0.1)

    def test_falls_back_when_detection_returns_none(self, tmp_path):
        artifact = tmp_path / "images" / "artifact.png"
        artifact.parent.mkdir()
        scales_dir = tmp_path / "scales"
        scales_dir.mkdir()

        cv2.imwrite(str(artifact), np.full((10, 10), 255, dtype=np.uint8))
        # Blank scale image so detection fails
        cv2.imwrite(
            str(scales_dir / "blank.png"),
            np.full((20, 20), 255, dtype=np.uint8),
        )

        scale_data = {"scale_id": "blank.png", "scale": "50"}
        with patch(
            "pylithics.image_processing.modules.scale_calibration.detect_scale_bar",
            return_value=None,
        ):
            factor, method, _ = get_calibration_factor(
                str(artifact), scale_data, {}
            )

        assert factor is None
        assert method == "pixels"
