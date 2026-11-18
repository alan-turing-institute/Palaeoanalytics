"""Tests for the scale calibration module."""

import os
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from pylithics.image_processing.modules.scale_calibration import (
    bar_box,
    calculate_conversion_factor,
    detect_scale_bar,
    get_calibration_factor,
)


SCALE_EXAMPLES_DIR = os.path.join(
    os.path.dirname(__file__), "fixtures", "scale_calibration",
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


def _binary(shape):
    return np.zeros(shape, dtype=np.uint8)


def _fill(binary, x0, y0, x1, y1):
    binary[y0:y1, x0:x1] = 255


class TestBarBox:
    """The measure is the bar alone, with or without a recorded box."""

    def test_with_a_box_the_caption_outside_it_is_not_measured(self):
        binary = _binary((400, 100))
        _fill(binary, 30, 10, 60, 40)            # "cm" above the bar
        _fill(binary, 20, 60, 80, 360)           # a vertical bar, 300 px
        assert bar_box(binary, (15, 55, 85, 365)) == (20, 60, 60, 300)
        assert bar_box(binary) == (20, 60, 60, 300)

    def test_separate_blocks_with_a_caption_above(self):
        binary = _binary((120, 400))
        for x0 in (20, 120, 220):                # three blocks of one thickness
            _fill(binary, x0, 60, x0 + 50, 80)
        _fill(binary, 120, 20, 135, 45)          # "5"
        _fill(binary, 145, 25, 175, 45)          # "cm"
        assert bar_box(binary) == (20, 60, 250, 20)

    def test_a_rule_with_ticks_and_numerals(self):
        binary = _binary((100, 400))
        _fill(binary, 20, 60, 320, 64)           # the rule
        for x0 in (20, 170, 316):
            _fill(binary, x0, 50, x0 + 4, 64)    # ticks attached to the rule
        _fill(binary, 15, 20, 30, 40)            # "0" above the first tick
        _fill(binary, 300, 20, 330, 40)          # "10" above the last
        assert bar_box(binary) == (20, 50, 300, 14)

    def test_a_speck_beside_a_solid_bar_is_left_out(self):
        binary = _binary((100, 400))
        _fill(binary, 20, 40, 320, 60)
        _fill(binary, 350, 45, 354, 49)
        assert bar_box(binary) == (20, 40, 300, 20)

    def test_a_frame_around_the_scale_does_not_win_by_size(self):
        binary = _binary((400, 200))
        _fill(binary, 10, 10, 190, 390)          # a thin frame
        binary[14:386, 14:186] = 0               # hollow
        _fill(binary, 60, 80, 140, 380)          # the bar inside it
        assert bar_box(binary) == (60, 80, 80, 300)

    def test_a_box_that_took_in_the_frame_and_caption_still_gives_the_bar(self):
        binary = _binary((400, 200))
        _fill(binary, 10, 10, 190, 390)
        binary[14:386, 14:186] = 0               # a frame around everything
        _fill(binary, 70, 30, 130, 60)           # "cm" inside the frame
        _fill(binary, 60, 80, 140, 380)          # the bar
        assert bar_box(binary, (10, 10, 190, 390)) == (60, 80, 80, 300)

    def test_a_numeral_as_tall_as_the_bar_is_thick_is_not_a_block(self):
        binary = _binary((100, 500))
        for x0 in (20, 120, 220):
            _fill(binary, x0, 40, x0 + 50, 60)   # three blocks, 20 thick
        _fill(binary, 300, 40, 330, 44)          # a hollow "5": top bar,
        _fill(binary, 300, 44, 304, 50)          # left stem,
        _fill(binary, 300, 50, 330, 54)          # middle bar,
        _fill(binary, 326, 54, 330, 60)          # right stem, bottom bar
        _fill(binary, 300, 56, 330, 60)
        assert bar_box(binary) == (20, 40, 250, 20)
        assert bar_box(binary, (15, 35, 275, 65)) == (20, 40, 250, 20)

    def test_a_block_the_box_cut_short_is_still_measured(self):
        binary = _binary((100, 400))
        for x0 in (20, 120, 220):
            _fill(binary, x0, 40, x0 + 50, 60)
        assert bar_box(binary, (110, 35, 280, 65)) == (20, 40, 250, 20)   # box misses block one

    def test_a_block_too_far_along_the_axis_is_another_mark(self):
        binary = _binary((100, 600))
        _fill(binary, 20, 40, 120, 60)
        _fill(binary, 170, 40, 270, 60)          # one gap on: a block
        _fill(binary, 500, 40, 600, 60)          # two and a half lengths on: not
        assert bar_box(binary) == (20, 40, 250, 20)

    def test_no_ink_gives_none(self):
        assert bar_box(_binary((50, 50))) is None


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
        assert method == "pixels_no_scale"
        assert confidence is None

    def test_falls_back_to_pixels_when_no_scale_data(self):
        factor, method, confidence = get_calibration_factor(
            "/any/path/img.png", {}, {}
        )
        assert factor is None
        assert method == "pixels_no_scale"
        assert confidence is None

    def test_falls_back_when_scale_image_missing(self, tmp_path):
        artifact = tmp_path / "images" / "artifact.png"
        artifact.parent.mkdir()
        cv2.imwrite(str(artifact), np.full((10, 10), 255, dtype=np.uint8))

        scale_data = {"scale_id": "missing.png", "scale": "50"}
        factor, method, _ = get_calibration_factor(str(artifact), scale_data, {})
        assert factor is None
        assert method == "pixels_detection_failed"

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

    def test_scale_box_in_the_metadata_bounds_the_measure(self, tmp_path):
        image = np.full((50, 300), 255, dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (220, 30), color=0, thickness=-1)   # the bar, 200 px
        cv2.rectangle(image, (240, 15), (280, 35), color=0, thickness=-1)  # "cm", as thick
        path = str(tmp_path / "bar_with_caption.png")
        cv2.imwrite(path, image)
        artifact = self._make_layout(tmp_path, "sc.png", path)
        with_box = {"scale_id": "sc.png", "scale": "50", "scale_box": "15 15 230 35"}
        factor, method, confidence = get_calibration_factor(artifact, with_box, {})
        assert method == "scale_bar" and factor == pytest.approx(201 / 50.0, abs=0.05)
        assert confidence == 0.95

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
        assert method == "pixels_detection_failed"
