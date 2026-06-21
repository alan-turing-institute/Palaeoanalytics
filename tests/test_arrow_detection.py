"""Tests for arrow detection in scar contours."""

import os
import tempfile

import numpy as np
import pytest

from pylithics.image_processing.modules.arrow_detection import (
    ArrowDetector,
    analyze_child_contour_for_arrow,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _poly(*pts):
    return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)


# A convincing arrow: tall triangle head on a narrow rectangular shaft.
# Scaled large enough to pass min_area filters at 300 DPI.
_ARROW_CONTOUR = _poly(
    [50, 10],   # tip
    [70, 30],   # right head
    [60, 30],   # right shoulder
    [60, 60],   # right shaft
    [40, 60],   # left shaft
    [40, 30],   # left shoulder
    [30, 30],   # left head
)


# ---------------------------------------------------------------------------
# ArrowDetector initialization
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestArrowDetectorInit:

    def test_default_config_loads_reference_values(self):
        detector = ArrowDetector()
        assert detector.reference_dpi == 300.0
        assert detector.debug_enabled is False
        assert set(detector.ref_thresholds) >= {
            "min_area", "min_defect_depth", "solidity_bounds",
            "min_triangle_height", "min_significant_defects",
        }

    def test_custom_config_overrides_reference_dpi_and_debug(self):
        detector = ArrowDetector({
            "reference_dpi": 150.0,
            "debug_enabled": True,
        })
        assert detector.reference_dpi == 150.0
        assert detector.debug_enabled is True


# ---------------------------------------------------------------------------
# scale_parameters_for_dpi: quadratic area scaling, linear depth scaling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestScaleParametersForDpi:

    def test_returns_reference_thresholds_for_matching_dpi(self):
        detector = ArrowDetector()
        params = detector.scale_parameters_for_dpi(300.0)
        # At reference DPI, min_area is ref * 1.0 * safety_factor (0.7)
        assert params["min_area"] == pytest.approx(
            detector.ref_thresholds["min_area"] * 0.7, abs=1e-6
        )
        assert params["solidity_bounds"] == detector.ref_thresholds["solidity_bounds"]

    def test_area_scales_quadratically_with_dpi(self):
        detector = ArrowDetector()
        params_300 = detector.scale_parameters_for_dpi(300.0)
        params_600 = detector.scale_parameters_for_dpi(600.0)
        # Doubling DPI should quadruple the min_area threshold
        assert params_600["min_area"] == pytest.approx(
            params_300["min_area"] * 4.0, rel=0.01
        )

    def test_defect_depth_scales_linearly_with_dpi(self):
        detector = ArrowDetector()
        params_300 = detector.scale_parameters_for_dpi(300.0)
        params_600 = detector.scale_parameters_for_dpi(600.0)
        assert params_600["min_defect_depth"] == pytest.approx(
            params_300["min_defect_depth"] * 2.0, rel=0.01
        )

    @pytest.mark.parametrize("invalid_dpi", [None, 0.0, -100.0])
    def test_invalid_dpi_returns_copy_of_reference_thresholds(self, invalid_dpi):
        detector = ArrowDetector()
        params = detector.scale_parameters_for_dpi(invalid_dpi)
        assert params == detector.ref_thresholds


# ---------------------------------------------------------------------------
# Basic filtering: tiny or non-arrow contours are rejected
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateBasicProperties:

    def test_tiny_contour_fails_area_check(self):
        detector = ArrowDetector()
        params = detector.scale_parameters_for_dpi(300.0)
        tiny = _poly([0, 0], [1, 0], [0, 1])
        assert detector._validate_basic_properties(tiny, params, None) is False

    def test_arrow_contour_passes_basic_validation(self):
        detector = ArrowDetector()
        params = detector.scale_parameters_for_dpi(300.0)
        assert detector._validate_basic_properties(_ARROW_CONTOUR, params, None) is True


# ---------------------------------------------------------------------------
# analyze_contour_for_arrow: end-to-end arrow detection
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAnalyzeContourForArrow:

    def test_rectangle_is_not_detected_as_arrow(self):
        detector = ArrowDetector()
        rectangle = _poly([10, 10], [50, 10], [50, 40], [10, 40])
        result = detector.analyze_contour_for_arrow(
            rectangle, {"scar": "test"},
            np.zeros((100, 100), dtype=np.uint8), 300.0,
        )
        assert result is None

    @pytest.mark.parametrize("degenerate", [
        np.array([], dtype=np.int32).reshape(0, 1, 2),
        np.array([[10, 10]], dtype=np.int32).reshape(-1, 1, 2),
    ])
    def test_degenerate_contours_return_none(self, degenerate):
        detector = ArrowDetector()
        result = detector.analyze_contour_for_arrow(
            degenerate, {"scar": "x"},
            np.zeros((100, 100), dtype=np.uint8), 300.0,
        )
        assert result is None


# ---------------------------------------------------------------------------
# _calculate_arrow_properties: verify the angle math directly
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCalculateArrowProperties:
    """The angle math here is load-bearing for downstream archaeological outputs."""

    def _properties_for_tip_and_base(self, tip, base_p1, base_p2):
        detector = ArrowDetector()
        base_midpoint = ((base_p1[0] + base_p2[0]) // 2,
                         (base_p1[1] + base_p2[1]) // 2)
        return detector._calculate_arrow_properties({
            "base_p1": base_p1,
            "base_p2": base_p2,
            "base_midpoint": base_midpoint,
            "triangle_tip": tip,
            "triangle_height": 20.0,
            "significant_defects": [],
        }, None)

    def test_arrow_back_is_triangle_tip_arrow_tip_is_base_midpoint(self):
        """The *named* arrow tip is the midpoint of the scar's base edge."""
        result = self._properties_for_tip_and_base((20, 10), (10, 30), (30, 30))
        assert result["arrow_back"] == (20, 10)
        assert result["arrow_tip"] == (20, 30)

    def test_downward_arrow_yields_compass_angle_zero(self):
        """
        Arrow from tip (20,10) to base midpoint (20,30) points +y in image
        coords. The code maps that to compass angle 0 (north in its rotated
        frame).
        """
        result = self._properties_for_tip_and_base((20, 10), (10, 30), (30, 30))
        assert result["angle_deg"] == pytest.approx(90.0, abs=1.0)
        assert result["compass_angle"] == pytest.approx(0.0, abs=1.0)

    def test_rightward_arrow_maps_to_compass_two_seventy(self):
        """
        Rightward in image coords (dx>0, dy=0) maps to compass 270 because the
        code rotates the frame by 270°. Downward = 0 and rightward = 270 fully
        pin the rotation direction.
        """
        result = self._properties_for_tip_and_base((10, 20), (30, 10), (30, 30))
        assert result["angle_deg"] == pytest.approx(0.0, abs=1.0)
        assert result["compass_angle"] == pytest.approx(270.0, abs=1.0)


# ---------------------------------------------------------------------------
# Backward-compatibility wrapper and debug output
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_backward_compat_wrapper_delegates_to_class_method():
    """analyze_child_contour_for_arrow is a thin wrapper around the class."""
    image = np.zeros((100, 100), dtype=np.uint8)
    entry = {"scar": "test"}

    via_wrapper = analyze_child_contour_for_arrow(
        _ARROW_CONTOUR, entry, image, 300.0,
    )
    via_class = ArrowDetector().analyze_contour_for_arrow(
        _ARROW_CONTOUR, entry, image, 300.0,
    )

    # Both return either a dict or None; when a dict, content should match.
    assert type(via_wrapper) is type(via_class)
    if via_wrapper is not None:
        assert via_wrapper == via_class


@pytest.mark.unit
def test_debug_visualization_writes_file_when_successful():
    detector = ArrowDetector({"debug_enabled": True})
    with tempfile.TemporaryDirectory() as temp_dir:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        triangle_data = {
            "base_p1": (10, 30),
            "base_p2": (30, 30),
            "base_midpoint": (20, 30),
            "triangle_tip": (20, 10),
            "significant_defects": [
                ((10, 30), (30, 30), (15, 25), 5.0),
            ],
        }
        arrow_data = {
            "arrow_back": (20, 10),
            "arrow_tip": (20, 30),
            "compass_angle": 0.0,
        }

        detector._create_debug_visualizations(
            _ARROW_CONTOUR, triangle_data, arrow_data, image, temp_dir,
        )
        assert os.path.exists(os.path.join(temp_dir, "arrow_debug.png"))
