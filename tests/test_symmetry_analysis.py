"""Tests for dorsal-surface symmetry analysis."""

import cv2
import numpy as np
import pytest

from pylithics.image_processing.modules.symmetry_analysis import (
    analyze_dorsal_symmetry,
)


EMPTY_RESULT = {
    "top_area": None, "bottom_area": None,
    "left_area": None, "right_area": None,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _poly(*pts):
    return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)


def _make_scene(contour, centroid, image_size=(100, 100)):
    """Return (metrics, contours_list, inverted_image) for a single Dorsal shape."""
    cx, cy = centroid
    metrics = [{
        "surface_type": "Dorsal",
        "parent": "parent 1",
        "scar": "parent 1",
        "centroid_x": cx,
        "centroid_y": cy,
    }]
    image = np.zeros(image_size, dtype=np.uint8)
    cv2.drawContours(image, [contour], -1, 255, thickness=cv2.FILLED)
    return metrics, [contour], image


# ---------------------------------------------------------------------------
# Null cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAnalyzeDorsalSymmetryNullCases:
    """Cases where no meaningful analysis can be done return an empty dict."""

    def test_no_dorsal_surface_returns_empty(self):
        metrics = [{
            "surface_type": "Ventral",
            "parent": "parent 1", "scar": "parent 1",
            "centroid_x": 50, "centroid_y": 50,
        }]
        result = analyze_dorsal_symmetry(
            metrics, [], np.zeros((100, 100), dtype=np.uint8)
        )
        assert result == EMPTY_RESULT

    def test_empty_contour_list_returns_empty(self):
        metrics = [{
            "surface_type": "Dorsal",
            "parent": "parent 1", "scar": "parent 1",
            "centroid_x": 50, "centroid_y": 50,
        }]
        result = analyze_dorsal_symmetry(
            metrics, [], np.zeros((100, 100), dtype=np.uint8)
        )
        assert result == EMPTY_RESULT

    def test_centroid_outside_contour_returns_empty(self):
        contour = _poly([20, 20], [40, 20], [40, 40], [20, 40])
        metrics, contours, image = _make_scene(contour, centroid=(10, 10))
        result = analyze_dorsal_symmetry(metrics, contours, image)
        assert result == EMPTY_RESULT


# ---------------------------------------------------------------------------
# Perfectly symmetric shapes: both axes should return 1.0
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAnalyzeDorsalSymmetryPerfectShapes:

    def test_centered_rectangle_is_perfectly_symmetric(self):
        """
        A 60x40 rectangle at (20..80, 30..70), centroid (50, 50), is split into
        four equal halves of 1200 pixels each. Both axes should hit 1.0 exactly.
        """
        contour = _poly([20, 30], [80, 30], [80, 70], [20, 70])
        metrics, contours, image = _make_scene(contour, centroid=(50, 50))

        result = analyze_dorsal_symmetry(metrics, contours, image)

        # OpenCV's drawContours boundary handling introduces a few-pixel slop;
        # 0.1 is still far tighter than the old 0.8–1.0 range.
        assert result["vertical_symmetry"] == pytest.approx(1.0, abs=0.1)
        assert result["horizontal_symmetry"] == pytest.approx(1.0, abs=0.1)

    def test_centered_diamond_is_perfectly_symmetric(self):
        contour = _poly([50, 20], [70, 50], [50, 80], [30, 50])
        metrics, contours, image = _make_scene(contour, centroid=(50, 50))

        result = analyze_dorsal_symmetry(metrics, contours, image)

        assert result["vertical_symmetry"] == pytest.approx(1.0, abs=0.1)
        assert result["horizontal_symmetry"] == pytest.approx(1.0, abs=0.1)


# ---------------------------------------------------------------------------
# Asymmetric shapes: symmetry scores must strictly decrease
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAnalyzeDorsalSymmetryAsymmetricShapes:

    def test_l_shape_is_asymmetric_on_both_axes(self):
        """
        An L-shape should score strictly below 1.0 on both axes.
        """
        l_shape = _poly(
            [20, 20], [60, 20], [60, 40], [40, 40], [40, 80], [20, 80],
        )
        # Use OpenCV's actual centroid so the point-in-polygon test passes
        M = cv2.moments(l_shape)
        centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        metrics, contours, image = _make_scene(l_shape, centroid=centroid)
        result = analyze_dorsal_symmetry(metrics, contours, image)

        assert result["vertical_symmetry"] < 1.0
        assert result["horizontal_symmetry"] < 1.0
        # Still positive — the shape isn't entirely to one side of the centroid
        assert result["vertical_symmetry"] > 0
        assert result["horizontal_symmetry"] > 0

    def test_offset_rectangle_vertical_axis_is_asymmetric(self):
        """
        A rectangle whose centroid is reported OFF-centre splits the mask
        unevenly. top != bottom and symmetry score drops below 1.0.
        """
        # 60x40 rectangle at (20..80, 30..70) — real centroid is (50, 50)
        contour = _poly([20, 30], [80, 30], [80, 70], [20, 70])
        # Report centroid 10px below actual centre to force asymmetry
        metrics, contours, image = _make_scene(contour, centroid=(50, 60))

        result = analyze_dorsal_symmetry(metrics, contours, image)

        assert result["top_area"] > result["bottom_area"]
        # Vertical symmetry noticeably drops — not just discretization noise
        assert result["vertical_symmetry"] < 0.75
        # Horizontal axis is still centered, so horizontal_symmetry stays ≈1.0
        assert result["horizontal_symmetry"] == pytest.approx(1.0, abs=0.1)


# ---------------------------------------------------------------------------
# Result schema and area conservation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_result_contains_all_six_keys_for_valid_input():
    contour = _poly([20, 20], [80, 20], [80, 80], [20, 80])
    metrics, contours, image = _make_scene(contour, centroid=(50, 50))

    result = analyze_dorsal_symmetry(metrics, contours, image)

    assert set(result.keys()) == {
        "top_area", "bottom_area", "left_area", "right_area",
        "vertical_symmetry", "horizontal_symmetry",
    }


@pytest.mark.unit
def test_top_plus_bottom_equals_left_plus_right_equals_contour_area():
    """The two halves of the mask must each sum to the full contour area."""
    contour = _poly([25, 25], [75, 25], [75, 75], [25, 75])
    metrics, contours, image = _make_scene(contour, centroid=(50, 50))

    result = analyze_dorsal_symmetry(metrics, contours, image)
    vertical_sum = result["top_area"] + result["bottom_area"]
    horizontal_sum = result["left_area"] + result["right_area"]

    assert vertical_sum == horizontal_sum
    # Should equal the number of filled pixels in the rendered mask
    filled = int(np.sum(image == 255))
    assert vertical_sum == filled


# ---------------------------------------------------------------------------
# Integration: realistic artifact still produces area-conservation invariants
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_irregular_contour_preserves_area_conservation():
    """For any shape, (top + bottom) and (left + right) must equal the mask area."""
    pts = []
    for i in range(20):
        angle = 2 * np.pi * i / 20
        radius = 30 + 5 * np.sin(3 * angle)
        x = int(50 + radius * np.cos(angle))
        y = int(50 + radius * np.sin(angle))
        pts.append([x, y])
    contour = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)

    M = cv2.moments(contour)
    centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    metrics, contours, image = _make_scene(contour, centroid=centroid)
    result = analyze_dorsal_symmetry(metrics, contours, image)

    filled = int(np.sum(image == 255))
    assert result["top_area"] + result["bottom_area"] == filled
    assert result["left_area"] + result["right_area"] == filled
    # Both symmetry scores must stay in [0, 1]
    assert 0.0 <= result["vertical_symmetry"] <= 1.0
    assert 0.0 <= result["horizontal_symmetry"] <= 1.0
