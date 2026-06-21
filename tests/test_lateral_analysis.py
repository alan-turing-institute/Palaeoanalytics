"""Tests for lateral-surface convexity and distance calculations."""

import numpy as np
import pytest

from pylithics.image_processing.modules.lateral_analysis import (
    _calculate_lateral_distance_to_max_width,
    _integrate_lateral_metrics,
    analyze_lateral_surface,
    detect_lateral_convexity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _poly(*pts):
    return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)


def _rect(x0, y0, x1, y1):
    return _poly([x0, y0], [x1, y0], [x1, y1], [x0, y1])


def _circle_contour(center=(50, 50), radius=30, n_points=32):
    pts = []
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        pts.append([
            int(center[0] + radius * np.cos(angle)),
            int(center[1] + radius * np.sin(angle)),
        ])
    return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)


def _star_contour(center=(50, 50), outer_r=30, inner_r=15, n_points=10):
    pts = []
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        radius = outer_r if i % 2 == 0 else inner_r
        pts.append([
            int(center[0] + radius * np.cos(angle)),
            int(center[1] + radius * np.sin(angle)),
        ])
    return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)


def _lateral_metric(area=1000.0, **overrides):
    base = {
        "surface_type": "Lateral",
        "parent": "parent 1",
        "scar": "parent 1",
        "area": area,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# detect_lateral_convexity: numerical properties of known shapes
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDetectLateralConvexity:
    """Convex shapes must give ≈1.0; concavity must reduce the ratio."""

    def test_rectangle_is_fully_convex(self):
        convexity = detect_lateral_convexity(_rect(10, 20, 60, 80))
        assert convexity == pytest.approx(1.0, abs=1e-6)

    def test_triangle_is_fully_convex(self):
        triangle = _poly([50, 20], [80, 70], [20, 70])
        convexity = detect_lateral_convexity(triangle)
        assert convexity == pytest.approx(1.0, abs=1e-6)

    def test_circle_approximates_one(self):
        # 32 points on a radius-30 circle, discretized to int pixels
        convexity = detect_lateral_convexity(_circle_contour(radius=30, n_points=32))
        # Discretization trims a tiny bit; accept >= 0.95
        assert 0.95 <= convexity <= 1.0

    def test_l_shape_matches_exact_area_ratio(self):
        """
        L-shape area = 1000, convex hull area = 1100; convexity = 10/11 ≈ 0.909.
        """
        l_shape = _poly(
            [10, 10], [40, 10], [40, 30], [30, 30], [30, 50], [10, 50],
        )
        convexity = detect_lateral_convexity(l_shape)
        assert convexity == pytest.approx(10 / 11, abs=0.01)

    def test_star_is_strongly_concave(self):
        convexity = detect_lateral_convexity(_star_contour())
        assert convexity < 0.7

    @pytest.mark.parametrize("bad_input", [
        None,
        np.array([], dtype=np.int32).reshape(0, 1, 2),
        np.array([[50, 50]], dtype=np.int32).reshape(-1, 1, 2),
    ])
    def test_invalid_contours_return_none(self, bad_input):
        assert detect_lateral_convexity(bad_input) is None

    def test_zero_area_line_returns_none(self):
        """A strictly colinear contour has zero area and should yield None."""
        line = _poly([10, 50], [30, 50], [50, 50], [70, 50])
        assert detect_lateral_convexity(line) is None


# ---------------------------------------------------------------------------
# _calculate_lateral_distance_to_max_width: geometry sanity
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLateralDistanceToMaxWidth:
    """Distance is from the top-point to the midpoint of the longest segment."""

    def test_vertical_rectangle_returns_positive_distance(self):
        # 20x60 rectangle, longest diagonal ≈ 63.25
        distance = _calculate_lateral_distance_to_max_width(
            _rect(40, 20, 60, 80)
        )
        assert distance is not None
        assert distance > 0

    def test_triangle_apex_to_base_center(self):
        triangle = _poly([50, 20], [20, 80], [80, 80])
        distance = _calculate_lateral_distance_to_max_width(triangle)
        assert distance is not None
        assert distance > 0

    @pytest.mark.parametrize("bad_input", [
        None,
        np.array([[50, 50]], dtype=np.int32).reshape(-1, 1, 2),
    ])
    def test_invalid_contour_returns_none(self, bad_input):
        assert _calculate_lateral_distance_to_max_width(bad_input) is None


# ---------------------------------------------------------------------------
# analyze_lateral_surface: high-level orchestration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAnalyzeLateralSurface:

    def test_returns_none_when_no_lateral_metric_present(self):
        metrics = [{
            "surface_type": "Dorsal",
            "parent": "parent 1",
            "scar": "parent 1",
            "area": 5000,
        }]
        result = analyze_lateral_surface(
            metrics, [_rect(0, 0, 100, 100)],
            np.zeros((200, 200), dtype=np.uint8),
        )
        assert result is None

    def test_returns_dict_with_rounded_values_for_lateral_surface(self):
        lateral_contour = _rect(10, 20, 60, 80)
        metrics = [_lateral_metric(area=3000.0)]
        result = analyze_lateral_surface(
            metrics, [lateral_contour],
            np.zeros((100, 100), dtype=np.uint8),
        )

        assert result is not None
        assert set(result.keys()) == {"lateral_convexity", "distance_to_max_width"}
        # Rectangle should be fully convex
        assert result["lateral_convexity"] == pytest.approx(1.0, abs=0.01)
        # Values are rounded to 2 decimal places
        assert result["distance_to_max_width"] == round(
            result["distance_to_max_width"], 2
        )

    def test_falls_back_to_area_match_when_index_exceeds_contours(self):
        """
        If the lateral metric's index is beyond the parent contour list,
        the code falls back to matching by area.
        """
        # Lateral is at index 1, but we only pass the matching contour at
        # index 0 — the fallback path should match by area.
        lateral_contour = _rect(0, 0, 40, 25)  # area 1000
        metrics = [
            {
                "surface_type": "Dorsal",
                "parent": "parent 1",
                "scar": "parent 1",
                "area": 5000,
            },
            _lateral_metric(area=1000.0),
        ]
        result = analyze_lateral_surface(
            metrics, [lateral_contour],
            np.zeros((50, 50), dtype=np.uint8),
        )
        assert result is not None
        assert result["lateral_convexity"] == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# _integrate_lateral_metrics: in-place update
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIntegrateLateralMetrics:

    def test_adds_lateral_fields_to_lateral_entry(self):
        metrics = [
            {"surface_type": "Dorsal", "parent": "parent 1", "scar": "parent 1"},
            _lateral_metric(area=1000.0),
        ]
        _integrate_lateral_metrics(
            metrics,
            {"lateral_convexity": 0.85, "distance_to_max_width": 12.5},
        )
        assert metrics[1]["lateral_convexity"] == 0.85
        assert metrics[1]["distance_to_max_width"] == 12.5

    def test_dorsal_entries_are_untouched(self):
        metrics = [
            {"surface_type": "Dorsal", "parent": "parent 1", "scar": "parent 1"},
            _lateral_metric(area=1000.0),
        ]
        _integrate_lateral_metrics(
            metrics,
            {"lateral_convexity": 0.75, "distance_to_max_width": 8.0},
        )
        assert "lateral_convexity" not in metrics[0]

    def test_no_lateral_metric_logs_warning(self):
        from unittest.mock import patch
        metrics = [
            {"surface_type": "Dorsal", "parent": "parent 1", "scar": "parent 1"},
        ]
        with patch(
            "pylithics.image_processing.modules.lateral_analysis.logging"
        ) as mock_log:
            _integrate_lateral_metrics(metrics, {"lateral_convexity": 0.9})
        mock_log.warning.assert_called()
