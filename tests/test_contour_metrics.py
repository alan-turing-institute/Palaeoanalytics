"""Tests for contour metric calculation and real-world unit conversion."""

import numpy as np
import pytest

from pylithics.image_processing.modules.contour_metrics import (
    calculate_contour_metrics,
    convert_metrics_to_real_world,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _poly(*pts):
    return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)


def _rect(x0, y0, x1, y1):
    return _poly([x0, y0], [x1, y0], [x1, y1], [x0, y1])


def _calc(sorted_contours, hierarchy, contours, image_shape=(200, 200)):
    return calculate_contour_metrics(
        sorted_contours, hierarchy, contours, image_shape
    )


# ---------------------------------------------------------------------------
# calculate_contour_metrics: parent-only geometry
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCalculateContourMetricsParent:
    """Geometric properties of parent contours."""

    def test_rectangle_produces_expected_area_and_centroid(self):
        parent = _rect(10, 20, 60, 80)  # 50 wide, 60 tall
        metrics = _calc(
            {"parents": [parent], "children": [], "nested_children": []},
            np.array([[-1, -1, -1, -1]]),
            [parent],
        )

        assert len(metrics) == 1
        m = metrics[0]
        assert m["parent"] == "parent 1"
        assert m["scar"] == "parent 1"
        assert m["area"] == pytest.approx(50 * 60, abs=1.0)
        assert m["centroid_x"] == pytest.approx(35, abs=1.0)
        assert m["centroid_y"] == pytest.approx(50, abs=1.0)
        assert m["technical_width"] == 50.0
        assert m["technical_length"] == 60.0
        assert m["aspect_ratio"] == pytest.approx(60.0 / 50.0, abs=1e-3)

    def test_triangle_centroid_matches_geometric_centroid(self):
        triangle = _poly([0, 0], [60, 0], [30, 60])
        metrics = _calc(
            {"parents": [triangle], "children": [], "nested_children": []},
            np.array([[-1, -1, -1, -1]]),
            [triangle],
        )

        # Exact centroid of (0,0), (60,0), (30,60) is (30, 20)
        m = metrics[0]
        assert m["centroid_x"] == pytest.approx(30.0, abs=1.0)
        assert m["centroid_y"] == pytest.approx(20.0, abs=1.0)

    def test_rectangle_max_length_equals_diagonal(self):
        rectangle = _rect(10, 20, 70, 40)  # 60x20
        metrics = _calc(
            {"parents": [rectangle], "children": [], "nested_children": []},
            np.array([[-1, -1, -1, -1]]),
            [rectangle],
        )
        m = metrics[0]
        # Max length is the longest point-to-point distance, i.e. the diagonal
        assert m["max_length"] == pytest.approx(np.sqrt(60 ** 2 + 20 ** 2), abs=1.0)
        assert m["max_length"] >= m["max_width"]

    def test_vertical_line_has_zero_width_and_null_aspect_ratio(self):
        line = _poly([25, 10], [25, 20], [25, 30], [25, 40])
        metrics = _calc(
            {"parents": [line], "children": [], "nested_children": []},
            np.array([[-1, -1, -1, -1]]),
            [line],
        )
        m = metrics[0]
        assert m["technical_width"] == 0.0
        assert m["technical_length"] == 30.0
        assert m["aspect_ratio"] is None

    def test_degenerate_contour_returns_nonnegative_metrics(self):
        degen = _poly([10, 10], [20, 10], [20, 10], [10, 10])
        metrics = _calc(
            {"parents": [degen], "children": [], "nested_children": []},
            np.array([[-1, -1, -1, -1]]),
            [degen],
        )
        m = metrics[0]
        assert m["area"] >= 0
        assert "centroid_x" in m and "centroid_y" in m


# ---------------------------------------------------------------------------
# calculate_contour_metrics: parent + child
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCalculateContourMetricsChild:
    """Child contours use `width`/`height` rather than the parent's technical dims."""

    def test_child_fields_and_sizes(self):
        parent = _rect(10, 10, 90, 90)
        child = _rect(30, 30, 50, 50)
        hierarchy = np.array([
            [-1, -1, 1, -1],
            [-1, -1, -1, 0],
        ])
        metrics = _calc(
            {"parents": [parent], "children": [child], "nested_children": []},
            hierarchy,
            [parent, child],
        )

        assert len(metrics) == 2
        parent_m, child_m = metrics
        assert parent_m["area"] == pytest.approx(80 * 80, abs=1.0)
        assert child_m["scar"] == "child 1"
        assert child_m["area"] == pytest.approx(20 * 20, abs=1.0)
        assert {"width", "height"}.issubset(child_m)
        # Children do not carry technical_width/length
        assert "technical_width" not in child_m

    def test_missing_index_mapping_logs_warning(self):
        # sorted contour doesn't appear in original_contours
        sorted_parent = _rect(10, 10, 50, 50)
        mismatch = _rect(20, 20, 60, 60)

        from unittest.mock import patch
        with patch(
            "pylithics.image_processing.modules.contour_metrics.logging"
        ) as mock_log:
            _calc(
                {"parents": [sorted_parent], "children": [], "nested_children": []},
                np.array([[-1, -1, -1, -1]]),
                [mismatch],
            )
        mock_log.warning.assert_called()


# ---------------------------------------------------------------------------
# calculate_contour_metrics: empty input
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_empty_sorted_contours_returns_empty_list():
    metrics = _calc(
        {"parents": [], "children": [], "nested_children": []},
        np.array([]), [],
    )
    assert metrics == []


# ---------------------------------------------------------------------------
# convert_metrics_to_real_world
# ---------------------------------------------------------------------------


@pytest.fixture
def pixel_metric():
    return {
        "parent": "parent 1",
        "scar": "parent 1",
        "centroid_x": 100.0,
        "centroid_y": 200.0,
        "technical_width": 50.0,
        "technical_length": 80.0,
        "max_length": 90.0,
        "max_width": 45.0,
        "perimeter": 260.0,
        "area": 4000.0,
        "convex_hull_area": 4200.0,
    }


@pytest.mark.unit
class TestConvertMetricsToRealWorld:

    def test_linear_fields_divided_by_pixels_per_mm(self, pixel_metric):
        [converted] = convert_metrics_to_real_world([pixel_metric], 10.0)
        assert converted["centroid_x"] == 10.0
        assert converted["centroid_y"] == 20.0
        assert converted["technical_width"] == 5.0
        assert converted["technical_length"] == 8.0
        assert converted["max_length"] == 9.0
        assert converted["perimeter"] == 26.0

    def test_area_fields_divided_by_square_of_factor(self, pixel_metric):
        [converted] = convert_metrics_to_real_world([pixel_metric], 10.0)
        # area: 4000 / 100 = 40
        assert converted["area"] == 40.0
        assert converted["convex_hull_area"] == 42.0

    def test_non_numeric_fields_preserved(self, pixel_metric):
        [converted] = convert_metrics_to_real_world([pixel_metric], 10.0)
        assert converted["parent"] == "parent 1"
        assert converted["scar"] == "parent 1"

    @pytest.mark.parametrize("invalid_factor", [0.0, -1.0, -10.5])
    def test_invalid_factor_returns_original(self, pixel_metric, invalid_factor):
        result = convert_metrics_to_real_world([pixel_metric], invalid_factor)
        # Function returns the input list unchanged (not a copy)
        assert result == [pixel_metric]

    def test_empty_metrics_yields_empty_list(self):
        assert convert_metrics_to_real_world([], 10.0) == []

    def test_missing_optional_fields_are_skipped(self):
        metric = {"parent": "parent 1", "scar": "parent 1", "centroid_x": 100.0}
        [converted] = convert_metrics_to_real_world([metric], 10.0)
        assert converted["centroid_x"] == 10.0
        assert "technical_width" not in converted

    def test_values_rounded_to_two_decimal_places(self):
        metric = {"parent": "p", "scar": "p", "centroid_x": 33.333}
        [converted] = convert_metrics_to_real_world([metric], 0.123456)
        # Exact expected: 33.333 / 0.123456 ≈ 269.987, rounded to 269.99
        assert converted["centroid_x"] == pytest.approx(269.99, abs=0.01)


# ---------------------------------------------------------------------------
# Integration: calculate + convert round-trip
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_calculate_then_convert_preserves_identifiers(
    sample_contours, sample_hierarchy
):
    sorted_contours = {
        "parents": [sample_contours[0]],
        "children": [sample_contours[1]],
        "nested_children": [],
    }

    metrics = calculate_contour_metrics(
        sorted_contours, sample_hierarchy, sample_contours, (200, 200)
    )
    parent_metrics = [m for m in metrics if "technical_width" in m]

    converted = convert_metrics_to_real_world(parent_metrics, pixels_per_mm=20.0)

    assert len(converted) == len(parent_metrics)
    for before, after in zip(parent_metrics, converted):
        assert after["parent"] == before["parent"]
        assert after["centroid_x"] == pytest.approx(
            round(before["centroid_x"] / 20.0, 2), abs=1e-6
        )
        assert after["area"] == pytest.approx(
            round(before["area"] / (20.0 ** 2), 2), abs=1e-6
        )
