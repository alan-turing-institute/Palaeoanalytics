"""Tests for the arrow-integration pipeline."""

from unittest.mock import patch

import cv2
import numpy as np
import pytest

from pylithics.image_processing.modules.arrow_integration import (
    detect_arrows_independently,
    integrate_arrows,
    process_nested_arrows,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _poly(*pts):
    return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)


def _rect(x0, y0, x1, y1):
    return _poly([x0, y0], [x1, y0], [x1, y1], [x0, y1])


def _parent_metric(area, **overrides):
    base = {
        "parent": "parent 1",
        "scar": "parent 1",
        "surface_type": "Dorsal",
        "centroid_x": 100.0,
        "centroid_y": 100.0,
        "area": area,
    }
    base.update(overrides)
    return base


def _scar_metric(name, area, **overrides):
    base = {
        "parent": "parent 1",
        "scar": name,
        "surface_type": "Dorsal",
        "centroid_x": 50.0,
        "centroid_y": 50.0,
        "width": 20.0,
        "height": 20.0,
        "area": area,
    }
    base.update(overrides)
    return base


ARROW_RESULT = {
    "arrow_back": (30, 35),
    "arrow_tip": (40, 45),
    "angle_rad": 0.785,
    "angle_deg": 45.0,
    "compass_angle": 45.0,
}


# ---------------------------------------------------------------------------
# integrate_arrows: orchestration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIntegrateArrows:
    """integrate_arrows coordinates nested and independent detection."""

    def test_skips_independent_detection_when_all_scars_have_arrows(self):
        """
        If process_nested_arrows already sets has_arrow=True on every scar,
        detect_arrows_independently must NOT be called.
        """
        parent = _rect(10, 10, 90, 90)
        scar = _rect(30, 30, 50, 50)
        metrics = [
            _parent_metric(area=6400.0),
            _scar_metric("scar 1", area=400.0, has_arrow=True),
        ]
        sorted_contours = {
            "parents": [parent], "children": [scar], "nested_children": [],
        }

        def mark_arrows(*_args, **_kwargs):
            metrics[1]["has_arrow"] = True
            return metrics

        with patch(
            "pylithics.image_processing.modules.arrow_integration.process_nested_arrows",
            side_effect=mark_arrows,
        ) as mock_nested, patch(
            "pylithics.image_processing.modules.arrow_integration.detect_arrows_independently"
        ) as mock_independent:
            result = integrate_arrows(
                sorted_contours, np.array([[-1, -1, 1, -1], [-1, -1, -1, 0]]),
                [parent, scar], metrics,
                np.zeros((100, 100), dtype=np.uint8),
                image_dpi=300.0,
            )

        assert mock_nested.called
        assert not mock_independent.called
        assert result is metrics

    def test_triggers_independent_detection_when_scars_lack_arrows(self):
        """If nested detection leaves a scar without has_arrow, fall back."""
        parent = _rect(10, 10, 90, 90)
        scar = _rect(30, 30, 50, 50)
        metrics = [
            _parent_metric(area=6400.0),
            _scar_metric("scar 1", area=400.0, has_arrow=False),
        ]
        sorted_contours = {
            "parents": [parent], "children": [scar], "nested_children": [],
        }

        with patch(
            "pylithics.image_processing.modules.arrow_integration.process_nested_arrows",
            return_value=metrics,
        ), patch(
            "pylithics.image_processing.modules.arrow_integration.detect_arrows_independently",
            return_value=metrics,
        ) as mock_independent:
            integrate_arrows(
                sorted_contours, np.array([[-1, -1, 1, -1], [-1, -1, -1, 0]]),
                [parent, scar], metrics,
                np.zeros((100, 100), dtype=np.uint8),
                image_dpi=300.0,
            )

        assert mock_independent.called

    def test_empty_input_returns_empty_list(self):
        sorted_contours = {"parents": [], "children": [], "nested_children": []}
        result = integrate_arrows(
            sorted_contours, np.array([]), [], [],
            np.zeros((100, 100), dtype=np.uint8), image_dpi=300.0,
        )
        assert result == []


# ---------------------------------------------------------------------------
# process_nested_arrows: arrow detection on nested children
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProcessNestedArrows:

    def test_arrow_detection_populates_metric_fields(self):
        """
        A successful arrow detection on a nested child should populate
        `has_arrow=True` plus all arrow geometry fields on the parent scar metric.
        """
        parent = _rect(10, 10, 190, 190)
        scar = _rect(40, 40, 120, 120)
        nested = _rect(50, 50, 100, 100)

        metrics = [
            _parent_metric(area=cv2.contourArea(parent)),
            _scar_metric("scar 1", area=cv2.contourArea(scar)),
        ]
        sorted_contours = {
            "parents": [parent],
            "children": [scar],
            "nested_children": [nested],
        }
        hierarchy = np.array([
            [-1, -1, 1, -1],   # parent
            [-1, -1, 2, 0],    # scar (has nested child at 2)
            [-1, -1, -1, 1],   # nested child
        ])

        with patch(
            "pylithics.image_processing.modules.arrow_integration."
            "analyze_child_contour_for_arrow",
            return_value=ARROW_RESULT,
        ):
            result = process_nested_arrows(
                sorted_contours, hierarchy, [parent, scar, nested],
                metrics, np.zeros((200, 200), dtype=np.uint8),
                image_dpi=300.0,
            )

        scar_metric = result[1]
        assert scar_metric["has_arrow"] is True
        assert scar_metric["arrow_back"] == (30, 35)
        assert scar_metric["arrow_tip"] == (40, 45)
        assert scar_metric["arrow_angle"] == 45.0

    def test_no_nested_children_leaves_metrics_unchanged(self):
        parent = _rect(10, 10, 90, 90)
        metrics = [_parent_metric(area=6400.0)]
        sorted_contours = {
            "parents": [parent], "children": [], "nested_children": [],
        }

        result = process_nested_arrows(
            sorted_contours, np.array([[-1, -1, -1, -1]]),
            [parent], metrics,
            np.zeros((100, 100), dtype=np.uint8), image_dpi=300.0,
        )
        assert result == [_parent_metric(area=6400.0)]

    def test_failed_arrow_detection_leaves_has_arrow_unset(self):
        parent = _rect(10, 10, 190, 190)
        scar = _rect(40, 40, 120, 120)
        nested = _rect(50, 50, 100, 100)

        metrics = [
            _parent_metric(area=cv2.contourArea(parent)),
            _scar_metric("scar 1", area=cv2.contourArea(scar)),
        ]
        sorted_contours = {
            "parents": [parent],
            "children": [scar],
            "nested_children": [nested],
        }
        hierarchy = np.array([
            [-1, -1, 1, -1],
            [-1, -1, 2, 0],
            [-1, -1, -1, 1],
        ])

        with patch(
            "pylithics.image_processing.modules.arrow_integration."
            "analyze_child_contour_for_arrow",
            return_value=None,
        ):
            result = process_nested_arrows(
                sorted_contours, hierarchy, [parent, scar, nested],
                metrics, np.zeros((200, 200), dtype=np.uint8),
                image_dpi=300.0,
            )

        assert "has_arrow" not in result[1] or result[1].get("has_arrow") is False


# ---------------------------------------------------------------------------
# detect_arrows_independently: scar-assignment path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDetectArrowsIndependently:

    def test_empty_inputs_return_empty_metrics(self):
        result = detect_arrows_independently(
            [], [], np.zeros((100, 100), dtype=np.uint8), 300.0
        )
        assert result == []

    def test_returns_same_metrics_object_for_in_place_update(self):
        """
        detect_arrows_independently mutates in place; the same list is returned
        so callers don't need to rebind.
        """
        parent = _rect(10, 10, 90, 90)
        scar = _rect(30, 30, 50, 50)
        metrics = [
            _parent_metric(area=cv2.contourArea(parent)),
            _scar_metric("scar 1", area=cv2.contourArea(scar)),
        ]

        result = detect_arrows_independently(
            [parent, scar], metrics,
            np.zeros((100, 100), dtype=np.uint8), 300.0,
        )
        assert result is metrics
