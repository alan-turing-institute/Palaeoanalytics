"""Tests for parent-contour surface classification."""

from unittest.mock import patch

import pytest

from pylithics.image_processing.modules.surface_classification import (
    classify_parent_contours,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parent(name, length, width, area):
    return {
        "parent": name,
        "scar": name,
        "technical_length": length,
        "technical_width": width,
        "area": area,
        "surface_type": None,
    }


def _child(parent_name, scar_name, area):
    return {
        "parent": parent_name,
        "scar": scar_name,
        "width": 10.0,
        "height": 15.0,
        "area": area,
        "surface_type": None,
    }


# ---------------------------------------------------------------------------
# Core classification rules
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClassifyParentContours:
    """classify_parent_contours assigns surface_type based on geometry."""

    def test_single_parent_is_classified_dorsal(self):
        metrics = [_parent("parent 1", length=60.0, width=50.0, area=3000.0)]
        result = classify_parent_contours(metrics)
        assert result[0]["surface_type"] == "Dorsal"

    def test_two_similar_parents_become_dorsal_and_ventral(self):
        """The larger one is Dorsal; within-tolerance second parent is Ventral."""
        metrics = [
            _parent("parent 1", length=60.0, width=50.0, area=3000.0),
            _parent("parent 2", length=61.0, width=49.0, area=2990.0),
        ]
        result = classify_parent_contours(metrics, tolerance=0.1)
        by_name = {m["parent"]: m["surface_type"] for m in result}
        assert by_name == {"parent 1": "Dorsal", "parent 2": "Ventral"}

    def test_three_parents_resolve_to_dorsal_platform_lateral(self):
        metrics = [
            # Largest → Dorsal
            _parent("parent 1", length=100.0, width=80.0, area=8000.0),
            # Much smaller both dims → Platform
            _parent("parent 2", length=40.0, width=30.0, area=1200.0),
            # Similar length to Dorsal, different width → Lateral
            _parent("parent 3", length=95.0, width=45.0, area=4275.0),
        ]
        result = classify_parent_contours(metrics, tolerance=0.1)
        by_name = {m["parent"]: m["surface_type"] for m in result}
        assert by_name == {
            "parent 1": "Dorsal",
            "parent 2": "Platform",
            "parent 3": "Lateral",
        }

    def test_four_parents_produce_all_four_surfaces(self):
        metrics = [
            _parent("parent 1", length=100.0, width=80.0, area=8000.0),
            _parent("parent 2", length=98.0, width=78.0, area=7640.0),
            _parent("parent 3", length=35.0, width=25.0, area=875.0),
            _parent("parent 4", length=95.0, width=45.0, area=4275.0),
        ]
        result = classify_parent_contours(metrics, tolerance=0.1)
        by_name = {m["parent"]: m["surface_type"] for m in result}
        assert by_name == {
            "parent 1": "Dorsal",
            "parent 2": "Ventral",
            "parent 3": "Platform",
            "parent 4": "Lateral",
        }

    def test_unmatched_remaining_parents_become_unclassified(self):
        """
        With a dorsal and 2 mystery parents that fit no rule, they should end
        up with surface_type == "Unclassified".
        """
        metrics = [
            _parent("parent 1", length=100.0, width=80.0, area=8000.0),
            _parent("parent 2", length=10.0, width=200.0, area=2000.0),
            _parent("parent 3", length=15.0, width=210.0, area=3150.0),
        ]
        result = classify_parent_contours(metrics, tolerance=0.1)
        # parent 1 is dorsal; 2 and 3 don't match any pattern here.
        assert result[0]["surface_type"] == "Dorsal"
        unclassified = [m for m in result if m["surface_type"] == "Unclassified"]
        assert len(unclassified) >= 1


# ---------------------------------------------------------------------------
# Tolerance sensitivity (the audit flagged this as duplicated before)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToleranceSensitivity:
    """One parameterized test replaces three with identical setup."""

    @pytest.fixture
    def metrics_diff_5pct(self):
        # Dimensions differ by ~5% in both axes
        return [
            _parent("parent 1", length=100.0, width=50.0, area=5000.0),
            _parent("parent 2", length=105.0, width=52.0, area=5460.0),
        ]

    def test_tolerance_above_diff_yields_ventral(self, metrics_diff_5pct):
        result = classify_parent_contours(metrics_diff_5pct, tolerance=0.1)
        types = {m["surface_type"] for m in result}
        assert types == {"Dorsal", "Ventral"}

    def test_tolerance_below_diff_prevents_ventral_match(self, metrics_diff_5pct):
        result = classify_parent_contours(metrics_diff_5pct, tolerance=0.01)
        # With a tight tolerance, the 5% gap breaks the Ventral match, so no
        # parent ends up tagged Ventral.
        types = {m["surface_type"] for m in result}
        assert "Ventral" not in types


# ---------------------------------------------------------------------------
# Null / child-only inputs
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClassifyNullInputs:

    def test_empty_metrics_returns_empty_and_warns(self):
        with patch(
            "pylithics.image_processing.modules.surface_classification.logging"
        ) as mock_log:
            assert classify_parent_contours([]) == []
        mock_log.warning.assert_called()

    def test_only_child_metrics_produces_no_classification(self):
        metrics = [_child("parent 1", "scar 1", area=150.0)]
        with patch(
            "pylithics.image_processing.modules.surface_classification.logging"
        ) as mock_log:
            result = classify_parent_contours(metrics)
        assert result[0]["surface_type"] is None
        mock_log.warning.assert_called()

    def test_child_metrics_are_never_touched(self):
        metrics = [
            _parent("parent 1", length=60.0, width=50.0, area=3000.0),
            _child("parent 1", "scar 1", area=150.0),
        ]
        result = classify_parent_contours(metrics)
        parent, child = result
        assert parent["surface_type"] == "Dorsal"
        assert child["surface_type"] is None
        assert "technical_width" not in child  # child dict untouched


# ---------------------------------------------------------------------------
# Non-classification fields are preserved
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_existing_metric_fields_are_preserved():
    metrics = [
        {
            "parent": "parent 1", "scar": "parent 1",
            "technical_length": 60.0, "technical_width": 50.0, "area": 3000.0,
            "surface_type": None,
            "centroid_x": 42.0, "centroid_y": 42.0,  # unrelated fields
            "image_id": "artifact.png",
        }
    ]
    result = classify_parent_contours(metrics)
    m = result[0]
    assert m["surface_type"] == "Dorsal"
    assert m["centroid_x"] == 42.0
    assert m["centroid_y"] == 42.0
    assert m["image_id"] == "artifact.png"
