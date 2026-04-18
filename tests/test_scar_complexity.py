"""
Tests for Scar Complexity Analysis Module
=========================================

Unit tests for scar_complexity.py focusing on border-sharing
detection and integration with the PyLithics pipeline.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from pylithics.image_processing.modules.scar_complexity import (
    analyze_scar_complexity,
    _create_polygon_from_contour,
    _create_fallback_complexity_results,
    _integrate_complexity_results
)


def _make_dorsal_parent():
    """Helper: create a dorsal parent metric."""
    return {
        "surface_type": "Dorsal",
        "surface_feature": "Dorsal",
        "parent": "parent 1",
        "scar": "parent 1",
        "contour": [
            [0, 0], [100, 0], [100, 100], [0, 100]
        ],
    }


def _make_scar(name, contour, parent="parent 1"):
    """Helper: create a scar metric on the dorsal surface."""
    return {
        "surface_type": "Dorsal",
        "surface_feature": name,
        "parent": parent,
        "scar": name,
        "contour": contour,
    }


class TestScarComplexityAnalysis:
    """Test scar complexity analysis functionality."""

    def test_no_dorsal_scars(self):
        """Test with no dorsal surface at all."""
        metrics = [
            {"surface_type": "Ventral", "surface_feature": "v",
             "parent": "p1", "scar": "p1",
             "contour": [[0, 0], [10, 0], [10, 10]]}
        ]
        result = analyze_scar_complexity(metrics, {})
        assert result == {}

    def test_single_scar(self):
        """Test with a single dorsal scar — needs at least 2."""
        metrics = [
            _make_dorsal_parent(),
            _make_scar("scar 1", [[5, 5], [15, 5], [15, 15], [5, 15]])
        ]
        result = analyze_scar_complexity(metrics, {})
        assert result == {"scar 1": 0}

    def test_isolated_scars(self):
        """Test scars that don't touch each other."""
        metrics = [
            _make_dorsal_parent(),
            _make_scar("scar 1", [[0, 0], [10, 0], [10, 10], [0, 10]]),
            _make_scar("scar 2", [[50, 50], [60, 50], [60, 60], [50, 60]]),
        ]
        result = analyze_scar_complexity(metrics, {})
        assert result["scar 1"] == 0
        assert result["scar 2"] == 0

    def test_adjacent_scars(self):
        """Test scars sharing a border."""
        metrics = [
            _make_dorsal_parent(),
            _make_scar("scar 1", [[0, 0], [10, 0], [10, 10], [0, 10]]),
            _make_scar("scar 2", [[10, 0], [20, 0], [20, 10], [10, 10]]),
        ]
        result = analyze_scar_complexity(
            metrics, {"scar_complexity": {"distance_threshold": 1.0}}
        )
        assert result["scar 1"] >= 1
        assert result["scar 2"] >= 1

    def test_multiple_adjacent(self):
        """Test one scar touching two others."""
        metrics = [
            _make_dorsal_parent(),
            _make_scar("scar 1", [[10, 0], [20, 0], [20, 10], [10, 10]]),
            _make_scar("scar 2", [[0, 0], [10, 0], [10, 10], [0, 10]]),
            _make_scar("scar 3", [[20, 0], [30, 0], [30, 10], [20, 10]]),
        ]
        result = analyze_scar_complexity(
            metrics, {"scar_complexity": {"distance_threshold": 1.0}}
        )
        assert result["scar 1"] >= 2

    def test_mixed_surfaces(self):
        """Only dorsal scars should be analysed."""
        metrics = [
            _make_dorsal_parent(),
            _make_scar("scar 1", [[0, 0], [10, 0], [10, 10], [0, 10]]),
            {"surface_type": "Ventral", "surface_feature": "scar 2",
             "parent": "parent 2", "scar": "scar 2",
             "contour": [[10, 0], [20, 0], [20, 10], [10, 10]]},
            _make_scar("scar 3", [[50, 50], [60, 50], [60, 60], [50, 60]]),
        ]
        result = analyze_scar_complexity(metrics, {})
        assert "scar 1" in result
        assert "scar 3" in result
        assert "scar 2" not in result

    def test_error_handling(self):
        """Invalid contour data should be handled gracefully."""
        metrics = [
            _make_dorsal_parent(),
            _make_scar("scar 1", None),
            _make_scar("scar 2", [[0, 0], [10, 0], [10, 10], [0, 10]]),
        ]
        result = analyze_scar_complexity(metrics, {})
        assert isinstance(result, dict)

    def test_disabled(self):
        """Analysis should be skipped when disabled."""
        metrics = [
            _make_dorsal_parent(),
            _make_scar("scar 1", [[0, 0], [10, 0], [10, 10], [0, 10]]),
        ]
        config = {"scar_complexity": {"enabled": False}}
        result = analyze_scar_complexity(metrics, config)
        assert result == {}


class TestPolygonCreation:
    """Test polygon creation utility functions."""

    def test_valid_contour(self):
        contour = [[0, 0], [10, 0], [10, 10], [0, 10]]
        polygon = _create_polygon_from_contour(contour)
        assert polygon is not None
        assert polygon.is_valid

    def test_numpy_array(self):
        contour = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        polygon = _create_polygon_from_contour(contour)
        assert polygon is not None
        assert polygon.is_valid

    def test_too_few_points(self):
        assert _create_polygon_from_contour([[0, 0], [10, 0]]) is None

    def test_none(self):
        assert _create_polygon_from_contour(None) is None

    def test_empty(self):
        assert _create_polygon_from_contour([]) is None


class TestFallbackResults:
    """Test fallback result creation."""

    def test_creates_zero_complexity(self):
        metrics = [
            {"surface_type": "Dorsal", "surface_feature": "scar 1"},
            {"surface_type": "Dorsal", "surface_feature": "scar 2"},
            {"surface_type": "Ventral", "surface_feature": "ventral"},
        ]
        result = _create_fallback_complexity_results(metrics)
        assert result == {"scar 1": 0, "scar 2": 0}

    def test_no_dorsal(self):
        metrics = [
            {"surface_type": "Ventral", "surface_feature": "v"},
        ]
        result = _create_fallback_complexity_results(metrics)
        assert result == {}


class TestIntegration:
    """Test integration with metrics."""

    def test_integrate_results(self):
        metrics = [
            {"surface_type": "Dorsal", "surface_feature": "scar 1"},
            {"surface_type": "Dorsal", "surface_feature": "scar 2"},
            {"surface_type": "Ventral", "surface_feature": "ventral"},
        ]
        _integrate_complexity_results(
            metrics, {"scar 1": 2, "scar 2": 0}
        )
        assert metrics[0]["scar_complexity"] == 2
        assert metrics[1]["scar_complexity"] == 0
        assert metrics[2]["scar_complexity"] is None


@pytest.mark.integration
class TestScarComplexityIntegration:
    """Integration tests with pipeline data format."""

    def test_pipeline_format_isolated_scars(self):
        """Test with typical pipeline output structure."""
        metrics = [
            _make_dorsal_parent(),
            _make_scar("scar 1", [[0, 0], [10, 0], [10, 10], [0, 10]]),
            _make_scar("scar 2", [[50, 50], [60, 50], [60, 60], [50, 60]]),
        ]
        result = analyze_scar_complexity(metrics, {})
        assert len(result) == 2
        assert result["scar 1"] == 0
        assert result["scar 2"] == 0


if __name__ == "__main__":
    pytest.main([__file__])
