"""
Tests for Scar Complexity Analysis Module
=========================================

Unit tests for scar_complexity.py focusing on border-sharing detection
and integration with the PyLithics pipeline.
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


class TestScarComplexityAnalysis:
    """Test scar complexity analysis functionality."""

    def test_analyze_scar_complexity_no_scars(self):
        """Test behavior with no dorsal scars."""
        metrics = [
            {"surface_type": "Ventral", "surface_feature": "ventral surface", "contour": [[0, 0], [10, 0], [10, 10]]},
            {"surface_type": "Platform", "surface_feature": "platform", "contour": [[0, 0], [5, 0], [5, 5]]}
        ]
        config = {"enabled": True}
        
        result = analyze_scar_complexity(metrics, config)
        
        assert result == {}

    def test_analyze_scar_complexity_single_scar(self):
        """Test behavior with single dorsal scar."""
        metrics = [
            {"surface_type": "Dorsal", "surface_feature": "scar 1", 
             "contour": [[0, 0], [10, 0], [10, 10], [0, 10]]}
        ]
        config = {"enabled": True}
        
        result = analyze_scar_complexity(metrics, config)
        
        assert result == {"scar 1": 0}

    def test_analyze_scar_complexity_isolated_scars(self):
        """Test behavior with multiple isolated scars."""
        metrics = [
            {"surface_type": "Dorsal", "surface_feature": "scar 1", 
             "contour": [[0, 0], [10, 0], [10, 10], [0, 10]]},
            {"surface_type": "Dorsal", "surface_feature": "scar 2", 
             "contour": [[20, 20], [30, 20], [30, 30], [20, 30]]}
        ]
        config = {"enabled": True}
        
        result = analyze_scar_complexity(metrics, config)
        
        # Neither scar should touch the other
        assert result == {"scar 1": 0, "scar 2": 0}

    def test_analyze_scar_complexity_adjacent_scars(self):
        """Test behavior with adjacent scars that share borders."""
        metrics = [
            {"surface_type": "Dorsal", "surface_feature": "scar 1", 
             "contour": [[0, 0], [10, 0], [10, 10], [0, 10]]},
            {"surface_type": "Dorsal", "surface_feature": "scar 2", 
             "contour": [[10, 0], [20, 0], [20, 10], [10, 10]]}  # Shares right edge with scar 1
        ]
        config = {"enabled": True}
        
        with patch('pylithics.image_processing.modules.scar_complexity._create_polygon_from_contour') as mock_create_polygon:
            # Mock polygon creation to return touching polygons
            mock_poly1 = MagicMock()
            mock_poly2 = MagicMock()
            mock_poly1.touches.return_value = True
            mock_poly2.touches.return_value = True
            mock_create_polygon.side_effect = [mock_poly1, mock_poly2, mock_poly2, mock_poly1]
            
            result = analyze_scar_complexity(metrics, config)
            
            # Each scar should have complexity of 1 (touches 1 other scar)
            assert result == {"scar 1": 1, "scar 2": 1}

    def test_analyze_scar_complexity_multiple_adjacent(self):
        """Test behavior with one scar adjacent to multiple others."""
        metrics = [
            {"surface_type": "Dorsal", "surface_feature": "scar 1", 
             "contour": [[10, 10], [20, 10], [20, 20], [10, 20]]},  # Center scar
            {"surface_type": "Dorsal", "surface_feature": "scar 2", 
             "contour": [[0, 10], [10, 10], [10, 20], [0, 20]]},   # Left adjacent
            {"surface_type": "Dorsal", "surface_feature": "scar 3", 
             "contour": [[20, 10], [30, 10], [30, 20], [20, 20]]}  # Right adjacent
        ]
        config = {"enabled": True}
        
        with patch('pylithics.image_processing.modules.scar_complexity._create_polygon_from_contour') as mock_create_polygon:
            # Mock polygon creation
            mock_poly1 = MagicMock()  # scar 1 (center)
            mock_poly2 = MagicMock()  # scar 2 (left)
            mock_poly3 = MagicMock()  # scar 3 (right)
            
            # scar 1 touches both scar 2 and scar 3
            mock_poly1.touches.side_effect = lambda other: other in [mock_poly2, mock_poly3]
            # scar 2 only touches scar 1
            mock_poly2.touches.side_effect = lambda other: other == mock_poly1
            # scar 3 only touches scar 1
            mock_poly3.touches.side_effect = lambda other: other == mock_poly1
            
            mock_create_polygon.side_effect = [mock_poly1, mock_poly2, mock_poly3,  # First iteration
                                             mock_poly2, mock_poly1, mock_poly3,    # Second iteration
                                             mock_poly3, mock_poly1, mock_poly2]    # Third iteration
            
            result = analyze_scar_complexity(metrics, config)
            
            # scar 1 should have complexity 2, others should have complexity 1
            assert result == {"scar 1": 2, "scar 2": 1, "scar 3": 1}

    def test_analyze_scar_complexity_mixed_surfaces(self):
        """Test that only dorsal scars are analyzed."""
        metrics = [
            {"surface_type": "Dorsal", "surface_feature": "scar 1", 
             "contour": [[0, 0], [10, 0], [10, 10], [0, 10]]},
            {"surface_type": "Ventral", "surface_feature": "scar 2", 
             "contour": [[10, 0], [20, 0], [20, 10], [10, 10]]},
            {"surface_type": "Dorsal", "surface_feature": "scar 3", 
             "contour": [[20, 20], [30, 20], [30, 30], [20, 30]]}
        ]
        config = {"enabled": True}
        
        result = analyze_scar_complexity(metrics, config)
        
        # Only dorsal scars should be included
        assert len(result) == 2
        assert "scar 1" in result
        assert "scar 3" in result
        assert "scar 2" not in result

    def test_analyze_scar_complexity_error_handling(self):
        """Test error handling with invalid contour data."""
        metrics = [
            {"surface_type": "Dorsal", "surface_feature": "scar 1", 
             "contour": None},  # Invalid contour
            {"surface_type": "Dorsal", "surface_feature": "scar 2", 
             "contour": [[0, 0], [10, 0], [10, 10], [0, 10]]}
        ]
        config = {"enabled": True}
        
        result = analyze_scar_complexity(metrics, config)
        
        # Should handle errors gracefully and return fallback values
        assert isinstance(result, dict)
        assert len(result) >= 1  # At least the valid scar should be processed


class TestPolygonCreation:
    """Test polygon creation utility functions."""

    def test_create_polygon_from_contour_valid(self):
        """Test polygon creation with valid contour."""
        contour = [[0, 0], [10, 0], [10, 10], [0, 10]]
        
        polygon = _create_polygon_from_contour(contour)
        
        assert polygon is not None
        assert polygon.is_valid

    def test_create_polygon_from_contour_numpy_array(self):
        """Test polygon creation with numpy array contour."""
        contour = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        
        polygon = _create_polygon_from_contour(contour)
        
        assert polygon is not None
        assert polygon.is_valid

    def test_create_polygon_from_contour_invalid_shape(self):
        """Test polygon creation with invalid contour shape."""
        contour = [[0, 0], [10, 0]]  # Only 2 points
        
        polygon = _create_polygon_from_contour(contour)
        
        assert polygon is None

    def test_create_polygon_from_contour_none(self):
        """Test polygon creation with None contour."""
        polygon = _create_polygon_from_contour(None)
        
        assert polygon is None

    def test_create_polygon_from_contour_empty(self):
        """Test polygon creation with empty contour."""
        polygon = _create_polygon_from_contour([])
        
        assert polygon is None


class TestFallbackResults:
    """Test fallback result creation."""

    def test_create_fallback_complexity_results(self):
        """Test fallback results creation."""
        metrics = [
            {"surface_type": "Dorsal", "surface_feature": "scar 1"},
            {"surface_type": "Dorsal", "surface_feature": "scar 2"},
            {"surface_type": "Ventral", "surface_feature": "ventral"}
        ]
        
        result = _create_fallback_complexity_results(metrics)
        
        # Should only include dorsal scars with zero complexity
        assert result == {"scar 1": 0, "scar 2": 0}

    def test_create_fallback_complexity_results_no_dorsal(self):
        """Test fallback results with no dorsal scars."""
        metrics = [
            {"surface_type": "Ventral", "surface_feature": "ventral"},
            {"surface_type": "Platform", "surface_feature": "platform"}
        ]
        
        result = _create_fallback_complexity_results(metrics)
        
        assert result == {}


class TestIntegration:
    """Test integration with metrics."""

    def test_integrate_complexity_results(self):
        """Test integration of complexity results into metrics."""
        metrics = [
            {"surface_type": "Dorsal", "surface_feature": "scar 1"},
            {"surface_type": "Dorsal", "surface_feature": "scar 2"},
            {"surface_type": "Ventral", "surface_feature": "ventral"}
        ]
        complexity_results = {"scar 1": 2, "scar 2": 0}
        
        _integrate_complexity_results(metrics, complexity_results)
        
        # Check that complexity values were added correctly
        assert metrics[0]["scar_complexity"] == 2
        assert metrics[1]["scar_complexity"] == 0
        assert metrics[2]["scar_complexity"] is None  # Non-dorsal should be None


@pytest.mark.integration
class TestScarComplexityIntegration:
    """Integration tests with full pipeline components."""

    def test_integration_with_surface_classification_output(self):
        """Test integration with typical surface classification output."""
        # Typical metrics structure after surface classification - only scars
        metrics = [
            {
                "surface_type": "Dorsal", 
                "surface_feature": "scar 1",
                "contour": [[0, 0], [10, 0], [10, 10], [0, 10]],  # Square scar
                "area": 100,
                "parent": "Dorsal",
                "scar": "scar 1"
            },
            {
                "surface_type": "Dorsal", 
                "surface_feature": "scar 2", 
                "contour": [[20, 20], [30, 20], [30, 30], [20, 30]],  # Isolated square scar
                "area": 100,
                "parent": "Dorsal",
                "scar": "scar 2"
            }
        ]
        config = {"enabled": True}
        
        result = analyze_scar_complexity(metrics, config)
        
        # Both scars should be isolated (no touching)
        assert len(result) == 2
        assert result.get("scar 1") == 0  # Isolated
        assert result.get("scar 2") == 0  # Isolated


if __name__ == "__main__":
    pytest.main([__file__])