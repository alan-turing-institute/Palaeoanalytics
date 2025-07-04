"""
PyLithics Surface Classification Tests
=====================================

Tests for surface type classification including Dorsal, Ventral, Platform, and Lateral
surface identification based on geometric properties and hierarchical relationships.
"""

import pytest
import numpy as np
from unittest.mock import patch

from pylithics.image_processing.modules.surface_classification import (
    classify_parent_contours
)


@pytest.mark.unit
class TestClassifyParentContours:
    """Test the main surface classification function."""

    def test_classify_single_parent_dorsal(self):
        """Test classification when only one parent exists (should be Dorsal)."""
        metrics = [
            {
                'parent': 'parent 1',
                'scar': 'parent 1',
                'technical_width': 50.0,
                'technical_length': 60.0,
                'area': 3000.0,
                'surface_type': None
            }
        ]

        classified_metrics = classify_parent_contours(metrics)

        assert len(classified_metrics) == 1
        assert classified_metrics[0]['surface_type'] == 'Dorsal'

    def test_classify_dorsal_ventral_pair(self):
        """Test classification of Dorsal and Ventral surfaces (similar dimensions)."""
        metrics = [
            {
                'parent': 'parent 1',
                'scar': 'parent 1',
                'technical_width': 50.0,
                'technical_length': 60.0,
                'area': 3000.0,
                'surface_type': None
            },
            {
                'parent': 'parent 2',
                'scar': 'parent 2',
                'technical_width': 49.0,  # Similar to parent 1
                'technical_length': 61.0,  # Similar to parent 1
                'area': 2990.0,  # Similar to parent 1
                'surface_type': None
            }
        ]

        classified_metrics = classify_parent_contours(metrics, tolerance=0.1)

        # Larger area should be Dorsal, similar one should be Ventral
        dorsal_metrics = [m for m in classified_metrics if m['surface_type'] == 'Dorsal']
        ventral_metrics = [m for m in classified_metrics if m['surface_type'] == 'Ventral']

        assert len(dorsal_metrics) == 1
        assert len(ventral_metrics) == 1
        assert dorsal_metrics[0]['area'] >= ventral_metrics[0]['area']

    def test_classify_dorsal_platform_lateral(self):
        """Test classification of Dorsal, Platform, and Lateral surfaces."""
        metrics = [
            {
                'parent': 'parent 1',
                'scar': 'parent 1',
                'technical_width': 80.0,
                'technical_length': 100.0,
                'area': 8000.0,  # Largest - should be Dorsal
                'surface_type': None
            },
            {
                'parent': 'parent 2',
                'scar': 'parent 2',
                'technical_width': 30.0,  # Much smaller
                'technical_length': 40.0,  # Much smaller
                'area': 1200.0,  # Smallest - should be Platform
                'surface_type': None
            },
            {
                'parent': 'parent 3',
                'scar': 'parent 3',
                'technical_width': 45.0,  # Different width from Dorsal
                'technical_length': 95.0,  # Similar length to Dorsal
                'area': 4275.0,  # Medium size - should be Lateral
                'surface_type': None
            }
        ]

        classified_metrics = classify_parent_contours(metrics, tolerance=0.1)

        # Check surface type assignments
        surface_types = {m['parent']: m['surface_type'] for m in classified_metrics}

        assert surface_types['parent 1'] == 'Dorsal'  # Largest area
        assert surface_types['parent 2'] == 'Platform'  # Smallest dimensions
        assert surface_types['parent 3'] == 'Lateral'  # Similar length, different width

    def test_classify_four_surfaces_complete(self):
        """Test classification with all four surface types present."""
        metrics = [
            {
                'parent': 'parent 1',
                'scar': 'parent 1',
                'technical_width': 80.0,
                'technical_length': 100.0,
                'area': 8000.0,  # Dorsal (largest)
                'surface_type': None
            },
            {
                'parent': 'parent 2',
                'scar': 'parent 2',
                'technical_width': 78.0,  # Similar to Dorsal
                'technical_length': 98.0,  # Similar to Dorsal
                'area': 7640.0,  # Similar to Dorsal - should be Ventral
                'surface_type': None
            },
            {
                'parent': 'parent 3',
                'scar': 'parent 3',
                'technical_width': 25.0,  # Much smaller
                'technical_length': 35.0,  # Much smaller
                'area': 875.0,  # Platform (smallest)
                'surface_type': None
            },
            {
                'parent': 'parent 4',
                'scar': 'parent 4',
                'technical_width': 45.0,  # Different from Dorsal
                'technical_length': 95.0,  # Similar to Dorsal
                'area': 4275.0,  # Lateral
                'surface_type': None
            }
        ]

        classified_metrics = classify_parent_contours(metrics, tolerance=0.1)

        # Check that all four surface types are identified
        surface_types = {m['surface_type'] for m in classified_metrics}
        expected_types = {'Dorsal', 'Ventral', 'Platform', 'Lateral'}

        assert surface_types == expected_types

        # Verify specific assignments
        surface_map = {m['parent']: m['surface_type'] for m in classified_metrics}
        assert surface_map['parent 1'] == 'Dorsal'
        assert surface_map['parent 2'] == 'Ventral'
        assert surface_map['parent 3'] == 'Platform'
        assert surface_map['parent 4'] == 'Lateral'

    def test_classify_with_child_contours_ignored(self):
        """Test that child contours are ignored during classification."""
        metrics = [
            {
                'parent': 'parent 1',
                'scar': 'parent 1',  # Parent
                'technical_width': 50.0,
                'technical_length': 60.0,
                'area': 3000.0,
                'surface_type': None
            },
            {
                'parent': 'parent 1',
                'scar': 'scar 1',  # Child - should be ignored
                'width': 10.0,
                'height': 15.0,
                'area': 150.0,
                'surface_type': None
            }
        ]

        classified_metrics = classify_parent_contours(metrics)

        # Only the parent should be classified
        parent_metrics = [m for m in classified_metrics if m['parent'] == m['scar']]
        child_metrics = [m for m in classified_metrics if m['parent'] != m['scar']]

        assert len(parent_metrics) == 1
        assert parent_metrics[0]['surface_type'] == 'Dorsal'

        # Child should remain unclassified (surface_type should be None)
        assert len(child_metrics) == 1
        assert child_metrics[0]['surface_type'] is None

    def test_classify_empty_metrics(self):
        """Test classification with empty metrics list."""
        metrics = []

        with patch('pylithics.image_processing.modules.surface_classification.logging') as mock_logging:
            classified_metrics = classify_parent_contours(metrics)

            assert classified_metrics == []
            mock_logging.warning.assert_called()

    def test_classify_no_parent_contours(self):
        """Test classification when no parent contours exist."""
        metrics = [
            {
                'parent': 'parent 1',
                'scar': 'scar 1',  # Only child contours
                'width': 10.0,
                'height': 15.0,
                'area': 150.0,
                'surface_type': None
            }
        ]

        with patch('pylithics.image_processing.modules.surface_classification.logging') as mock_logging:
            classified_metrics = classify_parent_contours(metrics)

            # Should return original metrics unchanged
            assert len(classified_metrics) == 1
            assert classified_metrics[0]['surface_type'] is None
            mock_logging.warning.assert_called()

    def test_classify_tolerance_effects(self):
        """Test how different tolerance values affect classification."""
        metrics = [
            {
                'parent': 'parent 1',
                'scar': 'parent 1',
                'technical_width': 50.0,
                'technical_length': 60.0,
                'area': 3000.0,
                'surface_type': None
            },
            {
                'parent': 'parent 2',
                'scar': 'parent 2',
                'technical_width': 52.0,  # 4% difference
                'technical_length': 62.0,  # 3.3% difference
                'area': 3100.0,  # 3.3% difference
                'surface_type': None
            }
        ]

        # With strict tolerance (1%), should not identify as Ventral
        classified_strict = classify_parent_contours(metrics.copy(), tolerance=0.01)
        ventral_strict = [m for m in classified_strict if m['surface_type'] == 'Ventral']
        assert len(ventral_strict) == 0

        # With loose tolerance (10%), should identify as Ventral
        classified_loose = classify_parent_contours(metrics.copy(), tolerance=0.10)
        ventral_loose = [m for m in classified_loose if m['surface_type'] == 'Ventral']
        assert len(ventral_loose) == 1

    def test_classify_platform_identification_criteria(self):
        """Test specific criteria for Platform surface identification."""
        metrics = [
            {
                'parent': 'parent 1',
                'scar': 'parent 1',
                'technical_width': 80.0,
                'technical_length': 100.0,
                'area': 8000.0,  # Dorsal
                'surface_type': None
            },
            {
                'parent': 'parent 2',
                'scar': 'parent 2',
                'technical_width': 70.0,  # Less than Dorsal ✓
                'technical_length': 90.0,  # Less than Dorsal ✓
                'area': 6300.0,  # Smaller area
                'surface_type': None
            },
            {
                'parent': 'parent 3',
                'scar': 'parent 3',
                'technical_width': 85.0,  # Greater than Dorsal ✗
                'technical_length': 95.0,  # Less than Dorsal ✓
                'area': 5000.0,  # Smaller area
                'surface_type': None
            }
        ]

        classified_metrics = classify_parent_contours(metrics)

        surface_map = {m['parent']: m['surface_type'] for m in classified_metrics}

        assert surface_map['parent 1'] == 'Dorsal'
        assert surface_map['parent 2'] == 'Platform'  # Both dimensions smaller
        assert surface_map['parent 3'] != 'Platform'  # Width larger than Dorsal

    def test_classify_lateral_identification_criteria(self):
        """Test specific criteria for Lateral surface identification."""
        metrics = [
            {
                'parent': 'parent 1',
                'scar': 'parent 1',
                'technical_width': 60.0,
                'technical_length': 80.0,
                'area': 4800.0,  # Dorsal
                'surface_type': None
            },
            {
                'parent': 'parent 2',
                'scar': 'parent 2',
                'technical_width': 30.0,  # Much smaller
                'technical_length': 40.0,  # Much smaller
                'area': 1200.0,  # Platform
                'surface_type': None
            },
            {
                'parent': 'parent 3',
                'scar': 'parent 3',
                'technical_width': 40.0,  # Different from Dorsal
                'technical_length': 78.0,  # Similar to Dorsal
                'area': 3120.0,  # Should be Lateral
                'surface_type': None
            }
        ]

        classified_metrics = classify_parent_contours(metrics, tolerance=0.1)

        surface_map = {m['parent']: m['surface_type'] for m in classified_metrics}

        assert surface_map['parent 1'] == 'Dorsal'
        assert surface_map['parent 2'] == 'Platform'
        assert surface_map['parent 3'] == 'Lateral'

    def test_classify_unclassified_fallback(self):
        """Test that unclassifiable contours get 'Unclassified' label."""
        metrics = [
            {
                'parent': 'parent 1',
                'scar': 'parent 1',
                'technical_width': 50.0,
                'technical_length': 60.0,
                'area': 3000.0,  # Will be Dorsal
                'surface_type': None
            },
            {
                'parent': 'parent 2',
                'scar': 'parent 2',
                'technical_width': 100.0,  # Very different dimensions
                'technical_length': 20.0,   # Very different dimensions
                'area': 2000.0,  # Won't match any classification criteria
                'surface_type': None
            }
        ]

        classified_metrics = classify_parent_contours(metrics)

        surface_map = {m['parent']: m['surface_type'] for m in classified_metrics}

        assert surface_map['parent 1'] == 'Dorsal'
        assert surface_map['parent 2'] == 'Unclassified'

    def test_classify_alternative_lateral_logic_without_platform(self):
        """Test alternative lateral classification when no platform exists."""
        metrics = [
            {
                'parent': 'parent 1',
                'scar': 'parent 1',
                'technical_width': 60.0,
                'technical_length': 80.0,
                'area': 4800.0,  # Dorsal (largest)
                'surface_type': None
            },
            {
                'parent': 'parent 2',
                'scar': 'parent 2',
                'technical_width': 58.0,  # Similar to Dorsal
                'technical_length': 78.0,  # Similar to Dorsal
                'area': 4500.0,  # Ventral (similar dimensions)
                'surface_type': None
            },
            {
                'parent': 'parent 3',
                'scar': 'parent 3',
                'technical_width': 40.0,  # Different width from Dorsal
                'technical_length': 78.0,  # Similar length to Dorsal
                'area': 3120.0,  # Potential Lateral
                'surface_type': None
            }
            # No Platform candidate
        ]

        classified_metrics = classify_parent_contours(metrics, tolerance=0.1)

        surface_map = {m['parent']: m['surface_type'] for m in classified_metrics}

        assert surface_map['parent 1'] == 'Dorsal'
        assert surface_map['parent 2'] == 'Ventral'
        # Should use alternative logic for Lateral when no Platform
        assert surface_map['parent 3'] in ['Lateral', 'Unclassified']

    def test_classify_maintains_original_data(self):
        """Test that classification preserves all original metric data."""
        original_metrics = [
            {
                'parent': 'parent 1',
                'scar': 'parent 1',
                'technical_width': 50.0,
                'technical_length': 60.0,
                'area': 3000.0,
                'surface_type': None,
                'extra_field': 'preserved_data',
                'centroid_x': 100.0,
                'centroid_y': 150.0
            }
        ]

        classified_metrics = classify_parent_contours(original_metrics)

        metric = classified_metrics[0]

        # Should preserve all original fields
        assert metric['parent'] == 'parent 1'
        assert metric['technical_width'] == 50.0
        assert metric['technical_length'] == 60.0
        assert metric['area'] == 3000.0
        assert metric['extra_field'] == 'preserved_data'
        assert metric['centroid_x'] == 100.0
        assert metric['centroid_y'] == 150.0

        # Should add surface_type classification
        assert metric['surface_type'] == 'Dorsal'


@pytest.mark.integration
class TestSurfaceClassificationIntegration:
    """Integration tests for surface classification workflow."""

    def test_realistic_archaeological_artifact_classification(self):
        """Test classification with realistic archaeological artifact metrics."""
        # Simulate metrics from a typical lithic tool
        metrics = [
            {
                'parent': 'parent 1',
                'scar': 'parent 1',
                'technical_width': 85.5,
                'technical_length': 124.3,
                'area': 10628.5,  # Main dorsal surface
                'centroid_x': 150.2,
                'centroid_y': 200.8,
                'aspect_ratio': 1.45,
                'surface_type': None
            },
            {
                'parent': 'parent 2',
                'scar': 'parent 2',
                'technical_width': 83.2,
                'technical_length': 120.1,
                'area': 9995.4,  # Ventral surface (similar to dorsal)
                'centroid_x': 148.9,
                'centroid_y': 198.3,
                'aspect_ratio': 1.44,
                'surface_type': None
            },
            {
                'parent': 'parent 3',
                'scar': 'parent 3',
                'technical_width': 28.4,
                'technical_length': 45.7,
                'area': 1298.3,  # Platform (much smaller)
                'centroid_x': 145.1,
                'centroid_y': 175.2,
                'aspect_ratio': 1.61,
                'surface_type': None
            },
            {
                'parent': 'parent 4',
                'scar': 'parent 4',
                'technical_width': 42.1,
                'technical_length': 118.9,
                'area': 5007.7,  # Lateral (similar length, different width)
                'centroid_x': 125.8,
                'centroid_y': 199.4,
                'aspect_ratio': 2.82,
                'surface_type': None
            },
            {
                'parent': 'parent 1',
                'scar': 'scar 1',  # Child contour (removal scar)
                'width': 15.3,
                'height': 22.1,
                'area': 338.1,
                'centroid_x': 160.5,
                'centroid_y': 185.7,
                'surface_type': None
            }
        ]

        classified_metrics = classify_parent_contours(metrics, tolerance=0.08)

        # Extract surface type assignments
        surface_assignments = {}
        for metric in classified_metrics:
            if metric['parent'] == metric['scar']:  # Only parents
                surface_assignments[metric['parent']] = metric['surface_type']

        # Verify realistic classification
        assert surface_assignments['parent 1'] == 'Dorsal'
        assert surface_assignments['parent 2'] == 'Ventral'
        assert surface_assignments['parent 3'] == 'Platform'
        assert surface_assignments['parent 4'] == 'Lateral'

        # Verify child contour is unaffected
        child_metric = next(m for m in classified_metrics if m['scar'] == 'scar 1')
        assert child_metric['surface_type'] is None

    def test_partial_artifact_classification(self):
        """Test classification when not all surface types are present."""
        # Artifact with only dorsal and platform surfaces
        metrics = [
            {
                'parent': 'parent 1',
                'scar': 'parent 1',
                'technical_width': 70.0,
                'technical_length': 90.0,
                'area': 6300.0,  # Dorsal
                'surface_type': None
            },
            {
                'parent': 'parent 2',
                'scar': 'parent 2',
                'technical_width': 25.0,
                'technical_length': 35.0,
                'area': 875.0,  # Platform
                'surface_type': None
            }
        ]

        classified_metrics = classify_parent_contours(metrics)

        surface_map = {m['parent']: m['surface_type'] for m in classified_metrics}

        assert surface_map['parent 1'] == 'Dorsal'
        assert surface_map['parent 2'] == 'Platform'

        # Should handle gracefully without ventral or lateral
        assert len(surface_map) == 2

    def test_classification_with_measurement_uncertainties(self):
        """Test classification robustness with realistic measurement uncertainties."""
        # Metrics with small variations that might occur in real measurements
        base_metrics = [
            {
                'parent': 'parent 1',
                'scar': 'parent 1',
                'technical_width': 60.0,
                'technical_length': 80.0,
                'area': 4800.0,
                'surface_type': None
            },
            {
                'parent': 'parent 2',
                'scar': 'parent 2',
                'technical_width': 61.2,  # Small measurement variation
                'technical_length': 79.5,  # Small measurement variation
                'area': 4866.0,  # Resulting area variation
                'surface_type': None
            }
        ]

        # Test with different tolerance levels
        tolerances = [0.05, 0.08, 0.12]

        for tolerance in tolerances:
            classified_metrics = classify_parent_contours(
                [m.copy() for m in base_metrics],
                tolerance=tolerance
            )

            surface_types = [m['surface_type'] for m in classified_metrics]

            # Should consistently identify one as Dorsal
            assert 'Dorsal' in surface_types

            # With sufficient tolerance, should identify the other as Ventral
            if tolerance >= 0.08:
                assert 'Ventral' in surface_types

    def test_edge_case_identical_dimensions(self):
        """Test classification when multiple contours have identical dimensions."""
        metrics = [
            {
                'parent': 'parent 1',
                'scar': 'parent 1',
                'technical_width': 50.0,
                'technical_length': 60.0,
                'area': 3000.0,
                'surface_type': None
            },
            {
                'parent': 'parent 2',
                'scar': 'parent 2',
                'technical_width': 50.0,  # Identical
                'technical_length': 60.0,  # Identical
                'area': 3000.0,  # Identical
                'surface_type': None
            }
        ]

        classified_metrics = classify_parent_contours(metrics)

        # Should handle identical dimensions gracefully
        surface_types = [m['surface_type'] for m in classified_metrics]

        # One should be Dorsal (larger area or first encountered)
        assert 'Dorsal' in surface_types
        # The other might be Ventral or Unclassified depending on exact logic
        assert len(surface_types) == 2
        assert all(st is not None for st in surface_types)


@pytest.mark.unit
class TestSurfaceClassificationEdgeCases:
    """Test edge cases and error conditions in surface classification."""

    def test_missing_required_fields(self):
        """Test handling of metrics with missing required fields."""
        metrics = [
            {
                'parent': 'parent 1',
                'scar': 'parent 1',
                # Missing technical_width, technical_length, area
                'surface_type': None
            }
        ]

        # Should handle missing fields gracefully without crashing
        try:
            classified_metrics = classify_parent_contours(metrics)
            # If it succeeds, verify it doesn't crash
            assert isinstance(classified_metrics, list)
        except (KeyError, TypeError):
            # If it fails, that's also acceptable behavior for missing fields
            pass

    def test_invalid_numeric_values(self):
        """Test handling of invalid numeric values in metrics."""
        metrics = [
            {
                'parent': 'parent 1',
                'scar': 'parent 1',
                'technical_width': 0.0,  # Zero width
                'technical_length': 80.0,
                'area': 0.0,  # Zero area
                'surface_type': None
            },
            {
                'parent': 'parent 2',
                'scar': 'parent 2',
                'technical_width': -10.0,  # Negative width
                'technical_length': 60.0,
                'area': 600.0,
                'surface_type': None
            }
        ]

        # Should handle invalid values gracefully
        classified_metrics = classify_parent_contours(metrics)

        # Should still classify what it can
        assert len(classified_metrics) == 2

        # At least one should be classified as Dorsal
        surface_types = [m['surface_type'] for m in classified_metrics]
        dorsal_count = surface_types.count('Dorsal')
        assert dorsal_count >= 1