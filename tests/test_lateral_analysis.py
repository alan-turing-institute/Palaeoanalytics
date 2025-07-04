"""
PyLithics Lateral Analysis Tests
================================

Tests for lateral surface analysis including convexity calculations,
distance to maximum width measurements, and metric integration for archaeological tools.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

from pylithics.image_processing.modules.lateral_analysis import (
    analyze_lateral_surface,
    detect_lateral_convexity,
    _integrate_lateral_metrics,
    _calculate_lateral_distance_to_max_width
)


@pytest.mark.unit
class TestAnalyzeLateralSurface:
    """Test the main lateral surface analysis function."""

    def test_analyze_lateral_no_lateral_surface(self):
        """Test lateral analysis when no lateral surface is found."""
        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'parent 1',
                'area': 5000.0
            },
            {
                'surface_type': 'Ventral',
                'parent': 'parent 2',
                'scar': 'parent 2',
                'area': 4800.0
            }
        ]

        parent_contours = []
        inverted_image = np.zeros((100, 100), dtype=np.uint8)

        result = analyze_lateral_surface(metrics, parent_contours, inverted_image)

        assert result is None

    def test_analyze_lateral_valid_surface(self):
        """Test lateral analysis with valid lateral surface."""
        # Create lateral contour (elongated rectangle)
        lateral_contour = np.array([
            [20, 30], [80, 30], [80, 70], [20, 70]
        ], dtype=np.int32).reshape(-1, 1, 2)

        metrics = [
            {
                'surface_type': 'Lateral',
                'parent': 'parent 1',
                'scar': 'parent 1',
                'area': 2400.0  # 60 * 40
            }
        ]

        parent_contours = [lateral_contour]
        inverted_image = np.zeros((100, 100), dtype=np.uint8)

        result = analyze_lateral_surface(metrics, parent_contours, inverted_image)

        assert result is not None
        assert 'lateral_convexity' in result
        assert 'distance_to_max_width' in result

        # Rectangle should have high convexity (close to 1.0)
        assert 0.95 <= result['lateral_convexity'] <= 1.0
        assert result['distance_to_max_width'] > 0

    def test_analyze_lateral_contour_index_matching(self):
        """Test lateral analysis with correct contour index matching."""
        # Create two contours
        contour1 = np.array([
            [10, 10], [30, 10], [30, 30], [10, 30]
        ], dtype=np.int32).reshape(-1, 1, 2)  # Area = 400

        contour2 = np.array([
            [40, 40], [80, 40], [80, 80], [40, 80]
        ], dtype=np.int32).reshape(-1, 1, 2)  # Area = 1600

        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'parent 1',
                'area': 400.0
            },
            {
                'surface_type': 'Lateral',
                'parent': 'parent 2',
                'scar': 'parent 2',
                'area': 1600.0  # Should match contour2
            }
        ]

        parent_contours = [contour1, contour2]
        inverted_image = np.zeros((100, 100), dtype=np.uint8)

        result = analyze_lateral_surface(metrics, parent_contours, inverted_image)

        assert result is not None
        # Should analyze the larger contour (contour2)
        assert result['lateral_convexity'] is not None

    def test_analyze_lateral_contour_area_fallback(self):
        """Test lateral analysis using area-based contour matching fallback."""
        lateral_contour = np.array([
            [20, 20], [60, 20], [60, 60], [20, 60]
        ], dtype=np.int32).reshape(-1, 1, 2)  # Area = 1600

        metrics = [
            {
                'surface_type': 'Lateral',
                'parent': 'parent 1',
                'scar': 'parent 1',
                'area': 1600.0  # Exact match
            }
        ]

        # Make contour index mismatch by having more contours than metrics
        extra_contour = np.array([
            [70, 70], [90, 70], [90, 90], [70, 90]
        ], dtype=np.int32).reshape(-1, 1, 2)

        parent_contours = [extra_contour, lateral_contour]  # Lateral is at index 1, not 0
        inverted_image = np.zeros((100, 100), dtype=np.uint8)

        result = analyze_lateral_surface(metrics, parent_contours, inverted_image)

        assert result is not None
        # Should find correct contour by area matching
        assert result['lateral_convexity'] is not None

    def test_analyze_lateral_no_matching_contour(self):
        """Test lateral analysis when no matching contour is found."""
        metrics = [
            {
                'surface_type': 'Lateral',
                'parent': 'parent 1',
                'scar': 'parent 1',
                'area': 5000.0  # Won't match any contour
            }
        ]

        small_contour = np.array([
            [10, 10], [20, 10], [20, 20], [10, 20]
        ], dtype=np.int32).reshape(-1, 1, 2)  # Area = 100

        parent_contours = [small_contour]
        inverted_image = np.zeros((100, 100), dtype=np.uint8)

        with patch('pylithics.image_processing.modules.lateral_analysis.logging') as mock_logging:
            result = analyze_lateral_surface(metrics, parent_contours, inverted_image)

            assert result is None
            mock_logging.error.assert_called()

    def test_analyze_lateral_convexity_calculation_error(self):
        """Test lateral analysis when convexity calculation fails."""
        # Create invalid contour that might cause convexity calculation to fail
        invalid_contour = np.array([
            [50, 50], [50, 50]  # Degenerate contour (single point repeated)
        ], dtype=np.int32).reshape(-1, 1, 2)

        metrics = [
            {
                'surface_type': 'Lateral',
                'parent': 'parent 1',
                'scar': 'parent 1',
                'area': 0.0
            }
        ]

        parent_contours = [invalid_contour]
        inverted_image = np.zeros((100, 100), dtype=np.uint8)

        with patch('pylithics.image_processing.modules.lateral_analysis.logging') as mock_logging:
            result = analyze_lateral_surface(metrics, parent_contours, inverted_image)

            # Should handle gracefully and return partial results
            if result is not None:
                assert 'lateral_convexity' in result
                # Convexity might be None due to error
            mock_logging.error.assert_called()

    def test_analyze_lateral_distance_calculation_error(self):
        """Test lateral analysis when distance calculation fails."""
        # Create contour that might cause distance calculation issues
        line_contour = np.array([
            [20, 50], [80, 50], [80, 50], [20, 50]  # Degenerate line
        ], dtype=np.int32).reshape(-1, 1, 2)

        metrics = [
            {
                'surface_type': 'Lateral',
                'parent': 'parent 1',
                'scar': 'parent 1',
                'area': 0.0
            }
        ]

        parent_contours = [line_contour]
        inverted_image = np.zeros((100, 100), dtype=np.uint8)

        with patch('pylithics.image_processing.modules.lateral_analysis.logging') as mock_logging:
            result = analyze_lateral_surface(metrics, parent_contours, inverted_image)

            # Should handle errors gracefully
            if result is not None:
                assert 'distance_to_max_width' in result
                # Distance might be None due to error


@pytest.mark.unit
class TestDetectLateralConvexity:
    """Test the lateral convexity detection function."""

    def test_convexity_perfect_rectangle(self):
        """Test convexity calculation for perfect rectangle (should be 1.0)."""
        rectangle = np.array([
            [10, 20], [60, 20], [60, 80], [10, 80]
        ], dtype=np.int32).reshape(-1, 1, 2)

        convexity = detect_lateral_convexity(rectangle)

        assert convexity is not None
        # Rectangle should have convexity very close to 1.0
        assert 0.98 <= convexity <= 1.0

    def test_convexity_perfect_circle(self):
        """Test convexity calculation for circle (should be 1.0)."""
        # Create circular contour
        center = (50, 50)
        radius = 30
        circle_points = []

        for i in range(32):  # 32 points around circle
            angle = 2 * np.pi * i / 32
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            circle_points.append([x, y])

        circle_contour = np.array(circle_points, dtype=np.int32).reshape(-1, 1, 2)

        convexity = detect_lateral_convexity(circle_contour)

        assert convexity is not None
        # Circle should have high convexity
        assert 0.95 <= convexity <= 1.0

    def test_convexity_concave_shape(self):
        """Test convexity calculation for concave shape."""
        # Create L-shaped contour (concave)
        l_shape = np.array([
            [10, 10], [40, 10], [40, 30], [30, 30], [30, 50], [10, 50]
        ], dtype=np.int32).reshape(-1, 1, 2)

        convexity = detect_lateral_convexity(l_shape)

        assert convexity is not None
        # L-shape should have lower convexity
        assert 0.3 <= convexity < 0.9

    def test_convexity_star_shape(self):
        """Test convexity calculation for star shape (highly concave)."""
        # Create star-shaped contour
        star_points = []
        center = (50, 50)

        for i in range(10):
            angle = 2 * np.pi * i / 10
            if i % 2 == 0:
                radius = 30  # Outer points
            else:
                radius = 15  # Inner points (creates concavity)

            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            star_points.append([x, y])

        star_contour = np.array(star_points, dtype=np.int32).reshape(-1, 1, 2)

        convexity = detect_lateral_convexity(star_contour)

        assert convexity is not None
        # Star should have low convexity due to deep concavities
        assert 0.1 <= convexity < 0.7

    def test_convexity_triangle(self):
        """Test convexity calculation for triangle (should be 1.0)."""
        triangle = np.array([
            [50, 20], [80, 70], [20, 70]
        ], dtype=np.int32).reshape(-1, 1, 2)

        convexity = detect_lateral_convexity(triangle)

        assert convexity is not None
        # Triangle should have convexity of 1.0 (convex)
        assert 0.98 <= convexity <= 1.0

    def test_convexity_invalid_contour(self):
        """Test convexity calculation with invalid contour."""
        # Empty contour
        empty_contour = np.array([], dtype=np.int32).reshape(0, 1, 2)

        convexity = detect_lateral_convexity(empty_contour)
        assert convexity is None

        # Single point
        point_contour = np.array([[50, 50]], dtype=np.int32).reshape(-1, 1, 2)

        convexity = detect_lateral_convexity(point_contour)
        assert convexity is None

        # Two points (line)
        line_contour = np.array([[20, 30], [80, 30]], dtype=np.int32).reshape(-1, 1, 2)

        convexity = detect_lateral_convexity(line_contour)
        # Might return None or a valid value depending on OpenCV handling
        assert convexity is None or isinstance(convexity, (int, float))

    def test_convexity_none_input(self):
        """Test convexity calculation with None input."""
        convexity = detect_lateral_convexity(None)
        assert convexity is None

    def test_convexity_zero_area_contour(self):
        """Test convexity calculation with zero area contour."""
        # Create degenerate contour (all points on a line)
        line_points = np.array([
            [10, 50], [30, 50], [50, 50], [70, 50]
        ], dtype=np.int32).reshape(-1, 1, 2)

        with patch('pylithics.image_processing.modules.lateral_analysis.logging') as mock_logging:
            convexity = detect_lateral_convexity(line_points)

            # Should handle zero area gracefully
            if convexity is not None:
                assert 0 <= convexity <= 1
            mock_logging.warning.assert_called()


@pytest.mark.unit
class TestCalculateLateralDistanceToMaxWidth:
    """Test the distance to maximum width calculation function."""

    def test_distance_rectangular_contour(self):
        """Test distance calculation for rectangular contour."""
        # Create vertical rectangle
        rectangle = np.array([
            [40, 20], [60, 20], [60, 80], [40, 80]
        ], dtype=np.int32).reshape(-1, 1, 2)

        distance = _calculate_lateral_distance_to_max_width(rectangle)

        assert distance is not None
        assert distance >= 0

        # For a rectangle, distance should be from top to middle
        # Top Y = 20, middle Y should be around 50, so distance ≈ 30
        expected_distance = 30.0
        assert abs(distance - expected_distance) < 5.0  # Allow some tolerance

    def test_distance_triangle_contour(self):
        """Test distance calculation for triangular contour."""
        # Create triangle with base at bottom
        triangle = np.array([
            [50, 20],   # Top point
            [20, 80],   # Bottom left
            [80, 80]    # Bottom right
        ], dtype=np.int32).reshape(-1, 1, 2)

        distance = _calculate_lateral_distance_to_max_width(triangle)

        assert distance is not None
        assert distance >= 0

        # For triangle, max width is at the base, distance should be from top to base center
        # Top Y = 20, base Y = 80, so distance ≈ 60
        expected_distance = 60.0
        assert abs(distance - expected_distance) < 10.0

    def test_distance_diamond_contour(self):
        """Test distance calculation for diamond-shaped contour."""
        # Create diamond shape
        diamond = np.array([
            [50, 20],   # Top
            [80, 50],   # Right (max width here)
            [50, 80],   # Bottom
            [20, 50]    # Left (max width here)
        ], dtype=np.int32).reshape(-1, 1, 2)

        distance = _calculate_lateral_distance_to_max_width(diamond)

        assert distance is not None
        assert distance >= 0

        # Max width is at Y=50, top is at Y=20, so distance = 30
        expected_distance = 30.0
        assert abs(distance - expected_distance) < 5.0

    def test_distance_irregular_contour(self):
        """Test distance calculation for irregular contour."""
        # Create irregular shape with clear max width
        irregular = np.array([
            [50, 10],   # Top point
            [60, 30],
            [80, 50],   # Wide part
            [90, 60],   # Maximum width here
            [80, 70],
            [70, 80],
            [30, 80],
            [20, 70],
            [10, 60],   # Maximum width here too
            [20, 50],
            [40, 30]
        ], dtype=np.int32).reshape(-1, 1, 2)

        distance = _calculate_lateral_distance_to_max_width(irregular)

        assert distance is not None
        assert distance >= 0
        # Should be reasonable distance given the shape
        assert distance < 100  # Sanity check

    def test_distance_invalid_contour(self):
        """Test distance calculation with invalid contour."""
        # Empty contour
        empty_contour = np.array([], dtype=np.int32).reshape(0, 1, 2)

        distance = _calculate_lateral_distance_to_max_width(empty_contour)
        assert distance is None

        # Single point
        point_contour = np.array([[50, 50]], dtype=np.int32).reshape(-1, 1, 2)

        distance = _calculate_lateral_distance_to_max_width(point_contour)
        assert distance is None

    def test_distance_none_input(self):
        """Test distance calculation with None input."""
        distance = _calculate_lateral_distance_to_max_width(None)
        assert distance is None

    def test_distance_calculation_error_handling(self):
        """Test distance calculation error handling."""
        # Create contour that might cause calculation errors
        problematic_contour = np.array([
            [50, 50], [50, 50], [50, 50]  # All same point
        ], dtype=np.int32).reshape(-1, 1, 2)

        with patch('pylithics.image_processing.modules.lateral_analysis.logging') as mock_logging:
            distance = _calculate_lateral_distance_to_max_width(problematic_contour)

            # Should handle gracefully
            if distance is not None:
                assert distance >= 0
            else:
                mock_logging.error.assert_called()


@pytest.mark.unit
class TestIntegrateLateralMetrics:
    """Test the lateral metrics integration function."""

    def test_integrate_lateral_metrics_success(self):
        """Test successful integration of lateral metrics."""
        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'parent 1'
            },
            {
                'surface_type': 'Lateral',
                'parent': 'parent 2',
                'scar': 'parent 2'
            }
        ]

        lateral_results = {
            'lateral_convexity': 0.85,
            'distance_to_max_width': 45.2
        }

        _integrate_lateral_metrics(metrics, lateral_results)

        # Check that lateral metric was updated
        lateral_metric = next(m for m in metrics if m['surface_type'] == 'Lateral')

        assert lateral_metric['lateral_convexity'] == 0.85
        assert lateral_metric['distance_to_max_width'] == 45.2

        # Check that dorsal metric was not affected
        dorsal_metric = next(m for m in metrics if m['surface_type'] == 'Dorsal')
        assert 'lateral_convexity' not in dorsal_metric

    def test_integrate_lateral_metrics_no_lateral_surface(self):
        """Test integration when no lateral surface exists."""
        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'parent 1'
            },
            {
                'surface_type': 'Ventral',
                'parent': 'parent 2',
                'scar': 'parent 2'
            }
        ]

        lateral_results = {
            'lateral_convexity': 0.85,
            'distance_to_max_width': 45.2
        }

        with patch('pylithics.image_processing.modules.lateral_analysis.logging') as mock_logging:
            _integrate_lateral_metrics(metrics, lateral_results)

            mock_logging.warning.assert_called()

            # No metrics should be modified
            for metric in metrics:
                assert 'lateral_convexity' not in metric

    def test_integrate_lateral_metrics_error_handling(self):
        """Test integration error handling."""
        metrics = [
            {
                'surface_type': 'Lateral',
                'parent': 'parent 1',
                'scar': 'parent 1'
            }
        ]

        # Malformed lateral results
        malformed_results = {
            'invalid_key': 'invalid_value'
        }

        try:
            _integrate_lateral_metrics(metrics, malformed_results)

            # Should handle gracefully
            lateral_metric = metrics[0]
            assert 'invalid_key' in lateral_metric

        except Exception as e:
            # If it raises an exception, should be handled gracefully
            with patch('pylithics.image_processing.modules.lateral_analysis.logging') as mock_logging:
                try:
                    _integrate_lateral_metrics(metrics, malformed_results)
                except:
                    pass
                mock_logging.error.assert_called()


@pytest.mark.integration
class TestLateralAnalysisIntegration:
    """Integration tests for lateral analysis workflow."""

    def test_complete_lateral_analysis_workflow(self):
        """Test complete lateral analysis workflow from contour to metrics."""
        # Create realistic lateral surface contour (elongated tool side)
        lateral_contour = np.array([
            [30, 20],   # Top narrow
            [35, 25],
            [40, 40],   # Widening
            [45, 60],   # Maximum width area
            [50, 80],
            [48, 100],  # Widening
            [45, 120],
            [40, 140],  # Narrowing
            [35, 155],
            [30, 160],  # Bottom narrow
            [25, 155],
            [20, 140],
            [15, 120],
            [12, 100],
            [10, 80],
            [15, 60],
            [20, 40],
            [25, 25]
        ], dtype=np.int32).reshape(-1, 1, 2)

        metrics = [
            {
                'surface_type': 'Lateral',
                'parent': 'parent 1',
                'scar': 'parent 1',
                'area': cv2.contourArea(lateral_contour)
            }
        ]

        parent_contours = [lateral_contour]
        inverted_image = np.zeros((200, 70), dtype=np.uint8)

        # Step 1: Analyze lateral surface
        result = analyze_lateral_surface(metrics, parent_contours, inverted_image)

        assert result is not None
        assert 'lateral_convexity' in result
        assert 'distance_to_max_width' in result

        # Step 2: Verify convexity makes sense for tool shape
        # Tool sides are typically somewhat concave due to shaping
        assert 0.4 <= result['lateral_convexity'] <= 0.9

        # Step 3: Verify distance measurement
        assert result['distance_to_max_width'] > 0
        # Should be reasonable distance within the contour bounds
        assert result['distance_to_max_width'] < 200

        # Step 4: Test integration
        _integrate_lateral_metrics(metrics, result)

        lateral_metric = metrics[0]
        assert lateral_metric['lateral_convexity'] == result['lateral_convexity']
        assert lateral_metric['distance_to_max_width'] == result['distance_to_max_width']

    def test_lateral_analysis_archaeological_scenarios(self):
        """Test lateral analysis with different archaeological tool scenarios."""
        scenarios = [
            {
                'name': 'blade_lateral',
                'contour': np.array([
                    [45, 10], [55, 10], [55, 200], [45, 200]  # Straight blade edge
                ], dtype=np.int32).reshape(-1, 1, 2),
                'expected_convexity_range': (0.95, 1.0)  # Should be very convex
            },
            {
                'name': 'notched_tool_lateral',
                'contour': np.array([
                    [40, 20], [60, 20], [60, 80], [50, 90], [40, 100],  # Notch
                    [50, 110], [60, 120], [60, 180], [40, 180]
                ], dtype=np.int32).reshape(-1, 1, 2),
                'expected_convexity_range': (0.3, 0.8)  # Less convex due to notch
            },
            {
                'name': 'scraper_lateral',
                'contour': np.array([
                    [30, 30], [70, 20], [80, 40], [85, 80], [80, 120],
                    [70, 140], [30, 130], [20, 110], [15, 80], [20, 50]
                ], dtype=np.int32).reshape(-1, 1, 2),
                'expected_convexity_range': (0.7, 0.95)  # Moderately convex
            }
        ]

        for scenario in scenarios:
            contour = scenario['contour']

            metrics = [
                {
                    'surface_type': 'Lateral',
                    'parent': 'parent 1',
                    'scar': 'parent 1',
                    'area': cv2.contourArea(contour)
                }
            ]

            parent_contours = [contour]
            inverted_image = np.zeros((220, 120), dtype=np.uint8)

            result = analyze_lateral_surface(metrics, parent_contours, inverted_image)

            assert result is not None, f"Failed for scenario: {scenario['name']}"

            convexity = result['lateral_convexity']
            expected_range = scenario['expected_convexity_range']

            assert expected_range[0] <= convexity <= expected_range[1], \
                f"Convexity {convexity} out of range {expected_range} for {scenario['name']}"

            assert result['distance_to_max_width'] > 0, \
                f"Invalid distance for scenario: {scenario['name']}"

    def test_lateral_analysis_error_recovery(self):
        """Test lateral analysis error recovery in complex scenarios."""
        # Create problematic contour that might cause issues
        problematic_contour = np.array([
            [50, 50], [51, 50], [50, 51], [50, 50]  # Near-degenerate
        ], dtype=np.int32).reshape(-1, 1, 2)

        metrics = [
            {
                'surface_type': 'Lateral',
                'parent': 'parent 1',
                'scar': 'parent 1',
                'area': cv2.contourArea(problematic_contour)
            }
        ]

        parent_contours = [problematic_contour]
        inverted_image = np.zeros((100, 100), dtype=np.uint8)

        with patch('pylithics.image_processing.modules.lateral_analysis.logging'):
            result = analyze_lateral_surface(metrics, parent_contours, inverted_image)

            # Should either succeed with reasonable values or fail gracefully
            if result is not None:
                assert 'lateral_convexity' in result
                assert 'distance_to_max_width' in result

                # Values should be reasonable or None
                if result['lateral_convexity'] is not None:
                    assert 0 <= result['lateral_convexity'] <= 1
                if result['distance_to_max_width'] is not None:
                    assert result['distance_to_max_width'] >= 0


@pytest.mark.performance
class TestLateralAnalysisPerformance:
    """Test performance aspects of lateral analysis."""

    def test_lateral_analysis_large_contour_performance(self):
        """Test lateral analysis performance with large complex contour."""
        # Create large complex contour
        large_points = []
        for i in range(1000):  # Many points
            angle = 2 * np.pi * i / 1000
            # Create irregular shape with varying radius
            base_radius = 100
            variation = 20 * np.sin(5 * angle) + 10 * np.cos(8 * angle)
            radius = base_radius + variation

            x = int(200 + radius * np.cos(angle))
            y = int(200 + radius * np.sin(angle))
            large_points.append([x, y])

        large_contour = np.array(large_points, dtype=np.int32).reshape(-1, 1, 2)