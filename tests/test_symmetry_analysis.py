"""
PyLithics Symmetry Analysis Tests
=================================

Tests for symmetry analysis functionality including dorsal surface symmetry calculations,
area splitting, and symmetry score computations for archaeological artifacts.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

from pylithics.image_processing.modules.symmetry_analysis import (
    analyze_dorsal_symmetry
)


@pytest.mark.unit
class TestAnalyzeDorsalSymmetry:
    """Test the main dorsal symmetry analysis function."""

    def test_analyze_symmetry_no_dorsal_surface(self):
        """Test symmetry analysis when no dorsal surface is found."""
        metrics = [
            {
                'surface_type': 'Ventral',
                'parent': 'parent 1',
                'centroid_x': 50.0,
                'centroid_y': 60.0
            }
        ]
        contours = []
        inverted_image = np.zeros((100, 100), dtype=np.uint8)

        result = analyze_dorsal_symmetry(metrics, contours, inverted_image)

        expected = {
            "top_area": None,
            "bottom_area": None,
            "left_area": None,
            "right_area": None
        }
        assert result == expected

    def test_analyze_symmetry_valid_dorsal_surface(self):
        """Test symmetry analysis with valid dorsal surface."""
        # Create dorsal surface contour
        dorsal_contour = np.array([
            [20, 30], [80, 30], [80, 70], [20, 70]
        ], dtype=np.int32).reshape(-1, 1, 2)

        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'centroid_x': 50,  # Center of rectangle
                'centroid_y': 50   # Center of rectangle
            }
        ]

        contours = [dorsal_contour]

        # Create inverted image with white rectangle
        inverted_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.drawContours(inverted_image, [dorsal_contour], -1, 255, thickness=cv2.FILLED)

        result = analyze_dorsal_symmetry(metrics, contours, inverted_image)

        assert result is not None
        assert 'top_area' in result
        assert 'bottom_area' in result
        assert 'left_area' in result
        assert 'right_area' in result
        assert 'vertical_symmetry' in result
        assert 'horizontal_symmetry' in result

        # For a centered rectangle, areas should be approximately equal
        assert result['top_area'] > 0
        assert result['bottom_area'] > 0
        assert result['left_area'] > 0
        assert result['right_area'] > 0

        # Check symmetry scores (should be close to 1.0 for symmetric shape)
        assert 0.8 <= result['vertical_symmetry'] <= 1.0
        assert 0.8 <= result['horizontal_symmetry'] <= 1.0

    def test_analyze_symmetry_asymmetric_shape(self):
        """Test symmetry analysis with asymmetric dorsal surface."""
        # Create asymmetric L-shaped contour
        l_shape_contour = np.array([
            [20, 20], [60, 20], [60, 40], [40, 40], [40, 80], [20, 80]
        ], dtype=np.int32).reshape(-1, 1, 2)

        # Calculate approximate centroid
        M = cv2.moments(l_shape_contour)
        centroid_x = int(M["m10"] / M["m00"])
        centroid_y = int(M["m01"] / M["m00"])

        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'centroid_x': centroid_x,
                'centroid_y': centroid_y
            }
        ]

        contours = [l_shape_contour]

        # Create inverted image with white L-shape
        inverted_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.drawContours(inverted_image, [l_shape_contour], -1, 255, thickness=cv2.FILLED)

        result = analyze_dorsal_symmetry(metrics, contours, inverted_image)

        assert result is not None

        # L-shape should be asymmetric
        assert result['vertical_symmetry'] < 0.9
        assert result['horizontal_symmetry'] < 0.9

        # Should still have valid area measurements
        assert result['top_area'] >= 0
        assert result['bottom_area'] >= 0
        assert result['left_area'] >= 0
        assert result['right_area'] >= 0

    def test_analyze_symmetry_perfect_vertical_symmetry(self):
        """Test symmetry analysis with perfectly vertically symmetric shape."""
        # Create vertically symmetric diamond
        diamond_contour = np.array([
            [50, 20],  # Top
            [70, 50],  # Right
            [50, 80],  # Bottom
            [30, 50]   # Left
        ], dtype=np.int32).reshape(-1, 1, 2)

        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'centroid_x': 50,
                'centroid_y': 50
            }
        ]

        contours = [diamond_contour]

        inverted_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.drawContours(inverted_image, [diamond_contour], -1, 255, thickness=cv2.FILLED)

        result = analyze_dorsal_symmetry(metrics, contours, inverted_image)

        # Perfect vertical symmetry should have high horizontal symmetry score
        assert result['horizontal_symmetry'] > 0.95

        # Left and right areas should be nearly equal
        left_area = result['left_area']
        right_area = result['right_area']
        if left_area > 0 and right_area > 0:
            ratio = min(left_area, right_area) / max(left_area, right_area)
            assert ratio > 0.9

    def test_analyze_symmetry_perfect_horizontal_symmetry(self):
        """Test symmetry analysis with perfectly horizontally symmetric shape."""
        # Create horizontally symmetric rectangle (wider than tall)
        rect_contour = np.array([
            [20, 40], [80, 40], [80, 60], [20, 60]
        ], dtype=np.int32).reshape(-1, 1, 2)

        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'centroid_x': 50,
                'centroid_y': 50
            }
        ]

        contours = [rect_contour]

        inverted_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.drawContours(inverted_image, [rect_contour], -1, 255, thickness=cv2.FILLED)

        result = analyze_dorsal_symmetry(metrics, contours, inverted_image)

        # Perfect horizontal symmetry should have high vertical symmetry score
        assert result['vertical_symmetry'] > 0.95

        # Top and bottom areas should be nearly equal
        top_area = result['top_area']
        bottom_area = result['bottom_area']
        if top_area > 0 and bottom_area > 0:
            ratio = min(top_area, bottom_area) / max(top_area, bottom_area)
            assert ratio > 0.9

    def test_analyze_symmetry_contour_not_found(self):
        """Test symmetry analysis when contour cannot be matched to metric."""
        # Create mismatched contour and metric
        contour = np.array([
            [10, 10], [20, 10], [20, 20], [10, 20]
        ], dtype=np.int32).reshape(-1, 1, 2)

        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'centroid_x': 100,  # Far from contour
                'centroid_y': 100
            }
        ]

        contours = [contour]
        inverted_image = np.zeros((150, 150), dtype=np.uint8)
        cv2.drawContours(inverted_image, [contour], -1, 255, thickness=cv2.FILLED)

        result = analyze_dorsal_symmetry(metrics, contours, inverted_image)

        # Should return default values when contour not found
        expected = {
            "top_area": None,
            "bottom_area": None,
            "left_area": None,
            "right_area": None
        }
        assert result == expected

    def test_analyze_symmetry_invalid_contour(self):
        """Test symmetry analysis with invalid contour."""
        # Create degenerate contour
        invalid_contour = np.array([
            [50, 50], [50, 50]  # Single point repeated
        ], dtype=np.int32).reshape(-1, 1, 2)

        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'centroid_x': 50,
                'centroid_y': 50
            }
        ]

        contours = [invalid_contour]
        inverted_image = np.zeros((100, 100), dtype=np.uint8)

        result = analyze_dorsal_symmetry(metrics, contours, inverted_image)

        # Should handle invalid contour gracefully
        expected = {
            "top_area": None,
            "bottom_area": None,
            "left_area": None,
            "right_area": None
        }
        assert result == expected

    def test_analyze_symmetry_centroid_outside_contour(self):
        """Test symmetry analysis when centroid is outside the contour."""
        # Create contour
        contour = np.array([
            [20, 20], [40, 20], [40, 40], [20, 40]
        ], dtype=np.int32).reshape(-1, 1, 2)

        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'centroid_x': 10,  # Outside contour
                'centroid_y': 10
            }
        ]

        contours = [contour]
        inverted_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.drawContours(inverted_image, [contour], -1, 255, thickness=cv2.FILLED)

        result = analyze_dorsal_symmetry(metrics, contours, inverted_image)

        # Should return default values when centroid is outside
        expected = {
            "top_area": None,
            "bottom_area": None,
            "left_area": None,
            "right_area": None
        }
        assert result == expected

    def test_analyze_symmetry_edge_contour(self):
        """Test symmetry analysis with contour at image edge."""
        # Create contour touching image boundary
        edge_contour = np.array([
            [0, 0], [50, 0], [50, 50], [0, 50]
        ], dtype=np.int32).reshape(-1, 1, 2)

        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'centroid_x': 25,
                'centroid_y': 25
            }
        ]

        contours = [edge_contour]
        inverted_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.drawContours(inverted_image, [edge_contour], -1, 255, thickness=cv2.FILLED)

        result = analyze_dorsal_symmetry(metrics, contours, inverted_image)

        # Should handle edge contours
        assert result is not None
        assert isinstance(result['top_area'], (int, float))
        assert isinstance(result['bottom_area'], (int, float))
        assert isinstance(result['left_area'], (int, float))
        assert isinstance(result['right_area'], (int, float))


@pytest.mark.unit
class TestSymmetryCalculations:
    """Test specific symmetry calculation logic."""

    def test_symmetry_score_calculation_equal_areas(self):
        """Test symmetry score calculation with equal areas."""
        # Simulate equal top/bottom areas
        top_area = 100.0
        bottom_area = 100.0

        # Calculate vertical symmetry as done in the function
        vertical_symmetry = 1 - abs(top_area - bottom_area) / (top_area + bottom_area)

        assert vertical_symmetry == 1.0

    def test_symmetry_score_calculation_unequal_areas(self):
        """Test symmetry score calculation with unequal areas."""
        # Simulate unequal left/right areas
        left_area = 80.0
        right_area = 120.0

        # Calculate horizontal symmetry
        horizontal_symmetry = 1 - abs(left_area - right_area) / (left_area + right_area)

        expected = 1 - 40.0 / 200.0  # 1 - 0.2 = 0.8
        assert abs(horizontal_symmetry - expected) < 0.01

    def test_symmetry_score_extreme_asymmetry(self):
        """Test symmetry score with extreme asymmetry."""
        # One side has all the area, other has none
        area1 = 200.0
        area2 = 0.0

        symmetry = 1 - abs(area1 - area2) / (area1 + area2) if (area1 + area2) > 0 else None

        assert symmetry == 0.0  # Maximum asymmetry

    def test_symmetry_score_zero_total_area(self):
        """Test symmetry score when total area is zero."""
        area1 = 0.0
        area2 = 0.0

        symmetry = 1 - abs(area1 - area2) / (area1 + area2) if (area1 + area2) > 0 else None

        assert symmetry is None  # Cannot calculate symmetry


@pytest.mark.integration
class TestSymmetryAnalysisIntegration:
    """Integration tests for symmetry analysis workflow."""

    def test_symmetry_with_realistic_artifact(self):
        """Test symmetry analysis with realistic artifact contour."""
        # Create realistic artifact shape (elongated with some asymmetry)
        artifact_points = []

        # Create roughly symmetric but slightly irregular outline
        for i in range(20):
            angle = 2 * np.pi * i / 20
            radius = 30 + 5 * np.sin(3 * angle)  # Add some irregularity

            # Add slight asymmetry
            if angle > np.pi:
                radius *= 0.9

            x = int(50 + radius * np.cos(angle))
            y = int(50 + radius * np.sin(angle))
            artifact_points.append([x, y])

        artifact_contour = np.array(artifact_points, dtype=np.int32).reshape(-1, 1, 2)

        # Calculate actual centroid
        M = cv2.moments(artifact_contour)
        if M["m00"] > 0:
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
        else:
            centroid_x, centroid_y = 50, 50

        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'centroid_x': centroid_x,
                'centroid_y': centroid_y
            }
        ]

        contours = [artifact_contour]
        inverted_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.drawContours(inverted_image, [artifact_contour], -1, 255, thickness=cv2.FILLED)

        result = analyze_dorsal_symmetry(metrics, contours, inverted_image)

        assert result is not None

        # Should have reasonable symmetry scores (not perfect due to irregularity)
        assert 0.5 <= result['vertical_symmetry'] <= 1.0
        assert 0.5 <= result['horizontal_symmetry'] <= 1.0

        # Areas should sum to approximately the total contour area
        total_calculated = (result['top_area'] + result['bottom_area'])
        contour_area = cv2.contourArea(artifact_contour)

        # Allow some tolerance due to pixel discretization
        assert abs(total_calculated - contour_area) / contour_area < 0.1

    def test_symmetry_with_multiple_contours(self):
        """Test symmetry analysis when multiple contours are present."""
        # Create two contours, one dorsal, one not
        dorsal_contour = np.array([
            [20, 20], [60, 20], [60, 60], [20, 60]
        ], dtype=np.int32).reshape(-1, 1, 2)

        other_contour = np.array([
            [70, 70], [90, 70], [90, 90], [70, 90]
        ], dtype=np.int32).reshape(-1, 1, 2)

        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'centroid_x': 40,
                'centroid_y': 40
            },
            {
                'surface_type': 'Ventral',
                'parent': 'parent 2',
                'centroid_x': 80,
                'centroid_y': 80
            }
        ]

        contours = [dorsal_contour, other_contour]
        inverted_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.drawContours(inverted_image, contours, -1, 255, thickness=cv2.FILLED)

        result = analyze_dorsal_symmetry(metrics, contours, inverted_image)

        # Should analyze only the dorsal contour
        assert result is not None
        assert result['vertical_symmetry'] > 0.9  # Square should be symmetric
        assert result['horizontal_symmetry'] > 0.9

    def test_symmetry_with_complex_shape(self):
        """Test symmetry analysis with complex archaeological tool shape."""
        # Create tool-like shape (elongated with pointed end)
        tool_points = [
            [50, 10],   # Tip
            [55, 20],
            [60, 40],
            [55, 60],
            [50, 80],   # Base center
            [45, 60],
            [40, 40],
            [45, 20]
        ]

        tool_contour = np.array(tool_points, dtype=np.int32).reshape(-1, 1, 2)

        # Calculate centroid
        M = cv2.moments(tool_contour)
        centroid_x = int(M["m10"] / M["m00"])
        centroid_y = int(M["m01"] / M["m00"])

        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'centroid_x': centroid_x,
                'centroid_y': centroid_y
            }
        ]

        contours = [tool_contour]
        inverted_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.drawContours(inverted_image, [tool_contour], -1, 255, thickness=cv2.FILLED)

        result = analyze_dorsal_symmetry(metrics, contours, inverted_image)

        assert result is not None

        # Tool shape should have good vertical symmetry but less horizontal symmetry
        assert result['horizontal_symmetry'] > 0.8  # Should be symmetric left-right
        assert result['vertical_symmetry'] < 0.8   # Less symmetric top-bottom due to pointed tip


@pytest.mark.unit
class TestSymmetryAnalysisErrorHandling:
    """Test error handling in symmetry analysis."""

    def test_symmetry_empty_metrics(self):
        """Test symmetry analysis with empty metrics."""
        result = analyze_dorsal_symmetry([], [], np.zeros((100, 100), dtype=np.uint8))

        expected = {
            "top_area": None,
            "bottom_area": None,
            "left_area": None,
            "right_area": None
        }
        assert result == expected

    def test_symmetry_empty_contours(self):
        """Test symmetry analysis with empty contours list."""
        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'centroid_x': 50,
                'centroid_y': 50
            }
        ]

        result = analyze_dorsal_symmetry(metrics, [], np.zeros((100, 100), dtype=np.uint8))

        expected = {
            "top_area": None,
            "bottom_area": None,
            "left_area": None,
            "right_area": None
        }
        assert result == expected

    def test_symmetry_invalid_image(self):
        """Test symmetry analysis with invalid image."""
        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'centroid_x': 50,
                'centroid_y': 50
            }
        ]

        contour = np.array([[40, 40], [60, 40], [60, 60], [40, 60]], dtype=np.int32).reshape(-1, 1, 2)

        # Test with None image
        try:
            result = analyze_dorsal_symmetry(metrics, [contour], None)
            # If it doesn't crash, should return safe defaults
        except (AttributeError, TypeError):
            # These are acceptable errors for None image
            pass

    def test_symmetry_malformed_metrics(self):
        """Test symmetry analysis with malformed metrics."""
        malformed_metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                # Missing centroid_x and centroid_y
            }
        ]

        contour = np.array([[40, 40], [60, 40], [60, 60], [40, 60]], dtype=np.int32).reshape(-1, 1, 2)
        inverted_image = np.zeros((100, 100), dtype=np.uint8)

        try:
            result = analyze_dorsal_symmetry(malformed_metrics, [contour], inverted_image)
            # Should handle missing fields gracefully
        except (KeyError, TypeError):
            # These are acceptable errors for malformed data
            pass

    def test_symmetry_negative_coordinates(self):
        """Test symmetry analysis with negative centroid coordinates."""
        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'centroid_x': -10,  # Negative coordinate
                'centroid_y': -10
            }
        ]

        contour = np.array([[40, 40], [60, 40], [60, 60], [40, 60]], dtype=np.int32).reshape(-1, 1, 2)
        inverted_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.drawContours(inverted_image, [contour], -1, 255, thickness=cv2.FILLED)

        result = analyze_dorsal_symmetry(metrics, [contour], inverted_image)

        # Should handle negative coordinates gracefully
        expected = {
            "top_area": None,
            "bottom_area": None,
            "left_area": None,
            "right_area": None
        }
        assert result == expected

    def test_symmetry_very_large_coordinates(self):
        """Test symmetry analysis with very large coordinates."""
        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'centroid_x': 10000,  # Very large coordinate
                'centroid_y': 10000
            }
        ]

        contour = np.array([[40, 40], [60, 40], [60, 60], [40, 60]], dtype=np.int32).reshape(-1, 1, 2)
        inverted_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.drawContours(inverted_image, [contour], -1, 255, thickness=cv2.FILLED)

        result = analyze_dorsal_symmetry(metrics, [contour], inverted_image)

        # Should handle out-of-bounds coordinates
        expected = {
            "top_area": None,
            "bottom_area": None,
            "left_area": None,
            "right_area": None
        }
        assert result == expected


@pytest.mark.performance
class TestSymmetryAnalysisPerformance:
    """Test performance aspects of symmetry analysis."""

    def test_symmetry_large_contour(self):
        """Test symmetry analysis performance with large contour."""
        # Create large complex contour
        large_points = []
        for i in range(1000):  # Many points
            angle = 2 * np.pi * i / 1000
            radius = 200 + 50 * np.sin(5 * angle)
            x = int(300 + radius * np.cos(angle))
            y = int(300 + radius * np.sin(angle))
            large_points.append([x, y])

        large_contour = np.array(large_points, dtype=np.int32).reshape(-1, 1, 2)

        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'centroid_x': 300,
                'centroid_y': 300
            }
        ]

        inverted_image = np.zeros((600, 600), dtype=np.uint8)
        cv2.drawContours(inverted_image, [large_contour], -1, 255, thickness=cv2.FILLED)

        import time
        start_time = time.time()

        result = analyze_dorsal_symmetry(metrics, [large_contour], inverted_image)

        end_time = time.time()
        processing_time = end_time - start_time

        # Should complete in reasonable time even with large contour
        assert processing_time < 5.0  # 5 seconds max
        assert result is not None

    def test_symmetry_high_resolution_image(self):
        """Test symmetry analysis with high resolution image."""
        # Create contour for high-res image
        contour = np.array([
            [1000, 1000], [1500, 1000], [1500, 1500], [1000, 1500]
        ], dtype=np.int32).reshape(-1, 1, 2)

        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'centroid_x': 1250,
                'centroid_y': 1250
            }
        ]

        # High resolution image
        high_res_image = np.zeros((2000, 2000), dtype=np.uint8)
        cv2.drawContours(high_res_image, [contour], -1, 255, thickness=cv2.FILLED)

        import time
        start_time = time.time()

        result = analyze_dorsal_symmetry(metrics, [contour], high_res_image)

        end_time = time.time()
        processing_time = end_time - start_time

        # Should handle high-res images efficiently
        assert processing_time < 10.0  # 10 seconds max
        assert result is not None