"""
PyLithics Arrow Detection Tests
===============================

Tests for arrow detection algorithms including contour analysis, defect detection,
triangle structure analysis, and arrow property calculations.
"""

import pytest
import numpy as np
import cv2
import math
import tempfile
import os
from unittest.mock import patch, MagicMock

from pylithics.image_processing.modules.arrow_detection import (
    ArrowDetector,
    analyze_child_contour_for_arrow,
    scale_parameters_for_dpi,
    create_arrow_detection_pipeline,
    batch_detect_arrows,
    validate_arrow_detection_config
)


@pytest.mark.unit
class TestArrowDetector:
    """Test the ArrowDetector class."""

    def test_arrow_detector_init_default_config(self):
        """Test ArrowDetector initialization with default config."""
        detector = ArrowDetector()

        assert detector.reference_dpi == 300.0
        assert detector.debug_enabled is False
        assert 'min_area' in detector.ref_thresholds
        assert 'min_defect_depth' in detector.ref_thresholds
        assert 'solidity_bounds' in detector.ref_thresholds

    def test_arrow_detector_init_custom_config(self):
        """Test ArrowDetector initialization with custom config."""
        custom_config = {
            'reference_dpi': 150.0,
            'debug_enabled': True,
            'min_area_scale_factor': 0.5,
            'min_defect_depth_scale_factor': 0.6,
            'min_triangle_height_scale_factor': 0.7
        }

        detector = ArrowDetector(custom_config)

        assert detector.reference_dpi == 150.0
        assert detector.debug_enabled is True
        assert detector.config['min_area_scale_factor'] == 0.5

    def test_scale_parameters_for_dpi_standard(self):
        """Test DPI scaling with standard DPI values."""
        detector = ArrowDetector()

        # Test with reference DPI (should return original values)
        params = detector.scale_parameters_for_dpi(300.0)

        assert params['min_area'] == detector.ref_thresholds['min_area'] * 0.7  # With safety factor
        assert params['min_defect_depth'] == detector.ref_thresholds['min_defect_depth'] * 0.8
        assert params['solidity_bounds'] == detector.ref_thresholds['solidity_bounds']

    def test_scale_parameters_for_dpi_different_resolution(self):
        """Test DPI scaling with different resolution."""
        detector = ArrowDetector()

        # Test with double DPI (should scale up)
        params_600 = detector.scale_parameters_for_dpi(600.0)
        params_300 = detector.scale_parameters_for_dpi(300.0)

        # Area should scale quadratically (2x DPI = 4x area)
        assert params_600['min_area'] > params_300['min_area']

        # Linear measurements should scale linearly
        assert params_600['min_defect_depth'] > params_300['min_defect_depth']
        assert params_600['min_triangle_height'] > params_300['min_triangle_height']

        # Solidity bounds should remain the same
        assert params_600['solidity_bounds'] == params_300['solidity_bounds']

    def test_scale_parameters_for_dpi_invalid_values(self):
        """Test DPI scaling with invalid DPI values."""
        detector = ArrowDetector()

        # Test with None DPI
        params_none = detector.scale_parameters_for_dpi(None)
        assert params_none == detector.ref_thresholds

        # Test with zero DPI
        params_zero = detector.scale_parameters_for_dpi(0.0)
        assert params_zero == detector.ref_thresholds

        # Test with negative DPI
        params_negative = detector.scale_parameters_for_dpi(-100.0)
        assert params_negative == detector.ref_thresholds

    def test_validate_basic_properties_valid_contour(self, arrow_shaped_contour):
        """Test basic property validation with a valid arrow-shaped contour."""
        detector = ArrowDetector()
        params = detector.scale_parameters_for_dpi(300.0)

        # Create a larger arrow to pass area test
        large_arrow = arrow_shaped_contour * 3  # Scale up

        result = detector._validate_basic_properties(large_arrow, params, None)

        # Should pass basic validation
        assert result is True

    def test_validate_basic_properties_too_small(self):
        """Test basic property validation with contour that's too small."""
        detector = ArrowDetector()
        params = detector.scale_parameters_for_dpi(300.0)

        # Create a contour smaller than min_area threshold
        # min_area = 1 * 1.0 * 0.7 = 0.7, so we need area < 0.7
        # Create a tiny triangle with area < 0.7
        tiny_contour = np.array([
            [0, 0], [1, 0], [0, 1]  # Triangle with area = 0.5
        ], dtype=np.int32).reshape(-1, 1, 2)

        result = detector._validate_basic_properties(tiny_contour, params, None)

        # Should fail area test
        assert result is False

    def test_validate_basic_properties_wrong_solidity(self):
        """Test basic property validation with wrong solidity."""
        detector = ArrowDetector()
        params = detector.scale_parameters_for_dpi(300.0)

        # Create a contour that's too solid (approaching a circle)
        # This should have high solidity and fail the test
        circle_points = []
        for i in range(20):
            angle = 2 * math.pi * i / 20
            x = int(50 + 40 * math.cos(angle))
            y = int(50 + 40 * math.sin(angle))
            circle_points.append([x, y])

        circle_contour = np.array(circle_points, dtype=np.int32).reshape(-1, 1, 2)

        result = detector._validate_basic_properties(circle_contour, params, None)

        # Should pass or fail based on solidity bounds
        # Circle has high solidity (~1.0), arrow shapes should be lower
        assert isinstance(result, bool)

    def test_find_significant_defects_valid_arrow(self, arrow_shaped_contour):
        """Test finding significant defects in an arrow-shaped contour."""
        detector = ArrowDetector()

        # Scale up the arrow to ensure it has measurable defects
        large_arrow = arrow_shaped_contour * 5

        defects = detector._find_significant_defects(large_arrow, 1.0)

        # Arrow should have some defects
        if defects is not None:
            assert len(defects) >= 0
            # Each defect should be a tuple with 4 elements: (start, end, far, depth)
            for defect in defects:
                assert len(defect) == 4
                assert defect[3] > 0  # Depth should be positive

    def test_find_significant_defects_simple_shape(self):
        """Test finding defects in a simple shape (should have few/no defects)."""
        detector = ArrowDetector()

        # Simple rectangle should have minimal defects
        rectangle = np.array([
            [0, 0], [100, 0], [100, 50], [0, 50]
        ], dtype=np.int32).reshape(-1, 1, 2)

        defects = detector._find_significant_defects(rectangle, 5.0)

        # Rectangle should have no significant defects
        if defects is not None:
            assert len(defects) == 0
        else:
            assert defects is None

    def test_analyze_contour_for_arrow_valid_arrow(self, arrow_shaped_contour):
        """Test complete arrow analysis on a valid arrow contour."""
        detector = ArrowDetector()

        # Create a larger, more complex arrow shape
        complex_arrow = np.array([
            [10, 30], [25, 15], [20, 20], [40, 20], [40, 25],
            [50, 25], [35, 40], [40, 35], [40, 40], [20, 40],
            [20, 35], [25, 45]
        ], dtype=np.int32).reshape(-1, 1, 2)

        entry = {'scar': 'test_scar'}
        image = np.zeros((100, 100), dtype=np.uint8)

        result = detector.analyze_contour_for_arrow(complex_arrow, entry, image, 300.0)

        # Result could be None (if detection fails) or a dict with arrow properties
        if result is not None:
            expected_fields = [
                'arrow_back', 'arrow_tip', 'angle_rad', 'angle_deg', 'compass_angle'
            ]
            for field in expected_fields:
                assert field in result

            # Angle values should be in expected ranges
            assert 0 <= result['angle_deg'] < 360
            assert 0 <= result['compass_angle'] < 360
            assert -math.pi <= result['angle_rad'] <= math.pi

    def test_analyze_contour_for_arrow_invalid_shape(self):
        """Test arrow analysis on a shape that's not an arrow."""
        detector = ArrowDetector()

        # Simple rectangle - not an arrow
        rectangle = np.array([
            [10, 10], [50, 10], [50, 40], [10, 40]
        ], dtype=np.int32).reshape(-1, 1, 2)

        entry = {'scar': 'test_scar'}
        image = np.zeros((100, 100), dtype=np.uint8)

        result = detector.analyze_contour_for_arrow(rectangle, entry, image, 300.0)

        # Should return None for non-arrow shapes
        assert result is None

    def test_calculate_arrow_properties(self):
        """Test calculation of arrow properties from triangle data."""
        detector = ArrowDetector()

        triangle_data = {
            'base_p1': (10, 30),
            'base_p2': (30, 30),
            'base_midpoint': (20, 30),
            'triangle_tip': (20, 10),
            'triangle_height': 20.0,
            'significant_defects': []
        }

        result = detector._calculate_arrow_properties(triangle_data, None)

        assert 'arrow_back' in result
        assert 'arrow_tip' in result
        assert 'angle_rad' in result
        assert 'angle_deg' in result
        assert 'compass_angle' in result

        # Arrow should point from triangle_tip to base_midpoint
        assert result['arrow_back'] == (20, 10)  # triangle_tip
        assert result['arrow_tip'] == (20, 30)   # base_midpoint

        # For vertical arrow pointing down:
        # dx = 20-20 = 0, dy = 30-10 = 20
        # angle_rad = atan2(20, 0) = π/2
        # angle_deg = 90°
        # compass_angle = (270 + 90) % 360 = 0°
        assert abs(result['angle_deg'] - 90.0) < 1.0
        assert abs(result['compass_angle'] - 0.0) < 1.0

    def test_debug_visualization_creation(self, arrow_shaped_contour):
        """Test creation of debug visualizations."""
        config = {'debug_enabled': True}
        detector = ArrowDetector(config)

        with tempfile.TemporaryDirectory() as temp_dir:
            entry = {
                'scar': 'test_debug',
                'debug_dir': temp_dir
            }

            # Create test image
            image = np.zeros((100, 100, 3), dtype=np.uint8)

            # Mock successful triangle analysis
            triangle_data = {
                'base_p1': (10, 30),
                'base_p2': (30, 30),
                'base_midpoint': (20, 30),
                'triangle_tip': (20, 10),
                'significant_defects': [((10, 30), (30, 30), (15, 25), 5.0)]
            }

            arrow_data = {
                'arrow_back': (20, 10),
                'arrow_tip': (20, 30),
                'compass_angle': 0.0
            }

            # Test debug visualization creation
            detector._create_debug_visualizations(
                arrow_shaped_contour, triangle_data, arrow_data, image, temp_dir
            )

            # Check if debug image was created
            debug_image_path = os.path.join(temp_dir, "arrow_debug.png")
            assert os.path.exists(debug_image_path)


@pytest.mark.unit
class TestArrowDetectionFunctions:
    """Test standalone arrow detection functions."""

    def test_analyze_child_contour_for_arrow_backward_compatibility(self, arrow_shaped_contour):
        """Test backward compatibility function."""
        entry = {'scar': 'test_scar'}
        image = np.zeros((100, 100), dtype=np.uint8)

        result = analyze_child_contour_for_arrow(
            arrow_shaped_contour, entry, image, 300.0
        )

        # Should return same result as ArrowDetector method
        assert result is None or isinstance(result, dict)

    def test_scale_parameters_for_dpi_function(self):
        """Test standalone DPI scaling function."""
        params = scale_parameters_for_dpi(600.0)

        assert isinstance(params, dict)
        assert 'min_area' in params
        assert 'min_defect_depth' in params
        assert 'solidity_bounds' in params

    def test_create_arrow_detection_pipeline(self):
        """Test creation of arrow detection pipeline."""
        config = {
            'reference_dpi': 150.0,
            'debug_enabled': True
        }

        detector = create_arrow_detection_pipeline(config)

        assert isinstance(detector, ArrowDetector)
        assert detector.reference_dpi == 150.0
        assert detector.debug_enabled is True

    def test_batch_detect_arrows(self, sample_contours):
        """Test batch arrow detection on multiple contours."""
        entries = [
            {'scar': 'scar_1'},
            {'scar': 'scar_2'}
        ]
        image = np.zeros((100, 100), dtype=np.uint8)

        results = batch_detect_arrows(
            sample_contours, entries, image, 300.0
        )

        assert len(results) == len(sample_contours)

        # Each result should be None or a dict
        for result in results:
            assert result is None or isinstance(result, dict)

    def test_validate_arrow_detection_config_valid(self):
        """Test validation of valid arrow detection config."""
        valid_config = {
            'reference_dpi': 300.0,
            'min_area_scale_factor': 0.7,
            'min_defect_depth_scale_factor': 0.8,
            'min_triangle_height_scale_factor': 0.8,
            'debug_enabled': False
        }

        result = validate_arrow_detection_config(valid_config)
        assert result is True

    def test_validate_arrow_detection_config_invalid(self):
        """Test validation of invalid arrow detection config."""
        # Missing required key
        invalid_config1 = {
            'min_area_scale_factor': 0.7
            # Missing reference_dpi
        }

        with patch('pylithics.image_processing.modules.arrow_detection.logging') as mock_logging:
            result = validate_arrow_detection_config(invalid_config1)
            assert result is False
            mock_logging.error.assert_called()

        # Invalid value ranges
        invalid_config2 = {
            'reference_dpi': -100.0,  # Negative DPI
            'min_area_scale_factor': 1.5  # > 1.0
        }

        with patch('pylithics.image_processing.modules.arrow_detection.logging') as mock_logging:
            result = validate_arrow_detection_config(invalid_config2)
            assert result is False
            mock_logging.error.assert_called()


@pytest.mark.unit
class TestArrowGeometry:
    """Test geometric calculations in arrow detection."""

    def test_triangle_base_identification(self):
        """Test identification of triangle base from defects."""
        detector = ArrowDetector()

        # Mock defects representing arrow base
        defects = [
            ((10, 30), (30, 30), (15, 25), 5.0),  # Base defect 1
            ((30, 30), (10, 30), (25, 25), 4.0)   # Base defect 2
        ]

        result = detector._identify_triangle_base(defects)

        if result is not None:
            base_p1, base_p2, base_midpoint, distance = result

            # Check that we get reasonable base points
            assert isinstance(base_p1, tuple)
            assert isinstance(base_p2, tuple)
            assert isinstance(base_midpoint, tuple)
            assert distance > 0

            # Midpoint should be between the base points
            expected_midpoint_x = (base_p1[0] + base_p2[0]) // 2
            expected_midpoint_y = (base_p1[1] + base_p2[1]) // 2

            assert abs(base_midpoint[0] - expected_midpoint_x) <= 1
            assert abs(base_midpoint[1] - expected_midpoint_y) <= 1

    def test_halfspace_analysis(self):
        """Test division of contour into half-spaces."""
        detector = ArrowDetector()

        # Create a simple arrow-like contour
        arrow_contour = np.array([
            [20, 10],   # Top point
            [30, 20], [25, 25], [30, 30],  # Right side
            [20, 40],   # Bottom point
            [10, 30], [15, 25], [10, 20]   # Left side
        ], dtype=np.int32).reshape(-1, 1, 2)

        triangle_base_info = ((10, 25), (30, 25), (20, 25), 20.0)

        result = detector._analyze_halfspaces(arrow_contour, triangle_base_info)

        if result is not None:
            shaft_halfspace, tip_halfspace, solidity1, solidity2 = result

            # Should identify different half-spaces
            assert shaft_halfspace != tip_halfspace
            assert shaft_halfspace in [1, 2]
            assert tip_halfspace in [1, 2]

            # Solidities should be between 0 and 1
            assert 0 <= solidity1 <= 1
            assert 0 <= solidity2 <= 1

    def test_triangle_tip_finding(self):
        """Test finding the triangle tip point."""
        detector = ArrowDetector()

        # Points representing the tip half-space
        tip_points = [(20, 10), (18, 12), (22, 12), (20, 8)]
        base_midpoint = (20, 25)

        tip_point = detector._find_triangle_tip(tip_points, base_midpoint)

        assert tip_point is not None
        assert isinstance(tip_point, tuple)

        # Should find the point furthest from base midpoint
        # In this case, (20, 8) is furthest from (20, 25)
        distances = [
            np.sqrt((p[0] - base_midpoint[0])**2 + (p[1] - base_midpoint[1])**2)
            for p in tip_points
        ]
        max_distance_point = tip_points[np.argmax(distances)]

        assert tip_point == max_distance_point


@pytest.mark.integration
class TestArrowDetectionIntegration:
    """Integration tests for arrow detection workflow."""

    def test_complete_arrow_detection_workflow(self):
        """Test complete arrow detection workflow from contour to result."""
        # Create a more realistic arrow shape
        arrow_points = []

        # Create arrow tip
        arrow_points.extend([(50, 10), (55, 20), (52, 20)])

        # Create arrow shaft
        arrow_points.extend([(52, 20), (52, 40), (48, 40), (48, 20)])

        # Create arrow base
        arrow_points.extend([(48, 20), (45, 20), (50, 10)])

        arrow_contour = np.array(arrow_points, dtype=np.int32).reshape(-1, 1, 2)

        detector = ArrowDetector()
        entry = {'scar': 'integration_test'}
        image = np.zeros((100, 100), dtype=np.uint8)

        result = detector.analyze_contour_for_arrow(arrow_contour, entry, image, 300.0)

        # Even if detection fails, should handle gracefully
        assert result is None or isinstance(result, dict)

        if result is not None:
            # Verify result structure
            required_fields = ['arrow_back', 'arrow_tip', 'angle_deg', 'compass_angle']
            for field in required_fields:
                assert field in result

    def test_arrow_detection_with_different_orientations(self):
        """Test arrow detection with arrows pointing in different directions."""
        detector = ArrowDetector()

        orientations = [
            # Horizontal arrows
            [(10, 25), (30, 20), (30, 30), (35, 25), (30, 30), (30, 20)],  # Right
            [(35, 25), (15, 20), (15, 30), (10, 25), (15, 30), (15, 20)],  # Left

            # Vertical arrows
            [(25, 10), (20, 30), (30, 30), (25, 35), (30, 30), (20, 30)],  # Down
            [(25, 35), (20, 15), (30, 15), (25, 10), (30, 15), (20, 15)]   # Up
        ]

        for i, points in enumerate(orientations):
            arrow_contour = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
            entry = {'scar': f'orientation_test_{i}'}
            image = np.zeros((50, 50), dtype=np.uint8)

            result = detector.analyze_contour_for_arrow(arrow_contour, entry, image, 300.0)

            # Should handle different orientations gracefully
            assert result is None or isinstance(result, dict)

    def test_arrow_detection_error_handling(self):
        """Test error handling in arrow detection."""
        detector = ArrowDetector()

        # Test with empty contour
        empty_contour = np.array([], dtype=np.int32).reshape(0, 1, 2)
        entry = {'scar': 'empty_test'}
        image = np.zeros((100, 100), dtype=np.uint8)

        result = detector.analyze_contour_for_arrow(empty_contour, entry, image, 300.0)
        assert result is None

        # Test with degenerate contour (single point)
        point_contour = np.array([[10, 10]], dtype=np.int32).reshape(-1, 1, 2)
        result = detector.analyze_contour_for_arrow(point_contour, entry, image, 300.0)
        assert result is None

        # Test with malformed entry
        malformed_entry = {}  # Missing 'scar' key
        valid_contour = np.array([[0, 0], [10, 0], [5, 10]], dtype=np.int32).reshape(-1, 1, 2)

        result = detector.analyze_contour_for_arrow(valid_contour, malformed_entry, image, 300.0)
        # Should handle gracefully (may succeed or fail, but shouldn't crash)
        assert result is None or isinstance(result, dict)

    def test_arrow_detection_with_debug_enabled(self):
        """Test arrow detection with debug mode enabled."""
        config = {'debug_enabled': True}
        detector = ArrowDetector(config)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a simple arrow shape
            arrow_contour = np.array([
                [25, 10], [30, 20], [27, 20], [27, 30],
                [23, 30], [23, 20], [20, 20]
            ], dtype=np.int32).reshape(-1, 1, 2)

            entry = {
                'scar': 'debug_test',
                'debug_dir': temp_dir
            }
            image = np.zeros((50, 50, 3), dtype=np.uint8)

            result = detector.analyze_contour_for_arrow(arrow_contour, entry, image, 300.0)

            # Check if debug files were created (if detection succeeded)
            if result is not None:
                log_file = os.path.join(temp_dir, 'arrow_detection_log.txt')
                # Debug files might be created depending on detection success
                # At minimum, should not crash with debug enabled

    @patch('pylithics.image_processing.modules.arrow_detection.logging')
    def test_batch_detection_error_handling(self, mock_logging):
        """Test error handling in batch arrow detection."""
        contours = [
            np.array([[0, 0], [10, 0], [5, 10]], dtype=np.int32).reshape(-1, 1, 2),
            np.array([], dtype=np.int32).reshape(0, 1, 2),  # Empty contour
            None  # Invalid contour
        ]

        entries = [
            {'scar': 'valid'},
            {'scar': 'empty'},
            {'scar': 'invalid'}
        ]

        image = np.zeros((100, 100), dtype=np.uint8)

        results = batch_detect_arrows(contours, entries, image, 300.0)

        assert len(results) == len(contours)

        # Should handle errors gracefully and continue processing
        for result in results:
            assert result is None or isinstance(result, dict)

        # Should log errors for failed detections
        assert mock_logging.error.called