"""
PyLithics Arrow Integration Tests
=================================

Tests for arrow detection pipeline integration including process_nested_arrows,
detect_arrows_independently, and integrate_arrows orchestrator functions.
"""

import pytest
import numpy as np
import cv2
import tempfile
import os
from unittest.mock import patch, MagicMock

from pylithics.image_processing.modules.arrow_integration import (
    process_nested_arrows,
    detect_arrows_independently,
    integrate_arrows
)


@pytest.mark.unit
class TestProcessNestedArrows:
    """Test the nested arrow processing function."""

    def test_process_nested_arrows_with_valid_input(self, sample_contours, sample_hierarchy):
        """Test nested arrow processing with valid contours and hierarchy."""
        image = np.zeros((200, 200), dtype=np.uint8)
        dpi = 300.0

        # Create proper sorted_contours dictionary structure
        sorted_contours = {
            "parents": [sample_contours[0]],  # First contour is parent
            "children": [sample_contours[1]] if len(sample_contours) > 1 else [],  # Second is child
            "nested_children": []  # No nested children in this test
        }

        # Create nested structure metrics
        metrics = [
            {
                'parent': 'parent 1',
                'scar': 'parent 1',
                'surface_type': 'Dorsal',
                'area': cv2.contourArea(sample_contours[0])  # Add area field
            },
            {
                'parent': 'parent 1',
                'scar': 'scar 1',
                'surface_type': 'Dorsal',
                'area': cv2.contourArea(sample_contours[1]) if len(sample_contours) > 1 else 0
            }
        ]

        with patch('pylithics.image_processing.modules.arrow_integration.analyze_child_contour_for_arrow') as mock_analyze:
            mock_analyze.return_value = {
                'arrow_back': (20, 25),
                'arrow_tip': (30, 35),
                'angle_deg': 45.0,
                'compass_angle': 45.0
            }

            result = process_nested_arrows(sorted_contours, sample_hierarchy,
                                        original_contours=sample_contours,
                                        metrics=metrics, image_shape=image, image_dpi=dpi)

            assert isinstance(result, list)

    def test_process_nested_arrows_no_children(self):
        """Test nested arrow processing when no child contours exist."""
        # Only parent contours (no children)
        parent_contour = np.array([
            [20, 20], [80, 20], [80, 80], [20, 80]
        ], dtype=np.int32).reshape(-1, 1, 2)

        sorted_contours = {"parents": [parent_contour], "children": [], "nested_children": []}
        hierarchy = np.array([[-1, -1, -1, -1]])  # No children
        metrics = [{'parent': 'parent 1', 'scar': 'parent 1', 'surface_type': 'Dorsal', 'area': cv2.contourArea(parent_contour)}]
        image = np.zeros((100, 100), dtype=np.uint8)

        result = process_nested_arrows(sorted_contours, hierarchy, [parent_contour], metrics, image, 300.0)

        assert isinstance(result, list)

    def test_process_nested_arrows_hierarchy_mismatch(self):
        """Test nested arrow processing with mismatched hierarchy data."""
        contour = np.array([
            [20, 20], [40, 20], [40, 40], [20, 40]
        ], dtype=np.int32).reshape(-1, 1, 2)

        sorted_contours = {"parents": [contour], "children": [], "nested_children": []}
        hierarchy = None  # Mismatched hierarchy
        metrics = [{'parent': 'parent 1', 'scar': 'scar 1', 'surface_type': 'Dorsal', 'area': cv2.contourArea(contour)}]
        image = np.zeros((100, 100), dtype=np.uint8)

        with patch('pylithics.image_processing.modules.arrow_integration.logging') as mock_logging:
            result = process_nested_arrows(sorted_contours, hierarchy, [contour], metrics, image, 300.0)

            # Should handle gracefully and return appropriate result
            assert isinstance(result, list)


@pytest.mark.unit
class TestDetectArrowsIndependently:
    """Test the independent arrow detection function."""

    def test_detect_arrows_independently_basic(self, sample_contours):
        """Test independent arrow detection with basic contours."""
        metrics = [
            {
                'parent': 'parent 1',
                'scar': 'parent 1',
                'surface_type': 'Dorsal',
                'area': cv2.contourArea(sample_contours[0])  # Add area
            },
            {
                'parent': 'parent 1',
                'scar': 'scar 1',
                'surface_type': 'Dorsal',
                'area': cv2.contourArea(sample_contours[1]) if len(sample_contours) > 1 else 100
            }
        ]
        image = np.zeros((200, 200), dtype=np.uint8)

        with patch('pylithics.image_processing.modules.arrow_integration.analyze_child_contour_for_arrow') as mock_analyze:
            mock_analyze.side_effect = [
                None,  # No arrow in parent
                {      # Arrow in child
                    'arrow_back': (30, 35),
                    'arrow_tip': (40, 45),
                    'angle_deg': 90.0,
                    'compass_angle': 90.0
                }
            ]

            result = detect_arrows_independently(sample_contours, metrics, image, 300.0)
            assert isinstance(result, list)

    def test_detect_arrows_independently_all_fail(self, sample_contours):
        """Test independent arrow detection when all detections fail."""
        metrics = [
            {
                'parent': 'parent 1',
                'scar': 'parent 1',
                'surface_type': 'Dorsal',
                'area': cv2.contourArea(sample_contours[0])
            },
            {
                'parent': 'parent 1',
                'scar': 'scar 1',
                'surface_type': 'Dorsal',
                'area': cv2.contourArea(sample_contours[1]) if len(sample_contours) > 1 else 100
            }
        ]
        image = np.zeros((200, 200), dtype=np.uint8)

        with patch('pylithics.image_processing.modules.arrow_integration.analyze_child_contour_for_arrow') as mock_analyze:
            mock_analyze.return_value = None

            result = detect_arrows_independently(sample_contours, metrics, image, 300.0)
            assert isinstance(result, list)

    def test_detect_arrows_independently_empty_input(self):
        """Test independent arrow detection with empty input."""
        result = detect_arrows_independently([], [], np.zeros((100, 100), dtype=np.uint8), 300.0)

        assert isinstance(result, list)
        assert len(result) == 0

    def test_detect_arrows_independently_mismatched_lengths(self, sample_contours):
        """Test independent arrow detection with mismatched contours and metrics."""
        # More contours than metrics
        metrics = [{
            'parent': 'parent 1',
            'scar': 'parent 1',
            'surface_type': 'Dorsal',
            'area': cv2.contourArea(sample_contours[0])
        }]
        image = np.zeros((200, 200), dtype=np.uint8)

        with patch('pylithics.image_processing.modules.arrow_integration.logging') as mock_logging:
            result = detect_arrows_independently(sample_contours, metrics, image, 300.0)

            # Should handle mismatch gracefully
            assert isinstance(result, list)


@pytest.mark.unit
class TestIntegrateArrows:
    """Test the main arrow integration orchestrator function."""

    def test_integrate_arrows_hierarchy_based(self, sample_contours, sample_hierarchy):
        """Test arrow integration using hierarchy-based approach."""
        # Create proper sorted_contours structure
        sorted_contours = {
            "parents": [sample_contours[0]],
            "children": [sample_contours[1]] if len(sample_contours) > 1 else [],
            "nested_children": []
        }

        metrics = [
            {
                'parent': 'parent 1',
                'scar': 'parent 1',
                'surface_type': 'Dorsal',
                'area': cv2.contourArea(sample_contours[0])
            },
            {
                'parent': 'parent 1',
                'scar': 'scar 1',
                'surface_type': 'Dorsal',
                'area': cv2.contourArea(sample_contours[1]) if len(sample_contours) > 1 else 100
            }
        ]
        image = np.zeros((200, 200), dtype=np.uint8)

        with patch('pylithics.image_processing.modules.arrow_integration.process_nested_arrows') as mock_nested:
            mock_nested.return_value = metrics

            result = integrate_arrows(sorted_contours, sample_hierarchy,
                                    original_contours=sample_contours,
                                    metrics=metrics, image_shape=image, image_dpi=300.0)

            assert isinstance(result, list)
            assert len(result) == len(metrics)
            mock_nested.assert_called_once()

            # Verify result structure
            for metric in result:
                assert isinstance(metric, dict)

    def test_integrate_arrows_independent_fallback(self, sample_contours):
        """Test arrow integration fallback to independent approach."""
        # Create proper sorted_contours structure
        sorted_contours = {
            "parents": [sample_contours[0]],
            "children": [sample_contours[1]] if len(sample_contours) > 1 else [],
            "nested_children": []
        }

        metrics = [
            {
                'parent': 'parent 1',
                'scar': 'parent 1',
                'surface_type': 'Dorsal',
                'area': cv2.contourArea(sample_contours[0])
            },
            {
                'parent': 'parent 1',
                'scar': 'scar 1',
                'surface_type': 'Dorsal',
                'area': cv2.contourArea(sample_contours[1]) if len(sample_contours) > 1 else 100
            }
        ]
        image = np.zeros((200, 200), dtype=np.uint8)
        hierarchy = None  # Force fallback

        with patch('pylithics.image_processing.modules.arrow_integration.process_nested_arrows') as mock_nested:
            mock_nested.return_value = metrics

            result = integrate_arrows(sorted_contours, hierarchy,
                                    original_contours=sample_contours,
                                    metrics=metrics, image_shape=image, image_dpi=300.0)

            assert isinstance(result, list)
            assert len(result) == len(metrics)

    def test_integrate_arrows_empty_input(self):
        """Test arrow integration with empty input."""
        sorted_contours = {
            "parents": [],
            "children": [],
            "nested_children": []
        }

        result = integrate_arrows(sorted_contours, None, original_contours=[],
                                metrics=[], image_shape=np.zeros((100, 100), dtype=np.uint8),
                                image_dpi=300.0)

        assert isinstance(result, list)
        assert len(result) == 0

    def test_integrate_arrows_config_propagation(self, sample_contours, sample_hierarchy):
        """Test that configuration is properly propagated through integration."""
        sorted_contours = {
            "parents": [sample_contours[0]],
            "children": [sample_contours[1]] if len(sample_contours) > 1 else [],
            "nested_children": []
        }

        metrics = [
            {
                'parent': 'parent 1',
                'scar': 'parent 1',
                'surface_type': 'Dorsal',
                'area': cv2.contourArea(sample_contours[0])
            },
            {
                'parent': 'parent 1',
                'scar': 'scar 1',
                'surface_type': 'Dorsal',
                'area': cv2.contourArea(sample_contours[1]) if len(sample_contours) > 1 else 100
            }
        ]
        image = np.zeros((200, 200), dtype=np.uint8)

        with patch('pylithics.image_processing.modules.arrow_integration.process_nested_arrows') as mock_nested:
            mock_nested.return_value = metrics

            integrate_arrows(sorted_contours, sample_hierarchy,
                           original_contours=sample_contours,
                           metrics=metrics, image_shape=image, image_dpi=300.0)

            mock_nested.assert_called_once()


@pytest.mark.integration
class TestArrowIntegrationWorkflow:
    """Integration tests for complete arrow integration workflow."""

    def test_complete_arrow_integration_workflow(self, sample_contours, sample_hierarchy):
        """Test complete arrow integration workflow from contours to final metrics."""
        # Create proper sorted_contours structure
        sorted_contours = {
            "parents": [sample_contours[0]],
            "children": [sample_contours[1]] if len(sample_contours) > 1 else [],
            "nested_children": []
        }

        # Create realistic metrics for archaeological artifact
        metrics = [
            {
                'parent': 'parent 1',
                'scar': 'parent 1',
                'surface_type': 'Dorsal',
                'centroid_x': 100.0,
                'centroid_y': 150.0,
                'technical_width': 120.0,
                'technical_length': 180.0,
                'area': 21600.0
            },
            {
                'parent': 'parent 1',
                'scar': 'scar 1',
                'surface_type': 'Dorsal',
                'centroid_x': 80.0,
                'centroid_y': 130.0,
                'width': 25.0,
                'height': 35.0,
                'area': 875.0
            }
        ]

        image = np.zeros((250, 200), dtype=np.uint8)

        # Test arrow integration
        result = integrate_arrows(sorted_contours, sample_hierarchy,
                                original_contours=sample_contours,
                                metrics=metrics, image_shape=image, image_dpi=300.0)

        assert isinstance(result, list)
        assert len(result) >= 1  # Should return some results

        # Verify structure of results
        for metric in result:
            assert isinstance(metric, dict)
            # Basic structure should be preserved
            assert 'parent' in metric
            assert 'scar' in metric

    def test_arrow_integration_with_realistic_scars(self):
        """Test arrow integration with realistic removal scar patterns."""
        # Create blade tool with systematic removal scars
        contours = []
        metrics = []

        # Main blade body
        blade_contour = np.array([
            [50, 30], [250, 35], [260, 50], [255, 180],
            [250, 190], [50, 185], [40, 175], [45, 50]
        ], dtype=np.int32).reshape(-1, 1, 2)
        contours.append(blade_contour)

        metrics.append({
            'parent': 'parent 1',
            'scar': 'parent 1',
            'surface_type': 'Dorsal',
            'centroid_x': 150.0,
            'centroid_y': 110.0,
            'technical_width': 220.0,
            'technical_length': 160.0,
            'area': 35200.0
        })

        # Add removal scars
        scar_positions = [
            (80, 80, 20, 30),   # x, y, width, height
            (120, 90, 25, 35),
            (180, 85, 22, 32),
            (220, 95, 18, 28)
        ]

        hierarchy_data = [[-1, -1, 1, -1]]  # Parent

        for i, (x, y, w, h) in enumerate(scar_positions):
            # Create scar contour
            scar_contour = np.array([
                [x, y], [x + w, y], [x + w, y + h], [x, y + h]
            ], dtype=np.int32).reshape(-1, 1, 2)
            contours.append(scar_contour)

            # Create scar metric
            metrics.append({
                'parent': 'parent 1',
                'scar': f'scar {i + 1}',
                'surface_type': 'Dorsal',
                'centroid_x': float(x + w // 2),
                'centroid_y': float(y + h // 2),
                'width': float(w),
                'height': float(h),
                'area': float(w * h)
            })

            # Add to hierarchy (all children of parent)
            next_sib = i + 2 if i < len(scar_positions) - 1 else -1
            prev_sib = i if i > 0 else -1
            hierarchy_data.append([next_sib, prev_sib, -1, 0])

        hierarchy = np.array(hierarchy_data)
        image = np.zeros((220, 300), dtype=np.uint8)

        # Create sorted_contours structure
        sorted_contours = {
            "parents": [blade_contour],
            "children": contours[1:],  # All scars are children
            "nested_children": []
        }

        # Test integration
        result = integrate_arrows(sorted_contours, hierarchy, original_contours=contours,
                                metrics=metrics, image_shape=image, image_dpi=300.0)

        assert len(result) >= 1  # Should process some results

        # Check basic structure is preserved
        for metric in result:
            assert isinstance(metric, dict)
            assert 'parent' in metric
            assert 'scar' in metric

    def test_arrow_integration_error_recovery(self):
        """Test arrow integration error recovery and robustness."""
        # Create problematic input that might cause errors
        contours = [
            np.array([[50, 50], [51, 50]], dtype=np.int32).reshape(-1, 1, 2),  # Degenerate contour
        ]

        sorted_contours = {
            "parents": contours,
            "children": [],
            "nested_children": []
        }

        metrics = [
            {'parent': 'parent 1', 'scar': 'parent 1', 'surface_type': 'Dorsal', 'area': 1.0},
        ]

        hierarchy = np.array([
            [-1, -1, -1, -1]
        ])

        image = np.zeros((100, 100), dtype=np.uint8)

        with patch('pylithics.image_processing.modules.arrow_integration.logging'):
            # Should handle errors gracefully and not crash
            result = integrate_arrows(sorted_contours, hierarchy, original_contours=contours,
                                    metrics=metrics, image_shape=image, image_dpi=300.0)

            # Should return some result even with problematic input
            assert isinstance(result, list)

            # Each result should be a valid metric dict
            for metric in result:
                assert isinstance(metric, dict)

    def test_arrow_integration_performance_many_scars(self):
        """Test arrow integration performance with many scars."""
        # Create many contours and metrics
        contours = []
        metrics = []
        hierarchy_data = []

        # Parent contour
        parent_contour = np.array([
            [50, 50], [450, 50], [450, 450], [50, 450]
        ], dtype=np.int32).reshape(-1, 1, 2)
        contours.append(parent_contour)

        metrics.append({
            'parent': 'parent 1',
            'scar': 'parent 1',
            'surface_type': 'Dorsal',
            'centroid_x': 250.0,
            'centroid_y': 250.0,
            'technical_width': 400.0,
            'technical_length': 400.0,
            'area': 160000.0
        })

        hierarchy_data.append([-1, -1, 1, -1])  # Parent

        # Add many child scars
        num_scars = 20  # Reduced for performance testing
        scar_contours = []
        for i in range(num_scars):
            x = 100 + (i % 10) * 30
            y = 100 + (i // 10) * 30

            scar_contour = np.array([
                [x, y], [x + 20, y], [x + 20, y + 20], [x, y + 20]
            ], dtype=np.int32).reshape(-1, 1, 2)
            contours.append(scar_contour)
            scar_contours.append(scar_contour)

            metrics.append({
                'parent': 'parent 1',
                'scar': f'scar {i + 1}',
                'surface_type': 'Dorsal',
                'centroid_x': float(x + 10),
                'centroid_y': float(y + 10),
                'width': 20.0,
                'height': 20.0,
                'area': 400.0
            })

            # Add to hierarchy
            next_sib = i + 2 if i < num_scars - 1 else -1
            prev_sib = i if i > 0 else -1
            hierarchy_data.append([next_sib, prev_sib, -1, 0])

        hierarchy = np.array(hierarchy_data)
        image = np.zeros((500, 500), dtype=np.uint8)

        # Create sorted_contours structure
        sorted_contours = {
            "parents": [parent_contour],
            "children": scar_contours,
            "nested_children": []
        }

        import time
        start_time = time.time()

        result = integrate_arrows(sorted_contours, hierarchy, original_contours=contours,
                                metrics=metrics, image_shape=image, image_dpi=300.0)

        end_time = time.time()
        processing_time = end_time - start_time

        # Should complete in reasonable time even with many scars
        assert processing_time < 30.0  # 30 seconds max
        assert isinstance(result, list)


@pytest.mark.integration
class TestArrowIntegrationRealWorldScenarios:
    """Test arrow integration with realistic archaeological scenarios."""

    def test_integration_lithic_core_scenario(self):
        """Test arrow integration with lithic core reduction scenario."""
        # Create core with radial flake removal pattern
        center_x, center_y = 150, 150

        # Core contour
        core_contour = np.array([
            [100, 100], [200, 100], [200, 200], [100, 200]
        ], dtype=np.int32).reshape(-1, 1, 2)

        contours = [core_contour]
        metrics = [{
            'parent': 'parent 1',
            'scar': 'parent 1',
            'surface_type': 'Dorsal',
            'centroid_x': 150.0,
            'centroid_y': 150.0,
            'technical_width': 100.0,
            'technical_length': 100.0,
            'area': 10000.0
        }]

        hierarchy_data = [[-1, -1, 1, -1]]  # Core

        # Add radial flake scars
        num_flakes = 6  # Reduced for simplicity
        flake_contours = []
        for i in range(num_flakes):
            angle = 2 * np.pi * i / num_flakes
            radius = 40
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)

            # Create flake scar
            flake_contour = np.array([
                [x - 8, y - 8], [x + 8, y - 8], [x + 8, y + 8], [x - 8, y + 8]
            ], dtype=np.int32).reshape(-1, 1, 2)
            contours.append(flake_contour)
            flake_contours.append(flake_contour)

            metrics.append({
                'parent': 'parent 1',
                'scar': f'flake {i + 1}',
                'surface_type': 'Dorsal',
                'centroid_x': x,
                'centroid_y': y,
                'width': 16.0,
                'height': 16.0,
                'area': 256.0
            })

            # Add to hierarchy
            next_sib = i + 2 if i < num_flakes - 1 else -1
            prev_sib = i if i > 0 else -1
            hierarchy_data.append([next_sib, prev_sib, -1, 0])

        hierarchy = np.array(hierarchy_data)
        image = np.zeros((300, 300), dtype=np.uint8)

        # Create sorted_contours structure
        sorted_contours = {
            "parents": [core_contour],
            "children": flake_contours,
            "nested_children": []
        }

        result = integrate_arrows(sorted_contours, hierarchy, original_contours=contours,
                                metrics=metrics, image_shape=image, image_dpi=300.0)

        assert len(result) >= 1

        # Basic structure should be preserved
        for flake_result in result:
            assert isinstance(flake_result, dict)
            assert 'parent' in flake_result
            assert 'scar' in flake_result