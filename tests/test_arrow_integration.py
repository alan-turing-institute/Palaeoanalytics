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
    integrate_arrows,
    _assign_arrows_to_scars,
    _resolve_arrow_conflicts,
    _update_metrics_with_arrows
)


@pytest.mark.unit
class TestProcessNestedArrows:
    """Test the nested arrow processing function."""

    def test_process_nested_arrows_with_valid_input(self, sample_contours, sample_hierarchy):
        """Test nested arrow processing with valid contours and hierarchy."""
        image = np.zeros((200, 200), dtype=np.uint8)
        dpi = 300.0

        # Create nested structure metrics
        metrics = [
            {
                'parent': 'parent 1',
                'scar': 'parent 1',
                'surface_type': 'Dorsal'
            },
            {
                'parent': 'parent 1',
                'scar': 'scar 1',
                'surface_type': 'Dorsal'
            }
        ]

        with patch('pylithics.image_processing.modules.arrow_integration.ArrowDetector') as mock_detector:
            mock_instance = MagicMock()
            mock_instance.analyze_contour_for_arrow.return_value = {
                'arrow_back': (20, 25),
                'arrow_tip': (30, 35),
                'angle_deg': 45.0,
                'compass_angle': 45.0
            }
            mock_detector.return_value = mock_instance

            result = process_nested_arrows(sample_contours, sample_hierarchy, metrics, image, dpi)

            assert isinstance(result, list)
            assert len(result) == len(sample_contours)

            # Check that arrow detection was called for child contours
            mock_instance.analyze_contour_for_arrow.assert_called()

    def test_process_nested_arrows_no_children(self):
        """Test nested arrow processing when no child contours exist."""
        # Only parent contours (no children)
        parent_contour = np.array([
            [20, 20], [80, 20], [80, 80], [20, 80]
        ], dtype=np.int32).reshape(-1, 1, 2)

        contours = [parent_contour]
        hierarchy = np.array([[-1, -1, -1, -1]])  # No children
        metrics = [{'parent': 'parent 1', 'scar': 'parent 1', 'surface_type': 'Dorsal'}]
        image = np.zeros((100, 100), dtype=np.uint8)

        result = process_nested_arrows(contours, hierarchy, metrics, image, 300.0)

        # Should return list of None values (no arrows found)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] is None

    def test_process_nested_arrows_hierarchy_mismatch(self):
        """Test nested arrow processing with mismatched hierarchy data."""
        contour = np.array([
            [20, 20], [40, 20], [40, 40], [20, 40]
        ], dtype=np.int32).reshape(-1, 1, 2)

        contours = [contour]
        hierarchy = None  # Mismatched hierarchy
        metrics = [{'parent': 'parent 1', 'scar': 'scar 1', 'surface_type': 'Dorsal'}]
        image = np.zeros((100, 100), dtype=np.uint8)

        with patch('pylithics.image_processing.modules.arrow_integration.logging') as mock_logging:
            result = process_nested_arrows(contours, hierarchy, metrics, image, 300.0)

            # Should handle gracefully and return appropriate result
            assert isinstance(result, list)
            mock_logging.warning.assert_called()

    def test_process_nested_arrows_detection_failure(self, sample_contours, sample_hierarchy):
        """Test nested arrow processing when arrow detection fails."""
        metrics = [
            {'parent': 'parent 1', 'scar': 'parent 1', 'surface_type': 'Dorsal'},
            {'parent': 'parent 1', 'scar': 'scar 1', 'surface_type': 'Dorsal'}
        ]
        image = np.zeros((200, 200), dtype=np.uint8)

        with patch('pylithics.image_processing.modules.arrow_integration.ArrowDetector') as mock_detector:
            mock_instance = MagicMock()
            mock_instance.analyze_contour_for_arrow.return_value = None  # Detection fails
            mock_detector.return_value = mock_instance

            result = process_nested_arrows(sample_contours, sample_hierarchy, metrics, image, 300.0)

            assert isinstance(result, list)
            assert len(result) == len(sample_contours)
            assert all(arrow is None for arrow in result)

    def test_process_nested_arrows_with_debug_mode(self, sample_contours, sample_hierarchy):
        """Test nested arrow processing with debug mode enabled."""
        metrics = [
            {'parent': 'parent 1', 'scar': 'parent 1', 'surface_type': 'Dorsal'},
            {'parent': 'parent 1', 'scar': 'scar 1', 'surface_type': 'Dorsal'}
        ]
        image = np.zeros((200, 200), dtype=np.uint8)

        config = {'debug_enabled': True}

        with tempfile.TemporaryDirectory() as temp_dir:
            # Add debug directory to metrics
            for metric in metrics:
                metric['debug_dir'] = temp_dir

            with patch('pylithics.image_processing.modules.arrow_integration.ArrowDetector') as mock_detector:
                mock_instance = MagicMock()
                mock_instance.analyze_contour_for_arrow.return_value = {
                    'arrow_back': (25, 30),
                    'arrow_tip': (35, 40),
                    'angle_deg': 60.0
                }
                mock_detector.return_value = mock_instance

                result = process_nested_arrows(
                    sample_contours, sample_hierarchy, metrics, image, 300.0, config
                )

                assert isinstance(result, list)
                # Debug mode should not affect basic functionality
                assert len(result) == len(sample_contours)


@pytest.mark.unit
class TestDetectArrowsIndependently:
    """Test the independent arrow detection function."""

    def test_detect_arrows_independently_basic(self, sample_contours):
        """Test independent arrow detection with basic contours."""
        metrics = [
            {'parent': 'parent 1', 'scar': 'parent 1', 'surface_type': 'Dorsal'},
            {'parent': 'parent 1', 'scar': 'scar 1', 'surface_type': 'Dorsal'}
        ]
        image = np.zeros((200, 200), dtype=np.uint8)

        with patch('pylithics.image_processing.modules.arrow_integration.ArrowDetector') as mock_detector:
            mock_instance = MagicMock()
            mock_instance.analyze_contour_for_arrow.side_effect = [
                None,  # No arrow in parent
                {      # Arrow in child
                    'arrow_back': (30, 35),
                    'arrow_tip': (40, 45),
                    'angle_deg': 90.0,
                    'compass_angle': 90.0
                }
            ]
            mock_detector.return_value = mock_instance

            result = detect_arrows_independently(sample_contours, metrics, image, 300.0)

            assert isinstance(result, list)
            assert len(result) == len(sample_contours)
            assert result[0] is None  # No arrow in parent
            assert result[1] is not None  # Arrow in child
            assert result[1]['angle_deg'] == 90.0

    def test_detect_arrows_independently_all_fail(self, sample_contours):
        """Test independent arrow detection when all detections fail."""
        metrics = [
            {'parent': 'parent 1', 'scar': 'parent 1', 'surface_type': 'Dorsal'},
            {'parent': 'parent 1', 'scar': 'scar 1', 'surface_type': 'Dorsal'}
        ]
        image = np.zeros((200, 200), dtype=np.uint8)

        with patch('pylithics.image_processing.modules.arrow_integration.ArrowDetector') as mock_detector:
            mock_instance = MagicMock()
            mock_instance.analyze_contour_for_arrow.return_value = None
            mock_detector.return_value = mock_instance

            result = detect_arrows_independently(sample_contours, metrics, image, 300.0)

            assert isinstance(result, list)
            assert len(result) == len(sample_contours)
            assert all(arrow is None for arrow in result)

    def test_detect_arrows_independently_empty_input(self):
        """Test independent arrow detection with empty input."""
        result = detect_arrows_independently([], [], np.zeros((100, 100), dtype=np.uint8), 300.0)

        assert isinstance(result, list)
        assert len(result) == 0

    def test_detect_arrows_independently_mismatched_lengths(self, sample_contours):
        """Test independent arrow detection with mismatched contours and metrics."""
        # More contours than metrics
        metrics = [{'parent': 'parent 1', 'scar': 'parent 1', 'surface_type': 'Dorsal'}]
        image = np.zeros((200, 200), dtype=np.uint8)

        with patch('pylithics.image_processing.modules.arrow_integration.logging') as mock_logging:
            result = detect_arrows_independently(sample_contours, metrics, image, 300.0)

            # Should handle mismatch gracefully
            assert isinstance(result, list)
            mock_logging.warning.assert_called()

    def test_detect_arrows_independently_with_config(self, sample_contours):
        """Test independent arrow detection with custom configuration."""
        metrics = [
            {'parent': 'parent 1', 'scar': 'parent 1', 'surface_type': 'Dorsal'},
            {'parent': 'parent 1', 'scar': 'scar 1', 'surface_type': 'Dorsal'}
        ]
        image = np.zeros((200, 200), dtype=np.uint8)
        config = {
            'reference_dpi': 150.0,
            'min_area_scale_factor': 0.5,
            'debug_enabled': False
        }

        with patch('pylithics.image_processing.modules.arrow_integration.ArrowDetector') as mock_detector:
            mock_instance = MagicMock()
            mock_detector.return_value = mock_instance

            detect_arrows_independently(sample_contours, metrics, image, 300.0, config)

            # Verify ArrowDetector was initialized with custom config
            mock_detector.assert_called_once_with(config)


@pytest.mark.unit
class TestIntegrateArrows:
    """Test the main arrow integration orchestrator function."""

    def test_integrate_arrows_hierarchy_based(self, sample_contours, sample_hierarchy):
        """Test arrow integration using hierarchy-based approach."""
        metrics = [
            {'parent': 'parent 1', 'scar': 'parent 1', 'surface_type': 'Dorsal'},
            {'parent': 'parent 1', 'scar': 'scar 1', 'surface_type': 'Dorsal'}
        ]
        image = np.zeros((200, 200), dtype=np.uint8)

        with patch('pylithics.image_processing.modules.arrow_integration.process_nested_arrows') as mock_nested:
            mock_nested.return_value = [
                None,  # Parent has no arrow
                {'arrow_back': (25, 30), 'arrow_tip': (35, 40), 'angle_deg': 45.0}  # Child has arrow
            ]

            result = integrate_arrows(sample_contours, sample_hierarchy, metrics, image, 300.0)

            assert isinstance(result, list)
            assert len(result) == len(metrics)

            # Check that hierarchy-based approach was used
            mock_nested.assert_called_once()

            # Verify result structure
            for i, metric in enumerate(result):
                assert isinstance(metric, dict)
                assert 'has_arrow' in metric

    def test_integrate_arrows_independent_fallback(self, sample_contours):
        """Test arrow integration fallback to independent approach."""
        metrics = [
            {'parent': 'parent 1', 'scar': 'parent 1', 'surface_type': 'Dorsal'},
            {'parent': 'parent 1', 'scar': 'scar 1', 'surface_type': 'Dorsal'}
        ]
        image = np.zeros((200, 200), dtype=np.uint8)
        hierarchy = None  # Force fallback to independent approach

        with patch('pylithics.image_processing.modules.arrow_integration.detect_arrows_independently') as mock_independent:
            mock_independent.return_value = [
                None,  # Parent has no arrow
                {'arrow_back': (30, 35), 'arrow_tip': (40, 45), 'angle_deg': 90.0}  # Child has arrow
            ]

            result = integrate_arrows(sample_contours, hierarchy, metrics, image, 300.0)

            assert isinstance(result, list)
            assert len(result) == len(metrics)

            # Check that independent approach was used
            mock_independent.assert_called_once()

    def test_integrate_arrows_empty_input(self):
        """Test arrow integration with empty input."""
        result = integrate_arrows([], None, [], np.zeros((100, 100), dtype=np.uint8), 300.0)

        assert isinstance(result, list)
        assert len(result) == 0

    def test_integrate_arrows_config_propagation(self, sample_contours, sample_hierarchy):
        """Test that configuration is properly propagated through integration."""
        metrics = [
            {'parent': 'parent 1', 'scar': 'parent 1', 'surface_type': 'Dorsal'},
            {'parent': 'parent 1', 'scar': 'scar 1', 'surface_type': 'Dorsal'}
        ]
        image = np.zeros((200, 200), dtype=np.uint8)
        config = {
            'reference_dpi': 150.0,
            'debug_enabled': True,
            'min_area_scale_factor': 0.6
        }

        with patch('pylithics.image_processing.modules.arrow_integration.process_nested_arrows') as mock_nested:
            mock_nested.return_value = [None, None]

            integrate_arrows(sample_contours, sample_hierarchy, metrics, image, 300.0, config)

            # Verify config was passed to nested arrow processing
            mock_nested.assert_called_once()
            args, kwargs = mock_nested.call_args
            assert len(args) >= 6  # Should include config parameter

    def test_integrate_arrows_metrics_update(self, sample_contours, sample_hierarchy):
        """Test that metrics are properly updated with arrow information."""
        metrics = [
            {'parent': 'parent 1', 'scar': 'parent 1', 'surface_type': 'Dorsal'},
            {'parent': 'parent 1', 'scar': 'scar 1', 'surface_type': 'Dorsal'}
        ]
        image = np.zeros((200, 200), dtype=np.uint8)

        with patch('pylithics.image_processing.modules.arrow_integration.process_nested_arrows') as mock_nested:
            mock_nested.return_value = [
                None,  # No arrow
                {      # Arrow found
                    'arrow_back': (25, 30),
                    'arrow_tip': (35, 40),
                    'angle_deg': 45.0,
                    'compass_angle': 225.0
                }
            ]

            result = integrate_arrows(sample_contours, sample_hierarchy, metrics, image, 300.0)

            # First metric should have no arrow
            assert result[0]['has_arrow'] is False
            assert 'arrow_angle' not in result[0] or result[0]['arrow_angle'] == 'NA'

            # Second metric should have arrow
            assert result[1]['has_arrow'] is True
            assert result[1]['arrow_angle'] == 45.0
            assert result[1]['compass_angle'] == 225.0


@pytest.mark.unit
class TestAssignArrowsToScars:
    """Test arrow assignment to scar contours."""

    def test_assign_arrows_basic_assignment(self):
        """Test basic arrow assignment to scars."""
        arrow_results = [
            None,  # Parent - no arrow
            {'arrow_back': (25, 30), 'arrow_tip': (35, 40), 'angle_deg': 45.0},  # Scar 1
            {'arrow_back': (55, 60), 'arrow_tip': (65, 70), 'angle_deg': 90.0}   # Scar 2
        ]

        metrics = [
            {'parent': 'parent 1', 'scar': 'parent 1', 'surface_type': 'Dorsal'},
            {'parent': 'parent 1', 'scar': 'scar 1', 'surface_type': 'Dorsal'},
            {'parent': 'parent 1', 'scar': 'scar 2', 'surface_type': 'Dorsal'}
        ]

        assignments = _assign_arrows_to_scars(arrow_results, metrics)

        assert len(assignments) == 2  # Two arrows assigned
        assert assignments[0] == (1, arrow_results[1])  # Scar 1 gets first arrow
        assert assignments[1] == (2, arrow_results[2])  # Scar 2 gets second arrow

    def test_assign_arrows_no_arrows_found(self):
        """Test arrow assignment when no arrows are found."""
        arrow_results = [None, None, None]
        metrics = [
            {'parent': 'parent 1', 'scar': 'parent 1', 'surface_type': 'Dorsal'},
            {'parent': 'parent 1', 'scar': 'scar 1', 'surface_type': 'Dorsal'},
            {'parent': 'parent 1', 'scar': 'scar 2', 'surface_type': 'Dorsal'}
        ]

        assignments = _assign_arrows_to_scars(arrow_results, metrics)

        assert len(assignments) == 0  # No assignments

    def test_assign_arrows_mismatched_lengths(self):
        """Test arrow assignment with mismatched arrow results and metrics."""
        arrow_results = [None, {'arrow_back': (25, 30), 'arrow_tip': (35, 40), 'angle_deg': 45.0}]
        metrics = [
            {'parent': 'parent 1', 'scar': 'parent 1', 'surface_type': 'Dorsal'},
            {'parent': 'parent 1', 'scar': 'scar 1', 'surface_type': 'Dorsal'},
            {'parent': 'parent 1', 'scar': 'scar 2', 'surface_type': 'Dorsal'}  # Extra metric
        ]

        with patch('pylithics.image_processing.modules.arrow_integration.logging') as mock_logging:
            assignments = _assign_arrows_to_scars(arrow_results, metrics)

            # Should handle mismatch gracefully
            assert isinstance(assignments, list)
            mock_logging.warning.assert_called()


@pytest.mark.unit
class TestResolveArrowConflicts:
    """Test arrow conflict resolution."""

    def test_resolve_arrow_conflicts_no_conflicts(self):
        """Test conflict resolution when no conflicts exist."""
        assignments = [
            (1, {'arrow_back': (25, 30), 'arrow_tip': (35, 40), 'angle_deg': 45.0}),
            (2, {'arrow_back': (55, 60), 'arrow_tip': (65, 70), 'angle_deg': 90.0})
        ]

        resolved = _resolve_arrow_conflicts(assignments)

        # Should return assignments unchanged
        assert len(resolved) == 2
        assert resolved == assignments

    def test_resolve_arrow_conflicts_duplicate_assignments(self):
        """Test conflict resolution with duplicate assignments."""
        arrow_data = {'arrow_back': (25, 30), 'arrow_tip': (35, 40), 'angle_deg': 45.0}
        assignments = [
            (1, arrow_data),
            (1, arrow_data)  # Duplicate assignment to same scar
        ]

        resolved = _resolve_arrow_conflicts(assignments)

        # Should keep only one assignment per scar
        assert len(resolved) == 1
        assert resolved[0] == (1, arrow_data)

    def test_resolve_arrow_conflicts_multiple_arrows_same_scar(self):
        """Test conflict resolution with multiple arrows assigned to same scar."""
        assignments = [
            (1, {'arrow_back': (25, 30), 'arrow_tip': (35, 40), 'angle_deg': 45.0}),
            (1, {'arrow_back': (26, 31), 'arrow_tip': (36, 41), 'angle_deg': 46.0}),  # Different arrow to same scar
            (2, {'arrow_back': (55, 60), 'arrow_tip': (65, 70), 'angle_deg': 90.0})
        ]

        resolved = _resolve_arrow_conflicts(assignments)

        # Should keep only first assignment for each scar
        assert len(resolved) == 2
        scar_indices = [assignment[0] for assignment in resolved]
        assert 1 in scar_indices
        assert 2 in scar_indices
        assert scar_indices.count(1) == 1  # Only one assignment to scar 1

    def test_resolve_arrow_conflicts_empty_input(self):
        """Test conflict resolution with empty input."""
        resolved = _resolve_arrow_conflicts([])

        assert len(resolved) == 0


@pytest.mark.unit
class TestUpdateMetricsWithArrows:
    """Test metrics update with arrow information."""

    def test_update_metrics_with_arrows_basic(self):
        """Test basic metrics update with arrow data."""
        metrics = [
            {'parent': 'parent 1', 'scar': 'parent 1', 'surface_type': 'Dorsal'},
            {'parent': 'parent 1', 'scar': 'scar 1', 'surface_type': 'Dorsal'},
            {'parent': 'parent 1', 'scar': 'scar 2', 'surface_type': 'Dorsal'}
        ]

        assignments = [
            (1, {
                'arrow_back': (25, 30),
                'arrow_tip': (35, 40),
                'angle_deg': 45.0,
                'compass_angle': 225.0
            }),
            (2, {
                'arrow_back': (55, 60),
                'arrow_tip': (65, 70),
                'angle_deg': 90.0,
                'compass_angle': 270.0
            })
        ]

        updated_metrics = _update_metrics_with_arrows(metrics, assignments)

        # Parent should have no arrow
        assert updated_metrics[0]['has_arrow'] is False
        assert updated_metrics[0]['arrow_angle'] == 'NA'

        # Scar 1 should have arrow
        assert updated_metrics[1]['has_arrow'] is True
        assert updated_metrics[1]['arrow_angle'] == 45.0
        assert updated_metrics[1]['compass_angle'] == 225.0
        assert updated_metrics[1]['arrow_back'] == (25, 30)
        assert updated_metrics[1]['arrow_tip'] == (35, 40)

        # Scar 2 should have arrow
        assert updated_metrics[2]['has_arrow'] is True
        assert updated_metrics[2]['arrow_angle'] == 90.0
        assert updated_metrics[2]['compass_angle'] == 270.0

    def test_update_metrics_with_arrows_no_assignments(self):
        """Test metrics update when no arrows are assigned."""
        metrics = [
            {'parent': 'parent 1', 'scar': 'parent 1', 'surface_type': 'Dorsal'},
            {'parent': 'parent 1', 'scar': 'scar 1', 'surface_type': 'Dorsal'}
        ]

        assignments = []  # No arrow assignments

        updated_metrics = _update_metrics_with_arrows(metrics, assignments)

        # All metrics should have no arrows
        for metric in updated_metrics:
            assert metric['has_arrow'] is False
            assert metric['arrow_angle'] == 'NA'

    def test_update_metrics_with_arrows_partial_assignment(self):
        """Test metrics update with partial arrow assignments."""
        metrics = [
            {'parent': 'parent 1', 'scar': 'parent 1', 'surface_type': 'Dorsal'},
            {'parent': 'parent 1', 'scar': 'scar 1', 'surface_type': 'Dorsal'},
            {'parent': 'parent 1', 'scar': 'scar 2', 'surface_type': 'Dorsal'}
        ]

        assignments = [
            (1, {  # Only scar 1 gets an arrow
                'arrow_back': (25, 30),
                'arrow_tip': (35, 40),
                'angle_deg': 45.0,
                'compass_angle': 225.0
            })
        ]

        updated_metrics = _update_metrics_with_arrows(metrics, assignments)

        # Parent should have no arrow
        assert updated_metrics[0]['has_arrow'] is False

        # Scar 1 should have arrow
        assert updated_metrics[1]['has_arrow'] is True
        assert updated_metrics[1]['arrow_angle'] == 45.0

        # Scar 2 should have no arrow
        assert updated_metrics[2]['has_arrow'] is False
        assert updated_metrics[2]['arrow_angle'] == 'NA'

    def test_update_metrics_with_arrows_invalid_index(self):
        """Test metrics update with invalid assignment index."""
        metrics = [
            {'parent': 'parent 1', 'scar': 'parent 1', 'surface_type': 'Dorsal'},
            {'parent': 'parent 1', 'scar': 'scar 1', 'surface_type': 'Dorsal'}
        ]

        assignments = [
            (5, {  # Invalid index (out of bounds)
                'arrow_back': (25, 30),
                'arrow_tip': (35, 40),
                'angle_deg': 45.0
            })
        ]

        with patch('pylithics.image_processing.modules.arrow_integration.logging') as mock_logging:
            updated_metrics = _update_metrics_with_arrows(metrics, assignments)

            # Should handle invalid index gracefully
            assert len(updated_metrics) == 2
            mock_logging.error.assert_called()

            # All metrics should have no arrows due to invalid assignment
            for metric in updated_metrics:
                assert metric['has_arrow'] is False


@pytest.mark.integration
class TestArrowIntegrationWorkflow:
    """Integration tests for complete arrow integration workflow."""

    def test_complete_arrow_integration_workflow(self, sample_contours, sample_hierarchy):
        """Test complete arrow integration workflow from contours to final metrics."""
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
            },
            {
                'parent': 'parent 1',
                'scar': 'scar 2',
                'surface_type': 'Dorsal',
                'centroid_x': 120.0,
                'centroid_y': 170.0,
                'width': 30.0,
                'height': 40.0,
                'area': 1200.0
            }
        ]

        # Add third contour for three-metric setup
        third_contour = np.array([
            [105, 150], [135, 150], [135, 190], [105, 190]
        ], dtype=np.int32).reshape(-1, 1, 2)

        test_contours = sample_contours + [third_contour]

        # Update hierarchy for three contours
        test_hierarchy = np.array([
            [-1, -1, 1, -1],  # Parent
            [2, -1, -1, 0],   # Child 1
            [-1, 1, -1, 0]    # Child 2
        ])

        image = np.zeros((250, 200), dtype=np.uint8)

        # Step 1: Test arrow integration
        result = integrate_arrows(test_contours, test_hierarchy, metrics, image, 300.0)

        assert isinstance(result, list)
        assert len(result) == 3

        # Step 2: Verify structure of results
        for metric in result:
            assert isinstance(metric, dict)
            assert 'has_arrow' in metric
            assert 'arrow_angle' in metric

            # Verify arrow fields are properly formatted
            if metric['has_arrow']:
                assert isinstance(metric['arrow_angle'], (int, float))
                assert 'arrow_back' in metric
                assert 'arrow_tip' in metric
                assert 'compass_angle' in metric
            else:
                assert metric['arrow_angle'] == 'NA'

        # Step 3: Verify parent vs child behavior
        parent_metric = result[0]
        child_metrics = result[1:]

        # Parent typically shouldn't have arrows in archaeological context
        assert parent_metric['has_arrow'] is False

        # Verify data integrity preservation
        assert parent_metric['parent'] == 'parent 1'
        assert parent_metric['surface_type'] == 'Dorsal'
        assert parent_metric['technical_width'] == 120.0

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

        # Test integration
        result = integrate_arrows(contours, hierarchy, metrics, image, 300.0)

        assert len(result) == 5  # 1 parent + 4 scars

        # Check that scars can have arrows but parent doesn't
        parent_result = result[0]
        scar_results = result[1:]

        assert parent_result['has_arrow'] is False

        # At least some scars should be capable of having arrows
        arrow_capable_scars = [metric for metric in scar_results if 'has_arrow' in metric]
        assert len(arrow_capable_scars) == 4

    def test_arrow_integration_error_recovery(self):
        """Test arrow integration error recovery and robustness."""
        # Create problematic input that might cause errors
        contours = [
            np.array([[50, 50], [51, 50]], dtype=np.int32).reshape(-1, 1, 2),  # Degenerate contour
            None,  # Invalid contour
            np.array([], dtype=np.int32).reshape(0, 1, 2)  # Empty contour
        ]

        metrics = [
            {'parent': 'parent 1', 'scar': 'parent 1', 'surface_type': 'Dorsal'},
            {'parent': 'parent 1', 'scar': 'scar 1', 'surface_type': 'Dorsal'},
            {'parent': 'parent 1', 'scar': 'scar 2', 'surface_type': 'Dorsal'}
        ]

        hierarchy = np.array([
            [-1, -1, 1, -1],
            [2, -1, -1, 0],
            [-1, 1, -1, 0]
        ])

        image = np.zeros((100, 100), dtype=np.uint8)

        with patch('pylithics.image_processing.modules.arrow_integration.logging'):
            # Should handle errors gracefully and not crash
            result = integrate_arrows(contours, hierarchy, metrics, image, 300.0)

            # Should return some result even with problematic input
            assert isinstance(result, list)
            assert len(result) <= len(metrics)

            # Each result should be a valid metric dict
            for metric in result:
                assert isinstance(metric, dict)
                assert 'has_arrow' in metric

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
        num_scars = 50
        for i in range(num_scars):
            x = 100 + (i % 10) * 30
            y = 100 + (i // 10) * 30

            scar_contour = np.array([
                [x, y], [x + 20, y], [x + 20, y + 20], [x, y + 20]
            ], dtype=np.int32).reshape(-1, 1, 2)
            contours.append(scar_contour)

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

        import time
        start_time = time.time()

        result = integrate_arrows(contours, hierarchy, metrics, image, 300.0)

        end_time = time.time()
        processing_time = end_time - start_time

        # Should complete in reasonable time even with many scars
        assert processing_time < 30.0  # 30 seconds max
        assert len(result) == num_scars + 1

        # Verify all results have arrow information
        for metric in result:
            assert 'has_arrow' in metric
            assert 'arrow_angle' in metric


@pytest.mark.unit
class TestArrowIntegrationConfigHandling:
    """Test configuration handling in arrow integration."""

    def test_integration_config_validation(self):
        """Test arrow integration with various configuration options."""
        contour = np.array([
            [30, 30], [70, 30], [70, 70], [30, 70]
        ], dtype=np.int32).reshape(-1, 1, 2)

        contours = [contour]
        hierarchy = np.array([[-1, -1, -1, -1]])
        metrics = [{'parent': 'parent 1', 'scar': 'scar 1', 'surface_type': 'Dorsal'}]
        image = np.zeros((100, 100), dtype=np.uint8)

        # Test different configurations
        configs = [
            {'reference_dpi': 150.0, 'debug_enabled': False},
            {'reference_dpi': 600.0, 'debug_enabled': True},
            {'min_area_scale_factor': 0.5, 'min_defect_depth_scale_factor': 0.6},
            {}  # Empty config (should use defaults)
        ]

        for config in configs:
            result = integrate_arrows(contours, hierarchy, metrics, image, 300.0, config)

            assert isinstance(result, list)
            assert len(result) == 1
            assert 'has_arrow' in result[0]

    def test_integration_debug_mode_handling(self):
        """Test arrow integration with debug mode enabled."""
        contour = np.array([
            [30, 30], [70, 30], [70, 70], [30, 70]
        ], dtype=np.int32).reshape(-1, 1, 2)

        contours = [contour]
        hierarchy = np.array([[-1, -1, -1, -1]])
        metrics = [{'parent': 'parent 1', 'scar': 'scar 1', 'surface_type': 'Dorsal'}]
        image = np.zeros((100, 100), dtype=np.uint8)

        config = {'debug_enabled': True}

        with tempfile.TemporaryDirectory() as temp_dir:
            # Add debug directory to metric
            metrics[0]['debug_dir'] = temp_dir

            result = integrate_arrows(contours, hierarchy, metrics, image, 300.0, config)

            assert isinstance(result, list)
            assert len(result) == 1

            # Debug mode should not affect basic functionality
            assert 'has_arrow' in result[0]


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
        num_flakes = 8
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

        result = integrate_arrows(contours, hierarchy, metrics, image, 300.0)

        assert len(result) == num_flakes + 1

        # Core should not have arrow
        core_result = result[0]
        assert core_result['has_arrow'] is False

        # Flake scars may or may not have arrows (depends on shape analysis)
        flake_results = result[1:]
        for flake_result in flake_results:
            assert 'has_arrow' in flake_result
            assert 'arrow_angle' in flake_result

    def test_integration_bifacial_tool_scenario(self):
        """Test arrow integration with bifacial tool scenario."""
        # Create bifacial tool with alternating edge scars
        tool_contour = np.array([
            [50, 100], [250, 90], [270, 110], [260, 190],
            [250, 200], [50, 210], [30, 190], [40, 110]
        ], dtype=np.int32).reshape(-1, 1, 2)

        contours = [tool_contour]
        metrics = [{
            'parent': 'parent 1',
            'scar': 'parent 1',
            'surface_type': 'Dorsal',
            'centroid_x': 150.0,
            'centroid_y': 150.0,
            'technical_width': 220.0,
            'technical_length': 110.0,
            'area': 24200.0
        }]

        hierarchy_data = [[-1, -1, 1, -1]]  # Tool

        # Add alternating edge scars
        left_edge_x = 70
        right_edge_x = 230

        for i in range(6):
            y_pos = 120 + i * 15

            # Left edge scar
            left_scar = np.array([
                [left_edge_x - 10, y_pos - 5], [left_edge_x + 10, y_pos - 5],
                [left_edge_x + 10, y_pos + 5], [left_edge_x - 10, y_pos + 5]
            ], dtype=np.int32).reshape(-1, 1, 2)
            contours.append(left_scar)

            metrics.append({
                'parent': 'parent 1',
                'scar': f'left_scar_{i + 1}',
                'surface_type': 'Dorsal',
                'centroid_x': float(left_edge_x),
                'centroid_y': float(y_pos),
                'width': 20.0,
                'height': 10.0,
                'area': 200.0
            })

            # Right edge scar (offset for alternating pattern)
            right_scar = np.array([
                [right_edge_x - 10, y_pos], [right_edge_x + 10, y_pos],
                [right_edge_x + 10, y_pos + 10], [right_edge_x - 10, y_pos + 10]
            ], dtype=np.int32).reshape(-1, 1, 2)
            contours.append(right_scar)

            metrics.append({
                'parent': 'parent 1',
                'scar': f'right_scar_{i + 1}',
                'surface_type': 'Dorsal',
                'centroid_x': float(right_edge_x),
                'centroid_y': float(y_pos + 5),
                'width': 20.0,
                'height': 10.0,
                'area': 200.0
            })

            # Add to hierarchy
            left_idx = len(hierarchy_data)
            right_idx = left_idx + 1

            # Left scar
            next_left = right_idx if True else -1
            prev_left = left_idx - 2 if i > 0 else -1
            hierarchy_data.append([next_left, prev_left, -1, 0])

            # Right scar
            next_right = left_idx + 2 if i < 5 else -1
            prev_right = left_idx
            hierarchy_data.append([next_right, prev_right, -1, 0])

        hierarchy = np.array(hierarchy_data)
        image = np.zeros((250, 320), dtype=np.uint8)

        result = integrate_arrows(contours, hierarchy, metrics, image, 300.0)

        assert len(result) == 13  # 1 tool + 12 scars

        # Tool should not have arrow
        tool_result = result[0]
        assert tool_result['has_arrow'] is False

        # Edge scars should be processed for arrows
        scar_results = result[1:]
        assert len(scar_results) == 12

        for scar_result in scar_results:
            assert 'has_arrow' in scar_result
            assert 'arrow_angle' in scar_result
            if scar_result['has_arrow']:
                assert isinstance(scar_result['arrow_angle'], (int, float))
                assert 'compass_angle' in scar_result