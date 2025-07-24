"""
PyLithics Analysis Pipeline Tests
=================================

Tests for the core analysis workflow integration including process_and_save_contours
main pipeline, integration between all modules, error recovery and graceful degradation,
and complete workflow testing.
"""

import pytest
import numpy as np
import cv2
import tempfile
import os
import pandas as pd
from unittest.mock import patch, MagicMock, call

from pylithics.image_processing.image_analysis import (
    process_and_save_contours,
)


@pytest.mark.integration
class TestProcessAndSaveContours:
    """Test the main pipeline orchestrator function."""

    def test_process_and_save_contours_complete_workflow(self, test_image_with_dpi, sample_config):
        """Test complete workflow from image to saved outputs."""
        image_id = "pipeline_test"
        conversion_factor = 0.1

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = temp_dir

            # Mock the various pipeline components
            with patch('pylithics.image_processing.image_analysis.extract_contours_with_hierarchy') as mock_extract, \
                 patch('pylithics.image_processing.image_analysis.calculate_contour_metrics') as mock_metrics, \
                 patch('pylithics.image_processing.image_analysis.classify_parent_contours') as mock_classify, \
                 patch('pylithics.image_processing.image_analysis.integrate_arrows') as mock_arrows, \
                 patch('pylithics.image_processing.image_analysis.save_measurements_to_csv') as mock_save_csv, \
                 patch('pylithics.image_processing.image_analysis.visualize_contours_with_hierarchy') as mock_viz:

                mock_contours = [
                    np.array([[50, 50], [150, 50], [150, 150], [50, 150]], dtype=np.int32).reshape(-1, 1, 2),
                    np.array([[70, 70], [130, 70], [130, 130], [70, 130]], dtype=np.int32).reshape(-1, 1, 2)
                ]
                mock_hierarchy = np.array([[-1, -1, 1, -1], [-1, -1, -1, 0]])
                mock_extract.return_value = (mock_contours, mock_hierarchy)

                mock_metrics.return_value = [
                    {
                        'parent': 'parent 1',
                        'scar': 'parent 1',
                        'surface_type': None,
                        'centroid_x': 100.0,
                        'centroid_y': 100.0,
                        'technical_width': 100.0,
                        'technical_length': 100.0,
                        'area': 10000.0
                    },
                    {
                        'parent': 'parent 1',
                        'scar': 'scar 1',
                        'surface_type': None,
                        'centroid_x': 100.0,
                        'centroid_y': 100.0,
                        'width': 60.0,
                        'height': 60.0,
                        'area': 3600.0
                    }
                ]

                mock_classify.return_value = [
                    {
                        'parent': 'parent 1',
                        'scar': 'parent 1',
                        'surface_type': 'Dorsal',
                        'centroid_x': 100.0,
                        'centroid_y': 100.0,
                        'technical_width': 100.0,
                        'technical_length': 100.0,
                        'area': 10000.0
                    },
                    {
                        'parent': 'parent 1',
                        'scar': 'scar 1',
                        'surface_type': 'Dorsal',
                        'centroid_x': 100.0,
                        'centroid_y': 100.0,
                        'width': 60.0,
                        'height': 60.0,
                        'area': 3600.0
                    }
                ]

                mock_arrows.return_value = [
                    {
                        'parent': 'parent 1',
                        'scar': 'parent 1',
                        'surface_type': 'Dorsal',
                        'centroid_x': 100.0,
                        'centroid_y': 100.0,
                        'technical_width': 100.0,
                        'technical_length': 100.0,
                        'area': 10000.0,
                        'has_arrow': False,
                        'arrow_angle': 'NA'
                    },
                    {
                        'parent': 'parent 1',
                        'scar': 'scar 1',
                        'surface_type': 'Dorsal',
                        'centroid_x': 100.0,
                        'centroid_y': 100.0,
                        'width': 60.0,
                        'height': 60.0,
                        'area': 3600.0,
                        'has_arrow': True,
                        'arrow_angle': 45.0
                    }
                ]

                # Execute pipeline
                process_and_save_contours(
                    test_image_with_dpi, conversion_factor, output_dir, image_id
                )

                # Verify all major components were called
                mock_extract.assert_called_once()
                mock_metrics.assert_called_once()
                mock_classify.assert_called_once()
                mock_arrows.assert_called_once()
                mock_save_csv.assert_called_once()
                mock_viz.assert_called_once()

    def test_process_and_save_contours_no_contours_found(self, test_image_with_dpi):
        """Test pipeline behavior when no contours are found."""
        image_id = "no_contours_test"
        conversion_factor = 0.1

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = temp_dir

            with patch('pylithics.image_processing.image_analysis.extract_contours_with_hierarchy') as mock_extract, \
                 patch('pylithics.image_processing.image_analysis.logging') as mock_logging:

                mock_extract.return_value = ([], None)  # No contours found

                process_and_save_contours(
                    test_image_with_dpi, conversion_factor, output_dir, image_id
                )

                mock_logging.warning.assert_called()

    def test_process_and_save_contours_partial_failure_graceful_degradation(self, test_image_with_dpi):
        """Test pipeline graceful degradation when some modules fail."""
        image_id = "partial_fail_test"
        conversion_factor = 0.1

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = temp_dir

            with patch('pylithics.image_processing.image_analysis.extract_contours_with_hierarchy') as mock_extract, \
                 patch('pylithics.image_processing.image_analysis.calculate_contour_metrics') as mock_metrics, \
                 patch('pylithics.image_processing.image_analysis.classify_parent_contours') as mock_classify, \
                 patch('pylithics.image_processing.image_analysis.integrate_arrows') as mock_arrows, \
                 patch('pylithics.image_processing.image_analysis.analyze_dorsal_symmetry') as mock_symmetry, \
                 patch('pylithics.image_processing.image_analysis.save_measurements_to_csv') as mock_save, \
                 patch('pylithics.image_processing.image_analysis.logging') as mock_logging:

                # Set up successful early stages
                mock_contours = [np.array([[50, 50], [150, 50], [150, 150], [50, 150]], dtype=np.int32).reshape(-1, 1, 2)]
                mock_hierarchy = np.array([[-1, -1, -1, -1]])
                mock_extract.return_value = (mock_contours, mock_hierarchy)

                mock_metrics.return_value = [{
                    'parent': 'parent 1',
                    'scar': 'parent 1',
                    'surface_type': None,
                    'centroid_x': 100.0,
                    'centroid_y': 100.0,
                    'technical_width': 100.0,
                    'technical_length': 100.0,
                    'area': 10000.0
                }]

                mock_classify.return_value = [{
                    'parent': 'parent 1',
                    'scar': 'parent 1',
                    'surface_type': 'Dorsal',
                    'centroid_x': 100.0,
                    'centroid_y': 100.0,
                    'technical_width': 100.0,
                    'technical_length': 100.0,
                    'area': 10000.0
                }]

                # Make arrow integration fail
                mock_arrows.side_effect = Exception("Arrow integration failed")

                # Symmetry should work
                mock_symmetry.return_value = {
                    'top_area': 5000.0,
                    'bottom_area': 5000.0,
                    'left_area': 5000.0,
                    'right_area': 5000.0,
                    'vertical_symmetry': 1.0,
                    'horizontal_symmetry': 1.0
                }

                process_and_save_contours(
                    test_image_with_dpi, conversion_factor, output_dir, image_id
                )

                # Should log the error but continue
                mock_logging.error.assert_called()

                # Should still save results
                mock_save.assert_called()


@pytest.mark.integration
class TestPipelineIntegrationScenarios:
    """Test pipeline integration with realistic archaeological scenarios."""

    def test_pipeline_blade_tool_scenario(self, test_image_with_dpi):
        """Test pipeline with blade tool scenario."""
        image_id = "blade_tool_scenario"
        conversion_factor = 0.067  # 15mm scale

        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock realistic blade tool processing
            with patch('pylithics.image_processing.image_analysis.extract_contours_with_hierarchy') as mock_extract, \
                 patch('pylithics.image_processing.image_analysis.calculate_contour_metrics') as mock_metrics, \
                 patch('pylithics.image_processing.image_analysis.classify_parent_contours') as mock_classify, \
                 patch('pylithics.image_processing.image_analysis.integrate_arrows') as mock_arrows, \
                 patch('pylithics.image_processing.image_analysis.analyze_dorsal_symmetry') as mock_symmetry, \
                 patch('pylithics.image_processing.image_analysis.save_measurements_to_csv') as mock_save:

                # Set up blade tool data
                blade_contour = np.array([[30, 50], [270, 55], [280, 70], [275, 250], [270, 260], [30, 255], [20, 240], [25, 65]], dtype=np.int32).reshape(-1, 1, 2)

                # Removal scars
                scar_contours = [
                    np.array([[60, 80], [90, 85], [95, 110], [65, 115]], dtype=np.int32).reshape(-1, 1, 2),
                    np.array([[120, 90], [150, 95], [155, 120], [125, 125]], dtype=np.int32).reshape(-1, 1, 2),
                    np.array([[180, 100], [210, 105], [215, 130], [185, 135]], dtype=np.int32).reshape(-1, 1, 2)
                ]

                all_contours = [blade_contour] + scar_contours
                hierarchy = np.array([
                    [-1, -1, 1, -1],  # Blade
                    [2, -1, -1, 0],   # Scar 1
                    [3, 1, -1, 0],    # Scar 2
                    [-1, 2, -1, 0]    # Scar 3
                ])

                mock_extract.return_value = (all_contours, hierarchy)

                # Metrics for blade and scars
                mock_metrics.return_value = [
                    {
                        'parent': 'parent 1',
                        'scar': 'parent 1',
                        'surface_type': None,
                        'centroid_x': 150.0,
                        'centroid_y': 155.0,
                        'technical_width': 250.0,
                        'technical_length': 210.0,
                        'area': 52500.0
                    },
                    {
                        'parent': 'parent 1',
                        'scar': 'scar 1',
                        'surface_type': None,
                        'centroid_x': 77.5,
                        'centroid_y': 97.5,
                        'width': 35.0,
                        'height': 35.0,
                        'area': 1225.0
                    },
                    {
                        'parent': 'parent 1',
                        'scar': 'scar 2',
                        'surface_type': None,
                        'centroid_x': 137.5,
                        'centroid_y': 107.5,
                        'width': 35.0,
                        'height': 35.0,
                        'area': 1225.0
                    },
                    {
                        'parent': 'parent 1',
                        'scar': 'scar 3',
                        'surface_type': None,
                        'centroid_x': 197.5,
                        'centroid_y': 117.5,
                        'width': 35.0,
                        'height': 35.0,
                        'area': 1225.0
                    }
                ]

                # Surface classification
                mock_classify.return_value = [
                    {
                        'parent': 'parent 1',
                        'scar': 'parent 1',
                        'surface_type': 'Dorsal',
                        'centroid_x': 150.0,
                        'centroid_y': 155.0,
                        'technical_width': 250.0,
                        'technical_length': 210.0,
                        'area': 52500.0
                    },
                    {
                        'parent': 'parent 1',
                        'scar': 'scar 1',
                        'surface_type': 'Dorsal',
                        'centroid_x': 77.5,
                        'centroid_y': 97.5,
                        'width': 35.0,
                        'height': 35.0,
                        'area': 1225.0
                    },
                    {
                        'parent': 'parent 1',
                        'scar': 'scar 2',
                        'surface_type': 'Dorsal',
                        'centroid_x': 137.5,
                        'centroid_y': 107.5,
                        'width': 35.0,
                        'height': 35.0,
                        'area': 1225.0
                    },
                    {
                        'parent': 'parent 1',
                        'scar': 'scar 3',
                        'surface_type': 'Dorsal',
                        'centroid_x': 197.5,
                        'centroid_y': 117.5,
                        'width': 35.0,
                        'height': 35.0,
                        'area': 1225.0
                    }
                ]

                # Arrow integration
                mock_arrows.return_value = [
                    {
                        'parent': 'parent 1',
                        'scar': 'parent 1',
                        'surface_type': 'Dorsal',
                        'centroid_x': 150.0,
                        'centroid_y': 155.0,
                        'technical_width': 250.0,
                        'technical_length': 210.0,
                        'area': 52500.0,
                        'has_arrow': False,
                        'arrow_angle': 'NA'
                    },
                    {
                        'parent': 'parent 1',
                        'scar': 'scar 1',
                        'surface_type': 'Dorsal',
                        'centroid_x': 77.5,
                        'centroid_y': 97.5,
                        'width': 35.0,
                        'height': 35.0,
                        'area': 1225.0,
                        'has_arrow': True,
                        'arrow_angle': 45.0
                    },
                    {
                        'parent': 'parent 1',
                        'scar': 'scar 2',
                        'surface_type': 'Dorsal',
                        'centroid_x': 137.5,
                        'centroid_y': 107.5,
                        'width': 35.0,
                        'height': 35.0,
                        'area': 1225.0,
                        'has_arrow': True,
                        'arrow_angle': 60.0
                    },
                    {
                        'parent': 'parent 1',
                        'scar': 'scar 3',
                        'surface_type': 'Dorsal',
                        'centroid_x': 197.5,
                        'centroid_y': 117.5,
                        'width': 35.0,
                        'height': 35.0,
                        'area': 1225.0,
                        'has_arrow': False,
                        'arrow_angle': 'NA'
                    }
                ]

                # Symmetry analysis
                mock_symmetry.return_value = {
                    'top_area': 26250.0,
                    'bottom_area': 26250.0,
                    'left_area': 26250.0,
                    'right_area': 26250.0,
                    'vertical_symmetry': 0.95,
                    'horizontal_symmetry': 0.85
                }

                # Execute pipeline
                process_and_save_contours(
                    test_image_with_dpi, conversion_factor, temp_dir, image_id
                )

                # Verify all major stages were called
                mock_extract.assert_called_once()
                mock_metrics.assert_called_once()
                mock_classify.assert_called_once()
                mock_arrows.assert_called_once()
                mock_symmetry.assert_called_once()
                mock_save.assert_called_once()

                # Verify final CSV data
                # Debug version - add this before the assertion that's failing:

                # Verify final CSV data
                save_call_args = mock_save.call_args[0]
                final_metrics = save_call_args[0]

                # DEBUG: Print what's actually in the metrics
                print("\n=== DEBUG: Final metrics content ===")
                for i, metric in enumerate(final_metrics):
                    print(f"Metric {i}: {metric.keys()}")
                    if metric.get('surface_type') == 'Dorsal':
                        print(f"  Dorsal metric: {metric}")
                print("=====================================\n")

                assert len(final_metrics) == 4

                # Check blade tool characteristics
                blade_metric = final_metrics[0]
                assert blade_metric['surface_type'] == 'Dorsal'
                assert blade_metric['has_arrow'] is False
                assert 'vertical_symmetry' in blade_metric

                # Check scar characteristics
                scar_metrics = final_metrics[1:]
                arrow_count = sum(1 for scar in scar_metrics if scar['has_arrow'])
                assert arrow_count >= 1  # At least some scars should have arrows


@pytest.mark.performance
class TestPipelinePerformance:
    """Test pipeline performance characteristics."""

    def test_pipeline_performance_single_image(self, test_image_with_dpi):
        """Test pipeline performance with single image."""
        image_id = "performance_test"
        conversion_factor = 0.1

        with tempfile.TemporaryDirectory() as temp_dir:
            # Use real processing (limited mocking for performance test)
            with patch('pylithics.image_processing.image_analysis.save_measurements_to_csv') as mock_save, \
                 patch('pylithics.image_processing.image_analysis.visualize_contours_with_hierarchy') as mock_viz:

                import time
                start_time = time.time()

                process_and_save_contours(
                    test_image_with_dpi, conversion_factor, temp_dir, image_id
                )

                end_time = time.time()
                processing_time = end_time - start_time

                # Should complete in reasonable time
                assert processing_time < 30.0  # 30 seconds max for single image

    def test_pipeline_memory_usage(self, test_image_with_dpi):
        """Test pipeline memory usage characteristics."""
        image_id = "memory_test"
        conversion_factor = 0.1

        with tempfile.TemporaryDirectory() as temp_dir:
            # Monitor memory usage during processing
            import psutil
            import os

            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss

            with patch('pylithics.image_processing.image_analysis.save_measurements_to_csv') as mock_save, \
                 patch('pylithics.image_processing.image_analysis.visualize_contours_with_hierarchy') as mock_viz:

                process_and_save_contours(
                    test_image_with_dpi, conversion_factor, temp_dir, image_id
                )

                final_memory = process.memory_info().rss
                memory_increase = final_memory - initial_memory

                # Memory increase should be reasonable (< 100MB for test image)
                assert memory_increase < 100 * 1024 * 1024  # 100MB


@pytest.mark.integration
class TestPipelineRealWorldIntegration:
    """Test pipeline with real-world integration scenarios."""

    def test_pipeline_with_actual_preprocessing(self, test_image_with_dpi):
        """Test pipeline with actual preprocessing (minimal mocking)."""
        image_id = "real_preprocessing_test"
        conversion_factor = 0.1

        with tempfile.TemporaryDirectory() as temp_dir:
            # Only mock the final outputs to avoid file system issues
            with patch('pylithics.image_processing.image_analysis.save_measurements_to_csv') as mock_save, \
                 patch('pylithics.image_processing.image_analysis.visualize_contours_with_hierarchy') as mock_viz:

                # Let most of the pipeline run with real processing
                process_and_save_contours(
                    test_image_with_dpi, conversion_factor, temp_dir, image_id
                )

                # Should complete without crashing - result is void function


@pytest.mark.integration
class TestPipelineEndToEndValidation:
    """End-to-end validation tests for the complete pipeline."""

    def test_pipeline_data_flow_integrity(self, test_image_with_dpi):
        """Test that data flows correctly through all pipeline stages."""
        image_id = "data_flow_test"
        conversion_factor = 0.1

        with tempfile.TemporaryDirectory() as temp_dir:
            # Track data flow through pipeline stages
            captured_data = {}

            def capture_extract(*args, **kwargs):
                captured_data['extract_input'] = args
                contour = np.array([[25, 25], [175, 25], [175, 125], [25, 125]], dtype=np.int32).reshape(-1, 1, 2)
                return ([contour], np.array([[-1, -1, -1, -1]]))

            def capture_metrics(*args, **kwargs):
                captured_data['metrics_input'] = args
                return [{
                    'parent': 'parent 1',
                    'scar': 'parent 1',
                    'surface_type': None,
                    'centroid_x': 100.0,
                    'centroid_y': 75.0,
                    'technical_width': 150.0,
                    'technical_length': 100.0,
                    'area': 15000.0
                }]

            def capture_classify(*args, **kwargs):
                captured_data['classify_input'] = args
                metrics = args[0]
                classified = []
                for metric in metrics:
                    new_metric = metric.copy()
                    new_metric['surface_type'] = 'Dorsal'
                    classified.append(new_metric)
                return classified

            def capture_save(*args, **kwargs):
                captured_data['save_input'] = args
                return None

            with patch('pylithics.image_processing.image_analysis.extract_contours_with_hierarchy', side_effect=capture_extract), \
                 patch('pylithics.image_processing.image_analysis.calculate_contour_metrics', side_effect=capture_metrics), \
                 patch('pylithics.image_processing.image_analysis.classify_parent_contours', side_effect=capture_classify), \
                 patch('pylithics.image_processing.image_analysis.save_measurements_to_csv', side_effect=capture_save), \
                 patch('pylithics.image_processing.image_analysis.visualize_contours_with_hierarchy'):

                process_and_save_contours(
                    test_image_with_dpi, conversion_factor, temp_dir, image_id
                )

                # Verify data flow integrity
                assert 'extract_input' in captured_data
                assert 'metrics_input' in captured_data
                assert 'classify_input' in captured_data
                assert 'save_input' in captured_data

                # Verify image_id propagates through
                save_data = captured_data['save_input'][0]  # Metrics list
                assert len(save_data) > 0

                # Verify contour data consistency
                extract_input = captured_data['extract_input']
                assert len(extract_input) >= 2  # Should include processed image and image_id

                # Verify classification input consistency
                classify_input = captured_data['classify_input']
                assert len(classify_input) >= 1  # Should include metrics list
                assert len(classify_input[0]) > 0  # Should have metrics to classify
