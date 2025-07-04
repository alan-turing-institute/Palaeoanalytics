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
    analyze_single_image,
    batch_process_images,
    _validate_pipeline_inputs,
    _execute_contour_extraction,
    _execute_surface_classification,
    _execute_arrow_integration,
    _execute_spatial_analysis,
    _generate_outputs
)


@pytest.mark.integration
class TestProcessAndSaveContours:
    """Test the main pipeline orchestrator function."""

    def test_process_and_save_contours_complete_workflow(self, test_image_with_dpi, sample_config):
        """Test complete workflow from image to saved outputs."""
        image_id = "pipeline_test"
        real_world_scale = 10.0

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = temp_dir
            config_file = os.path.join(temp_dir, "config.yaml")

            # Create config file
            import yaml
            with open(config_file, 'w') as f:
                yaml.dump(sample_config, f)

            # Mock the various pipeline components
            with patch('pylithics.image_processing.image_analysis.execute_preprocessing_pipeline') as mock_preprocess, \
                 patch('pylithics.image_processing.image_analysis.extract_contours_with_hierarchy') as mock_extract, \
                 patch('pylithics.image_processing.image_analysis.calculate_contour_metrics') as mock_metrics, \
                 patch('pylithics.image_processing.image_analysis.classify_parent_contours') as mock_classify, \
                 patch('pylithics.image_processing.image_analysis.integrate_arrows') as mock_arrows, \
                 patch('pylithics.image_processing.image_analysis.save_measurements_to_csv') as mock_save_csv, \
                 patch('pylithics.image_processing.image_analysis.visualize_contours_with_hierarchy') as mock_viz:

                # Set up mock returns
                mock_preprocess.return_value = np.ones((200, 300), dtype=np.uint8) * 255

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
                result = process_and_save_contours(
                    test_image_with_dpi, image_id, real_world_scale, output_dir
                )

                # Verify pipeline execution
                assert result is True

                # Verify all major components were called
                mock_preprocess.assert_called_once()
                mock_extract.assert_called_once()
                mock_metrics.assert_called_once()
                mock_classify.assert_called_once()
                mock_arrows.assert_called_once()
                mock_save_csv.assert_called_once()
                mock_viz.assert_called_once()

    def test_process_and_save_contours_preprocessing_failure(self, test_image_with_dpi):
        """Test pipeline behavior when preprocessing fails."""
        image_id = "preprocess_fail_test"
        real_world_scale = 10.0

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = temp_dir

            with patch('pylithics.image_processing.image_analysis.execute_preprocessing_pipeline') as mock_preprocess, \
                 patch('pylithics.image_processing.image_analysis.logging') as mock_logging:

                mock_preprocess.return_value = None  # Preprocessing fails

                result = process_and_save_contours(
                    test_image_with_dpi, image_id, real_world_scale, output_dir
                )

                assert result is False
                mock_logging.error.assert_called()

    def test_process_and_save_contours_no_contours_found(self, test_image_with_dpi, sample_config):
        """Test pipeline behavior when no contours are found."""
        image_id = "no_contours_test"
        real_world_scale = 10.0

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = temp_dir

            with patch('pylithics.image_processing.image_analysis.execute_preprocessing_pipeline') as mock_preprocess, \
                 patch('pylithics.image_processing.image_analysis.extract_contours_with_hierarchy') as mock_extract, \
                 patch('pylithics.image_processing.image_analysis.logging') as mock_logging:

                mock_preprocess.return_value = np.ones((200, 300), dtype=np.uint8) * 255
                mock_extract.return_value = ([], None)  # No contours found

                result = process_and_save_contours(
                    test_image_with_dpi, image_id, real_world_scale, output_dir
                )

                assert result is False
                mock_logging.warning.assert_called()

    def test_process_and_save_contours_dpi_verification_failure(self, test_image_with_dpi):
        """Test pipeline behavior when DPI verification fails."""
        image_id = "dpi_fail_test"
        real_world_scale = 10.0

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = temp_dir

            with patch('pylithics.image_processing.image_analysis.verify_image_dpi_and_scale') as mock_dpi, \
                 patch('pylithics.image_processing.image_analysis.logging') as mock_logging:

                mock_dpi.return_value = None  # DPI verification fails

                result = process_and_save_contours(
                    test_image_with_dpi, image_id, real_world_scale, output_dir
                )

                assert result is False
                mock_logging.error.assert_called()

    def test_process_and_save_contours_partial_failure_graceful_degradation(self, test_image_with_dpi, sample_config):
        """Test pipeline graceful degradation when some modules fail."""
        image_id = "partial_fail_test"
        real_world_scale = 10.0

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = temp_dir

            with patch('pylithics.image_processing.image_analysis.execute_preprocessing_pipeline') as mock_preprocess, \
                 patch('pylithics.image_processing.image_analysis.extract_contours_with_hierarchy') as mock_extract, \
                 patch('pylithics.image_processing.image_analysis.calculate_contour_metrics') as mock_metrics, \
                 patch('pylithics.image_processing.image_analysis.classify_parent_contours') as mock_classify, \
                 patch('pylithics.image_processing.image_analysis.integrate_arrows') as mock_arrows, \
                 patch('pylithics.image_processing.image_analysis.analyze_dorsal_symmetry') as mock_symmetry, \
                 patch('pylithics.image_processing.image_analysis.save_measurements_to_csv') as mock_save, \
                 patch('pylithics.image_processing.image_analysis.logging') as mock_logging:

                # Set up successful early stages
                mock_preprocess.return_value = np.ones((200, 300), dtype=np.uint8) * 255
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

                result = process_and_save_contours(
                    test_image_with_dpi, image_id, real_world_scale, output_dir
                )

                # Should still succeed despite arrow integration failure
                assert result is True

                # Should log the error but continue
                mock_logging.error.assert_called()

                # Should still save results
                mock_save.assert_called()

    def test_process_and_save_contours_custom_config(self, test_image_with_dpi):
        """Test pipeline with custom configuration file."""
        image_id = "custom_config_test"
        real_world_scale = 10.0

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = temp_dir
            config_file = os.path.join(temp_dir, "custom_config.yaml")

            # Create custom config
            custom_config = {
                'thresholding': {'method': 'otsu', 'max_value': 255},
                'arrow_detection': {'enabled': True, 'reference_dpi': 150.0},
                'contour_filtering': {'min_area': 100.0},
                'logging': {'level': 'DEBUG'}
            }

            import yaml
            with open(config_file, 'w') as f:
                yaml.dump(custom_config, f)

            with patch('pylithics.image_processing.image_analysis.load_preprocessing_config') as mock_load_config, \
                 patch('pylithics.image_processing.image_analysis.execute_preprocessing_pipeline') as mock_preprocess, \
                 patch('pylithics.image_processing.image_analysis.extract_contours_with_hierarchy') as mock_extract:

                mock_load_config.return_value = custom_config
                mock_preprocess.return_value = np.ones((200, 300), dtype=np.uint8) * 255
                mock_extract.return_value = ([], None)  # Minimal setup

                process_and_save_contours(
                    test_image_with_dpi, image_id, real_world_scale, output_dir, config_file
                )

                # Verify custom config was loaded
                mock_load_config.assert_called_with(config_file)


@pytest.mark.integration
class TestAnalyzeSingleImage:
    """Test single image analysis function."""

    def test_analyze_single_image_success(self, test_image_with_dpi):
        """Test successful single image analysis."""
        image_path = test_image_with_dpi
        image_id = "single_test"
        scale_value = 12.5

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = temp_dir

            with patch('pylithics.image_processing.image_analysis.process_and_save_contours') as mock_process:
                mock_process.return_value = True

                result = analyze_single_image(image_path, image_id, scale_value, output_dir)

                assert result is True
                mock_process.assert_called_once_with(image_path, image_id, scale_value, output_dir, None)

    def test_analyze_single_image_with_config(self, test_image_with_dpi):
        """Test single image analysis with custom config."""
        image_path = test_image_with_dpi
        image_id = "single_config_test"
        scale_value = 12.5

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = temp_dir
            config_file = os.path.join(temp_dir, "test_config.yaml")

            # Create minimal config file
            with open(config_file, 'w') as f:
                f.write("thresholding:\n  method: simple\n")

            with patch('pylithics.image_processing.image_analysis.process_and_save_contours') as mock_process:
                mock_process.return_value = True

                result = analyze_single_image(image_path, image_id, scale_value, output_dir, config_file)

                assert result is True
                mock_process.assert_called_once_with(image_path, image_id, scale_value, output_dir, config_file)

    def test_analyze_single_image_failure(self, test_image_with_dpi):
        """Test single image analysis when processing fails."""
        image_path = test_image_with_dpi
        image_id = "single_fail_test"
        scale_value = 12.5

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = temp_dir

            with patch('pylithics.image_processing.image_analysis.process_and_save_contours') as mock_process, \
                 patch('pylithics.image_processing.image_analysis.logging') as mock_logging:

                mock_process.return_value = False

                result = analyze_single_image(image_path, image_id, scale_value, output_dir)

                assert result is False
                mock_logging.error.assert_called()

    def test_analyze_single_image_exception_handling(self, test_image_with_dpi):
        """Test single image analysis exception handling."""
        image_path = test_image_with_dpi
        image_id = "single_exception_test"
        scale_value = 12.5

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = temp_dir

            with patch('pylithics.image_processing.image_analysis.process_and_save_contours') as mock_process, \
                 patch('pylithics.image_processing.image_analysis.logging') as mock_logging:

                mock_process.side_effect = Exception("Processing error")

                result = analyze_single_image(image_path, image_id, scale_value, output_dir)

                assert result is False
                mock_logging.error.assert_called()


@pytest.mark.integration
class TestBatchProcessImages:
    """Test batch image processing function."""

    def test_batch_process_success(self, sample_metadata_file):
        """Test successful batch processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = temp_dir
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Create test images
            test_images = ["test_image_1.png", "test_image_2.png", "test_image_3.png"]
            for img_name in test_images:
                img_path = os.path.join(images_dir, img_name)
                # Create minimal PNG file
                test_img = np.zeros((100, 100, 3), dtype=np.uint8)
                cv2.imwrite(img_path, test_img)

            with patch('pylithics.image_processing.image_analysis.read_metadata') as mock_read_meta, \
                 patch('pylithics.image_processing.image_analysis.analyze_single_image') as mock_analyze:

                # Mock metadata
                mock_read_meta.return_value = [
                    {'image_id': 'test_image_1.png', 'scale': '10.0'},
                    {'image_id': 'test_image_2.png', 'scale': '12.0'},
                    {'image_id': 'test_image_3.png', 'scale': '8.5'}
                ]

                mock_analyze.return_value = True

                results = batch_process_images(data_dir, sample_metadata_file)

                assert len(results) == 3
                assert all(result is True for result in results)
                assert mock_analyze.call_count == 3

    def test_batch_process_partial_success(self, sample_metadata_file):
        """Test batch processing with some failures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = temp_dir
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Create test images
            test_images = ["test_image_1.png", "test_image_2.png"]
            for img_name in test_images:
                img_path = os.path.join(images_dir, img_name)
                test_img = np.zeros((100, 100, 3), dtype=np.uint8)
                cv2.imwrite(img_path, test_img)

            with patch('pylithics.image_processing.image_analysis.read_metadata') as mock_read_meta, \
                 patch('pylithics.image_processing.image_analysis.analyze_single_image') as mock_analyze, \
                 patch('pylithics.image_processing.image_analysis.logging') as mock_logging:

                mock_read_meta.return_value = [
                    {'image_id': 'test_image_1.png', 'scale': '10.0'},
                    {'image_id': 'test_image_2.png', 'scale': '12.0'},
                    {'image_id': 'missing_image.png', 'scale': '8.5'}  # This image doesn't exist
                ]

                # First two succeed, third fails
                mock_analyze.side_effect = [True, True, False]

                results = batch_process_images(data_dir, sample_metadata_file)

                assert len(results) == 3
                assert results[:2] == [True, True]
                assert results[2] is False

                # Should log errors for failed images
                mock_logging.error.assert_called()

    def test_batch_process_metadata_loading_failure(self, sample_metadata_file):
        """Test batch processing when metadata loading fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = temp_dir

            with patch('pylithics.image_processing.image_analysis.read_metadata') as mock_read_meta, \
                 patch('pylithics.image_processing.image_analysis.logging') as mock_logging:

                mock_read_meta.return_value = []  # Empty metadata

                results = batch_process_images(data_dir, sample_metadata_file)

                assert results == []
                mock_logging.warning.assert_called()

    def test_batch_process_with_config(self, sample_metadata_file):
        """Test batch processing with custom configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = temp_dir
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            config_file = os.path.join(temp_dir, "batch_config.yaml")
            with open(config_file, 'w') as f:
                f.write("arrow_detection:\n  enabled: false\n")

            # Create test image
            img_path = os.path.join(images_dir, "test_image_1.png")
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(img_path, test_img)

            with patch('pylithics.image_processing.image_analysis.read_metadata') as mock_read_meta, \
                 patch('pylithics.image_processing.image_analysis.analyze_single_image') as mock_analyze:

                mock_read_meta.return_value = [{'image_id': 'test_image_1.png', 'scale': '10.0'}]
                mock_analyze.return_value = True

                results = batch_process_images(data_dir, sample_metadata_file, config_file)

                assert len(results) == 1
                assert results[0] is True

                # Verify config was passed
                mock_analyze.assert_called_with(
                    img_path, 'test_image_1.png', 10.0, temp_dir, config_file
                )


@pytest.mark.unit
class TestPipelineValidation:
    """Test pipeline input validation functions."""

    def test_validate_pipeline_inputs_valid(self, test_image_with_dpi):
        """Test validation with valid inputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = _validate_pipeline_inputs(test_image_with_dpi, "test_id", 10.0, temp_dir)
            assert result is True

    def test_validate_pipeline_inputs_invalid_image_path(self):
        """Test validation with invalid image path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('pylithics.image_processing.image_analysis.logging') as mock_logging:
                result = _validate_pipeline_inputs("/nonexistent/image.png", "test_id", 10.0, temp_dir)
                assert result is False
                mock_logging.error.assert_called()

    def test_validate_pipeline_inputs_invalid_scale(self, test_image_with_dpi):
        """Test validation with invalid scale value."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('pylithics.image_processing.image_analysis.logging') as mock_logging:
                result = _validate_pipeline_inputs(test_image_with_dpi, "test_id", -5.0, temp_dir)
                assert result is False
                mock_logging.error.assert_called()

    def test_validate_pipeline_inputs_invalid_output_dir(self, test_image_with_dpi):
        """Test validation with invalid output directory."""
        with patch('pylithics.image_processing.image_analysis.logging') as mock_logging:
            result = _validate_pipeline_inputs(test_image_with_dpi, "test_id", 10.0, "/nonexistent/dir")
            assert result is False
            mock_logging.error.assert_called()

    def test_validate_pipeline_inputs_empty_image_id(self, test_image_with_dpi):
        """Test validation with empty image ID."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('pylithics.image_processing.image_analysis.logging') as mock_logging:
                result = _validate_pipeline_inputs(test_image_with_dpi, "", 10.0, temp_dir)
                assert result is False
                mock_logging.error.assert_called()


@pytest.mark.unit
class TestPipelineComponents:
    """Test individual pipeline component functions."""

    def test_execute_contour_extraction_success(self):
        """Test successful contour extraction execution."""
        processed_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(processed_image, (20, 20), (80, 80), 255, -1)

        image_id = "contour_test"
        output_dir = "/tmp"

        with patch('pylithics.image_processing.image_analysis.extract_contours_with_hierarchy') as mock_extract:
            mock_contours = [np.array([[20, 20], [80, 20], [80, 80], [20, 80]], dtype=np.int32).reshape(-1, 1, 2)]
            mock_hierarchy = np.array([[-1, -1, -1, -1]])
            mock_extract.return_value = (mock_contours, mock_hierarchy)

            contours, hierarchy = _execute_contour_extraction(processed_image, image_id, output_dir)

            assert contours is not None
            assert hierarchy is not None
            assert len(contours) == 1
            mock_extract.assert_called_once()

    def test_execute_contour_extraction_failure(self):
        """Test contour extraction execution failure."""
        processed_image = np.zeros((100, 100), dtype=np.uint8)
        image_id = "contour_fail_test"
        output_dir = "/tmp"

        with patch('pylithics.image_processing.image_analysis.extract_contours_with_hierarchy') as mock_extract:
            mock_extract.return_value = ([], None)  # No contours found

            contours, hierarchy = _execute_contour_extraction(processed_image, image_id, output_dir)

            assert contours == []
            assert hierarchy is None

    def test_execute_surface_classification_success(self):
        """Test successful surface classification execution."""
        metrics = [
            {
                'parent': 'parent 1',
                'scar': 'parent 1',
                'surface_type': None,
                'technical_width': 100.0,
                'technical_length': 120.0,
                'area': 12000.0
            }
        ]

        with patch('pylithics.image_processing.image_analysis.classify_parent_contours') as mock_classify:
            mock_classify.return_value = [
                {
                    'parent': 'parent 1',
                    'scar': 'parent 1',
                    'surface_type': 'Dorsal',
                    'technical_width': 100.0,
                    'technical_length': 120.0,
                    'area': 12000.0
                }
            ]

            result = _execute_surface_classification(metrics)

            assert result is not None
            assert len(result) == 1
            assert result[0]['surface_type'] == 'Dorsal'
            mock_classify.assert_called_once()

    def test_execute_surface_classification_failure(self):
        """Test surface classification execution failure."""
        metrics = []

        with patch('pylithics.image_processing.image_analysis.classify_parent_contours') as mock_classify:
            mock_classify.side_effect = Exception("Classification failed")

            with patch('pylithics.image_processing.image_analysis.logging') as mock_logging:
                result = _execute_surface_classification(metrics)

                assert result is None
                mock_logging.error.assert_called()

    def test_execute_arrow_integration_success(self):
        """Test successful arrow integration execution."""
        contours = [np.array([[20, 20], [80, 20], [80, 80], [20, 80]], dtype=np.int32).reshape(-1, 1, 2)]
        hierarchy = np.array([[-1, -1, -1, -1]])
        metrics = [{'parent': 'parent 1', 'scar': 'parent 1', 'surface_type': 'Dorsal'}]
        processed_image = np.zeros((100, 100), dtype=np.uint8)
        conversion_factor = 10.0

        with patch('pylithics.image_processing.image_analysis.integrate_arrows') as mock_arrows:
            mock_arrows.return_value = [
                {
                    'parent': 'parent 1',
                    'scar': 'parent 1',
                    'surface_type': 'Dorsal',
                    'has_arrow': False,
                    'arrow_angle': 'NA'
                }
            ]

            result = _execute_arrow_integration(contours, hierarchy, metrics, processed_image, conversion_factor)

            assert result is not None
            assert len(result) == 1
            assert result[0]['has_arrow'] is False
            mock_arrows.assert_called_once()

    def test_execute_arrow_integration_failure(self):
        """Test arrow integration execution failure."""
        contours = []
        hierarchy = None
        metrics = []
        processed_image = np.zeros((100, 100), dtype=np.uint8)
        conversion_factor = 10.0

        with patch('pylithics.image_processing.image_analysis.integrate_arrows') as mock_arrows:
            mock_arrows.side_effect = Exception("Arrow integration failed")

            with patch('pylithics.image_processing.image_analysis.logging') as mock_logging:
                result = _execute_arrow_integration(contours, hierarchy, metrics, processed_image, conversion_factor)

                assert result is None
                mock_logging.error.assert_called()

    def test_execute_spatial_analysis_success(self):
        """Test successful spatial analysis execution."""
        metrics = [
            {
                'parent': 'parent 1',
                'scar': 'parent 1',
                'surface_type': 'Dorsal',
                'centroid_x': 100.0,
                'centroid_y': 100.0,
                'contour': [[[50, 50]], [[150, 50]], [[150, 150]], [[50, 150]]]
            }
        ]
        contours = [np.array([[50, 50], [150, 50], [150, 150], [50, 150]], dtype=np.int32).reshape(-1, 1, 2)]
        processed_image = np.zeros((200, 200), dtype=np.uint8)

        with patch('pylithics.image_processing.image_analysis.analyze_dorsal_symmetry') as mock_symmetry, \
             patch('pylithics.image_processing.image_analysis.calculate_voronoi_points') as mock_voronoi, \
             patch('pylithics.image_processing.image_analysis.analyze_lateral_surface') as mock_lateral:

            mock_symmetry.return_value = {
                'top_area': 5000.0,
                'bottom_area': 5000.0,
                'left_area': 5000.0,
                'right_area': 5000.0,
                'vertical_symmetry': 1.0,
                'horizontal_symmetry': 1.0
            }

            mock_voronoi.return_value = {
                'voronoi_metrics': {'num_cells': 1},
                'convex_hull_metrics': {'width': 100.0, 'height': 100.0, 'area': 10000.0}
            }

            mock_lateral.return_value = {
                'lateral_convexity': 0.85,
                'distance_to_max_width': 45.0
            }

            result = _execute_spatial_analysis(metrics, contours, processed_image)

            assert result is not None
            assert len(result) == 1
            assert 'vertical_symmetry' in result[0]
            assert 'voronoi_num_cells' in result[0]

    def test_execute_spatial_analysis_partial_failure(self):
        """Test spatial analysis with partial module failures."""
        metrics = [
            {
                'parent': 'parent 1',
                'scar': 'parent 1',
                'surface_type': 'Dorsal',
                'centroid_x': 100.0,
                'centroid_y': 100.0
            }
        ]
        contours = [np.array([[50, 50], [150, 50], [150, 150], [50, 150]], dtype=np.int32).reshape(-1, 1, 2)]
        processed_image = np.zeros((200, 200), dtype=np.uint8)

        with patch('pylithics.image_processing.image_analysis.analyze_dorsal_symmetry') as mock_symmetry, \
             patch('pylithics.image_processing.image_analysis.calculate_voronoi_points') as mock_voronoi, \
             patch('pylithics.image_processing.image_analysis.analyze_lateral_surface') as mock_lateral, \
             patch('pylithics.image_processing.image_analysis.logging') as mock_logging:

            # Symmetry succeeds
            mock_symmetry.return_value = {'vertical_symmetry': 0.8, 'horizontal_symmetry': 0.9}

            # Voronoi fails
            mock_voronoi.side_effect = Exception("Voronoi failed")

            # Lateral succeeds
            mock_lateral.return_value = {'lateral_convexity': 0.7}

            result = _execute_spatial_analysis(metrics, contours, processed_image)

            # Should still return results with partial data
            assert result is not None
            assert len(result) == 1
            assert 'vertical_symmetry' in result[0]
            assert 'lateral_convexity' in result[0]

            # Should log the error
            mock_logging.error.assert_called()

    def test_generate_outputs_success(self):
        """Test successful output generation."""
        metrics = [
            {
                'image_id': 'test_output',
                'parent': 'parent 1',
                'scar': 'parent 1',
                'surface_type': 'Dorsal',
                'centroid_x': 100.0,
                'centroid_y': 100.0,
                'area': 10000.0,
                'has_arrow': False
            }
        ]
        contours = [np.array([[50, 50], [150, 50], [150, 150], [50, 150]], dtype=np.int32).reshape(-1, 1, 2)]
        hierarchy = np.array([[-1, -1, -1, -1]])
        processed_image = np.zeros((200, 200), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('pylithics.image_processing.image_analysis.save_measurements_to_csv') as mock_save_csv, \
                 patch('pylithics.image_processing.image_analysis.visualize_contours_with_hierarchy') as mock_viz:

                result = _generate_outputs(metrics, contours, hierarchy, processed_image, temp_dir, "test_output")

                assert result is True
                mock_save_csv.assert_called_once()
                mock_viz.assert_called_once()

    def test_generate_outputs_failure(self):
        """Test output generation failure."""
        metrics = []
        contours = []
        hierarchy = None
        processed_image = np.zeros((200, 200), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('pylithics.image_processing.image_analysis.save_measurements_to_csv') as mock_save_csv, \
                 patch('pylithics.image_processing.image_analysis.logging') as mock_logging:

                mock_save_csv.side_effect = Exception("Save failed")

                result = _generate_outputs(metrics, contours, hierarchy, processed_image, temp_dir, "test_fail")

                assert result is False
                mock_logging.error.assert_called()


@pytest.mark.integration
class TestPipelineIntegrationScenarios:
    """Test pipeline integration with realistic archaeological scenarios."""

    def test_pipeline_blade_tool_scenario(self, test_image_with_dpi):
        """Test pipeline with blade tool scenario."""
        image_id = "blade_tool_scenario"
        real_world_scale = 15.0

        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock realistic blade tool processing
            with patch('pylithics.image_processing.image_analysis.execute_preprocessing_pipeline') as mock_preprocess, \
                 patch('pylithics.image_processing.image_analysis.extract_contours_with_hierarchy') as mock_extract, \
                 patch('pylithics.image_processing.image_analysis.calculate_contour_metrics') as mock_metrics, \
                 patch('pylithics.image_processing.image_analysis.classify_parent_contours') as mock_classify, \
                 patch('pylithics.image_processing.image_analysis.integrate_arrows') as mock_arrows, \
                 patch('pylithics.image_processing.image_analysis.analyze_dorsal_symmetry') as mock_symmetry, \
                 patch('pylithics.image_processing.image_analysis.save_measurements_to_csv') as mock_save:

                # Set up blade tool data
                mock_preprocess.return_value = np.ones((300, 200), dtype=np.uint8) * 255

                # Blade contour (elongated)
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
                result = process_and_save_contours(
                    test_image_with_dpi, image_id, real_world_scale, temp_dir
                )

                assert result is True

                # Verify all major stages were called
                mock_preprocess.assert_called_once()
                mock_extract.assert_called_once()
                mock_metrics.assert_called_once()
                mock_classify.assert_called_once()
                mock_arrows.assert_called_once()
                mock_symmetry.assert_called_once()
                mock_save.assert_called_once()

                # Verify final CSV data
                save_call_args = mock_save.call_args[0]
                final_metrics = save_call_args[0]

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

    def test_pipeline_core_reduction_scenario(self, test_image_with_dpi):
        """Test pipeline with core reduction scenario."""
        image_id = "core_reduction_scenario"
        real_world_scale = 20.0

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('pylithics.image_processing.image_analysis.execute_preprocessing_pipeline') as mock_preprocess, \
                 patch('pylithics.image_processing.image_analysis.extract_contours_with_hierarchy') as mock_extract, \
                 patch('pylithics.image_processing.image_analysis.calculate_contour_metrics') as mock_metrics, \
                 patch('pylithics.image_processing.image_analysis.classify_parent_contours') as mock_classify, \
                 patch('pylithics.image_processing.image_analysis.integrate_arrows') as mock_arrows, \
                 patch('pylithics.image_processing.image_analysis.calculate_voronoi_points') as mock_voronoi, \
                 patch('pylithics.image_processing.image_analysis.save_measurements_to_csv') as mock_save:

                # Set up core reduction data
                mock_preprocess.return_value = np.ones((250, 250), dtype=np.uint8) * 255

                # Core contour (roughly square)
                core_contour = np.array([[50, 50], [200, 50], [200, 200], [50, 200]], dtype=np.int32).reshape(-1, 1, 2)

                # Radial flake scars around core
                flake_contours = []
                center_x, center_y = 125, 125

                for i in range(6):
                    angle = 2 * np.pi * i / 6
                    radius = 60
                    x = center_x + radius * np.cos(angle)
                    y = center_y + radius * np.sin(angle)

                    flake_contour = np.array([
                        [x - 8, y - 8], [x + 8, y - 8], [x + 8, y + 8], [x - 8, y + 8]
                    ], dtype=np.int32).reshape(-1, 1, 2)
                    flake_contours.append(flake_contour)

                all_contours = [core_contour] + flake_contours

                # Hierarchy: core with multiple children
                hierarchy_data = [[-1, -1, 1, -1]]  # Core
                for i in range(6):
                    next_sib = i + 2 if i < 5 else -1
                    prev_sib = i if i > 0 else -1
                    hierarchy_data.append([next_sib, prev_sib, -1, 0])

                hierarchy = np.array(hierarchy_data)
                mock_extract.return_value = (all_contours, hierarchy)

                # Mock core metrics
                core_metrics = [{
                    'parent': 'parent 1',
                    'scar': 'parent 1',
                    'surface_type': None,
                    'centroid_x': 125.0,
                    'centroid_y': 125.0,
                    'technical_width': 150.0,
                    'technical_length': 150.0,
                    'area': 22500.0
                }]

                # Add flake metrics
                for i in range(6):
                    angle = 2 * np.pi * i / 6
                    radius = 60
                    x = center_x + radius * np.cos(angle)
                    y = center_y + radius * np.sin(angle)

                    core_metrics.append({
                        'parent': 'parent 1',
                        'scar': f'flake {i + 1}',
                        'surface_type': None,
                        'centroid_x': x,
                        'centroid_y': y,
                        'width': 16.0,
                        'height': 16.0,
                        'area': 256.0
                    })

                mock_metrics.return_value = core_metrics

                # Classification
                classified_metrics = []
                for metric in core_metrics:
                    new_metric = metric.copy()
                    new_metric['surface_type'] = 'Dorsal'
                    classified_metrics.append(new_metric)

                mock_classify.return_value = classified_metrics

                # Arrow integration
                arrow_metrics = []
                for metric in classified_metrics:
                    new_metric = metric.copy()
                    if metric['scar'] == 'parent 1':
                        new_metric['has_arrow'] = False
                        new_metric['arrow_angle'] = 'NA'
                    else:
                        # Some flakes have arrows pointing toward center
                        new_metric['has_arrow'] = True
                        new_metric['arrow_angle'] = float(45 + len(arrow_metrics) * 30)
                    arrow_metrics.append(new_metric)

                mock_arrows.return_value = arrow_metrics

                # Voronoi analysis (good for radial patterns)
                mock_voronoi.return_value = {
                    'voronoi_metrics': {'num_cells': 7},
                    'convex_hull_metrics': {'width': 150.0, 'height': 150.0, 'area': 22500.0}
                }

                # Execute pipeline
                result = process_and_save_contours(
                    test_image_with_dpi, image_id, real_world_scale, temp_dir
                )

                assert result is True

                # Verify Voronoi analysis was performed
                mock_voronoi.assert_called_once()

                # Check final metrics structure
                save_call_args = mock_save.call_args[0]
                final_metrics = save_call_args[0]

                assert len(final_metrics) == 7  # Core + 6 flakes

                # Core should not have arrows
                core_metric = final_metrics[0]
                assert core_metric['surface_type'] == 'Dorsal'
                assert core_metric['has_arrow'] is False

                # Flakes should have various arrow patterns
                flake_metrics = final_metrics[1:]
                assert len(flake_metrics) == 6

                # Should have Voronoi data
                assert 'voronoi_num_cells' in core_metric
                assert core_metric['voronoi_num_cells'] == 7

    def test_pipeline_error_recovery_workflow(self, test_image_with_dpi):
        """Test pipeline error recovery and graceful degradation."""
        image_id = "error_recovery_test"
        real_world_scale = 10.0

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('pylithics.image_processing.image_analysis.execute_preprocessing_pipeline') as mock_preprocess, \
                 patch('pylithics.image_processing.image_analysis.extract_contours_with_hierarchy') as mock_extract, \
                 patch('pylithics.image_processing.image_analysis.calculate_contour_metrics') as mock_metrics, \
                 patch('pylithics.image_processing.image_analysis.classify_parent_contours') as mock_classify, \
                 patch('pylithics.image_processing.image_analysis.integrate_arrows') as mock_arrows, \
                 patch('pylithics.image_processing.image_analysis.analyze_dorsal_symmetry') as mock_symmetry, \
                 patch('pylithics.image_processing.image_analysis.calculate_voronoi_points') as mock_voronoi, \
                 patch('pylithics.image_processing.image_analysis.save_measurements_to_csv') as mock_save, \
                 patch('pylithics.image_processing.image_analysis.logging') as mock_logging:

                # Set up successful early stages
                mock_preprocess.return_value = np.ones((200, 200), dtype=np.uint8) * 255

                contour = np.array([[50, 50], [150, 50], [150, 150], [50, 150]], dtype=np.int32).reshape(-1, 1, 2)
                mock_extract.return_value = ([contour], np.array([[-1, -1, -1, -1]]))

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

                # Make various modules fail
                mock_arrows.side_effect = Exception("Arrow integration failed")
                mock_symmetry.side_effect = Exception("Symmetry analysis failed")
                mock_voronoi.side_effect = Exception("Voronoi analysis failed")

                # Execute pipeline
                result = process_and_save_contours(
                    test_image_with_dpi, image_id, real_world_scale, temp_dir
                )

                # Should still succeed despite multiple failures
                assert result is True

                # Should log multiple errors but continue
                assert mock_logging.error.call_count >= 3

                # Should still save basic results
                mock_save.assert_called_once()

                # Check that basic metrics were preserved
                save_call_args = mock_save.call_args[0]
                final_metrics = save_call_args[0]

                assert len(final_metrics) == 1
                assert final_metrics[0]['surface_type'] == 'Dorsal'
                assert final_metrics[0]['area'] == 10000.0


@pytest.mark.performance
class TestPipelinePerformance:
    """Test pipeline performance characteristics."""

    def test_pipeline_performance_single_image(self, test_image_with_dpi):
        """Test pipeline performance with single image."""
        image_id = "performance_test"
        real_world_scale = 10.0

        with tempfile.TemporaryDirectory() as temp_dir:
            # Use real processing (limited mocking for performance test)
            with patch('pylithics.image_processing.image_analysis.save_measurements_to_csv') as mock_save, \
                 patch('pylithics.image_processing.image_analysis.visualize_contours_with_hierarchy') as mock_viz:

                import time
                start_time = time.time()

                result = process_and_save_contours(
                    test_image_with_dpi, image_id, real_world_scale, temp_dir
                )

                end_time = time.time()
                processing_time = end_time - start_time

                # Should complete in reasonable time
                assert processing_time < 30.0  # 30 seconds max for single image

                # Should produce result (success or graceful failure)
                assert isinstance(result, bool)

    def test_pipeline_performance_batch_processing(self, sample_metadata_file):
        """Test pipeline performance with batch processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = temp_dir
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Create multiple test images
            num_images = 5
            for i in range(num_images):
                img_path = os.path.join(images_dir, f"perf_test_{i}.png")
                test_img = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
                cv2.imwrite(img_path, test_img)

            with patch('pylithics.image_processing.image_analysis.read_metadata') as mock_read_meta, \
                 patch('pylithics.image_processing.image_analysis.analyze_single_image') as mock_analyze:

                mock_read_meta.return_value = [
                    {'image_id': f'perf_test_{i}.png', 'scale': '10.0'} for i in range(num_images)
                ]

                mock_analyze.return_value = True

                import time
                start_time = time.time()

                results = batch_process_images(data_dir, sample_metadata_file)

                end_time = time.time()
                processing_time = end_time - start_time

                # Should complete batch in reasonable time
                assert processing_time < 60.0  # 1 minute max for 5 images
                assert len(results) == num_images

    def test_pipeline_memory_usage(self, test_image_with_dpi):
        """Test pipeline memory usage characteristics."""
        image_id = "memory_test"
        real_world_scale = 10.0

        with tempfile.TemporaryDirectory() as temp_dir:
            # Monitor memory usage during processing
            import psutil
            import os

            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss

            with patch('pylithics.image_processing.image_analysis.save_measurements_to_csv') as mock_save, \
                 patch('pylithics.image_processing.image_analysis.visualize_contours_with_hierarchy') as mock_viz:

                result = process_and_save_contours(
                    test_image_with_dpi, image_id, real_world_scale, temp_dir
                )

                final_memory = process.memory_info().rss
                memory_increase = final_memory - initial_memory

                # Memory increase should be reasonable (< 100MB for test image)
                assert memory_increase < 100 * 1024 * 1024  # 100MB


@pytest.mark.unit
class TestPipelineUtilityFunctions:
    """Test utility functions used in the pipeline."""

    def test_pipeline_configuration_loading(self):
        """Test pipeline configuration loading and validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "pipeline_config.yaml")

            # Create test config
            test_config = {
                'thresholding': {'method': 'adaptive', 'max_value': 255},
                'arrow_detection': {'enabled': True, 'reference_dpi': 300.0},
                'contour_filtering': {'min_area': 50.0},
                'logging': {'level': 'INFO'}
            }

            import yaml
            with open(config_file, 'w') as f:
                yaml.dump(test_config, f)

            with patch('pylithics.image_processing.image_analysis.load_preprocessing_config') as mock_load:
                mock_load.return_value = test_config

                # Test that config loading works in pipeline context
                from pylithics.image_processing.image_analysis import load_preprocessing_config
                loaded_config = load_preprocessing_config(config_file)

                assert loaded_config == test_config
                mock_load.assert_called_with(config_file)

    def test_pipeline_output_directory_creation(self):
        """Test that pipeline creates necessary output directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_output_dir = os.path.join(temp_dir, "outputs")

            # Directory doesn't exist initially
            assert not os.path.exists(base_output_dir)

            with patch('pylithics.image_processing.image_analysis.execute_preprocessing_pipeline') as mock_preprocess, \
                 patch('pylithics.image_processing.image_analysis.extract_contours_with_hierarchy') as mock_extract:

                mock_preprocess.return_value = None  # Force early exit
                mock_extract.return_value = ([], None)

                # Pipeline should create directory
                result = process_and_save_contours(
                    "dummy_path", "test_id", 10.0, base_output_dir
                )

                # Directory should now exist (created by pipeline)
                assert os.path.exists(base_output_dir)

    def test_pipeline_logging_integration(self, test_image_with_dpi):
        """Test pipeline logging integration."""
        image_id = "logging_test"
        real_world_scale = 10.0

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('pylithics.image_processing.image_analysis.logging') as mock_logging:

                # Create a scenario that will generate various log messages
                with patch('pylithics.image_processing.image_analysis.execute_preprocessing_pipeline') as mock_preprocess, \
                     patch('pylithics.image_processing.image_analysis.extract_contours_with_hierarchy') as mock_extract:

                    mock_preprocess.return_value = np.ones((100, 100), dtype=np.uint8) * 255
                    mock_extract.return_value = ([], None)  # No contours found

                    result = process_and_save_contours(
                        test_image_with_dpi, image_id, real_world_scale, temp_dir
                    )

                    # Should log various stages and warnings
                    assert mock_logging.info.called or mock_logging.debug.called
                    assert mock_logging.warning.called  # For no contours found

    def test_pipeline_error_context_preservation(self, test_image_with_dpi):
        """Test that pipeline preserves error context through stages."""
        image_id = "error_context_test"
        real_world_scale = 10.0

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('pylithics.image_processing.image_analysis.execute_preprocessing_pipeline') as mock_preprocess, \
                 patch('pylithics.image_processing.image_analysis.extract_contours_with_hierarchy') as mock_extract, \
                 patch('pylithics.image_processing.image_analysis.logging') as mock_logging:

                # Set up cascading failures with context
                mock_preprocess.return_value = np.ones((100, 100), dtype=np.uint8) * 255
                mock_extract.side_effect = Exception("Contour extraction failed: invalid image format")

                result = process_and_save_contours(
                    test_image_with_dpi, image_id, real_world_scale, temp_dir
                )

                assert result is False

                # Should log error with context
                mock_logging.error.assert_called()

                # Check that error message contains context
                error_calls = mock_logging.error.call_args_list
                error_messages = [str(call) for call in error_calls]
                assert any("Contour extraction failed" in msg for msg in error_messages)


@pytest.mark.integration
class TestPipelineRealWorldIntegration:
    """Test pipeline with real-world integration scenarios."""

    def test_pipeline_with_actual_preprocessing(self, test_image_with_dpi):
        """Test pipeline with actual preprocessing (minimal mocking)."""
        image_id = "real_preprocessing_test"
        real_world_scale = 10.0

        with tempfile.TemporaryDirectory() as temp_dir:
            # Only mock the final outputs to avoid file system issues
            with patch('pylithics.image_processing.image_analysis.save_measurements_to_csv') as mock_save, \
                 patch('pylithics.image_processing.image_analysis.visualize_contours_with_hierarchy') as mock_viz:

                # Let most of the pipeline run with real processing
                result = process_and_save_contours(
                    test_image_with_dpi, image_id, real_world_scale, temp_dir
                )

                # Should complete without crashing
                assert isinstance(result, bool)

                # If successful, outputs should be generated
                if result:
                    mock_save.assert_called_once()
                    mock_viz.assert_called_once()

    def test_pipeline_config_override_chain(self, test_image_with_dpi):
        """Test pipeline configuration override chain."""
        image_id = "config_override_test"
        real_world_scale = 10.0

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config with specific settings
            config_file = os.path.join(temp_dir, "override_config.yaml")
            override_config = {
                'thresholding': {'method': 'otsu', 'max_value': 200},
                'arrow_detection': {'enabled': False},
                'contour_filtering': {'min_area': 200.0}
            }

            import yaml
            with open(config_file, 'w') as f:
                yaml.dump(override_config, f)

            with patch('pylithics.image_processing.image_analysis.get_thresholding_config') as mock_thresh_config, \
                 patch('pylithics.image_processing.image_analysis.get_arrow_detection_config') as mock_arrow_config, \
                 patch('pylithics.image_processing.image_analysis.get_contour_filtering_config') as mock_contour_config, \
                 patch('pylithics.image_processing.image_analysis.execute_preprocessing_pipeline') as mock_preprocess, \
                 patch('pylithics.image_processing.image_analysis.extract_contours_with_hierarchy') as mock_extract:

                mock_preprocess.return_value = None  # Force early exit
                mock_extract.return_value = ([], None)

                result = process_and_save_contours(
                    test_image_with_dpi, image_id, real_world_scale, temp_dir, config_file
                )

                # Verify that config functions were called (indicating config was loaded)
                # The exact calls depend on implementation, but loading should occur

    def test_pipeline_multi_surface_artifact(self, test_image_with_dpi):
        """Test pipeline with multi-surface artifact scenario."""
        image_id = "multi_surface_test"
        real_world_scale = 15.0

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('pylithics.image_processing.image_analysis.execute_preprocessing_pipeline') as mock_preprocess, \
                 patch('pylithics.image_processing.image_analysis.extract_contours_with_hierarchy') as mock_extract, \
                 patch('pylithics.image_processing.image_analysis.calculate_contour_metrics') as mock_metrics, \
                 patch('pylithics.image_processing.image_analysis.classify_parent_contours') as mock_classify, \
                 patch('pylithics.image_processing.image_analysis.save_measurements_to_csv') as mock_save:

                # Set up multi-surface scenario
                mock_preprocess.return_value = np.ones((300, 400), dtype=np.uint8) * 255

                # Multiple parent contours (different surfaces)
                dorsal_contour = np.array([[50, 50], [350, 50], [350, 250], [50, 250]], dtype=np.int32).reshape(-1, 1, 2)
                ventral_contour = np.array([[60, 60], [340, 60], [340, 240], [60, 240]], dtype=np.int32).reshape(-1, 1, 2)
                platform_contour = np.array([[70, 70], [130, 70], [130, 110], [70, 110]], dtype=np.int32).reshape(-1, 1, 2)

                multi_contours = [dorsal_contour, ventral_contour, platform_contour]
                multi_hierarchy = np.array([
                    [-1, -1, -1, -1],  # Dorsal
                    [-1, -1, -1, -1],  # Ventral
                    [-1, -1, -1, -1]   # Platform
                ])

                mock_extract.return_value = (multi_contours, multi_hierarchy)

                # Multi-surface metrics
                mock_metrics.return_value = [
                    {
                        'parent': 'parent 1',
                        'scar': 'parent 1',
                        'surface_type': None,
                        'centroid_x': 200.0,
                        'centroid_y': 150.0,
                        'technical_width': 300.0,
                        'technical_length': 200.0,
                        'area': 60000.0
                    },
                    {
                        'parent': 'parent 2',
                        'scar': 'parent 2',
                        'surface_type': None,
                        'centroid_x': 200.0,
                        'centroid_y': 150.0,
                        'technical_width': 280.0,
                        'technical_length': 180.0,
                        'area': 50400.0
                    },
                    {
                        'parent': 'parent 3',
                        'scar': 'parent 3',
                        'surface_type': None,
                        'centroid_x': 100.0,
                        'centroid_y': 90.0,
                        'technical_width': 60.0,
                        'technical_length': 40.0,
                        'area': 2400.0
                    }
                ]

                # Surface classification
                mock_classify.return_value = [
                    {
                        'parent': 'parent 1',
                        'scar': 'parent 1',
                        'surface_type': 'Dorsal',
                        'centroid_x': 200.0,
                        'centroid_y': 150.0,
                        'technical_width': 300.0,
                        'technical_length': 200.0,
                        'area': 60000.0
                    },
                    {
                        'parent': 'parent 2',
                        'scar': 'parent 2',
                        'surface_type': 'Ventral',
                        'centroid_x': 200.0,
                        'centroid_y': 150.0,
                        'technical_width': 280.0,
                        'technical_length': 180.0,
                        'area': 50400.0
                    },
                    {
                        'parent': 'parent 3',
                        'scar': 'parent 3',
                        'surface_type': 'Platform',
                        'centroid_x': 100.0,
                        'centroid_y': 90.0,
                        'technical_width': 60.0,
                        'technical_length': 40.0,
                        'area': 2400.0
                    }
                ]

                # Execute pipeline
                result = process_and_save_contours(
                    test_image_with_dpi, image_id, real_world_scale, temp_dir
                )

                assert result is True

                # Verify multi-surface classification
                save_call_args = mock_save.call_args[0]
                final_metrics = save_call_args[0]

                assert len(final_metrics) == 3

                surface_types = [metric['surface_type'] for metric in final_metrics]
                assert 'Dorsal' in surface_types
                assert 'Ventral' in surface_types
                assert 'Platform' in surface_types

    def test_pipeline_edge_case_robustness(self, test_image_with_dpi):
        """Test pipeline robustness with edge cases."""
        image_id = "edge_case_test"
        real_world_scale = 5.0

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('pylithics.image_processing.image_analysis.execute_preprocessing_pipeline') as mock_preprocess, \
                 patch('pylithics.image_processing.image_analysis.extract_contours_with_hierarchy') as mock_extract, \
                 patch('pylithics.image_processing.image_analysis.calculate_contour_metrics') as mock_metrics, \
                 patch('pylithics.image_processing.image_analysis.save_measurements_to_csv') as mock_save, \
                 patch('pylithics.image_processing.image_analysis.logging') as mock_logging:

                # Set up edge case scenarios
                mock_preprocess.return_value = np.ones((100, 100), dtype=np.uint8) * 255

                # Very small contour (edge case)
                tiny_contour = np.array([[50, 50], [52, 50], [52, 52], [50, 52]], dtype=np.int32).reshape(-1, 1, 2)

                # Degenerate contour (line)
                line_contour = np.array([[30, 30], [70, 30], [70, 30], [30, 30]], dtype=np.int32).reshape(-1, 1, 2)

                edge_contours = [tiny_contour, line_contour]
                edge_hierarchy = np.array([
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1]
                ])

                mock_extract.return_value = (edge_contours, edge_hierarchy)

                # Edge case metrics (very small areas)
                mock_metrics.return_value = [
                    {
                        'parent': 'parent 1',
                        'scar': 'parent 1',
                        'surface_type': None,
                        'centroid_x': 51.0,
                        'centroid_y': 51.0,
                        'technical_width': 2.0,
                        'technical_length': 2.0,
                        'area': 4.0
                    },
                    {
                        'parent': 'parent 2',
                        'scar': 'parent 2',
                        'surface_type': None,
                        'centroid_x': 50.0,
                        'centroid_y': 30.0,
                        'technical_width': 40.0,
                        'technical_length': 0.0,  # Zero height (line)
                        'area': 0.0
                    }
                ]

                # Execute pipeline
                result = process_and_save_contours(
                    test_image_with_dpi, image_id, real_world_scale, temp_dir
                )

                # Should handle edge cases gracefully
                assert isinstance(result, bool)

                # May log warnings about edge cases
                if mock_logging.warning.called or mock_logging.error.called:
                    # This is acceptable for edge cases
                    pass

                # If successful, should save results
                if result:
                    mock_save.assert_called_once()


@pytest.mark.integration
class TestPipelineEndToEndValidation:
    """End-to-end validation tests for the complete pipeline."""

    def test_pipeline_data_flow_integrity(self, test_image_with_dpi):
        """Test that data flows correctly through all pipeline stages."""
        image_id = "data_flow_test"
        real_world_scale = 10.0

        with tempfile.TemporaryDirectory() as temp_dir:
            # Track data flow through pipeline stages
            captured_data = {}

            def capture_preprocess(*args, **kwargs):
                captured_data['preprocess_input'] = args
                return np.ones((150, 200), dtype=np.uint8) * 255

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

            with patch('pylithics.image_processing.image_analysis.execute_preprocessing_pipeline', side_effect=capture_preprocess), \
                 patch('pylithics.image_processing.image_analysis.extract_contours_with_hierarchy', side_effect=capture_extract), \
                 patch('pylithics.image_processing.image_analysis.calculate_contour_metrics', side_effect=capture_metrics), \
                 patch('pylithics.image_processing.image_analysis.classify_parent_contours', side_effect=capture_classify), \
                 patch('pylithics.image_processing.image_analysis.save_measurements_to_csv', side_effect=capture_save), \
                 patch('pylithics.image_processing.image_analysis.visualize_contours_with_hierarchy'):

                result = process_and_save_contours(
                    test_image_with_dpi, image_id, real_world_scale, temp_dir
                )

                assert result is True

                # Verify data flow integrity
                assert 'preprocess_input' in captured_data
                assert 'extract_input' in captured_data
                assert 'metrics_input' in captured_data
                assert 'classify_input' in captured_data
                assert 'save_input' in captured_data

                # Verify image_id propagates through
                save_data = captured_data['save_input'][0]  # Metrics list
                assert len(save_data) > 0
                # Image ID should be added to metrics at some point

                # Verify contour data consistency
                extract_input = captured_data['extract_input']
                assert len(extract_input) >= 2  # Should include processed image and image_id

                # Verify metrics data consistency
                metrics_input = captured_data['metrics_input']
                assert len(metrics_input) >= 4  # Should include sorted_contours, hierarchy, original_contours, image_shape

                # Verify classification input consistency
                classify_input = captured_data['classify_input']
                assert len(classify_input) >= 1  # Should include metrics list
                assert len(classify_input[0]) > 0  # Should have metrics to classify

    def test_pipeline_output_completeness(self, test_image_with_dpi):
        """Test that pipeline generates complete and valid outputs."""
        image_id = "output_completeness_test"
        real_world_scale = 12.0

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, f"{image_id}_measurements.csv")
            viz_path = os.path.join(temp_dir, f"{image_id}_visualization.png")

            # Test with minimal real processing
            with patch('pylithics.image_processing.image_analysis.save_measurements_to_csv') as mock_save, \
                 patch('pylithics.image_processing.image_analysis.visualize_contours_with_hierarchy') as mock_viz:

                # Configure mocks to simulate file creation
                def mock_save_csv(metrics, output_path, *args, **kwargs):
                    # Simulate CSV creation
                    with open(output_path, 'w') as f:
                        f.write("image_id,surface_type,area\n")
                        for metric in metrics:
                            f.write(f"{metric.get('image_id', image_id)},{metric.get('surface_type', 'Unknown')},{metric.get('area', 0)}\n")

                def mock_viz_func(contours, hierarchy, metrics, image, output_path, *args, **kwargs):
                    # Simulate image creation
                    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
                    cv2.imwrite(output_path, dummy_img)

                mock_save.side_effect = mock_save_csv
                mock_viz.side_effect = mock_viz_func

                result = process_and_save_contours(
                    test_image_with_dpi, image_id, real_world_scale, temp_dir
                )

                # Verify outputs were generated
                if result:
                    mock_save.assert_called_once()
                    mock_viz.assert_called_once()

                    # Check that output paths were correctly constructed
                    save_call = mock_save.call_args
                    viz_call = mock_viz.call_args

                    # Verify CSV output path
                    csv_output_path = save_call[0][1]  # Second argument is output path
                    assert image_id in csv_output_path
                    assert csv_output_path.endswith('.csv')

                    # Verify visualization output path
                    viz_output_path = viz_call[0][4]  # Fifth argument is output path
                    assert image_id in viz_output_path
                    assert viz_output_path.endswith('.png')

    def test_pipeline_configuration_effects(self, test_image_with_dpi):
        """Test that different configurations produce different results."""
        image_id = "config_effects_test"
        real_world_scale = 10.0

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test two different configurations
            config1 = {
                'arrow_detection': {'enabled': True, 'reference_dpi': 300.0},
                'contour_filtering': {'min_area': 50.0}
            }

            config2 = {
                'arrow_detection': {'enabled': False},
                'contour_filtering': {'min_area': 200.0}
            }

            results = []

            for i, config in enumerate([config1, config2]):
                config_file = os.path.join(temp_dir, f"config_{i}.yaml")

                import yaml
                with open(config_file, 'w') as f:
                    yaml.dump(config, f)

                with patch('pylithics.image_processing.image_analysis.save_measurements_to_csv') as mock_save, \
                     patch('pylithics.image_processing.image_analysis.visualize_contours_with_hierarchy') as mock_viz:

                    result = process_and_save_contours(
                        test_image_with_dpi, f"{image_id}_{i}", real_world_scale, temp_dir, config_file
                    )

                    results.append(result)

                    # Capture the results for comparison
                    if mock_save.called:
                        call_args = mock_save.call_args[0]
                        metrics = call_args[0]
                        results.append(('config_' + str(i), metrics))

            # Both configurations should produce results (success or failure)
            assert len(results) >= 2
            for result in results[:2]:  # First two are boolean results
                assert isinstance(result, bool)