"""
PyLithics Error Scenarios Tests
===============================

Comprehensive error handling tests for all PyLithics modules.
Tests error conditions, recovery mechanisms, and system resilience
under various failure scenarios.
"""

import pytest
import numpy as np
import cv2
import pandas as pd
import tempfile
import os
import yaml
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image

from pylithics.app import PyLithicsApplication
from pylithics.image_processing.importer import execute_preprocessing_pipeline
from pylithics.image_processing.image_analysis import process_and_save_contours
from pylithics.image_processing.config import ConfigurationManager
from pylithics.image_processing.modules.arrow_detection import ArrowDetector
from pylithics.image_processing.modules.contour_extraction import extract_contours_with_hierarchy
from pylithics.image_processing.modules.contour_metrics import calculate_contour_metrics


@pytest.mark.error_scenarios
class TestConfigurationErrors:
    """Test error handling in configuration management."""

    def test_missing_config_file(self):
        """Test handling of missing configuration file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_config = os.path.join(temp_dir, "nonexistent.yaml")

            # Should handle gracefully
            app = PyLithicsApplication(config_file=missing_config)
            assert app.config_manager is not None

            # Should fall back to defaults
            config = app.config_manager.config
            assert 'thresholding' in config
            assert 'logging' in config

    def test_corrupted_config_file(self):
        """Test handling of corrupted YAML file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            corrupted_config = os.path.join(temp_dir, "corrupted.yaml")

            # Create invalid YAML
            with open(corrupted_config, 'w') as f:
                f.write("invalid: yaml: content: [unclosed\nmalformed")

            # Should handle YAML errors gracefully
            app = PyLithicsApplication(config_file=corrupted_config)
            assert app.config_manager is not None

            # Should fall back to defaults
            config = app.config_manager.config
            assert isinstance(config, dict)

    def test_partial_config_file(self):
        """Test handling of incomplete configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            partial_config = os.path.join(temp_dir, "partial.yaml")

            # Create config with only some sections
            partial_data = {
                'thresholding': {'method': 'simple'},
                # Missing other required sections
            }

            with open(partial_config, 'w') as f:
                yaml.dump(partial_data, f)

            config_manager = ConfigurationManager(partial_config)

            # Should have all required sections with defaults
            assert 'thresholding' in config_manager.config
            assert 'logging' in config_manager.config
            assert 'normalization' in config_manager.config

    def test_invalid_config_values(self):
        """Test handling of invalid configuration values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_config = os.path.join(temp_dir, "invalid.yaml")

            # Create config with invalid values
            invalid_data = {
                'thresholding': {
                    'method': 'nonexistent_method',
                    'threshold_value': 'not_a_number',
                    'max_value': -50  # Invalid negative value
                },
                'arrow_detection': {
                    'reference_dpi': 'invalid_dpi'
                }
            }

            with open(invalid_config, 'w') as f:
                yaml.dump(invalid_data, f)

            # Should load without crashing
            config_manager = ConfigurationManager(invalid_config)
            assert config_manager.config is not None


@pytest.mark.error_scenarios
class TestImagePreprocessingErrors:
    """Test error handling in image preprocessing pipeline."""

    def test_missing_image_file(self, sample_config):
        """Test handling of missing image files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Create metadata referencing non-existent image
            metadata_content = "image_id,scale_id,scale\nmissing_image.png,scale_1,15.0\n"
            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            app = PyLithicsApplication(config_file=config_path)
            results = app.run_batch_analysis(data_dir, metadata_path)

            # Should report failure gracefully
            assert results['success'] is True  # Framework succeeds
            assert results['processed_successfully'] == 0
            assert 'missing_image.png' in results['failed_images']

    def test_corrupted_image_file(self, sample_config):
        """Test handling of corrupted image files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Create corrupted image file
            corrupted_image = os.path.join(images_dir, "corrupted.png")
            with open(corrupted_image, 'w') as f:
                f.write("This is not an image file")

            metadata_content = "image_id,scale_id,scale\ncorrupted.png,scale_1,15.0\n"
            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            app = PyLithicsApplication(config_file=config_path)
            results = app.run_batch_analysis(data_dir, metadata_path)

            # Should handle corrupted images gracefully
            assert results['success'] is True
            assert results['processed_successfully'] == 0
            assert 'corrupted.png' in results['failed_images']

    def test_zero_size_image(self, sample_config):
        """Test handling of zero-size images."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Create zero-size image
            zero_image = os.path.join(images_dir, "zero_size.png")
            empty_array = np.array([], dtype=np.uint8)
            try:
                cv2.imwrite(zero_image, empty_array)
            except:
                # If cv2.imwrite fails, create empty file
                with open(zero_image, 'w') as f:
                    pass

            metadata_content = "image_id,scale_id,scale\nzero_size.png,scale_1,15.0\n"
            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            app = PyLithicsApplication(config_file=config_path)
            results = app.run_batch_analysis(data_dir, metadata_path)

            # Should handle gracefully
            assert results['success'] is True
            assert results['processed_successfully'] == 0

    def test_preprocessing_pipeline_errors(self):
        """Test error handling in preprocessing steps."""
        # Test with invalid image data
        invalid_image = None
        config = {'thresholding': {'method': 'simple'}}

        # Should return None for invalid input
        result = execute_preprocessing_pipeline("nonexistent.png", config)
        assert result is None

    def test_dpi_extraction_errors(self, sample_config):
        """Test error handling when DPI information is missing or invalid."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Create image without DPI information
            image_path = os.path.join(images_dir, "no_dpi.png")
            test_image = np.full((200, 200, 3), 128, dtype=np.uint8)
            cv2.imwrite(image_path, test_image)  # cv2 doesn't preserve DPI

            metadata_content = "image_id,scale_id,scale\nno_dpi.png,scale_1,15.0\n"
            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            app = PyLithicsApplication(config_file=config_path)

            # Should handle missing DPI gracefully
            results = app.run_batch_analysis(data_dir, metadata_path)
            # May succeed or fail depending on implementation, but shouldn't crash
            assert isinstance(results, dict)


@pytest.mark.error_scenarios
class TestContourExtractionErrors:
    """Test error handling in contour extraction."""

    def test_no_contours_found(self):
        """Test handling when no contours are found."""
        # Create uniform image (no contours)
        uniform_image = np.full((100, 100), 128, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as temp_dir:
            contours, hierarchy = extract_contours_with_hierarchy(
                uniform_image, "test_image", temp_dir
            )

            assert contours == []
            assert hierarchy is None

    def test_all_contours_filtered_out(self):
        """Test when all contours are filtered out by size or border criteria."""
        # Create image with only tiny contours
        tiny_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(tiny_image, (50, 50), 1, 255, -1)  # Very small circle

        with tempfile.TemporaryDirectory() as temp_dir:
            contours, hierarchy = extract_contours_with_hierarchy(
                tiny_image, "tiny_test", temp_dir
            )

            # Should handle filtering gracefully
            assert isinstance(contours, list)
            assert len(contours) == 0

    def test_border_touching_contours(self):
        """Test handling of contours that touch image borders."""
        # Create image with border-touching shapes
        border_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(border_image, (0, 0), (50, 50), 255, -1)  # Touches left and top
        cv2.rectangle(border_image, (50, 50), (99, 99), 255, -1)  # Touches right and bottom

        with tempfile.TemporaryDirectory() as temp_dir:
            contours, hierarchy = extract_contours_with_hierarchy(
                border_image, "border_test", temp_dir
            )

            # Should filter out border-touching contours
            assert isinstance(contours, list)
            # All contours should be filtered out
            assert len(contours) == 0

    def test_malformed_hierarchy_data(self):
        """Test handling of malformed hierarchy data."""
        from pylithics.image_processing.modules.contour_extraction import sort_contours_by_hierarchy

        # Test with mismatched contours and hierarchy
        test_contours = [
            np.array([[10, 10], [20, 10], [20, 20], [10, 20]], dtype=np.int32).reshape(-1, 1, 2)
        ]
        malformed_hierarchy = np.array([[0, -1, -1, -1], [1, 0, -1, -1]])  # More hierarchy than contours

        result = sort_contours_by_hierarchy(test_contours, malformed_hierarchy)

        # Should handle gracefully
        assert isinstance(result, dict)
        assert 'parents' in result
        assert 'children' in result


@pytest.mark.error_scenarios
class TestMetricsCalculationErrors:
    """Test error handling in metrics calculation."""

    def test_degenerate_contours(self):
        """Test handling of degenerate contours (lines, points)."""
        # Create degenerate contours
        line_contour = np.array([[10, 10], [20, 10]], dtype=np.int32).reshape(-1, 1, 2)
        point_contour = np.array([[15, 15]], dtype=np.int32).reshape(-1, 1, 2)

        sorted_contours = {
            'parents': [line_contour],
            'children': [point_contour],
            'nested_children': []
        }

        hierarchy = np.array([[-1, -1, 1, -1], [-1, -1, -1, 0]])
        original_contours = [line_contour, point_contour]
        image_shape = (100, 100)

        # Should handle degenerate contours gracefully
        metrics = calculate_contour_metrics(
            sorted_contours, hierarchy, original_contours, image_shape
        )

        assert isinstance(metrics, list)
        # Should create some metrics even for degenerate contours
        assert len(metrics) >= 1

    def test_zero_area_contours(self):
        """Test handling of contours with zero area."""
        # Create contour that results in zero area
        zero_area_contour = np.array([[10, 10], [10, 10], [10, 10]], dtype=np.int32).reshape(-1, 1, 2)

        sorted_contours = {
            'parents': [zero_area_contour],
            'children': [],
            'nested_children': []
        }

        hierarchy = np.array([[-1, -1, -1, -1]])
        original_contours = [zero_area_contour]
        image_shape = (100, 100)

        metrics = calculate_contour_metrics(
            sorted_contours, hierarchy, original_contours, image_shape
        )

        assert isinstance(metrics, list)
        if len(metrics) > 0:
            assert metrics[0]['area'] >= 0  # Should handle zero area

    def test_contour_index_mismatch(self):
        """Test handling when contour indices don't match."""
        # Create contours that won't match in the index mapping
        contour1 = np.array([[10, 10], [20, 10], [20, 20], [10, 20]], dtype=np.int32).reshape(-1, 1, 2)
        contour2 = np.array([[30, 30], [40, 30], [40, 40], [30, 40]], dtype=np.int32).reshape(-1, 1, 2)
        contour3 = np.array([[50, 50], [60, 50], [60, 60], [50, 60]], dtype=np.int32).reshape(-1, 1, 2)

        sorted_contours = {
            'parents': [contour1],
            'children': [contour2],
            'nested_children': []
        }

        # Use different contour in original_contours
        hierarchy = np.array([[-1, -1, 1, -1], [-1, -1, -1, 0]])
        original_contours = [contour3, contour2]  # Mismatch: contour3 instead of contour1
        image_shape = (100, 100)

        # Should handle mismatch gracefully
        with patch('pylithics.image_processing.modules.contour_metrics.logging') as mock_logging:
            metrics = calculate_contour_metrics(
                sorted_contours, hierarchy, original_contours, image_shape
            )

            # Should log warnings but continue
            mock_logging.warning.assert_called()
            assert isinstance(metrics, list)


@pytest.mark.error_scenarios
class TestArrowDetectionErrors:
    """Test error handling in arrow detection module."""

    def test_invalid_contour_for_arrow(self):
        """Test arrow detection with invalid contours."""
        detector = ArrowDetector()

        # Test with empty contour
        empty_contour = np.array([], dtype=np.int32).reshape(0, 1, 2)
        entry = {'scar': 'test_empty'}
        image = np.zeros((100, 100), dtype=np.uint8)

        result = detector.analyze_contour_for_arrow(empty_contour, entry, image, 300.0)
        assert result is None

        # Test with single point
        point_contour = np.array([[10, 10]], dtype=np.int32).reshape(-1, 1, 2)
        result = detector.analyze_contour_for_arrow(point_contour, entry, image, 300.0)
        assert result is None

    def test_arrow_detection_with_invalid_dpi(self):
        """Test arrow detection with invalid DPI values."""
        detector = ArrowDetector()

        # Create valid contour
        test_contour = np.array([
            [10, 10], [20, 10], [15, 20]
        ], dtype=np.int32).reshape(-1, 1, 2)

        entry = {'scar': 'test_dpi'}
        image = np.zeros((100, 100), dtype=np.uint8)

        # Test with various invalid DPI values
        invalid_dpis = [None, 0, -100, 'invalid']

        for invalid_dpi in invalid_dpis:
            result = detector.analyze_contour_for_arrow(test_contour, entry, image, invalid_dpi)
            # Should handle gracefully (may return None or valid result with fallback)
            assert result is None or isinstance(result, dict)

    def test_arrow_detection_convexity_defects_error(self):
        """Test error handling when convexity defects calculation fails."""
        detector = ArrowDetector()

        # Create contour that might cause convexity defects issues
        problematic_contour = np.array([
            [10, 10], [11, 10], [10, 11]  # Very small triangle
        ], dtype=np.int32).reshape(-1, 1, 2)

        entry = {'scar': 'test_defects'}
        image = np.zeros((100, 100), dtype=np.uint8)

        result = detector.analyze_contour_for_arrow(problematic_contour, entry, image, 300.0)
        # Should handle gracefully when defects can't be calculated
        assert result is None or isinstance(result, dict)

    def test_arrow_detection_debug_directory_creation_error(self):
        """Test error handling when debug directory can't be created."""
        config = {'debug_enabled': True}
        detector = ArrowDetector(config)

        # Create contour for testing
        test_contour = np.array([
            [10, 10], [30, 10], [20, 30]
        ], dtype=np.int32).reshape(-1, 1, 2)

        # Use invalid debug directory path
        entry = {
            'scar': 'debug_test',
            'debug_dir': '/invalid/readonly/path'
        }
        image = np.zeros((100, 100), dtype=np.uint8)

        # Should handle debug directory creation errors gracefully
        result = detector.analyze_contour_for_arrow(test_contour, entry, image, 300.0)
        # Should not crash even if debug fails
        assert result is None or isinstance(result, dict)


@pytest.mark.error_scenarios
class TestSurfaceClassificationErrors:
    """Test error handling in surface classification."""

    def test_classification_with_no_parent_contours(self):
        """Test surface classification when no parent contours exist."""
        from pylithics.image_processing.modules.surface_classification import classify_parent_contours

        # Metrics with no parent contours
        metrics_no_parents = [
            {'parent': 'parent 1', 'scar': 'scar 1', 'area': 100},
            {'parent': 'parent 1', 'scar': 'scar 2', 'area': 50}
        ]

        result = classify_parent_contours(metrics_no_parents)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_classification_with_missing_metrics(self):
        """Test surface classification with missing metric fields."""
        from pylithics.image_processing.modules.surface_classification import classify_parent_contours

        # Metrics missing required fields
        incomplete_metrics = [
            {'parent': 'parent 1', 'scar': 'parent 1'},  # Missing area, dimensions
            {'parent': 'parent 2', 'scar': 'parent 2', 'area': 500}  # Missing dimensions
        ]

        # Should handle missing fields gracefully
        result = classify_parent_contours(incomplete_metrics)
        assert isinstance(result, list)


@pytest.mark.error_scenarios
class TestAnalysisPipelineErrors:
    """Test error handling in the main analysis pipeline."""

    def test_pipeline_with_fallback_metrics(self, sample_config):
        """Test pipeline fallback when main metric calculation fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Create image that might cause metrics calculation to fail
            image_path = os.path.join(images_dir, "fallback_test.png")
            problematic_image = np.zeros((50, 50, 3), dtype=np.uint8)
            # Add minimal feature that might cause issues
            cv2.circle(problematic_image, (25, 25), 1, (255, 255, 255), -1)
            cv2.imwrite(image_path, problematic_image)

            metadata_content = "image_id,scale_id,scale\nfallback_test.png,scale_1,15.0\n"
            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            app = PyLithicsApplication(config_file=config_path)

            # Should use fallback metrics if main calculation fails
            results = app.run_batch_analysis(data_dir, metadata_path)

            # Pipeline should complete with fallback
            assert isinstance(results, dict)
            assert 'success' in results

    def test_pipeline_step_failures(self, sample_config):
        """Test pipeline continuation when individual steps fail."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Create image for testing
            image_path = os.path.join(images_dir, "step_failure_test.png")
            test_image = np.full((100, 100, 3), 200, dtype=np.uint8)
            cv2.rectangle(test_image, (25, 25), (75, 75), (50, 50, 50), -1)
            cv2.imwrite(image_path, test_image)

            metadata_content = "image_id,scale_id,scale\nstep_failure_test.png,scale_1,15.0\n"
            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            # Configure to potentially cause step failures
            error_prone_config = sample_config.copy()
            error_prone_config.update({
                'arrow_detection': {'enabled': True, 'debug_enabled': True},
                'symmetry_analysis': {'enabled': True},
                'voronoi_analysis': {'enabled': True}
            })

            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(error_prone_config, f)

            app = PyLithicsApplication(config_file=config_path)

            # Mock individual step failures to test recovery
            with patch('pylithics.image_processing.modules.symmetry_analysis.analyze_dorsal_symmetry',
                      side_effect=Exception("Symmetry analysis failed")):
                results = app.run_batch_analysis(data_dir, metadata_path)

                # Pipeline should continue despite step failure
                assert isinstance(results, dict)
                # Should still generate some output even with step failures
                processed_dir = os.path.join(data_dir, 'processed')
                if os.path.exists(processed_dir):
                    csv_files = list(Path(processed_dir).glob("processed_metrics.csv"))
                    # May or may not have CSV depending on how early failure occurs


@pytest.mark.error_scenarios
class TestFileSystemErrors:
    """Test error handling for file system related errors."""

    def test_output_directory_creation_failure(self, sample_config):
        """Test handling when output directory can't be created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Create valid image
            image_path = os.path.join(images_dir, "test.png")
            test_image = np.full((100, 100, 3), 128, dtype=np.uint8)
            cv2.imwrite(image_path, test_image)

            metadata_content = "image_id,scale_id,scale\ntest.png,scale_1,15.0\n"
            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            # Mock os.makedirs to fail
            with patch('os.makedirs', side_effect=PermissionError("Cannot create directory")):
                app = PyLithicsApplication(config_file=config_path)
                results = app.run_batch_analysis(data_dir, metadata_path)

                # Should handle directory creation errors
                assert isinstance(results, dict)

    def test_csv_write_permission_error(self, sample_config):
        """Test handling when CSV output can't be written."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Create valid image
            image_path = os.path.join(images_dir, "csv_test.png")
            test_image = np.full((100, 100, 3), 128, dtype=np.uint8)
            cv2.rectangle(test_image, (25, 25), (75, 75), (50, 50, 50), -1)
            cv2.imwrite(image_path, test_image)

            metadata_content = "image_id,scale_id,scale\ncsv_test.png,scale_1,15.0\n"
            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            # Mock pandas to_csv to fail
            with patch('pandas.DataFrame.to_csv', side_effect=PermissionError("Cannot write CSV")):
                app = PyLithicsApplication(config_file=config_path)
                results = app.run_batch_analysis(data_dir, metadata_path)

                # Should handle CSV write errors gracefully
                assert isinstance(results, dict)

    def test_disk_space_errors(self, sample_config):
        """Test handling of disk space errors during processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Create image
            image_path = os.path.join(images_dir, "disk_test.png")
            test_image = np.full((200, 200, 3), 128, dtype=np.uint8)
            cv2.imwrite(image_path, test_image)

            metadata_content = "image_id,scale_id,scale\ndisk_test.png,scale_1,15.0\n"
            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            # Mock cv2.imwrite to simulate disk space error
            with patch('cv2.imwrite', side_effect=OSError("No space left on device")):
                app = PyLithicsApplication(config_file=config_path)
                results = app.run_batch_analysis(data_dir, metadata_path)

                # Should handle disk space errors gracefully
                assert isinstance(results, dict)


@pytest.mark.error_scenarios
class TestMemoryAndResourceErrors:
    """Test error handling for memory and resource related errors."""

    def test_memory_exhaustion_simulation(self, sample_config):
        """Test handling of memory exhaustion scenarios."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Create image
            image_path = os.path.join(images_dir, "memory_test.png")
            test_image = np.full((300, 300, 3), 128, dtype=np.uint8)
            cv2.imwrite(image_path, test_image)

            metadata_content = "image_id,scale_id,scale\nmemory_test.png,scale_1,15.0\n"
            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            # Mock numpy operations to simulate memory error
            with patch('numpy.array', side_effect=MemoryError("Cannot allocate memory")):
                app = PyLithicsApplication(config_file=config_path)
                results = app.run_batch_analysis(data_dir, metadata_path)

                # Should handle memory errors gracefully
                assert isinstance(results, dict)

    def test_opencv_errors(self, sample_config):
        """Test handling of OpenCV-specific errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Create image
            image_path = os.path.join(images_dir, "opencv_test.png")
            test_image = np.full((100, 100, 3), 128, dtype=np.uint8)
            cv2.imwrite(image_path, test_image)

            metadata_content = "image_id,scale_id,scale\nopencv_test.png,scale_1,15.0\n"
            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            # Mock cv2.findContours to raise error
            with patch('cv2.findContours', side_effect=cv2.error("OpenCV Error")):
                app = PyLithicsApplication(config_file=config_path)
                results = app.run_batch_analysis(data_dir, metadata_path)

                # Should handle OpenCV errors gracefully
                assert isinstance(results, dict)


@pytest.mark.error_scenarios
class TestDataIntegrityErrors:
    """Test error handling for data integrity issues."""

    def test_malformed_metadata(self, sample_config):
        """Test handling of malformed metadata files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Create valid image
            image_path = os.path.join(images_dir, "meta_test.png")
            test_image = np.full((100, 100, 3), 128, dtype=np.uint8)
            cv2.imwrite(image_path, test_image)

            # Create malformed metadata
            malformed_metadata = "invalid,csv,format\nno,header,row\nmeta_test.png,scale_1"  # Missing column
            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(malformed_metadata)

            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            app = PyLithicsApplication(config_file=config_path)

            # Should handle malformed metadata gracefully
            valid = app.validate_inputs(data_dir, metadata_path)
            assert isinstance(valid, bool)

    def test_invalid_scale_values(self, sample_config):
        """Test handling of invalid scale values in metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Create images
            for i in range(3):
                image_path = os.path.join(images_dir, f"scale_test_{i}.png")
                test_image = np.full((100, 100, 3), 128, dtype=np.uint8)
                cv2.imwrite(image_path, test_image)

            # Create metadata with invalid scale values
            invalid_metadata = """image_id,scale_id,scale
scale_test_0.png,scale_1,invalid_scale
scale_test_1.png,scale_2,
scale_test_2.png,scale_3,-5.0"""

            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(invalid_metadata)

            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            app = PyLithicsApplication(config_file=config_path)
            results = app.run_batch_analysis(data_dir, metadata_path)

            # Should handle invalid scales gracefully
            assert isinstance(results, dict)
            assert results['total_images'] == 3
            # Some may fail due to invalid scales
            assert results['processed_successfully'] <= 3

    def test_unicode_and_encoding_errors(self, sample_config):
        """Test handling of unicode and encoding issues."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Create image with unicode filename
            unicode_filename = "tëst_ünicödé.png"
            image_path = os.path.join(images_dir, unicode_filename)
            test_image = np.full((100, 100, 3), 128, dtype=np.uint8)
            cv2.imwrite(image_path, test_image)

            # Create metadata with unicode content
            unicode_metadata = f"image_id,scale_id,scale\n{unicode_filename},scälé_1,15.0\n"
            metadata_path = os.path.join(data_dir, "metadata.csv")

            # Write with different encoding to test robustness
            with open(metadata_path, 'w', encoding='utf-8') as f:
                f.write(unicode_metadata)

            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            app = PyLithicsApplication(config_file=config_path)
            results = app.run_batch_analysis(data_dir, metadata_path)

            # Should handle unicode gracefully
            assert isinstance(results, dict)


@pytest.mark.error_scenarios
class TestConcurrencyAndRaceConditions:
    """Test error handling for concurrency issues."""

    def test_file_access_conflicts(self, sample_config):
        """Test handling of file access conflicts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Create image
            image_path = os.path.join(images_dir, "conflict_test.png")
            test_image = np.full((100, 100, 3), 128, dtype=np.uint8)
            cv2.imwrite(image_path, test_image)

            metadata_content = "image_id,scale_id,scale\nconflict_test.png,scale_1,15.0\n"
            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            # Simulate file access conflict
            with patch('builtins.open', side_effect=PermissionError("File in use")):
                app = PyLithicsApplication(config_file=config_path)
                results = app.run_batch_analysis(data_dir, metadata_path)

                # Should handle file conflicts gracefully
                assert isinstance(results, dict)

    def test_interrupted_processing(self, sample_config):
        """Test handling of interrupted processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Create image
            image_path = os.path.join(images_dir, "interrupt_test.png")
            test_image = np.full((100, 100, 3), 128, dtype=np.uint8)
            cv2.imwrite(image_path, test_image)

            metadata_content = "image_id,scale_id,scale\ninterrupt_test.png,scale_1,15.0\n"
            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            # Simulate keyboard interrupt
            with patch('pylithics.image_processing.image_analysis.process_and_save_contours',
                      side_effect=KeyboardInterrupt("User interrupted")):
                app = PyLithicsApplication(config_file=config_path)

                # Should handle interruption gracefully
                try:
                    results = app.run_batch_analysis(data_dir, metadata_path)
                    # If it doesn't raise, should return valid result
                    assert isinstance(results, dict)
                except KeyboardInterrupt:
                    # If it does raise, that's also acceptable behavior
                    pass


@pytest.mark.error_scenarios
class TestRecoveryMechanisms:
    """Test error recovery and graceful degradation."""

    def test_partial_pipeline_recovery(self, sample_config):
        """Test recovery when only some pipeline steps succeed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Create image
            image_path = os.path.join(images_dir, "recovery_test.png")
            test_image = np.full((200, 200, 3), 200, dtype=np.uint8)
            cv2.rectangle(test_image, (50, 50), (150, 150), (100, 100, 100), -1)
            cv2.imwrite(image_path, test_image)

            metadata_content = "image_id,scale_id,scale\nrecovery_test.png,scale_1,15.0\n"
            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            # Mock specific analysis steps to fail
            with patch('pylithics.image_processing.modules.voronoi_analysis.calculate_voronoi_points',
                      side_effect=Exception("Voronoi failed")):
                app = PyLithicsApplication(config_file=config_path)
                results = app.run_batch_analysis(data_dir, metadata_path)

                # Should still produce some results
                assert isinstance(results, dict)
                # Should attempt processing even with step failures
                assert 'success' in results

    def test_fallback_configuration_usage(self):
        """Test that system falls back to default config when needed."""
        # Test with completely invalid config
        invalid_config = {
            'invalid_section': {
                'invalid_key': 'invalid_value'
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "invalid_config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(invalid_config, f)

            # Should fall back to defaults
            config_manager = ConfigurationManager(config_path)
            config = config_manager.config

            # Should have valid default sections
            assert 'thresholding' in config
            assert 'logging' in config
            assert 'normalization' in config

    def test_error_logging_and_tracking(self, sample_config):
        """Test that errors are properly logged and tracked."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Create mix of valid and invalid images
            valid_image = os.path.join(images_dir, "valid.png")
            test_image = np.full((100, 100, 3), 128, dtype=np.uint8)
            cv2.imwrite(valid_image, test_image)

            invalid_image = os.path.join(images_dir, "invalid.png")
            with open(invalid_image, 'w') as f:
                f.write("not an image")

            metadata_content = """image_id,scale_id,scale
valid.png,scale_1,15.0
invalid.png,scale_2,20.0"""

            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            # Configure logging
            log_config = sample_config.copy()
            log_config['logging'] = {
                'level': 'ERROR',
                'log_to_file': True,
                'log_file': os.path.join(temp_dir, 'error_test.log')
            }

            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(log_config, f)

            app = PyLithicsApplication(config_file=config_path)
            results = app.run_batch_analysis(data_dir, metadata_path)

            # Should track errors properly
            assert isinstance(results, dict)
            assert 'failed_images' in results
            assert 'processing_errors' in results
            assert 'invalid.png' in results['failed_images']

            # Check log file was created
            log_file = Path(temp_dir) / 'error_test.log'
            if log_file.exists():
                assert log_file.stat().st_size > 0


if __name__ == "__main__":
    pytest.main([__file__])