"""
PyLithics Full Pipeline Tests
=============================

End-to-end functional tests for the complete PyLithics processing pipeline.
Tests the entire workflow from raw images to final outputs, focusing on
software engineering concerns: integration, data flow, error handling,
and system behavior validation.
"""

import pytest
import numpy as np
import cv2
import pandas as pd
import tempfile
import os
import yaml
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image

# Import main application components
from pylithics.app import PyLithicsApplication, main
from pylithics.image_processing.importer import execute_preprocessing_pipeline
from pylithics.image_processing.image_analysis import process_and_save_contours


@pytest.mark.functional
class TestEndToEndPipeline:
    """Test complete end-to-end pipeline functionality."""

    def test_single_image_complete_workflow(self, sample_config):
        """Test complete workflow from single image to final outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data structure
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Create simple test image
            image_path = os.path.join(images_dir, "test_artifact.png")
            test_image = self._create_test_image()
            cv2.imwrite(image_path, test_image)

            # Create metadata file
            metadata_content = "image_id,scale_id,scale\ntest_artifact.png,scale_1,15.0\n"
            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            # Create configuration file
            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            # Execute pipeline
            app = PyLithicsApplication(config_file=config_path)
            result = app.run_batch_analysis(
                data_dir=data_dir,
                meta_file=metadata_path
            )

            # Verify processing completed successfully
            assert result['success'] is True
            assert result['processed_successfully'] == 1
            assert result['total_images'] == 1
            assert len(result['failed_images']) == 0

            # Verify expected output files exist
            processed_dir = os.path.join(data_dir, 'processed')
            assert os.path.exists(processed_dir)

            # Check CSV output
            csv_files = list(Path(processed_dir).glob("processed_metrics.csv"))
            assert len(csv_files) == 1, "Expected exactly one CSV output file"

            # Check visualization output
            viz_files = list(Path(processed_dir).glob("*_labeled.png"))
            assert len(viz_files) >= 1, "Expected visualization output files"

            # Verify CSV structure
            df = pd.read_csv(csv_files[0])
            required_columns = ['image_id', 'parent', 'scar', 'centroid_x', 'centroid_y', 'area']
            for col in required_columns:
                assert col in df.columns, f"Missing required column: {col}"

            assert len(df) > 0, "No measurement data in CSV"
            assert df['image_id'].iloc[0] == 'test_artifact.png'

    def test_batch_processing_workflow(self, sample_config):
        """Test batch processing of multiple images."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create batch test structure
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Create multiple test images
            test_images = ["image_001.png", "image_002.png", "image_003.png"]
            metadata_content = "image_id,scale_id,scale\n"

            for i, filename in enumerate(test_images):
                image_path = os.path.join(images_dir, filename)
                test_image = self._create_test_image(variation=i)
                cv2.imwrite(image_path, test_image)

                scale = 15.0 + i * 2.0
                metadata_content += f"{filename},scale_{i+1},{scale}\n"

            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            # Execute batch processing
            app = PyLithicsApplication(config_file=config_path)
            results = app.run_batch_analysis(
                data_dir=data_dir,
                meta_file=metadata_path
            )

            # Verify batch results
            assert results['success'] is True
            assert results['total_images'] == len(test_images)
            assert results['processed_successfully'] >= 1  # At least one should succeed

            # Verify output files
            processed_dir = os.path.join(data_dir, 'processed')
            csv_files = list(Path(processed_dir).glob("processed_metrics.csv"))
            assert len(csv_files) == 1

            # Verify batch data in CSV
            if csv_files:
                df = pd.read_csv(csv_files[0])
                unique_images = df['image_id'].nunique()
                assert unique_images >= results['processed_successfully']

    def test_cli_interface_integration(self, sample_config):
        """Test command line interface integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup test data
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            image_path = os.path.join(images_dir, "cli_test.png")
            test_image = self._create_test_image()
            cv2.imwrite(image_path, test_image)

            metadata_content = "image_id,scale_id,scale\ncli_test.png,scale_1,20.0\n"
            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            # Test CLI execution
            test_args = [
                'pylithics',
                '--data_dir', data_dir,
                '--meta_file', metadata_path,
                '--config_file', config_path,
                '--log_level', 'INFO'
            ]

            with patch('sys.argv', test_args):
                exit_code = main()

            assert exit_code == 0, "CLI execution should succeed"

            # Verify outputs were created
            processed_dir = os.path.join(data_dir, 'processed')
            output_files = list(Path(processed_dir).glob("*"))
            assert len(output_files) > 0, "CLI should generate output files"

    def test_configuration_integration(self):
        """Test that configuration changes properly affect pipeline behavior."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup test data
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            image_path = os.path.join(images_dir, "config_test.png")
            test_image = self._create_test_image()
            cv2.imwrite(image_path, test_image)

            metadata_content = "image_id,scale_id,scale\nconfig_test.png,scale_1,15.0\n"
            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            # Test different configurations
            configs = {
                'arrows_enabled': {
                    'arrow_detection': {'enabled': True, 'reference_dpi': 300.0},
                    'thresholding': {'method': 'simple', 'threshold_value': 127}
                },
                'arrows_disabled': {
                    'arrow_detection': {'enabled': False},
                    'thresholding': {'method': 'otsu', 'max_value': 255}
                }
            }

            results = {}
            for config_name, config_data in configs.items():
                config_path = os.path.join(temp_dir, f"{config_name}_config.yaml")
                with open(config_path, 'w') as f:
                    yaml.dump(config_data, f)

                app = PyLithicsApplication(config_file=config_path)
                result = app.run_batch_analysis(
                    data_dir=data_dir,
                    meta_file=metadata_path
                )

                results[config_name] = result

            # Verify both configurations work
            assert results['arrows_enabled']['success']
            assert results['arrows_disabled']['success']

    def _create_test_image(self, variation=0):
        """Create simple test image with controllable variations."""
        size = (400, 300)
        height, width = size

        # Create background
        image = np.full((height, width, 3), 240, dtype=np.uint8)

        # Create main artifact (rectangle with slight variations)
        rect_x = 50 + variation * 10
        rect_y = 50
        rect_w = 200
        rect_h = 150

        cv2.rectangle(image, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (80, 70, 60), -1)

        # Add a simple scar (circle)
        scar_x = rect_x + 60 + variation * 5
        scar_y = rect_y + 50
        cv2.circle(image, (scar_x, scar_y), 20, (50, 45, 40), -1)

        return image


@pytest.mark.functional
class TestPipelineIntegration:
    """Test integration between pipeline components."""

    def test_preprocessing_to_analysis_integration(self, sample_config):
        """Test data flow from preprocessing through analysis."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test image with known characteristics
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            image_path = os.path.join(images_dir, "integration_test.png")
            test_image = self._create_integration_test_image()

            pil_image = Image.fromarray(test_image)
            pil_image.save(image_path, dpi=(300, 300))

            metadata_content = "image_id,scale_id,scale\nintegration_test.png,scale_1,20.0\n"
            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            # Configure for detailed analysis
            integration_config = {
                'thresholding': {'method': 'simple', 'threshold_value': 150, 'max_value': 255},
                'normalization': {'enabled': True, 'method': 'minmax'},
                'grayscale_conversion': {'enabled': True, 'method': 'standard'},
                'morphological_closing': {'enabled': True, 'kernel_size': 3},
                'contour_filtering': {'min_area': 100.0, 'exclude_border': True},
                'arrow_detection': {'enabled': True, 'reference_dpi': 300.0, 'debug_enabled': False},
                'logging': {'level': 'DEBUG'}
            }

            config_path = os.path.join(temp_dir, "integration_config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(integration_config, f)

            # Execute pipeline
            app = PyLithicsApplication(config_file=config_path)
            result = app.run_batch_analysis(
                data_dir=data_dir,
                meta_file=metadata_path
            )

            # Verify integration worked
            assert result['success'], "Integration test should succeed"

            processed_dir = os.path.join(data_dir, 'processed')
            csv_files = list(Path(processed_dir).glob("processed_metrics.csv"))
            viz_files = list(Path(processed_dir).glob("*_labeled.png"))

            assert len(csv_files) > 0, "No CSV output from integrated pipeline"
            assert len(viz_files) > 0, "No visualization output from integrated pipeline"

            # Verify data structure
            df = pd.read_csv(csv_files[0])
            assert len(df) >= 2, "Should detect multiple contours (parent + children)"

            # Check that surface classification ran
            if 'surface_type' in df.columns:
                surface_types = df['surface_type'].unique()
                assert len(surface_types) >= 1, "Surface classification should produce results"

    def test_module_communication(self, sample_config):
        """Test that modules properly pass data between each other."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Create test image
            image_path = os.path.join(images_dir, "module_test.png")
            test_image = self._create_integration_test_image()
            cv2.imwrite(image_path, test_image)

            metadata_content = "image_id,scale_id,scale\nmodule_test.png,scale_1,15.0\n"
            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            config_path = os.path.join(temp_dir, "module_config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            # Test individual components work together
            app = PyLithicsApplication(config_file=config_path)

            # Validate inputs
            assert app.validate_inputs(data_dir, metadata_path), "Input validation should pass"

            # Test processing
            result = app.run_batch_analysis(data_dir, metadata_path)
            assert result['success'], "Module communication test should succeed"

    def test_error_propagation(self, sample_config):
        """Test how errors propagate through the pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Create problematic test cases
            test_cases = [
                ("empty_image.png", self._create_empty_image()),
                ("minimal_image.png", self._create_minimal_image()),
                ("normal_image.png", self._create_integration_test_image())
            ]

            metadata_content = "image_id,scale_id,scale\n"
            for filename, image_data in test_cases:
                image_path = os.path.join(images_dir, filename)
                cv2.imwrite(image_path, image_data)
                metadata_content += f"{filename},scale_1,15.0\n"

            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            config_path = os.path.join(temp_dir, "error_config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            # Execute with error tolerance
            app = PyLithicsApplication(config_file=config_path)
            result = app.run_batch_analysis(data_dir, metadata_path)

            # Pipeline should handle errors gracefully
            assert isinstance(result, dict), "Should return result dictionary"
            assert 'success' in result, "Should have success flag"
            assert 'failed_images' in result, "Should track failed images"

            # At least one image should process successfully
            assert result['processed_successfully'] >= 1, "At least normal image should process"

    def _create_integration_test_image(self):
        """Create test image designed for integration testing."""
        image = np.full((300, 400, 3), 240, dtype=np.uint8)

        # Main artifact (parent contour)
        cv2.rectangle(image, (50, 50), (350, 250), (80, 70, 60), -1)

        # Child contours (scars)
        cv2.circle(image, (150, 120), 25, (50, 45, 40), -1)
        cv2.circle(image, (250, 180), 20, (55, 50, 45), -1)

        # Small arrow-like feature
        arrow_points = np.array([
            [145, 115], [155, 115], [153, 120], [160, 120],
            [153, 125], [155, 125], [145, 125], [147, 120]
        ], dtype=np.int32)
        cv2.fillPoly(image, [arrow_points], (90, 80, 70))

        return image

    def _create_empty_image(self):
        """Create empty/uniform image for error testing."""
        return np.full((100, 100, 3), 128, dtype=np.uint8)

    def _create_minimal_image(self):
        """Create minimal image with tiny features."""
        image = np.full((50, 50, 3), 240, dtype=np.uint8)
        cv2.rectangle(image, (10, 10), (40, 40), (80, 70, 60), -1)
        return image


@pytest.mark.functional
class TestDataFlowValidation:
    """Test data flow and format validation through pipeline."""

    def test_csv_output_format(self, sample_config):
        """Test CSV output format and data integrity."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            image_path = os.path.join(images_dir, "format_test.png")
            test_image = self._create_test_image_with_known_properties()
            cv2.imwrite(image_path, test_image)

            metadata_content = "image_id,scale_id,scale\nformat_test.png,scale_1,20.0\n"
            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            config_path = os.path.join(temp_dir, "format_config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            app = PyLithicsApplication(config_file=config_path)
            result = app.run_batch_analysis(data_dir, metadata_path)

            assert result['success'], "Format test should succeed"

            # Validate CSV format
            processed_dir = os.path.join(data_dir, 'processed')
            csv_files = list(Path(processed_dir).glob("processed_metrics.csv"))
            assert len(csv_files) == 1

            df = pd.read_csv(csv_files[0])

            # Test required columns exist
            required_columns = ['image_id', 'parent', 'scar', 'area', 'centroid_x', 'centroid_y']
            for col in required_columns:
                assert col in df.columns, f"Missing required column: {col}"

            # Test data types
            assert df['area'].dtype in [np.float64, np.int64], "Area should be numeric"
            assert df['centroid_x'].dtype in [np.float64, np.int64], "Centroid X should be numeric"
            assert df['centroid_y'].dtype in [np.float64, np.int64], "Centroid Y should be numeric"

            # Test data validity
            assert all(df['area'] >= 0), "All areas should be non-negative"
            assert all(df['centroid_x'] >= 0), "All X coordinates should be non-negative"
            assert all(df['centroid_y'] >= 0), "All Y coordinates should be non-negative"

    def test_visualization_output_generation(self, sample_config):
        """Test visualization file generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            image_path = os.path.join(images_dir, "viz_test.png")
            test_image = self._create_test_image_with_known_properties()
            cv2.imwrite(image_path, test_image)

            metadata_content = "image_id,scale_id,scale\nviz_test.png,scale_1,15.0\n"
            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            config_path = os.path.join(temp_dir, "viz_config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            app = PyLithicsApplication(config_file=config_path)
            result = app.run_batch_analysis(data_dir, metadata_path)

            assert result['success'], "Visualization test should succeed"

            # Check visualization files
            processed_dir = os.path.join(data_dir, 'processed')
            labeled_files = list(Path(processed_dir).glob("*_labeled.png"))

            assert len(labeled_files) >= 1, "Should generate labeled visualization"

            # Verify files are valid images
            for viz_file in labeled_files:
                assert viz_file.stat().st_size > 0, "Visualization file should not be empty"

                # Try to read the image to verify it's valid
                try:
                    test_read = cv2.imread(str(viz_file))
                    assert test_read is not None, "Should be readable image file"
                except Exception as e:
                    pytest.fail(f"Generated visualization file is not valid: {e}")

    def test_metadata_processing(self, sample_config):
        """Test metadata reading and processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Create test images
            test_data = [
                ("meta_test_1.png", 15.5),
                ("meta_test_2.png", 22.3),
                ("meta_test_3.png", 18.7)
            ]

            metadata_content = "image_id,scale_id,scale\n"
            for filename, scale in test_data:
                image_path = os.path.join(images_dir, filename)
                test_image = self._create_test_image_with_known_properties()
                cv2.imwrite(image_path, test_image)
                metadata_content += f"{filename},scale_1,{scale}\n"

            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            config_path = os.path.join(temp_dir, "meta_config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            app = PyLithicsApplication(config_file=config_path)

            # Test metadata validation
            assert app.validate_inputs(data_dir, metadata_path), "Metadata validation should pass"

            # Test processing with metadata
            result = app.run_batch_analysis(data_dir, metadata_path)
            assert result['success'], "Metadata processing should succeed"
            assert result['total_images'] == len(test_data)

    def _create_test_image_with_known_properties(self):
        """Create test image with known, measurable properties."""
        image = np.full((200, 300, 3), 240, dtype=np.uint8)

        # Create rectangle with known dimensions
        cv2.rectangle(image, (50, 50), (250, 150), (80, 70, 60), -1)

        # Add circular feature
        cv2.circle(image, (100, 100), 20, (50, 45, 40), -1)

        return image


@pytest.mark.performance
class TestPipelinePerformance:
    """Test pipeline performance characteristics."""

    def test_single_image_processing_time(self, sample_config):
        """Test processing time for single image."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            image_path = os.path.join(images_dir, "perf_test.png")
            test_image = self._create_performance_test_image()
            cv2.imwrite(image_path, test_image)

            metadata_content = "image_id,scale_id,scale\nperf_test.png,scale_1,20.0\n"
            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            config_path = os.path.join(temp_dir, "perf_config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            app = PyLithicsApplication(config_file=config_path)

            # Measure processing time
            start_time = time.time()
            result = app.run_batch_analysis(data_dir, metadata_path)
            end_time = time.time()

            processing_time = end_time - start_time

            # Performance assertions
            assert processing_time < 30.0, f"Processing took too long: {processing_time:.1f}s"
            assert result['success'], "Performance test should succeed"

    def test_batch_processing_performance(self, sample_config):
        """Test batch processing performance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Create multiple test images
            num_images = 3  # Keep small for test speed
            metadata_content = "image_id,scale\n"

            for i in range(num_images):
                image_name = f"batch_test_{i:02d}.png"
                image_path = os.path.join(images_dir, image_name)
                test_image = self._create_performance_test_image()
                cv2.imwrite(image_path, test_image)

                scale = 15.0 + i * 2.0
                metadata_content += f"{image_name},{scale}\n"

            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            config_path = os.path.join(temp_dir, "batch_config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            app = PyLithicsApplication(config_file=config_path)

            # Measure batch processing time
            start_time = time.time()
            results = app.run_batch_analysis(data_dir, metadata_path)
            end_time = time.time()

            total_time = end_time - start_time
            avg_time_per_image = total_time / num_images

            # Performance assertions
            assert avg_time_per_image < 20.0, f"Average processing time too high: {avg_time_per_image:.1f}s"
            assert results['success'], "Batch performance test should succeed"

    def test_memory_usage(self, sample_config):
        """Test memory usage during processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            image_path = os.path.join(images_dir, "memory_test.png")
            # Create larger image for memory testing
            test_image = self._create_performance_test_image(size=(800, 600))
            cv2.imwrite(image_path, test_image)

            metadata_content = "image_id,scale_id,scale\nmemory_test.png,scale_1,25.0\n"
            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            config_path = os.path.join(temp_dir, "memory_config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            # Monitor memory usage
            import psutil
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss

            app = PyLithicsApplication(config_file=config_path)
            result = app.run_batch_analysis(data_dir, metadata_path)

            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory

            # Memory usage assertions
            max_acceptable_increase = 100 * 1024 * 1024  # 100MB
            assert memory_increase < max_acceptable_increase, \
                f"Memory usage too high: {memory_increase / (1024*1024):.1f}MB"

    def _create_performance_test_image(self, size=(400, 300)):
        """Create test image for performance testing."""
        height, width = size
        image = np.full((height, width, 3), 235, dtype=np.uint8)

        # Create artifact with multiple features
        cv2.ellipse(image, (width//2, height//2), (width//3, height//4), 0, 0, 360, (85, 75, 65), -1)

        # Add several scars
        for i in range(5):
            x = width//3 + i * width//12
            y = height//3 + (i % 2) * height//6
            cv2.circle(image, (x, y), 15, (60, 55, 50), -1)

        return image


@pytest.mark.functional
class TestErrorHandling:
    """Test error handling and robustness."""

    def test_invalid_input_handling(self, sample_config):
        """Test handling of invalid inputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Test missing metadata file
            app = PyLithicsApplication()
            missing_metadata = os.path.join(temp_dir, "missing.csv")
            assert not app.validate_inputs(data_dir, missing_metadata)

            # Test missing images directory
            missing_images_dir = os.path.join(temp_dir, "missing_data")
            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write("image_id,scale\ntest.png,15.0\n")

            assert not app.validate_inputs(missing_images_dir, metadata_path)

    def test_corrupted_image_handling(self, sample_config):
        """Test handling of corrupted or problematic images."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Create corrupted files
            corrupted_files = [
                ("empty_file.png", "empty"),
                ("tiny_image.png", "tiny"),
                ("uniform_image.png", "uniform"),
                ("valid_image.png", "valid")
            ]

            metadata_content = "image_id,scale\n"

            for filename, file_type in corrupted_files:
                image_path = os.path.join(images_dir, filename)

                if file_type == "empty":
                    with open(image_path, 'w') as f:
                        pass
                elif file_type == "tiny":
                    tiny_img = np.array([[[255, 255, 255]]], dtype=np.uint8)
                    cv2.imwrite(image_path, tiny_img)
                elif file_type == "uniform":
                    uniform_img = np.full((100, 100, 3), 128, dtype=np.uint8)
                    cv2.imwrite(image_path, uniform_img)
                else:  # valid
                    valid_img = np.full((200, 200, 3), 240, dtype=np.uint8)
                    cv2.rectangle(valid_img, (50, 50), (150, 150), (80, 70, 60), -1)
                    cv2.imwrite(image_path, valid_img)

                metadata_content += f"{filename},15.0\n"

            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            config_path = os.path.join(temp_dir, "error_config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            app = PyLithicsApplication(config_file=config_path)
            result = app.run_batch_analysis(data_dir, metadata_path)

            # Should handle errors gracefully
            assert isinstance(result, dict)
            assert 'success' in result
            assert 'failed_images' in result

            # At least the valid image should process
            assert result.get('processed_successfully', 0) >= 1

    def test_configuration_error_handling(self):
        """Test handling of configuration errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test invalid configuration file
            invalid_config_path = os.path.join(temp_dir, "invalid_config.yaml")
            with open(invalid_config_path, 'w') as f:
                f.write("invalid: yaml: content: [unclosed")

            # Should handle invalid config gracefully
            app = PyLithicsApplication(config_file=invalid_config_path)
            assert app.config_manager is not None

            # Test missing configuration file
            missing_config_path = os.path.join(temp_dir, "missing_config.yaml")
            app = PyLithicsApplication(config_file=missing_config_path)
            assert app.config_manager is not None


@pytest.mark.functional
class TestSystemIntegration:
    """Test overall system integration."""

    def test_complete_system_workflow(self, sample_config):
        """Test complete system from start to finish."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup complete test environment
            data_dir = os.path.join(temp_dir, "complete_test")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Create comprehensive test data
            test_images = [
                "artifact_001.png",
                "artifact_002.png"
            ]

            metadata_content = "image_id,scale_id,scale,notes\n"
            for i, filename in enumerate(test_images):
                image_path = os.path.join(images_dir, filename)
                test_image = self._create_comprehensive_test_image(variation=i)
                cv2.imwrite(image_path, test_image)

                scale = 18.0 + i * 3.0
                metadata_content += f"{filename},scale_{i+1},{scale},test_artifact_{i+1}\n"

            metadata_path = os.path.join(data_dir, "complete_metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            # Configure comprehensive analysis
            complete_config = sample_config.copy()
            complete_config.update({
                'arrow_detection': {'enabled': True, 'reference_dpi': 300.0},
                'symmetry_analysis': {'enabled': True},
                'voronoi_analysis': {'enabled': True},
                'logging': {'level': 'INFO', 'log_to_file': True}
            })

            config_path = os.path.join(temp_dir, "complete_config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(complete_config, f)

            # Execute complete workflow
            app = PyLithicsApplication(config_file=config_path)
            result = app.run_batch_analysis(data_dir, metadata_path)

            # Comprehensive validation
            assert result['success'], "Complete system test should succeed"
            assert result['total_images'] == len(test_images)
            assert result['processed_successfully'] >= 1

            # Validate all expected outputs
            processed_dir = os.path.join(data_dir, 'processed')

            # Check CSV output
            csv_files = list(Path(processed_dir).glob("processed_metrics.csv"))
            assert len(csv_files) == 1, "Should have exactly one CSV output"

            # Check visualizations
            viz_files = list(Path(processed_dir).glob("*_labeled.png"))
            assert len(viz_files) >= result['processed_successfully']

            # Validate CSV data completeness
            df = pd.read_csv(csv_files[0])
            assert len(df) > 0, "CSV should contain measurement data"

            # Check for expected analysis results
            expected_columns = [
                'image_id', 'parent', 'scar', 'surface_type',
                'area', 'centroid_x', 'centroid_y'
            ]

            for col in expected_columns:
                if col in df.columns:  # Some columns may be optional
                    assert not df[col].isnull().all(), f"Column {col} should have data"

    def test_resource_cleanup(self, sample_config):
        """Test that system properly cleans up resources."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "cleanup_test")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            image_path = os.path.join(images_dir, "cleanup_test.png")
            test_image = self._create_comprehensive_test_image()
            cv2.imwrite(image_path, test_image)

            metadata_content = "image_id,scale\ncleanup_test.png,20.0\n"
            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            config_path = os.path.join(temp_dir, "cleanup_config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            # Monitor resources
            import psutil
            process = psutil.Process()
            initial_files = len(process.open_files())

            app = PyLithicsApplication(config_file=config_path)
            result = app.run_batch_analysis(data_dir, metadata_path)

            final_files = len(process.open_files())

            # Should not leak file handles
            file_leak = final_files - initial_files
            assert file_leak < 5, f"Potential file handle leak: {file_leak} files"

    def _create_comprehensive_test_image(self, variation=0):
        """Create comprehensive test image for system testing."""
        size = (500, 400)
        height, width = size

        image = np.full((height, width, 3), 240, dtype=np.uint8)

        # Create main artifact
        artifact_x = 50 + variation * 20
        artifact_y = 50
        artifact_w = 300
        artifact_h = 200

        cv2.rectangle(image, (artifact_x, artifact_y),
                     (artifact_x + artifact_w, artifact_y + artifact_h),
                     (80, 70, 60), -1)

        # Add multiple scars for comprehensive testing
        scars = [
            (artifact_x + 80, artifact_y + 60, 25),
            (artifact_x + 180, artifact_y + 120, 20),
            (artifact_x + 120, artifact_y + 160, 18)
        ]

        for scar_x, scar_y, radius in scars:
            cv2.circle(image, (scar_x, scar_y), radius, (50, 45, 40), -1)

        # Add some edge features for surface classification
        cv2.rectangle(image, (artifact_x + 250, artifact_y + 10),
                     (artifact_x + 280, artifact_y + 40), (60, 55, 50), -1)

        return image


if __name__ == "__main__":
    pytest.main([__file__])