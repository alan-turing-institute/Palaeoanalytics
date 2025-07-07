"""
PyLithics Batch Processing Tests
===============================

Tests for multi-image batch processing workflows, focusing on:
- Batch result aggregation and tracking
- Mixed success/failure scenarios
- Batch-specific error handling
- Performance characteristics for multiple images
- Resource management across batches
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

from pylithics.app import PyLithicsApplication


@pytest.mark.unit
class TestBatchResultAggregation:
    """Test how batch results are collected and aggregated."""

    def test_successful_batch_result_structure(self, sample_config):
        """Test result structure for completely successful batch."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create batch with 3 valid images
            data_dir, metadata_path = self._create_test_batch(temp_dir, 3, all_valid=True)

            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            app = PyLithicsApplication(config_file=config_path)
            results = app.run_batch_analysis(data_dir, metadata_path)

            # Verify result structure
            assert isinstance(results, dict)
            assert 'success' in results
            assert 'total_images' in results
            assert 'processed_successfully' in results
            assert 'failed_images' in results
            assert 'processing_errors' in results

            # Verify successful batch values
            assert results['success'] is True
            assert results['total_images'] == 3
            assert results['processed_successfully'] == 3
            assert len(results['failed_images']) == 0
            assert len(results['processing_errors']) == 0

    def test_mixed_success_failure_batch(self, sample_config):
        """Test result aggregation with mixed success/failure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create batch with mix of valid and invalid images
            data_dir, metadata_path = self._create_mixed_batch(temp_dir)

            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            app = PyLithicsApplication(config_file=config_path)
            results = app.run_batch_analysis(data_dir, metadata_path)

            # Should still report success for partial completion
            assert results['success'] is True  # Batch framework succeeded
            assert results['total_images'] == 4
            assert results['processed_successfully'] >= 1  # At least valid images
            assert len(results['failed_images']) >= 1  # Some should fail
            assert isinstance(results['processing_errors'], list)

    def test_completely_failed_batch(self, sample_config):
        """Test result aggregation when all images fail."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create batch with only invalid images
            data_dir, metadata_path = self._create_test_batch(temp_dir, 3, all_valid=False)

            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            app = PyLithicsApplication(config_file=config_path)
            results = app.run_batch_analysis(data_dir, metadata_path)

            # Verify failed batch handling
            assert isinstance(results, dict)
            assert results['total_images'] == 3
            assert results['processed_successfully'] == 0
            assert len(results['failed_images']) == 3
            assert len(results['processing_errors']) == 3

    def test_empty_batch_handling(self, sample_config):
        """Test handling of empty batch (no images)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create empty batch
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Empty metadata
            metadata_content = "image_id,scale_id,scale\n"
            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            app = PyLithicsApplication(config_file=config_path)
            results = app.run_batch_analysis(data_dir, metadata_path)

            # Should handle empty batch gracefully
            assert results['success'] is True
            assert results['total_images'] == 0
            assert results['processed_successfully'] == 0
            assert len(results['failed_images']) == 0


@pytest.mark.integration
class TestBatchProcessingWorkflows:
    """Test complete batch processing workflows."""

    def test_small_batch_processing(self, sample_config):
        """Test processing small batch (2-5 images)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            batch_size = 3
            data_dir, metadata_path = self._create_test_batch(temp_dir, batch_size, all_valid=True)

            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            app = PyLithicsApplication(config_file=config_path)

            start_time = time.time()
            results = app.run_batch_analysis(data_dir, metadata_path)
            end_time = time.time()

            # Verify successful processing
            assert results['success'] is True
            assert results['processed_successfully'] == batch_size

            # Verify outputs
            processed_dir = os.path.join(data_dir, 'processed')
            csv_files = list(Path(processed_dir).glob("processed_metrics.csv"))
            assert len(csv_files) == 1

            # Verify all images in CSV
            df = pd.read_csv(csv_files[0])
            unique_images = df['image_id'].nunique()
            assert unique_images == batch_size

            # Performance check for small batch
            assert end_time - start_time < 60.0, "Small batch took too long"

    def test_medium_batch_processing(self, sample_config):
        """Test processing medium batch (10-15 images)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            batch_size = 10
            data_dir, metadata_path = self._create_test_batch(temp_dir, batch_size, all_valid=True)

            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            app = PyLithicsApplication(config_file=config_path)

            start_time = time.time()
            results = app.run_batch_analysis(data_dir, metadata_path)
            end_time = time.time()

            # Verify batch completed
            assert results['success'] is True
            assert results['total_images'] == batch_size

            # Allow for some failures in larger batches
            success_rate = results['processed_successfully'] / results['total_images']
            assert success_rate >= 0.7, f"Success rate too low: {success_rate:.1%}"

            # Performance check for medium batch
            avg_time_per_image = (end_time - start_time) / batch_size
            assert avg_time_per_image < 20.0, f"Average time too high: {avg_time_per_image:.1f}s"

    def test_batch_with_progressive_difficulty(self, sample_config):
        """Test batch with images of increasing processing difficulty."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Create images with increasing complexity
            difficulties = ["simple", "medium", "complex", "very_complex"]
            metadata_content = "image_id,scale_id,scale\n"

            for i, difficulty in enumerate(difficulties):
                filename = f"image_{difficulty}_{i:02d}.png"
                image_path = os.path.join(images_dir, filename)

                test_image = self._create_difficulty_test_image(difficulty)
                cv2.imwrite(image_path, test_image)

                scale = 15.0 + i * 2.0
                metadata_content += f"{filename},scale_{i+1},{scale}\n"

            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            app = PyLithicsApplication(config_file=config_path)
            results = app.run_batch_analysis(data_dir, metadata_path)

            # Should handle varying difficulty levels
            assert results['success'] is True
            assert results['total_images'] == len(difficulties)

            # Simpler images should definitely succeed
            assert results['processed_successfully'] >= 2


@pytest.mark.integration
class TestBatchErrorHandling:
    """Test error handling in batch processing scenarios."""

    def test_batch_continues_after_single_failure(self, sample_config):
        """Test that batch processing continues after individual image failures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Create mix: valid, corrupted, valid, corrupted, valid
            images_data = [
                ("valid_01.png", "valid"),
                ("corrupted_01.png", "corrupted"),
                ("valid_02.png", "valid"),
                ("corrupted_02.png", "corrupted"),
                ("valid_03.png", "valid")
            ]

            metadata_content = "image_id,scale_id,scale\n"
            for i, (filename, image_type) in enumerate(images_data):
                image_path = os.path.join(images_dir, filename)

                if image_type == "valid":
                    test_image = self._create_valid_test_image()
                    cv2.imwrite(image_path, test_image)
                else:  # corrupted
                    # Create corrupted file
                    with open(image_path, 'w') as f:
                        f.write("not an image file")

                scale = 15.0 + i * 1.0
                metadata_content += f"{filename},scale_{i+1},{scale}\n"

            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            app = PyLithicsApplication(config_file=config_path)
            results = app.run_batch_analysis(data_dir, metadata_path)

            # Batch should continue and process valid images
            assert results['success'] is True
            assert results['total_images'] == 5
            assert results['processed_successfully'] == 3  # 3 valid images
            assert len(results['failed_images']) == 2  # 2 corrupted images

            # Failed images should be tracked
            failed_names = results['failed_images']
            assert 'corrupted_01.png' in failed_names
            assert 'corrupted_02.png' in failed_names

    def test_batch_missing_image_files(self, sample_config):
        """Test batch processing when some referenced images don't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Create metadata referencing both existing and missing files
            metadata_content = "image_id,scale_id,scale\n"
            metadata_content += "existing_01.png,scale_1,15.0\n"
            metadata_content += "missing_01.png,scale_2,18.0\n"  # This file won't exist
            metadata_content += "existing_02.png,scale_3,20.0\n"
            metadata_content += "missing_02.png,scale_4,22.0\n"  # This file won't exist

            # Only create the "existing" files
            for filename in ["existing_01.png", "existing_02.png"]:
                image_path = os.path.join(images_dir, filename)
                test_image = self._create_valid_test_image()
                cv2.imwrite(image_path, test_image)

            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            app = PyLithicsApplication(config_file=config_path)
            results = app.run_batch_analysis(data_dir, metadata_path)

            # Should process existing files and track missing ones
            assert results['success'] is True
            assert results['total_images'] == 4
            assert results['processed_successfully'] == 2
            assert len(results['failed_images']) == 2
            assert 'missing_01.png' in results['failed_images']
            assert 'missing_02.png' in results['failed_images']

    def test_batch_with_invalid_metadata_entries(self, sample_config):
        """Test batch processing with some invalid metadata entries."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Create metadata with valid and invalid entries
            metadata_content = "image_id,scale_id,scale\n"
            metadata_content += "valid_01.png,scale_1,15.0\n"
            metadata_content += "valid_02.png,scale_2,invalid_scale\n"  # Invalid scale
            metadata_content += "valid_03.png,scale_3,\n"  # Empty scale
            metadata_content += "valid_04.png,scale_4,25.0\n"

            # Create all image files
            for i in range(1, 5):
                filename = f"valid_{i:02d}.png"
                image_path = os.path.join(images_dir, filename)
                test_image = self._create_valid_test_image()
                cv2.imwrite(image_path, test_image)

            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            app = PyLithicsApplication(config_file=config_path)
            results = app.run_batch_analysis(data_dir, metadata_path)

            # Should handle invalid metadata gracefully
            assert results['success'] is True
            assert results['total_images'] == 4

            # Some images should process successfully despite metadata issues
            assert results['processed_successfully'] >= 2


@pytest.mark.performance
class TestBatchPerformance:
    """Test performance characteristics of batch processing."""

    def test_batch_memory_usage(self, sample_config):
        """Test memory usage during batch processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            batch_size = 5
            data_dir, metadata_path = self._create_test_batch(temp_dir, batch_size)

            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            # Monitor memory usage
            import psutil
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss

            app = PyLithicsApplication(config_file=config_path)
            results = app.run_batch_analysis(data_dir, metadata_path)

            peak_memory = process.memory_info().rss
            memory_increase = peak_memory - initial_memory

            # Memory should not grow excessively
            max_acceptable = 150 * 1024 * 1024  # 150MB
            assert memory_increase < max_acceptable, \
                f"Memory usage too high: {memory_increase / (1024*1024):.1f}MB"

    def test_batch_processing_time_scaling(self, sample_config):
        """Test that processing time scales reasonably with batch size."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            times = []
            batch_sizes = [2, 4, 6]

            for batch_size in batch_sizes:
                batch_dir = os.path.join(temp_dir, f"batch_{batch_size}")
                data_dir, metadata_path = self._create_test_batch(batch_dir, batch_size)

                app = PyLithicsApplication(config_file=config_path)

                start_time = time.time()
                results = app.run_batch_analysis(data_dir, metadata_path)
                end_time = time.time()

                processing_time = end_time - start_time
                times.append((batch_size, processing_time))

            # Verify reasonable scaling
            for batch_size, processing_time in times:
                avg_time_per_image = processing_time / batch_size
                assert avg_time_per_image < 25.0, \
                    f"Batch size {batch_size}: {avg_time_per_image:.1f}s per image too slow"

    def test_batch_resource_cleanup(self, sample_config):
        """Test that resources are properly cleaned up during batch processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            batch_size = 8
            data_dir, metadata_path = self._create_test_batch(temp_dir, batch_size)

            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            import psutil
            process = psutil.Process()
            initial_open_files = len(process.open_files())

            app = PyLithicsApplication(config_file=config_path)
            results = app.run_batch_analysis(data_dir, metadata_path)

            final_open_files = len(process.open_files())
            file_handle_leak = final_open_files - initial_open_files

            # Should not leak file handles
            assert file_handle_leak < 10, f"File handle leak: {file_handle_leak} files"


@pytest.mark.integration
class TestBatchOutputValidation:
    """Test validation of batch processing outputs."""

    def test_batch_csv_consolidation(self, sample_config):
        """Test that batch results are properly consolidated in CSV."""
        with tempfile.TemporaryDirectory() as temp_dir:
            batch_size = 4
            data_dir, metadata_path = self._create_test_batch(temp_dir, batch_size, all_valid=True)

            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            app = PyLithicsApplication(config_file=config_path)
            results = app.run_batch_analysis(data_dir, metadata_path)

            assert results['success'] is True

            # Verify CSV consolidation
            processed_dir = os.path.join(data_dir, 'processed')
            csv_files = list(Path(processed_dir).glob("processed_metrics.csv"))
            assert len(csv_files) == 1, "Should have exactly one consolidated CSV"

            df = pd.read_csv(csv_files[0])

            # Should have data from all processed images
            unique_images = df['image_id'].nunique()
            assert unique_images == results['processed_successfully']

            # Should have consistent data structure
            required_columns = ['image_id', 'parent', 'scar', 'area']
            for col in required_columns:
                assert col in df.columns, f"Missing column: {col}"

    def test_batch_visualization_generation(self, sample_config):
        """Test visualization generation for batch processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            batch_size = 3
            data_dir, metadata_path = self._create_test_batch(temp_dir, batch_size, all_valid=True)

            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            app = PyLithicsApplication(config_file=config_path)
            results = app.run_batch_analysis(data_dir, metadata_path)

            assert results['success'] is True

            # Check visualization files
            processed_dir = os.path.join(data_dir, 'processed')
            viz_files = list(Path(processed_dir).glob("*_labeled.png"))

            # Should have visualizations for processed images
            assert len(viz_files) >= results['processed_successfully']

            # Verify visualization files are valid
            for viz_file in viz_files:
                assert viz_file.stat().st_size > 0
                # Try to read to verify it's a valid image
                test_img = cv2.imread(str(viz_file))
                assert test_img is not None, f"Invalid visualization: {viz_file}"

    def test_batch_processing_logging(self, sample_config):
        """Test logging during batch processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            batch_size = 3
            data_dir, metadata_path = self._create_mixed_batch(temp_dir)

            # Configure logging
            log_config = sample_config.copy()
            log_config['logging'] = {
                'level': 'INFO',
                'log_to_file': True,
                'log_file': os.path.join(temp_dir, 'batch_test.log')
            }

            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(log_config, f)

            app = PyLithicsApplication(config_file=config_path)
            results = app.run_batch_analysis(data_dir, metadata_path)

            # Check log file was created
            log_file = Path(temp_dir) / 'batch_test.log'
            assert log_file.exists(), "Log file should be created"
            assert log_file.stat().st_size > 0, "Log file should contain data"

            # Check log content
            log_content = log_file.read_text()
            assert "Starting batch processing" in log_content
            assert "Batch processing completed" in log_content


# Helper methods for test data creation
def _create_test_batch(self, temp_dir, batch_size, all_valid=True):
    """Create test batch with specified characteristics."""
    data_dir = os.path.join(temp_dir, "data")
    images_dir = os.path.join(data_dir, "images")
    os.makedirs(images_dir)

    metadata_content = "image_id,scale_id,scale\n"

    for i in range(batch_size):
        filename = f"batch_test_{i:03d}.png"
        image_path = os.path.join(images_dir, filename)

        if all_valid:
            test_image = self._create_valid_test_image(variation=i)
            cv2.imwrite(image_path, test_image)
        else:
            # Create invalid/corrupted file
            with open(image_path, 'w') as f:
                f.write(f"corrupted_data_{i}")

        scale = 15.0 + i * 1.5
        metadata_content += f"{filename},scale_{i+1},{scale}\n"

    metadata_path = os.path.join(data_dir, "metadata.csv")
    with open(metadata_path, 'w') as f:
        f.write(metadata_content)

    return data_dir, metadata_path

def _create_mixed_batch(self, temp_dir):
    """Create batch with mix of valid and invalid images."""
    data_dir = os.path.join(temp_dir, "data")
    images_dir = os.path.join(data_dir, "images")
    os.makedirs(images_dir)

    # Mix of valid and invalid files
    files_data = [
        ("valid_01.png", "valid"),
        ("invalid_01.png", "invalid"),
        ("valid_02.png", "valid"),
        ("invalid_02.png", "invalid")
    ]

    metadata_content = "image_id,scale_id,scale\n"

    for i, (filename, file_type) in enumerate(files_data):
        image_path = os.path.join(images_dir, filename)

        if file_type == "valid":
            test_image = self._create_valid_test_image(variation=i)
            cv2.imwrite(image_path, test_image)
        else:
            # Create corrupted file
            with open(image_path, 'w') as f:
                f.write("not_an_image")

        scale = 15.0 + i * 2.0
        metadata_content += f"{filename},scale_{i+1},{scale}\n"

    metadata_path = os.path.join(data_dir, "metadata.csv")
    with open(metadata_path, 'w') as f:
        f.write(metadata_content)

    return data_dir, metadata_path

def _create_valid_test_image(self, variation=0):
    """Create valid test image with optional variation."""
    size = (300, 400)
    height, width = size

    image = np.full((height, width, 3), 240, dtype=np.uint8)

    # Main artifact with variation
    rect_x = 50 + variation * 5
    rect_y = 50 + variation * 3
    rect_w = 200
    rect_h = 150

    cv2.rectangle(image, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (80, 70, 60), -1)

    # Add scar
    scar_x = rect_x + 60 + variation * 2
    scar_y = rect_y + 50 + variation * 2
    cv2.circle(image, (scar_x, scar_y), 20, (50, 45, 40), -1)

    return image

def _create_difficulty_test_image(self, difficulty):
    """Create test images with varying processing difficulty."""
    size = (300, 400)
    height, width = size

    if difficulty == "simple":
        # Simple rectangle
        image = np.full((height, width, 3), 240, dtype=np.uint8)
        cv2.rectangle(image, (100, 100), (200, 200), (80, 70, 60), -1)

    elif difficulty == "medium":
        # Rectangle with scar
        image = np.full((height, width, 3), 240, dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (250, 200), (80, 70, 60), -1)
        cv2.circle(image, (150, 125), 25, (50, 45, 40), -1)

    elif difficulty == "complex":
        # Multiple features
        image = np.full((height, width, 3), 240, dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (250, 200), (80, 70, 60), -1)
        cv2.circle(image, (120, 100), 20, (50, 45, 40), -1)
        cv2.circle(image, (180, 140), 18, (55, 50, 45), -1)
        cv2.circle(image, (150, 170), 15, (45, 40, 35), -1)

    else:  # very_complex
        # Many small features
        image = np.full((height, width, 3), 240, dtype=np.uint8)
        cv2.rectangle(image, (40, 40), (260, 220), (80, 70, 60), -1)

        # Add many small scars
        for i in range(8):
            x = 70 + (i % 4) * 45
            y = 70 + (i // 4) * 35
            radius = 8 + (i % 3) * 3
            cv2.circle(image, (x, y), radius, (50 - i, 45 - i, 40 - i), -1)

    return image

# Add helper methods to test classes
TestBatchResultAggregation._create_test_batch = _create_test_batch
TestBatchResultAggregation._create_mixed_batch = _create_mixed_batch
TestBatchResultAggregation._create_valid_test_image = _create_valid_test_image

TestBatchProcessingWorkflows._create_test_batch = _create_test_batch
TestBatchProcessingWorkflows._create_difficulty_test_image = _create_difficulty_test_image
TestBatchProcessingWorkflows._create_valid_test_image = _create_valid_test_image

TestBatchErrorHandling._create_valid_test_image = _create_valid_test_image
TestBatchErrorHandling._create_mixed_batch = _create_mixed_batch

TestBatchPerformance._create_test_batch = _create_test_batch
TestBatchPerformance._create_valid_test_image = _create_valid_test_image

TestBatchOutputValidation._create_test_batch = _create_test_batch
TestBatchOutputValidation._create_mixed_batch = _create_mixed_batch
TestBatchOutputValidation._create_valid_test_image = _create_valid_test_image


if __name__ == "__main__":
    pytest.main([__file__])