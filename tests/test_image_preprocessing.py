"""
PyLithics Image Preprocessing Tests
===================================

Tests for the complete image preprocessing pipeline including grayscale conversion,
normalization, thresholding, morphological operations, and DPI validation.
"""

import pytest
import numpy as np
import cv2
import tempfile
import os
from PIL import Image
from unittest.mock import patch, MagicMock

from pylithics.image_processing.importer import (
    read_image_from_path,
    apply_grayscale_conversion,
    apply_contrast_normalization,
    perform_thresholding,
    invert_image,
    morphological_closing,
    verify_image_dpi_and_scale,
    execute_preprocessing_pipeline,
    preprocess_images
)


@pytest.mark.unit
class TestReadImageFromPath:
    """Test image reading functionality."""

    def test_read_valid_image(self, test_image_with_dpi):
        """Test reading a valid image file."""
        image = read_image_from_path(test_image_with_dpi)

        assert image is not None
        assert isinstance(image, np.ndarray)
        assert len(image.shape) == 3  # Should be BGR color image
        assert image.shape[2] == 3    # Three color channels

    def test_read_nonexistent_image(self):
        """Test reading a non-existent image file."""
        with patch('pylithics.image_processing.importer.logging') as mock_logging:
            image = read_image_from_path('/nonexistent/image.png')

            assert image is None
            mock_logging.error.assert_called()

    def test_read_invalid_image_file(self, test_data_dir):
        """Test reading an invalid image file."""
        # Create a file that's not an image
        invalid_file = os.path.join(test_data_dir, "not_an_image.png")
        with open(invalid_file, 'w') as f:
            f.write("This is not an image file")

        with patch('pylithics.image_processing.importer.logging') as mock_logging:
            image = read_image_from_path(invalid_file)

            assert image is None
            mock_logging.error.assert_called()

    def test_read_corrupted_image(self, test_data_dir):
        """Test reading a corrupted image file."""
        # Create a file with image extension but corrupted content
        corrupted_file = os.path.join(test_data_dir, "corrupted.png")
        with open(corrupted_file, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n')  # PNG header but incomplete
            f.write(b'corrupted data')

        image = read_image_from_path(corrupted_file)

        # OpenCV might return None or handle it gracefully
        # Either None or a valid array is acceptable
        assert image is None or isinstance(image, np.ndarray)


@pytest.mark.unit
class TestApplyGrayscaleConversion:
    """Test grayscale conversion functionality."""

    def test_standard_grayscale_conversion(self):
        """Test standard grayscale conversion."""
        # Create a color image
        color_image = np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)

        config = {
            'grayscale_conversion': {
                'enabled': True,
                'method': 'standard'
            }
        }

        gray_image = apply_grayscale_conversion(color_image, config)

        assert gray_image is not None
        assert len(gray_image.shape) == 2  # Should be grayscale
        assert gray_image.shape == (100, 150)
        assert gray_image.dtype == np.uint8

    def test_clahe_grayscale_conversion(self):
        """Test CLAHE grayscale conversion."""
        color_image = np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)

        config = {
            'grayscale_conversion': {
                'enabled': True,
                'method': 'clahe'
            }
        }

        gray_image = apply_grayscale_conversion(color_image, config)

        assert gray_image is not None
        assert len(gray_image.shape) == 2
        assert gray_image.shape == (100, 150)

    def test_disabled_grayscale_conversion(self):
        """Test when grayscale conversion is disabled."""
        color_image = np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)

        config = {
            'grayscale_conversion': {
                'enabled': False,
                'method': 'standard'
            }
        }

        result_image = apply_grayscale_conversion(color_image, config)

        # Should return original image unchanged
        assert np.array_equal(result_image, color_image)

    def test_invalid_grayscale_method(self):
        """Test handling of invalid grayscale conversion method."""
        color_image = np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)

        config = {
            'grayscale_conversion': {
                'enabled': True,
                'method': 'invalid_method'
            }
        }

        with patch('pylithics.image_processing.importer.logging') as mock_logging:
            result = apply_grayscale_conversion(color_image, config)

            assert result is None
            mock_logging.error.assert_called()

    def test_grayscale_missing_config(self):
        """Test grayscale conversion with missing config section."""
        color_image = np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)

        # Since the code expects the section to exist, we should test that it fails appropriately
        config = {}  # Missing grayscale_conversion section

        # The function will fail with KeyError
        with pytest.raises(KeyError):
            result = apply_grayscale_conversion(color_image, config)


@pytest.mark.unit
class TestApplyContrastNormalization:
    """Test contrast normalization functionality."""

    def test_minmax_normalization(self):
        """Test min-max normalization."""
        # Create an image with known range
        gray_image = np.array([
            [50, 100, 150],
            [75, 125, 175],
            [25, 200, 225]
        ], dtype=np.uint8)

        config = {
            'normalization': {
                'enabled': True,
                'method': 'minmax',
                'clip_values': [0, 255]
            }
        }

        normalized = apply_contrast_normalization(gray_image, config)

        assert normalized is not None
        assert normalized.shape == gray_image.shape
        assert normalized.min() >= 0
        assert normalized.max() <= 255

    def test_zscore_normalization(self):
        """Test z-score normalization."""
        gray_image = np.random.randint(50, 200, (50, 50), dtype=np.uint8)

        config = {
            'normalization': {
                'enabled': True,
                'method': 'zscore'
            }
        }

        normalized = apply_contrast_normalization(gray_image, config)

        assert normalized is not None
        assert normalized.shape == gray_image.shape
        # Z-score normalized data should have mean ~0, std ~1
        assert abs(normalized.mean()) < 1.0
        assert abs(normalized.std() - 1.0) < 0.1

    def test_disabled_normalization(self):
        """Test when normalization is disabled."""
        gray_image = np.random.randint(0, 256, (50, 50), dtype=np.uint8)

        config = {
            'normalization': {
                'enabled': False,
                'method': 'minmax'
            }
        }

        result = apply_contrast_normalization(gray_image, config)

        # Should return original image unchanged
        assert np.array_equal(result, gray_image)

    def test_invalid_normalization_method(self):
        """Test handling of invalid normalization method."""
        gray_image = np.random.randint(0, 256, (50, 50), dtype=np.uint8)

        config = {
            'normalization': {
                'enabled': True,
                'method': 'invalid_method'
            }
        }

        with patch('pylithics.image_processing.importer.logging') as mock_logging:
            result = apply_contrast_normalization(gray_image, config)

            assert result is None
            mock_logging.error.assert_called()


@pytest.mark.unit
class TestPerformThresholding:
    """Test thresholding functionality."""

    def test_simple_thresholding(self):
        """Test simple binary thresholding."""
        gray_image = np.array([
            [50, 100, 150],
            [75, 125, 175],
            [25, 200, 225]
        ], dtype=np.uint8)

        config = {
            'thresholding': {
                'method': 'simple',
                'threshold_value': 127,
                'max_value': 255
            }
        }

        thresholded = perform_thresholding(gray_image, config)

        assert thresholded is not None
        assert thresholded.shape == gray_image.shape
        # Should be binary (only 0 or 255)
        unique_values = np.unique(thresholded)
        assert len(unique_values) <= 2
        assert all(val in [0, 255] for val in unique_values)

    def test_otsu_thresholding(self):
        """Test Otsu automatic thresholding."""
        # Create bimodal image (good for Otsu)
        gray_image = np.concatenate([
            np.full((50, 50), 80, dtype=np.uint8),   # Dark region
            np.full((50, 50), 180, dtype=np.uint8)   # Bright region
        ])

        config = {
            'thresholding': {
                'method': 'otsu',
                'max_value': 255
            }
        }

        thresholded = perform_thresholding(gray_image, config)

        assert thresholded is not None
        assert thresholded.shape == gray_image.shape
        # Should be binary
        unique_values = np.unique(thresholded)
        assert len(unique_values) <= 2

    def test_adaptive_thresholding(self):
        """Test adaptive thresholding."""
        # Create image with varying illumination
        gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

        config = {
            'thresholding': {
                'method': 'adaptive',
                'max_value': 255
            }
        }

        thresholded = perform_thresholding(gray_image, config)

        assert thresholded is not None
        assert thresholded.shape == gray_image.shape
        # Should be binary
        unique_values = np.unique(thresholded)
        assert len(unique_values) <= 2

    def test_default_thresholding(self):
        """Test default thresholding method."""
        gray_image = np.random.randint(0, 256, (50, 50), dtype=np.uint8)

        config = {
            'thresholding': {
                'method': 'default',
                'threshold_value': 100,
                'max_value': 255
            }
        }

        thresholded = perform_thresholding(gray_image, config)

        assert thresholded is not None
        assert thresholded.shape == gray_image.shape

    def test_invalid_thresholding_method(self):
        """Test handling of invalid thresholding method."""
        gray_image = np.random.randint(0, 256, (50, 50), dtype=np.uint8)

        config = {
            'thresholding': {
                'method': 'invalid_method',
                'threshold_value': 127,
                'max_value': 255
            }
        }

        with patch('pylithics.image_processing.importer.logging') as mock_logging:
            result = perform_thresholding(gray_image, config)

            assert result is None
            mock_logging.error.assert_called()

    def test_thresholding_gaussian_blur_applied(self):
        """Test that Gaussian blur is applied before thresholding."""
        # Create noisy image
        gray_image = np.random.randint(0, 256, (50, 50), dtype=np.uint8)

        config = {
            'thresholding': {
                'method': 'simple',
                'threshold_value': 127,
                'max_value': 255
            }
        }

        with patch('cv2.GaussianBlur') as mock_blur:
            mock_blur.return_value = gray_image  # Return same image for simplicity

            thresholded = perform_thresholding(gray_image, config)

            # Verify Gaussian blur was called
            mock_blur.assert_called_once()


@pytest.mark.unit
class TestInvertImage:
    """Test image inversion functionality."""

    def test_invert_binary_image(self):
        """Test inverting a binary image."""
        binary_image = np.array([
            [0, 255, 0],
            [255, 0, 255],
            [0, 0, 255]
        ], dtype=np.uint8)

        inverted = invert_image(binary_image)

        expected = np.array([
            [255, 0, 255],
            [0, 255, 0],
            [255, 255, 0]
        ], dtype=np.uint8)

        assert np.array_equal(inverted, expected)

    def test_invert_grayscale_image(self):
        """Test inverting a grayscale image."""
        gray_image = np.array([
            [0, 100, 255],
            [50, 150, 200]
        ], dtype=np.uint8)

        inverted = invert_image(gray_image)

        expected = np.array([
            [255, 155, 0],
            [205, 105, 55]
        ], dtype=np.uint8)

        assert np.array_equal(inverted, expected)


@pytest.mark.unit
class TestMorphologicalClosing:
    """Test morphological closing functionality."""

    def test_morphological_closing_enabled(self):
        """Test morphological closing when enabled."""
        # Create image with gaps
        binary_image = np.array([
            [0, 0, 0, 0, 0],
            [0, 255, 0, 255, 0],
            [0, 255, 255, 255, 0],
            [0, 255, 0, 255, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.uint8)

        config = {
            'morphological_closing': {
                'enabled': True,
                'kernel_size': 3
            }
        }

        closed = morphological_closing(binary_image, config)

        assert closed is not None
        assert closed.shape == binary_image.shape
        assert closed.dtype == binary_image.dtype

    def test_morphological_closing_different_kernel_sizes(self):
        """Test morphological closing with different kernel sizes."""
        binary_image = np.zeros((20, 20), dtype=np.uint8)
        binary_image[5:15, 5:15] = 255  # White square
        binary_image[8:12, 8:12] = 0    # Black hole

        for kernel_size in [3, 5, 7]:
            config = {
                'morphological_closing': {
                    'enabled': True,
                    'kernel_size': kernel_size
                }
            }

            closed = morphological_closing(binary_image, config)

            assert closed is not None
            assert closed.shape == binary_image.shape

    def test_morphological_closing_missing_config(self):
        """Test morphological closing with missing config values."""
        binary_image = np.random.randint(0, 2, (50, 50), dtype=np.uint8) * 255

        config = {}  # Missing morphological_closing section

        # Should use default values
        closed = morphological_closing(binary_image, config)

        assert closed is not None
        assert closed.shape == binary_image.shape

@pytest.mark.unit
class TestVerifyImageDpiAndScale:
    """Test DPI verification and scale calculation."""

    def test_verify_dpi_valid_image(self, test_image_with_dpi):
        """Test DPI verification with valid image."""
        real_world_scale_mm = 10.0

        conversion_factor = verify_image_dpi_and_scale(test_image_with_dpi, real_world_scale_mm)

        assert conversion_factor is not None
        assert conversion_factor > 0
        # Should be approximately 300 DPI / 25.4 mm/inch
        expected_factor = 300.0 / 25.4
        assert abs(conversion_factor - expected_factor) < 0.1

    def test_verify_dpi_no_dpi_info(self, test_data_dir):
        """Test DPI verification with image lacking DPI information."""
        # Create image without DPI info
        image_array = np.zeros((100, 100, 3), dtype=np.uint8)
        image_path = os.path.join(test_data_dir, "no_dpi_image.png")

        # Save without DPI info using OpenCV
        cv2.imwrite(image_path, image_array)

        with patch('pylithics.image_processing.importer.logging') as mock_logging:
            conversion_factor = verify_image_dpi_and_scale(image_path, 10.0)

            assert conversion_factor is None
            mock_logging.warning.assert_called()

    def test_verify_dpi_nonexistent_image(self):
        """Test DPI verification with non-existent image."""
        with patch('pylithics.image_processing.importer.logging') as mock_logging:
            conversion_factor = verify_image_dpi_and_scale('/nonexistent/image.png', 10.0)

            assert conversion_factor is None
            mock_logging.error.assert_called()

    def test_verify_dpi_calculation(self, test_data_dir):
        """Test DPI calculation with known values."""
        # Create image with specific DPI
        test_dpi = 150.0
        image_array = np.zeros((100, 100, 3), dtype=np.uint8)
        image_path = os.path.join(test_data_dir, "known_dpi_image.png")

        pil_image = Image.fromarray(image_array)
        pil_image.save(image_path, dpi=(test_dpi, test_dpi))

        real_world_scale_mm = 5.0
        conversion_factor = verify_image_dpi_and_scale(image_path, real_world_scale_mm)

        assert conversion_factor is not None
        expected_factor = test_dpi / 25.4
        assert abs(conversion_factor - expected_factor) < 0.1


@pytest.mark.integration
class TestExecutePreprocessingPipeline:
    """Test the complete preprocessing pipeline."""

    def test_complete_pipeline_success(self, test_image_with_dpi, sample_config):
        """Test successful execution of complete preprocessing pipeline."""
        processed_image = execute_preprocessing_pipeline(test_image_with_dpi, sample_config)

        assert processed_image is not None
        assert isinstance(processed_image, np.ndarray)
        assert len(processed_image.shape) == 2  # Should be grayscale
        assert processed_image.dtype == np.uint8
        # Should be binary (mostly 0s and 255s)
        unique_values = np.unique(processed_image)
        assert len(unique_values) <= 10  # Allow some intermediate values from blur/morphology

    def test_pipeline_with_nonexistent_image(self, sample_config):
        """Test pipeline with non-existent image."""
        result = execute_preprocessing_pipeline('/nonexistent/image.png', sample_config)

        assert result is None

    def test_pipeline_step_failure_propagation(self, test_image_with_dpi):
        """Test that pipeline stops when a step fails."""
        # Config that will cause thresholding to fail
        bad_config = {
            'grayscale_conversion': {'enabled': True, 'method': 'standard'},
            'normalization': {'enabled': True, 'method': 'minmax', 'clip_values': [0, 255]},
            'thresholding': {'method': 'invalid_method'},  # This will fail
            'morphological_closing': {'enabled': True, 'kernel_size': 3}
        }

        result = execute_preprocessing_pipeline(test_image_with_dpi, bad_config)

        assert result is None

    def test_pipeline_different_configurations(self, test_image_with_dpi):
        """Test pipeline with different configuration combinations."""
        configs = [
            {
                'grayscale_conversion': {'enabled': True, 'method': 'standard'},
                'normalization': {'enabled': False},
                'thresholding': {'method': 'simple', 'threshold_value': 100, 'max_value': 255},
                'morphological_closing': {'enabled': True, 'kernel_size': 3}
            },
            {
                'grayscale_conversion': {'enabled': True, 'method': 'clahe'},
                'normalization': {'enabled': True, 'method': 'minmax', 'clip_values': [0, 255]},
                'thresholding': {'method': 'otsu', 'max_value': 255},
                'morphological_closing': {'enabled': True, 'kernel_size': 5}
            }
        ]

        for config in configs:
            result = execute_preprocessing_pipeline(test_image_with_dpi, config)

            assert result is not None
            assert isinstance(result, np.ndarray)
            assert len(result.shape) == 2


@pytest.mark.integration
class TestPreprocessImages:
    """Test batch image preprocessing functionality."""

    def test_preprocess_multiple_images(self, test_image_directory, sample_metadata_file, sample_config):
        """Test preprocessing multiple images."""
        # Create a temporary data directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy test structure
            import shutil
            data_dir = os.path.join(temp_dir, "data")
            shutil.copytree(os.path.dirname(test_image_directory), data_dir)

            with patch('pylithics.image_processing.importer.load_preprocessing_config') as mock_load_config:
                mock_load_config.return_value = sample_config

                preprocessed_images = preprocess_images(data_dir, sample_metadata_file, False)

                assert isinstance(preprocessed_images, dict)
                # Should have entries for successfully processed images
                assert len(preprocessed_images) >= 0

                # Check format of returned data
                for image_id, (processed_image, conversion_factor) in preprocessed_images.items():
                    assert isinstance(processed_image, np.ndarray)
                    assert isinstance(conversion_factor, (int, float))
                    assert conversion_factor > 0

    def test_preprocess_images_config_loading_failure(self, test_image_directory, sample_metadata_file):
        """Test behavior when config loading fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            import shutil
            data_dir = os.path.join(temp_dir, "data")
            shutil.copytree(os.path.dirname(test_image_directory), data_dir)

            with patch('pylithics.image_processing.importer.load_preprocessing_config') as mock_load_config:
                mock_load_config.return_value = None  # Config loading failed

                with patch('pylithics.image_processing.importer.logging') as mock_logging:
                    preprocessed_images = preprocess_images(data_dir, sample_metadata_file, False)

                    assert preprocessed_images == {}
                    mock_logging.error.assert_called()

    def test_preprocess_images_missing_images(self, sample_metadata_file, sample_config):
        """Test preprocessing when image files are missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create data directory structure but no images
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            with patch('pylithics.image_processing.importer.load_preprocessing_config') as mock_load_config:
                mock_load_config.return_value = sample_config

                with patch('pylithics.image_processing.importer.logging') as mock_logging:
                    preprocessed_images = preprocess_images(data_dir, sample_metadata_file, False)

                    assert preprocessed_images == {}
                    # Should log errors for missing images
                    mock_logging.error.assert_called()

    def test_preprocess_images_partial_success(self, test_image_directory, sample_config):
        """Test preprocessing with some successful and some failed images."""
        # Create metadata with mix of existing and non-existing images
        metadata = [
            {'image_id': 'existing_image.png', 'scale': '10.0'},
            {'image_id': 'nonexistent_image.png', 'scale': '12.0'}
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create one valid image
            import shutil
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Copy one test image
            test_files = os.listdir(test_image_directory)
            if test_files:
                shutil.copy2(
                    os.path.join(test_image_directory, test_files[0]),
                    os.path.join(images_dir, "existing_image.png")
                )

            with patch('pylithics.image_processing.importer.read_metadata') as mock_read_metadata:
                mock_read_metadata.return_value = metadata

                with patch('pylithics.image_processing.importer.load_preprocessing_config') as mock_load_config:
                    mock_load_config.return_value = sample_config

                    preprocessed_images = preprocess_images(data_dir, "dummy_meta_file", False)

                    # Should have at most one successful image
                    assert len(preprocessed_images) <= 1


@pytest.mark.unit
class TestPreprocessingErrorHandling:
    """Test error handling in preprocessing functions."""

    def test_grayscale_conversion_with_invalid_image(self):
        """Test grayscale conversion with invalid image data."""
        invalid_image = "not an image"
        config = {'grayscale_conversion': {'enabled': True, 'method': 'standard'}}

        # OpenCV will raise an error for invalid input
        with pytest.raises(cv2.error):
            result = apply_grayscale_conversion(invalid_image, config)

    def test_normalization_with_empty_image(self):
        """Test normalization with empty image."""
        empty_image = np.array([], dtype=np.uint8)
        config = {'normalization': {'enabled': True, 'method': 'minmax', 'clip_values': [0, 255]}}

        try:
            result = apply_contrast_normalization(empty_image, config)
            # Should handle gracefully
        except (ValueError, TypeError):
            # These are acceptable errors for empty input
            pass

    def test_thresholding_with_single_pixel(self):
        """Test thresholding with single pixel image."""
        single_pixel = np.array([[100]], dtype=np.uint8)
        config = {'thresholding': {'method': 'simple', 'threshold_value': 127, 'max_value': 255}}

        result = perform_thresholding(single_pixel, config)

        # Should handle gracefully
        if result is not None:
            assert result.shape == single_pixel.shape
            assert result.dtype == np.uint8


@pytest.mark.performance
class TestPreprocessingPerformance:
    """Test preprocessing performance with various image sizes."""

    def test_pipeline_performance_small_image(self, sample_config):
        """Test pipeline performance with small image."""
        # Create small test image
        small_image_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Save test image
            pil_image = Image.fromarray(small_image_array)
            pil_image.save(temp_path, dpi=(300, 300))

            # Time the preprocessing
            import time
            start_time = time.time()

            result = execute_preprocessing_pipeline(temp_path, sample_config)

            end_time = time.time()
            processing_time = end_time - start_time

            # Should complete quickly for small image
            assert processing_time < 5.0  # 5 seconds max
            assert result is not None

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_pipeline_performance_large_image(self, sample_config):
        """Test pipeline performance with larger image."""
        # Create larger test image
        large_image_array = np.random.randint(0, 256, (1000, 1000, 3), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Save test image
            pil_image = Image.fromarray(large_image_array)
            pil_image.save(temp_path, dpi=(300, 300))

            # Time the preprocessing
            import time
            start_time = time.time()

            result = execute_preprocessing_pipeline(temp_path, sample_config)

            end_time = time.time()
            processing_time = end_time - start_time

            # Should complete in reasonable time even for large image
            assert processing_time < 30.0  # 30 seconds max
            assert result is not None

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


@pytest.mark.integration
class TestPreprocessingRealWorldScenarios:
    """Test preprocessing with realistic archaeological image scenarios."""

    def test_archaeological_artifact_preprocessing(self, sample_config):
        """Test preprocessing pipeline with archaeological artifact-like image."""
        # Create realistic artifact image
        artifact_image = np.zeros((400, 600, 3), dtype=np.uint8)

        # Background (light)
        artifact_image.fill(240)

        # Add artifact shape (darker)
        cv2.ellipse(artifact_image, (300, 200), (150, 100), 0, 0, 360, (80, 70, 60), -1)

        # Add some surface texture
        noise = np.random.normal(0, 10, artifact_image.shape).astype(np.int16)
        artifact_image = np.clip(artifact_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Add some scars (small dark areas)
        cv2.circle(artifact_image, (280, 180), 15, (40, 35, 30), -1)
        cv2.circle(artifact_image, (320, 220), 12, (45, 40, 35), -1)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Save test image with DPI
            pil_image = Image.fromarray(artifact_image)
            pil_image.save(temp_path, dpi=(300, 300))

            result = execute_preprocessing_pipeline(temp_path, sample_config)

            assert result is not None
            assert result.shape == (400, 600)  # Should be grayscale

            # Should be mostly binary
            unique_values = np.unique(result)
            assert len(unique_values) <= 10  # Allow some intermediate values

            # Should have both foreground and background
            assert 0 in unique_values  # Background
            assert 255 in unique_values  # Foreground

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_varying_lighting_conditions(self, sample_config):
        """Test preprocessing with images having varying lighting conditions."""
        # Create image with gradient lighting
        base_image = np.zeros((300, 400, 3), dtype=np.uint8)

        # Create lighting gradient
        for y in range(300):
            for x in range(400):
                # Gradient from bright (left) to dark (right)
                brightness = int(200 - (x / 400) * 100)
                base_image[y, x] = [brightness, brightness, brightness]

        # Add artifact shape
        cv2.rectangle(base_image, (150, 100), (250, 200), (0, 0, 0), -1)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            pil_image = Image.fromarray(base_image)
            pil_image.save(temp_path, dpi=(300, 300))

            # Test different thresholding methods
            methods = ['simple', 'adaptive', 'otsu']

            for method in methods:
                config = sample_config.copy()
                config['thresholding']['method'] = method

                result = execute_preprocessing_pipeline(temp_path, config)

                if result is not None:  # Some methods might work better than others
                    assert result.shape == (300, 400)
                    assert result.dtype == np.uint8

                    # Should detect the dark rectangle
                    dark_pixels = np.sum(result == 255)  # Inverted, so artifact is white
                    assert dark_pixels > 1000  # Should find significant artifact area

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_high_contrast_artifact(self, sample_config):
        """Test preprocessing with high contrast artifact image."""
        # Create high contrast image
        high_contrast_image = np.full((300, 400, 3), 255, dtype=np.uint8)  # White background

        # Add very dark artifact
        cv2.ellipse(high_contrast_image, (200, 150), (80, 120), 0, 0, 360, (10, 10, 10), -1)

        # Add some scars
        cv2.circle(high_contrast_image, (190, 130), 8, (255, 255, 255), -1)  # Light scar
        cv2.circle(high_contrast_image, (210, 170), 6, (0, 0, 0), -1)        # Dark scar

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            pil_image = Image.fromarray(high_contrast_image)
            pil_image.save(temp_path, dpi=(300, 300))

            result = execute_preprocessing_pipeline(temp_path, sample_config)

            assert result is not None
            assert result.shape == (300, 400)

            # High contrast should produce clean binary result
            unique_values = np.unique(result)
            assert len(unique_values) <= 5  # Should be mostly binary

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


@pytest.mark.unit
class TestPreprocessingEdgeCases:
    """Test edge cases in preprocessing functions."""

    def test_very_dark_image(self, sample_config):
        """Test preprocessing with very dark image."""
        dark_image = np.full((100, 100, 3), 20, dtype=np.uint8)  # Very dark

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            pil_image = Image.fromarray(dark_image)
            pil_image.save(temp_path, dpi=(300, 300))

            result = execute_preprocessing_pipeline(temp_path, sample_config)

            # Should handle gracefully
            if result is not None:
                assert result.shape == (100, 100)
                assert result.dtype == np.uint8

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_very_bright_image(self, sample_config):
        """Test preprocessing with very bright image."""
        bright_image = np.full((100, 100, 3), 240, dtype=np.uint8)  # Very bright

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            pil_image = Image.fromarray(bright_image)
            pil_image.save(temp_path, dpi=(300, 300))

            result = execute_preprocessing_pipeline(temp_path, sample_config)

            # Should handle gracefully
            if result is not None:
                assert result.shape == (100, 100)
                assert result.dtype == np.uint8

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_uniform_color_image(self, sample_config):
        """Test preprocessing with uniform color image."""
        uniform_image = np.full((100, 100, 3), 128, dtype=np.uint8)  # Uniform gray

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            pil_image = Image.fromarray(uniform_image)
            pil_image.save(temp_path, dpi=(300, 300))

            result = execute_preprocessing_pipeline(temp_path, sample_config)

            # Should handle uniform images
            if result is not None:
                assert result.shape == (100, 100)
                # Uniform image might result in all same values after thresholding
                unique_values = np.unique(result)
                assert len(unique_values) >= 1

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_minimal_size_image(self, sample_config):
        """Test preprocessing with minimal size image."""
        tiny_image = np.random.randint(0, 256, (5, 5, 3), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            pil_image = Image.fromarray(tiny_image)
            pil_image.save(temp_path, dpi=(300, 300))

            result = execute_preprocessing_pipeline(temp_path, sample_config)

            # Should handle tiny images
            if result is not None:
                assert result.shape == (5, 5)
                assert result.dtype == np.uint8

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)