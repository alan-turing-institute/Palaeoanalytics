"""
PyLithics Test Configuration and Fixtures
=========================================

This file provides pytest fixtures and configuration for the PyLithics test suite.
It includes sample data generation, test image creation, and common test utilities.
"""

import pytest
import numpy as np
import cv2
import os
import tempfile
import shutil
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple
from PIL import Image

# Test data constants
TEST_IMAGE_SIZE = (200, 300)  # (height, width)
TEST_DPI = 300


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data that persists for the session."""
    temp_dir = tempfile.mkdtemp(prefix="pylithics_test_")
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def sample_config():
    """Provide a sample configuration dictionary for testing."""
    return {
        'thresholding': {
            'method': 'simple',
            'threshold_value': 127,
            'max_value': 255
        },
        'normalization': {
            'enabled': True,
            'method': 'minmax',
            'clip_values': [0, 255]
        },
        'grayscale_conversion': {
            'enabled': True,
            'method': 'standard'
        },
        'morphological_closing': {
            'enabled': True,
            'kernel_size': 3
        },
        'logging': {
            'level': 'INFO',
            'log_to_file': False,
            'log_file': 'test_pylithics.log'
        },
        'contour_filtering': {
            'min_area': 50.0,
            'exclude_border': True
        },
        'arrow_detection': {
            'enabled': True,
            'reference_dpi': 300.0,
            'min_area_scale_factor': 0.7,
            'min_defect_depth_scale_factor': 0.8,
            'min_triangle_height_scale_factor': 0.8,
            'debug_enabled': False
        }
    }


@pytest.fixture
def sample_config_file(test_data_dir, sample_config):
    """Create a temporary YAML config file for testing."""
    config_path = os.path.join(test_data_dir, "test_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(sample_config, f)
    return config_path


@pytest.fixture
def sample_metadata():
    """Provide sample metadata for testing."""
    return [
        {'image_id': 'test_image_1.png', 'scale_id': 'scale_1', 'scale': '10.0'},
        {'image_id': 'test_image_2.png', 'scale_id': 'scale_2', 'scale': '15.0'},
        {'image_id': 'test_image_3.png', 'scale_id': 'scale_3', 'scale': '12.5'}
    ]


@pytest.fixture
def sample_metadata_file(test_data_dir, sample_metadata):
    """Create a temporary CSV metadata file for testing."""
    import csv
    metadata_path = os.path.join(test_data_dir, "test_metadata.csv")

    with open(metadata_path, 'w', newline='') as csvfile:
        fieldnames = ['image_id', 'scale_id', 'scale']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in sample_metadata:
            writer.writerow(row)

    return metadata_path


@pytest.fixture
def simple_binary_image():
    """Create a simple binary image with a rectangular shape for testing."""
    image = np.zeros(TEST_IMAGE_SIZE, dtype=np.uint8)
    # Add a rectangular shape in the center
    cv2.rectangle(image, (50, 75), (150, 225), 255, -1)
    return image


@pytest.fixture
def complex_binary_image():
    """Create a more complex binary image with multiple shapes for testing."""
    image = np.zeros(TEST_IMAGE_SIZE, dtype=np.uint8)

    # Main artifact shape (irregular polygon)
    main_contour = np.array([
        [50, 100], [80, 80], [120, 90], [140, 120],
        [130, 180], [100, 200], [70, 190], [40, 150]
    ], dtype=np.int32)
    cv2.fillPoly(image, [main_contour], 255)

    # Add some "scars" (smaller shapes inside)
    cv2.circle(image, (90, 130), 15, 0, -1)  # Circular scar
    cv2.ellipse(image, (110, 160), (8, 12), 45, 0, 360, 0, -1)  # Elliptical scar

    return image


@pytest.fixture
def arrow_shaped_contour():
    """Create a contour that resembles an arrow shape for arrow detection testing."""
    # Arrow pointing right
    arrow_points = np.array([
        [10, 20],   # Left tip of arrow
        [25, 10],   # Top of arrowhead
        [20, 15],   # Inner top of arrowhead
        [35, 15],   # Right shaft top
        [35, 25],   # Right shaft bottom
        [20, 25],   # Inner bottom of arrowhead
        [25, 30]    # Bottom of arrowhead
    ], dtype=np.int32)

    return arrow_points.reshape(-1, 1, 2)


@pytest.fixture
def sample_contours():
    """Provide sample contours for testing."""
    # Main contour (parent)
    parent = np.array([
        [20, 30], [50, 25], [80, 40], [85, 70],
        [75, 100], [45, 105], [15, 90], [10, 60]
    ], dtype=np.int32).reshape(-1, 1, 2)

    # Child contour (scar)
    child = np.array([
        [40, 50], [55, 48], [58, 65], [45, 67]
    ], dtype=np.int32).reshape(-1, 1, 2)

    return [parent, child]


@pytest.fixture
def sample_hierarchy():
    """Provide sample hierarchy data for testing."""
    # Hierarchy format: [next, previous, first_child, parent]
    # Parent contour has no parent (-1), child contour has parent (0)
    return np.array([
        [-1, -1, 1, -1],  # Parent contour (index 0)
        [-1, -1, -1, 0]   # Child contour (index 1, parent is 0)
    ])


@pytest.fixture
def sample_metrics():
    """Provide sample metrics data for testing."""
    return [
        {
            'parent': 'parent 1',
            'scar': 'parent 1',
            'surface_type': 'Dorsal',
            'centroid_x': 50.0,
            'centroid_y': 65.0,
            'technical_width': 60.0,
            'technical_length': 75.0,
            'area': 4500.0,
            'aspect_ratio': 1.25,
            'max_length': 80.0,
            'max_width': 55.0,
            'has_arrow': False
        },
        {
            'parent': 'parent 1',
            'scar': 'scar 1',
            'surface_type': 'Dorsal',
            'centroid_x': 50.0,
            'centroid_y': 58.0,
            'width': 18.0,
            'height': 17.0,
            'area': 306.0,
            'aspect_ratio': 0.94,
            'max_length': 20.0,
            'max_width': 15.0,
            'has_arrow': True,
            'arrow_angle': 45.0
        }
    ]


@pytest.fixture
def test_image_with_dpi(test_data_dir):
    """Create a test image file with DPI information."""
    image_path = os.path.join(test_data_dir, "test_image_with_dpi.png")

    # Create a simple test image
    image_array = np.zeros((200, 300, 3), dtype=np.uint8)
    cv2.rectangle(image_array, (50, 50), (250, 150), (255, 255, 255), -1)

    # Save with DPI information using PIL
    pil_image = Image.fromarray(image_array)
    pil_image.save(image_path, dpi=(TEST_DPI, TEST_DPI))

    return image_path


@pytest.fixture
def test_image_directory(test_data_dir, test_image_with_dpi):
    """Create a test directory structure with images."""
    images_dir = os.path.join(test_data_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Copy the test image to the images directory
    image_dest = os.path.join(images_dir, "test_image_1.png")
    shutil.copy2(test_image_with_dpi, image_dest)

    # Create additional test images
    for i in range(2, 4):
        image_path = os.path.join(images_dir, f"test_image_{i}.png")
        image_array = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
        pil_image = Image.fromarray(image_array)
        pil_image.save(image_path, dpi=(TEST_DPI, TEST_DPI))

    return images_dir


@pytest.fixture
def processed_output_dir(test_data_dir):
    """Create a processed output directory for testing."""
    processed_dir = os.path.join(test_data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    return processed_dir


class ImageComparisonHelper:
    """Helper class for comparing images in tests."""

    @staticmethod
    def images_similar(img1: np.ndarray, img2: np.ndarray, tolerance: float = 0.01) -> bool:
        """
        Compare two images and return True if they are similar within tolerance.

        Args:
            img1, img2: Images to compare
            tolerance: Allowed difference as fraction (0-1)
        """
        if img1.shape != img2.shape:
            return False

        diff = np.abs(img1.astype(float) - img2.astype(float))
        max_diff = np.max(diff)
        normalized_diff = max_diff / 255.0

        return normalized_diff <= tolerance

    @staticmethod
    def contours_similar(cnt1: np.ndarray, cnt2: np.ndarray, tolerance: float = 2.0) -> bool:
        """
        Compare two contours and return True if they are similar within tolerance.

        Args:
            cnt1, cnt2: Contours to compare
            tolerance: Maximum allowed pixel difference
        """
        if len(cnt1) != len(cnt2):
            return False

        # Compare contour points
        diff = np.abs(cnt1.astype(float) - cnt2.astype(float))
        max_diff = np.max(diff)

        return max_diff <= tolerance


@pytest.fixture
def image_comparison():
    """Provide image comparison utilities for tests."""
    return ImageComparisonHelper()


class MockConfigManager:
    """Mock configuration manager for testing."""

    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    def get_section(self, section: str) -> Dict[str, Any]:
        return self._config.get(section, {})

    def get_value(self, section: str, key: str, default: Any = None) -> Any:
        return self._config.get(section, {}).get(key, default)

    def update_value(self, section: str, key: str, value: Any) -> None:
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value


@pytest.fixture
def mock_config_manager(sample_config):
    """Provide a mock configuration manager for testing."""
    return MockConfigManager(sample_config)


# Utility functions for test assertions
def assert_config_section_valid(config: Dict[str, Any], section: str, required_keys: List[str]):
    """Assert that a config section contains all required keys."""
    assert section in config, f"Missing config section: {section}"
    section_config = config[section]
    for key in required_keys:
        assert key in section_config, f"Missing key '{key}' in section '{section}'"


def assert_contour_valid(contour: np.ndarray):
    """Assert that a contour is valid."""
    assert contour is not None, "Contour is None"
    assert len(contour.shape) == 3, f"Invalid contour shape: {contour.shape}"
    assert contour.shape[1] == 1, f"Invalid contour format: {contour.shape}"
    assert contour.shape[2] == 2, f"Invalid contour coordinates: {contour.shape}"
    assert len(contour) >= 3, f"Contour has too few points: {len(contour)}"


def assert_metrics_valid(metrics: List[Dict[str, Any]]):
    """Assert that metrics list is valid."""
    assert isinstance(metrics, list), "Metrics should be a list"
    assert len(metrics) > 0, "Metrics list is empty"

    required_keys = ['parent', 'scar', 'centroid_x', 'centroid_y', 'area']
    for i, metric in enumerate(metrics):
        assert isinstance(metric, dict), f"Metric {i} is not a dictionary"
        for key in required_keys:
            assert key in metric, f"Missing key '{key}' in metric {i}"