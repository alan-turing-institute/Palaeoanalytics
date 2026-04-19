"""Pytest fixtures for the PyLithics test suite."""

import csv
import os
import shutil
import tempfile

import cv2
import numpy as np
import pytest
import yaml
from PIL import Image


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
            'max_value': 255,
        },
        'normalization': {
            'enabled': True,
            'method': 'minmax',
            'clip_values': [0, 255],
        },
        'grayscale_conversion': {
            'enabled': True,
            'method': 'standard',
        },
        'morphological_closing': {
            'enabled': True,
            'kernel_size': 3,
        },
        'logging': {
            'level': 'INFO',
            'log_to_file': False,
            'log_file': 'test_pylithics.log',
        },
        'contour_filtering': {
            'min_area': 50.0,
            'exclude_border': True,
        },
        'arrow_detection': {
            'enabled': True,
            'reference_dpi': 300.0,
            'min_area_scale_factor': 0.7,
            'min_defect_depth_scale_factor': 0.8,
            'min_triangle_height_scale_factor': 0.8,
            'debug_enabled': False,
        },
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
        {'image_id': 'test_image_3.png', 'scale_id': 'scale_3', 'scale': '12.5'},
    ]


@pytest.fixture
def sample_metadata_file(test_data_dir, sample_metadata):
    """Create a temporary CSV metadata file for testing."""
    metadata_path = os.path.join(test_data_dir, "test_metadata.csv")
    with open(metadata_path, 'w', newline='') as csvfile:
        fieldnames = ['image_id', 'scale_id', 'scale']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in sample_metadata:
            writer.writerow(row)
    return metadata_path


@pytest.fixture
def complex_binary_image():
    """Create a binary image with a main shape and internal scars."""
    image = np.zeros(TEST_IMAGE_SIZE, dtype=np.uint8)

    main_contour = np.array([
        [50, 100], [80, 80], [120, 90], [140, 120],
        [130, 180], [100, 200], [70, 190], [40, 150],
    ], dtype=np.int32)
    cv2.fillPoly(image, [main_contour], 255)

    cv2.circle(image, (90, 130), 15, 0, -1)
    cv2.ellipse(image, (110, 160), (8, 12), 45, 0, 360, 0, -1)

    return image


@pytest.fixture
def arrow_shaped_contour():
    """Create a contour shaped like a right-pointing arrow."""
    arrow_points = np.array([
        [10, 20], [25, 10], [20, 15], [35, 15],
        [35, 25], [20, 25], [25, 30],
    ], dtype=np.int32)
    return arrow_points.reshape(-1, 1, 2)


@pytest.fixture
def sample_contours():
    """Provide a parent and child contour pair for testing."""
    parent = np.array([
        [20, 30], [50, 25], [80, 40], [85, 70],
        [75, 100], [45, 105], [15, 90], [10, 60],
    ], dtype=np.int32).reshape(-1, 1, 2)
    child = np.array([
        [40, 50], [55, 48], [58, 65], [45, 67],
    ], dtype=np.int32).reshape(-1, 1, 2)
    return [parent, child]


@pytest.fixture
def sample_hierarchy():
    """Provide OpenCV-style hierarchy for the sample_contours fixture."""
    # [next, previous, first_child, parent]
    return np.array([
        [-1, -1, 1, -1],
        [-1, -1, -1, 0],
    ])


@pytest.fixture
def test_image_with_dpi(test_data_dir):
    """Create a test image file with DPI metadata."""
    image_path = os.path.join(test_data_dir, "test_image_with_dpi.png")
    image_array = np.zeros((200, 300, 3), dtype=np.uint8)
    cv2.rectangle(image_array, (50, 50), (250, 150), (255, 255, 255), -1)

    pil_image = Image.fromarray(image_array)
    pil_image.save(image_path, dpi=(TEST_DPI, TEST_DPI))
    return image_path


@pytest.fixture
def test_image_directory(test_data_dir, test_image_with_dpi):
    """Create a test directory populated with DPI-tagged images."""
    images_dir = os.path.join(test_data_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    shutil.copy2(test_image_with_dpi, os.path.join(images_dir, "test_image_1.png"))

    for i in range(2, 4):
        image_path = os.path.join(images_dir, f"test_image_{i}.png")
        image_array = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
        Image.fromarray(image_array).save(image_path, dpi=(TEST_DPI, TEST_DPI))

    return images_dir
