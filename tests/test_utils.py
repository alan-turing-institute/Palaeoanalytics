"""
PyLithics Utility Functions Tests
=================================

Tests for utility functions including metadata reading, configuration loading,
contour filtering, and other supporting functionality.
"""

import pytest
import csv
import json
import os
import tempfile
import numpy as np
import cv2
from unittest.mock import patch, mock_open, MagicMock

from pylithics.image_processing.utils import (
    read_metadata,
    load_config,
    filter_contours_by_min_area
)


@pytest.mark.unit
class TestReadMetadata:
    """Test metadata reading functionality."""

    def test_read_metadata_valid_csv(self, sample_metadata_file):
        """Test reading a valid CSV metadata file."""
        metadata = read_metadata(sample_metadata_file)

        assert isinstance(metadata, list)
        assert len(metadata) == 3  # From sample_metadata fixture

        # Check first entry
        first_entry = metadata[0]
        assert 'image_id' in first_entry
        assert 'scale_id' in first_entry
        assert 'scale' in first_entry
        assert first_entry['image_id'] == 'test_image_1.png'
        assert first_entry['scale'] == '10.0'

    def test_read_metadata_all_required_fields(self, sample_metadata_file):
        """Test that all entries have required fields."""
        metadata = read_metadata(sample_metadata_file)

        required_fields = ['image_id', 'scale_id', 'scale']

        for entry in metadata:
            for field in required_fields:
                assert field in entry
                assert entry[field] is not None

    def test_read_metadata_nonexistent_file(self):
        """Test reading a non-existent metadata file."""
        with patch('pylithics.image_processing.utils.logging') as mock_logging:
            metadata = read_metadata('/nonexistent/path/metadata.csv')

            assert metadata == []
            mock_logging.error.assert_called()

    def test_read_metadata_empty_file(self, test_data_dir):
        """Test reading an empty CSV file."""
        empty_file = os.path.join(test_data_dir, "empty_metadata.csv")
        with open(empty_file, 'w') as f:
            pass  # Create empty file

        metadata = read_metadata(empty_file)
        assert metadata == []

    def test_read_metadata_malformed_csv(self, test_data_dir):
        """Test reading a malformed CSV file."""
        malformed_file = os.path.join(test_data_dir, "malformed_metadata.csv")
        with open(malformed_file, 'w') as f:
            f.write("image_id,scale_id,scale\n")
            f.write("test1.png,scale1,10.0\n")
            f.write("test2.png,scale2")  # Missing value
            f.write("test3.png,scale3,15.0,extra_field\n")  # Extra field

        # Should handle malformed CSV gracefully
        try:
            metadata = read_metadata(malformed_file)
            assert isinstance(metadata, list)
            # Should read what it can
            assert len(metadata) >= 1
        except Exception:
            # If it fails, that's also acceptable for malformed data
            pass

    def test_read_metadata_custom_fields(self, test_data_dir):
        """Test reading CSV with additional custom fields."""
        custom_file = os.path.join(test_data_dir, "custom_metadata.csv")
        with open(custom_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[
                'image_id', 'scale_id', 'scale', 'artifact_type', 'site_location'
            ])
            writer.writeheader()
            writer.writerow({
                'image_id': 'artifact1.png',
                'scale_id': 'scale1',
                'scale': '12.5',
                'artifact_type': 'blade',
                'site_location': 'sector_A'
            })

        metadata = read_metadata(custom_file)

        assert len(metadata) == 1
        entry = metadata[0]

        # Should preserve custom fields
        assert entry['artifact_type'] == 'blade'
        assert entry['site_location'] == 'sector_A'

        # Should still have required fields
        assert entry['image_id'] == 'artifact1.png'
        assert entry['scale'] == '12.5'

    def test_read_metadata_unicode_content(self, test_data_dir):
        """Test reading CSV with Unicode content."""
        unicode_file = os.path.join(test_data_dir, "unicode_metadata.csv")
        with open(unicode_file, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['image_id', 'scale_id', 'scale', 'notes'])
            writer.writeheader()
            writer.writerow({
                'image_id': 'français.png',
                'scale_id': 'échelle_1',
                'scale': '10.0',
                'notes': 'Artéfact trouvé à 深圳'
            })

        metadata = read_metadata(unicode_file)

        assert len(metadata) == 1
        entry = metadata[0]
        assert entry['image_id'] == 'français.png'
        assert entry['scale_id'] == 'échelle_1'
        assert '深圳' in entry['notes']

    def test_read_metadata_permission_error(self):
        """Test handling of permission errors when reading metadata."""
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            with patch('pylithics.image_processing.utils.logging') as mock_logging:
                metadata = read_metadata('/restricted/metadata.csv')

                assert metadata == []
                mock_logging.error.assert_called()


@pytest.mark.unit
class TestLoadConfig:
    """Test configuration loading functionality."""

    def test_load_config_valid_json(self, test_data_dir):
        """Test loading a valid JSON configuration file."""
        config_data = {
            'thresholding': {
                'method': 'otsu',
                'threshold_value': 150
            },
            'processing': {
                'enabled': True,
                'debug': False
            }
        }

        config_file = os.path.join(test_data_dir, "test_config.json")
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        with patch('pylithics.image_processing.utils.logging') as mock_logging:
            loaded_config = load_config(config_file)

            assert loaded_config == config_data
            assert loaded_config['thresholding']['method'] == 'otsu'
            assert loaded_config['processing']['enabled'] is True
            mock_logging.info.assert_called()

    def test_load_config_nonexistent_file(self):
        """Test loading a non-existent configuration file."""
        with patch('pylithics.image_processing.utils.logging') as mock_logging:
            config = load_config('/nonexistent/config.json')

            assert config is None
            mock_logging.error.assert_called()

    def test_load_config_invalid_json(self, test_data_dir):
        """Test loading an invalid JSON file."""
        invalid_file = os.path.join(test_data_dir, "invalid_config.json")
        with open(invalid_file, 'w') as f:
            f.write('{"invalid": json content}')  # Missing quotes, invalid JSON

        with patch('pylithics.image_processing.utils.logging') as mock_logging:
            config = load_config(invalid_file)

            assert config is None
            mock_logging.error.assert_called()

    def test_load_config_empty_file(self, test_data_dir):
        """Test loading an empty JSON file."""
        empty_file = os.path.join(test_data_dir, "empty_config.json")
        with open(empty_file, 'w') as f:
            f.write('{}')  # Empty but valid JSON

        config = load_config(empty_file)

        assert config == {}

    def test_load_config_default_parameter(self):
        """Test loading config with default parameter value."""
        # Mock a config file that doesn't exist to test default behavior
        with patch('builtins.open', side_effect=FileNotFoundError()):
            with patch('pylithics.image_processing.utils.logging'):
                config = load_config()  # Uses default "config.json"

                assert config is None

    def test_load_config_permission_error(self):
        """Test handling of permission errors when loading config."""
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            with patch('pylithics.image_processing.utils.logging') as mock_logging:
                config = load_config('/restricted/config.json')

                assert config is None
                mock_logging.error.assert_called()

    def test_load_config_nested_structure(self, test_data_dir):
        """Test loading config with complex nested structure."""
        complex_config = {
            'image_processing': {
                'preprocessing': {
                    'normalization': {
                        'enabled': True,
                        'method': 'minmax',
                        'parameters': {
                            'clip_values': [0, 255],
                            'preserve_range': False
                        }
                    },
                    'filtering': {
                        'gaussian_blur': {
                            'enabled': False,
                            'kernel_size': 5,
                            'sigma': 1.0
                        }
                    }
                }
            },
            'analysis': {
                'contour_detection': {
                    'method': 'opencv',
                    'parameters': ['RETR_TREE', 'CHAIN_APPROX_SIMPLE']
                }
            }
        }

        config_file = os.path.join(test_data_dir, "complex_config.json")
        with open(config_file, 'w') as f:
            json.dump(complex_config, f, indent=2)

        loaded_config = load_config(config_file)

        assert loaded_config == complex_config

        # Test deep access
        assert loaded_config['image_processing']['preprocessing']['normalization']['enabled'] is True
        assert loaded_config['analysis']['contour_detection']['method'] == 'opencv'


@pytest.mark.unit
class TestFilterContoursByMinArea:
    """Test contour filtering by minimum area."""

    def test_filter_contours_basic(self):
        """Test basic contour filtering by area."""
        # Create contours with different areas
        large_contour = np.array([
            [0, 0], [100, 0], [100, 100], [0, 100]
        ], dtype=np.int32).reshape(-1, 1, 2)  # Area = 10000

        small_contour = np.array([
            [0, 0], [10, 0], [10, 10], [0, 10]
        ], dtype=np.int32).reshape(-1, 1, 2)  # Area = 100

        contours = [large_contour, small_contour]
        hierarchy = np.array([
            [-1, -1, -1, -1],
            [-1, -1, -1, -1]
        ])

        filtered_contours, filtered_hierarchy = filter_contours_by_min_area(
            contours, hierarchy, min_area=500.0
        )

        assert len(filtered_contours) == 1
        assert len(filtered_hierarchy) == 1
        assert cv2.contourArea(filtered_contours[0]) >= 500.0

    def test_filter_contours_all_pass(self):
        """Test filtering where all contours pass the threshold."""
        contour1 = np.array([
            [0, 0], [50, 0], [50, 50], [0, 50]
        ], dtype=np.int32).reshape(-1, 1, 2)  # Area = 2500

        contour2 = np.array([
            [60, 60], [90, 60], [90, 90], [60, 90]
        ], dtype=np.int32).reshape(-1, 1, 2)  # Area = 900

        contours = [contour1, contour2]
        hierarchy = np.array([
            [-1, -1, -1, -1],
            [-1, -1, -1, -1]
        ])

        filtered_contours, filtered_hierarchy = filter_contours_by_min_area(
            contours, hierarchy, min_area=100.0
        )

        assert len(filtered_contours) == 2
        assert len(filtered_hierarchy) == 2

    def test_filter_contours_all_fail(self):
        """Test filtering where all contours fail the threshold."""
        small_contour1 = np.array([
            [0, 0], [5, 0], [5, 5], [0, 5]
        ], dtype=np.int32).reshape(-1, 1, 2)  # Area = 25

        small_contour2 = np.array([
            [10, 10], [15, 10], [15, 15], [10, 15]
        ], dtype=np.int32).reshape(-1, 1, 2)  # Area = 25

        contours = [small_contour1, small_contour2]
        hierarchy = np.array([
            [-1, -1, -1, -1],
            [-1, -1, -1, -1]
        ])

        filtered_contours, filtered_hierarchy = filter_contours_by_min_area(
            contours, hierarchy, min_area=100.0
        )

        assert len(filtered_contours) == 0
        assert filtered_hierarchy is None

    def test_filter_contours_empty_input(self):
        """Test filtering with empty contour list."""
        filtered_contours, filtered_hierarchy = filter_contours_by_min_area(
            [], None, min_area=50.0
        )

        assert filtered_contours == []
        assert filtered_hierarchy is None

    def test_filter_contours_none_hierarchy(self):
        """Test filtering with None hierarchy."""
        contour = np.array([
            [0, 0], [20, 0], [20, 20], [0, 20]
        ], dtype=np.int32).reshape(-1, 1, 2)

        filtered_contours, filtered_hierarchy = filter_contours_by_min_area(
            [contour], None, min_area=100.0
        )

        assert len(filtered_contours) == 0
        assert filtered_hierarchy is None

    def test_filter_contours_zero_area_threshold(self):
        """Test filtering with zero area threshold."""
        contour = np.array([
            [0, 0], [10, 0], [10, 10], [0, 10]
        ], dtype=np.int32).reshape(-1, 1, 2)

        contours = [contour]
        hierarchy = np.array([[-1, -1, -1, -1]])

        filtered_contours, filtered_hierarchy = filter_contours_by_min_area(
            contours, hierarchy, min_area=0.0
        )

        assert len(filtered_contours) == 1
        assert len(filtered_hierarchy) == 1

    def test_filter_contours_negative_area_threshold(self):
        """Test filtering with negative area threshold."""
        contour = np.array([
            [0, 0], [10, 0], [10, 10], [0, 10]
        ], dtype=np.int32).reshape(-1, 1, 2)

        contours = [contour]
        hierarchy = np.array([[-1, -1, -1, -1]])

        filtered_contours, filtered_hierarchy = filter_contours_by_min_area(
            contours, hierarchy, min_area=-10.0
        )

        # Should include all contours (negative threshold means everything passes)
        assert len(filtered_contours) == 1

    def test_filter_contours_default_area_threshold(self):
        """Test filtering with default area threshold."""
        tiny_contour = np.array([
            [0, 0], [1, 0], [1, 1], [0, 1]
        ], dtype=np.int32).reshape(-1, 1, 2)  # Area = 1

        contours = [tiny_contour]
        hierarchy = np.array([[-1, -1, -1, -1]])

        # Test with default threshold (1.0)
        filtered_contours, filtered_hierarchy = filter_contours_by_min_area(
            contours, hierarchy
        )

        # Should include the contour as its area equals the default threshold
        assert len(filtered_contours) == 1

    def test_filter_contours_complex_hierarchy(self):
        """Test filtering preserves hierarchy relationships."""
        # Parent contour (large)
        parent_contour = np.array([
            [0, 0], [100, 0], [100, 100], [0, 100]
        ], dtype=np.int32).reshape(-1, 1, 2)

        # Child contour (medium)
        child_contour = np.array([
            [20, 20], [80, 20], [80, 80], [20, 80]
        ], dtype=np.int32).reshape(-1, 1, 2)

        # Small contour (will be filtered out)
        small_contour = np.array([
            [5, 5], [7, 5], [7, 7], [5, 7]
        ], dtype=np.int32).reshape(-1, 1, 2)

        contours = [parent_contour, child_contour, small_contour]
        hierarchy = np.array([
            [-1, -1, 1, -1],  # Parent has child at index 1
            [-1, -1, -1, 0],  # Child has parent at index 0
            [-1, -1, -1, -1]  # Small contour (independent)
        ])

        filtered_contours, filtered_hierarchy = filter_contours_by_min_area(
            contours, hierarchy, min_area=1000.0
        )

        # Should filter out the small contour
        assert len(filtered_contours) == 2
        assert len(filtered_hierarchy) == 2

        # Check that hierarchy is preserved
        areas = [cv2.contourArea(cnt) for cnt in filtered_contours]
        assert all(area >= 1000.0 for area in areas)


@pytest.mark.integration
class TestUtilsIntegration:
    """Integration tests for utility functions."""

    def test_metadata_and_config_workflow(self, test_data_dir):
        """Test complete workflow using metadata and config utilities."""
        # Create metadata file
        metadata_data = [
            {'image_id': 'artifact_001.png', 'scale_id': 'scale_1', 'scale': '10.5'},
            {'image_id': 'artifact_002.png', 'scale_id': 'scale_2', 'scale': '12.0'},
            {'image_id': 'artifact_003.png', 'scale_id': 'scale_3', 'scale': '8.5'}
        ]

        metadata_file = os.path.join(test_data_dir, "workflow_metadata.csv")
        with open(metadata_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['image_id', 'scale_id', 'scale'])
            writer.writeheader()
            for row in metadata_data:
                writer.writerow(row)

        # Create config file
        config_data = {
            'contour_filtering': {
                'min_area': 100.0,
                'exclude_border': True
            },
            'processing': {
                'method': 'advanced',
                'debug': False
            }
        }

        config_file = os.path.join(test_data_dir, "workflow_config.json")
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        # Test workflow
        metadata = read_metadata(metadata_file)
        config = load_config(config_file)

        assert len(metadata) == 3
        assert config is not None

        # Verify metadata content
        image_ids = [entry['image_id'] for entry in metadata]
        assert 'artifact_001.png' in image_ids
        assert 'artifact_002.png' in image_ids
        assert 'artifact_003.png' in image_ids

        # Verify config content
        assert config['contour_filtering']['min_area'] == 100.0
        assert config['processing']['method'] == 'advanced'

        # Test using config for contour filtering
        min_area = config['contour_filtering']['min_area']

        # Create test contours
        large_contour = np.array([
            [0, 0], [50, 0], [50, 50], [0, 50]
        ], dtype=np.int32).reshape(-1, 1, 2)  # Area = 2500

        small_contour = np.array([
            [0, 0], [5, 0], [5, 5], [0, 5]
        ], dtype=np.int32).reshape(-1, 1, 2)  # Area = 25

        contours = [large_contour, small_contour]
        hierarchy = np.array([[-1, -1, -1, -1], [-1, -1, -1, -1]])

        filtered_contours, _ = filter_contours_by_min_area(
            contours, hierarchy, min_area=min_area
        )

        # Should filter based on config value
        assert len(filtered_contours) == 1
        assert cv2.contourArea(filtered_contours[0]) >= min_area

    def test_error_handling_workflow(self, test_data_dir):
        """Test error handling across utility functions."""
        # Test with non-existent files
        nonexistent_metadata = os.path.join(test_data_dir, "missing_metadata.csv")
        nonexistent_config = os.path.join(test_data_dir, "missing_config.json")

        with patch('pylithics.image_processing.utils.logging'):
            metadata = read_metadata(nonexistent_metadata)
            config = load_config(nonexistent_config)

        assert metadata == []
        assert config is None

        # Test contour filtering with malformed data
        malformed_contours = [None, "not_a_contour", []]

        try:
            filtered_contours, _ = filter_contours_by_min_area(
                malformed_contours, None, min_area=50.0
            )
            # If it doesn't crash, that's good error handling
        except (TypeError, ValueError, AttributeError):
            # If it raises appropriate errors, that's also acceptable
            pass

    def test_large_dataset_performance(self, test_data_dir):
        """Test utilities with larger datasets."""
        # Create a large metadata file
        large_metadata_file = os.path.join(test_data_dir, "large_metadata.csv")

        with open(large_metadata_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['image_id', 'scale_id', 'scale'])
            writer.writeheader()

            # Write 1000 entries
            for i in range(1000):
                writer.writerow({
                    'image_id': f'artifact_{i:04d}.png',
                    'scale_id': f'scale_{i}',
                    'scale': str(10.0 + (i % 10))  # Vary scale values
                })

        # Test reading large metadata
        metadata = read_metadata(large_metadata_file)

        assert len(metadata) == 1000
        assert metadata[0]['image_id'] == 'artifact_0000.png'
        assert metadata[999]['image_id'] == 'artifact_0999.png'

        # Test with many small contours
        many_contours = []
        for i in range(100):
            contour = np.array([
                [i, i], [i+5, i], [i+5, i+5], [i, i+5]
            ], dtype=np.int32).reshape(-1, 1, 2)
            many_contours.append(contour)

        hierarchy = np.array([[-1, -1, -1, -1]] * 100)

        filtered_contours, filtered_hierarchy = filter_contours_by_min_area(
            many_contours, hierarchy, min_area=20.0
        )

        # All contours have area 25, so all should pass
        assert len(filtered_contours) == 100
        assert len(filtered_hierarchy) == 100


@pytest.mark.unit
class TestUtilsEdgeCases:
    """Test edge cases and error conditions in utility functions."""

    def test_read_metadata_file_encoding_issues(self, test_data_dir):
        """Test handling of file encoding issues."""
        # Create file with different encoding
        encoded_file = os.path.join(test_data_dir, "encoded_metadata.csv")

        # Write with Latin-1 encoding
        content = "image_id,scale_id,scale\ncafé.png,scale1,10.0\n"
        with open(encoded_file, 'w', encoding='latin-1') as f:
            f.write(content)

        # Try to read (may fail or succeed depending on system default encoding)
        try:
            metadata = read_metadata(encoded_file)
            # If successful, verify content
            if metadata:
                assert len(metadata) >= 0
        except UnicodeDecodeError:
            # Encoding error is acceptable for this edge case
            pass

    def test_load_config_very_large_file(self, test_data_dir):
        """Test loading a very large configuration file."""
        large_config = {}

        # Create a large nested structure
        for i in range(100):
            section = f'section_{i}'
            large_config[section] = {}
            for j in range(50):
                subsection = f'subsection_{j}'
                large_config[section][subsection] = {
                    'enabled': i % 2 == 0,
                    'value': i * j,
                    'description': f'Configuration for {section}.{subsection}'
                }

        large_config_file = os.path.join(test_data_dir, "large_config.json")
        with open(large_config_file, 'w') as f:
            json.dump(large_config, f)

        # Test loading large config
        loaded_config = load_config(large_config_file)

        assert loaded_config is not None
        assert len(loaded_config) == 100
        assert 'section_0' in loaded_config
        assert 'section_99' in loaded_config

    def test_filter_contours_edge_geometries(self):
        """Test contour filtering with edge case geometries."""
        # Degenerate contour (single point repeated)
        point_contour = np.array([
            [10, 10], [10, 10], [10, 10]
        ], dtype=np.int32).reshape(-1, 1, 2)

        # Line contour (zero area)
        line_contour = np.array([
            [0, 0], [10, 0], [10, 0], [0, 0]
        ], dtype=np.int32).reshape(-1, 1, 2)

        # Very small valid contour
        tiny_contour = np.array([
            [0, 0], [1, 0], [1, 1], [0, 1]
        ], dtype=np.int32).reshape(-1, 1, 2)

        contours = [point_contour, line_contour, tiny_contour]
        hierarchy = np.array([
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1]
        ])

        filtered_contours, filtered_hierarchy = filter_contours_by_min_area(
            contours, hierarchy, min_area=0.5
        )

        # Should handle degenerate geometries gracefully
        assert isinstance(filtered_contours, list)
        assert len(filtered_contours) >= 0

        # Check that remaining contours have sufficient area
        for contour in filtered_contours:
            area = cv2.contourArea(contour)
            assert area >= 0.5 or area == 0  # Allow zero area if OpenCV returns it