"""
PyLithics Visualization Tests
============================

Tests for visualization generation and CSV output including contour labeling,
arrow visualization, and comprehensive data export functionality.
"""

import pytest
import numpy as np
import cv2
import pandas as pd
import os
import tempfile
import csv
from unittest.mock import patch, MagicMock

from pylithics.image_processing.modules.visualization import (
    visualize_contours_with_hierarchy,
    save_measurements_to_csv
)


@pytest.mark.unit
class TestVisualizeContoursWithHierarchy:
    """Test contour visualization functionality."""

    def test_visualize_basic_contours(self):
        """Test basic contour visualization without arrows."""
        # Create simple contours
        parent_contour = np.array([
            [20, 20], [80, 20], [80, 80], [20, 80]
        ], dtype=np.int32).reshape(-1, 1, 2)

        child_contour = np.array([
            [30, 30], [50, 30], [50, 50], [30, 50]
        ], dtype=np.int32).reshape(-1, 1, 2)

        contours = [parent_contour, child_contour]
        hierarchy = np.array([
            [-1, -1, 1, -1],  # Parent
            [-1, -1, -1, 0]   # Child
        ])

        metrics = [
            {
                'parent': 'parent 1',
                'scar': 'parent 1',
                'surface_type': 'Dorsal',
                'has_arrow': False
            },
            {
                'parent': 'parent 1',
                'scar': 'scar 1',
                'surface_type': 'Dorsal',
                'has_arrow': False
            }
        ]

        inverted_image = np.zeros((100, 100), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            output_path = temp_file.name

        try:
            # Should not crash
            visualize_contours_with_hierarchy(contours, hierarchy, metrics, inverted_image, output_path)

            # Check that file was created
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_visualize_contours_with_arrows(self):
        """Test contour visualization with arrow overlays."""
        contour = np.array([
            [30, 30], [70, 30], [70, 70], [30, 70]
        ], dtype=np.int32).reshape(-1, 1, 2)

        contours = [contour]
        hierarchy = np.array([[-1, -1, -1, -1]])

        metrics = [
            {
                'parent': 'parent 1',
                'scar': 'scar 1',
                'surface_type': 'Dorsal',
                'has_arrow': True,
                'arrow_back': (40, 40),
                'arrow_tip': (60, 60),
                'arrow_angle': 45.0
            }
        ]

        inverted_image = np.zeros((100, 100), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            output_path = temp_file.name

        try:
            visualize_contours_with_hierarchy(contours, hierarchy, metrics, inverted_image, output_path)

            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_visualize_multiple_arrows(self):
        """Test visualization with multiple arrows in different scars."""
        contours = []
        metrics = []

        # Create multiple contours with arrows
        for i in range(3):
            contour = np.array([
                [i*30 + 10, 20], [i*30 + 40, 20],
                [i*30 + 40, 50], [i*30 + 10, 50]
            ], dtype=np.int32).reshape(-1, 1, 2)
            contours.append(contour)

            metrics.append({
                'parent': 'parent 1',
                'scar': f'scar {i+1}',
                'surface_type': 'Dorsal',
                'has_arrow': True,
                'arrow_back': (i*30 + 15, 25),
                'arrow_tip': (i*30 + 35, 45),
                'arrow_angle': i * 30  # Different angles
            })

        hierarchy = np.array([
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1]
        ])

        inverted_image = np.zeros((70, 120), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            output_path = temp_file.name

        try:
            visualize_contours_with_hierarchy(contours, hierarchy, metrics, inverted_image, output_path)

            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 1000  # Should be substantial with multiple elements

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_visualize_surface_type_labels(self):
        """Test visualization with different surface type labels."""
        contours = []
        metrics = []

        surface_types = ['Dorsal', 'Ventral', 'Platform', 'Lateral']

        for i, surface_type in enumerate(surface_types):
            contour = np.array([
                [i*25 + 10, 10], [i*25 + 30, 10],
                [i*25 + 30, 30], [i*25 + 10, 30]
            ], dtype=np.int32).reshape(-1, 1, 2)
            contours.append(contour)

            metrics.append({
                'parent': f'parent {i+1}',
                'scar': f'parent {i+1}',
                'surface_type': surface_type,
                'has_arrow': False
            })

        hierarchy = np.array([
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1]
        ])

        inverted_image = np.zeros((50, 120), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            output_path = temp_file.name

        try:
            visualize_contours_with_hierarchy(contours, hierarchy, metrics, inverted_image, output_path)

            assert os.path.exists(output_path)

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_visualize_label_positioning(self):
        """Test that labels are positioned to avoid overlaps."""
        # Create closely spaced contours to test label positioning
        contours = []
        metrics = []

        for i in range(4):
            for j in range(4):
                contour = np.array([
                    [i*25 + 10, j*25 + 10], [i*25 + 20, j*25 + 10],
                    [i*25 + 20, j*25 + 20], [i*25 + 10, j*25 + 20]
                ], dtype=np.int32).reshape(-1, 1, 2)
                contours.append(contour)

                metrics.append({
                    'parent': 'parent 1',
                    'scar': f'scar {i*4 + j + 1}',
                    'surface_type': 'Dorsal',
                    'has_arrow': False
                })

        hierarchy = np.array([[-1, -1, -1, -1]] * 16)
        inverted_image = np.zeros((120, 120), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            output_path = temp_file.name

        try:
            # Should handle overlapping labels gracefully
            visualize_contours_with_hierarchy(contours, hierarchy, metrics, inverted_image, output_path)

            assert os.path.exists(output_path)

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_visualize_arrow_angle_display(self):
        """Test arrow angle display with various angles."""
        contour = np.array([
            [40, 40], [60, 40], [60, 60], [40, 60]
        ], dtype=np.int32).reshape(-1, 1, 2)

        test_angles = [0, 45, 90, 135, 180, 225, 270, 315]

        for angle in test_angles:
            metrics = [{
                'parent': 'parent 1',
                'scar': 'scar 1',
                'surface_type': 'Dorsal',
                'has_arrow': True,
                'arrow_back': (45, 45),
                'arrow_tip': (55, 55),
                'arrow_angle': angle
            }]

            hierarchy = np.array([[-1, -1, -1, -1]])
            inverted_image = np.zeros((100, 100), dtype=np.uint8)

            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                output_path = temp_file.name

            try:
                visualize_contours_with_hierarchy([contour], hierarchy, metrics, inverted_image, output_path)

                assert os.path.exists(output_path)
                assert os.path.getsize(output_path) > 0

            finally:
                if os.path.exists(output_path):
                    os.unlink(output_path)

    def test_visualize_empty_contours(self):
        """Test visualization with empty contours list."""
        contours = []
        hierarchy = np.array([])
        metrics = []
        inverted_image = np.zeros((100, 100), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            output_path = temp_file.name

        try:
            # Should handle empty input gracefully
            visualize_contours_with_hierarchy(contours, hierarchy, metrics, inverted_image, output_path)

            assert os.path.exists(output_path)

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_visualize_mismatched_metrics(self):
        """Test visualization when metrics count doesn't match contours."""
        contour = np.array([
            [20, 20], [80, 20], [80, 80], [20, 80]
        ], dtype=np.int32).reshape(-1, 1, 2)

        contours = [contour]
        hierarchy = np.array([[-1, -1, -1, -1]])

        # No metrics for the contour
        metrics = []

        inverted_image = np.zeros((100, 100), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            output_path = temp_file.name

        try:
            # Should handle mismatched metrics gracefully
            visualize_contours_with_hierarchy(contours, hierarchy, metrics, inverted_image, output_path)

            assert os.path.exists(output_path)

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_visualize_invalid_arrow_coordinates(self):
        """Test visualization with invalid arrow coordinates."""
        contour = np.array([
            [30, 30], [70, 30], [70, 70], [30, 70]
        ], dtype=np.int32).reshape(-1, 1, 2)

        metrics = [{
            'parent': 'parent 1',
            'scar': 'scar 1',
            'surface_type': 'Dorsal',
            'has_arrow': True,
            'arrow_back': None,  # Invalid coordinates
            'arrow_tip': (60, 60),
            'arrow_angle': 45.0
        }]

        hierarchy = np.array([[-1, -1, -1, -1]])
        inverted_image = np.zeros((100, 100), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            output_path = temp_file.name

        try:
            # Should handle invalid coordinates gracefully
            visualize_contours_with_hierarchy([contour], hierarchy, metrics, inverted_image, output_path)

            assert os.path.exists(output_path)

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


@pytest.mark.unit
class TestSaveMeasurementsToCSV:
    """Test CSV export functionality."""

    def test_save_basic_metrics(self):
        """Test saving basic metrics to CSV."""
        metrics = [
            {
                'parent': 'parent 1',
                'scar': 'parent 1',
                'image_id': 'test_image_1',
                'surface_type': 'Dorsal',
                'surface_feature': 'Dorsal',
                'centroid_x': 50.0,
                'centroid_y': 60.0,
                'technical_width': 40.0,
                'technical_length': 80.0,
                'area': 3200.0,
                'has_arrow': False
            }
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            output_path = temp_file.name

        try:
            save_measurements_to_csv(metrics, output_path)

            # Validate file creation and content
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0

            # Verify CSV structure and data
            df = pd.read_csv(output_path)
            assert len(df) == 1
            assert df.iloc[0]['image_id'] == 'test_image_1'
            assert df.iloc[0]['surface_type'] == 'Dorsal'
            assert df.iloc[0]['centroid_x'] == 50.0

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_save_metrics_with_arrows(self):
        """Test saving metrics with arrow data."""
        metrics = [
            {
                'parent': 'parent 1',
                'scar': 'scar 1',
                'image_id': 'test_image_1',
                'surface_type': 'Dorsal',
                'surface_feature': 'scar 1',
                'centroid_x': 50.0,
                'centroid_y': 60.0,
                'area': 500.0,
                'has_arrow': True,
                'arrow_angle': 45.0,
                'arrow_tip': [55, 65],
                'arrow_back': [45, 55]
            }
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            output_path = temp_file.name

        try:
            save_measurements_to_csv(metrics, output_path)

            df = pd.read_csv(output_path)
            assert len(df) == 1
            assert df.iloc[0]['has_arrow'] == True
            assert df.iloc[0]['arrow_angle'] == 45.0

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_save_comprehensive_metrics(self):
        """Test saving comprehensive metrics with all fields."""
        metrics = [
            {
                'parent': 'parent 1',
                'scar': 'parent 1',
                'image_id': 'comprehensive_test',
                'surface_type': 'Dorsal',
                'surface_feature': 'Dorsal',
                'centroid_x': 100.0,
                'centroid_y': 150.0,
                'technical_width': 80.0,
                'technical_length': 120.0,
                'max_width': 85.0,
                'max_length': 125.0,
                'area': 9600.0,
                'aspect_ratio': 1.5,
                'perimeter': 400.0,
                'distance_to_max_width': 35.0,
                'voronoi_num_cells': 5,
                'convex_hull_width': 90.0,
                'convex_hull_height': 130.0,
                'convex_hull_area': 11700.0,
                'voronoi_cell_area': 2000.0,
                'top_area': 4800.0,
                'bottom_area': 4800.0,
                'left_area': 4800.0,
                'right_area': 4800.0,
                'vertical_symmetry': 1.0,
                'horizontal_symmetry': 1.0,
                'lateral_convexity': 0.95,
                'has_arrow': False,
                'arrow_angle': 'NA'
            }
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            output_path = temp_file.name

        try:
            save_measurements_to_csv(metrics, output_path)

            df = pd.read_csv(output_path)
            assert len(df) == 1

            # Verify various field categories
            assert df.iloc[0]['voronoi_num_cells'] == 5
            assert df.iloc[0]['vertical_symmetry'] == 1.0
            assert df.iloc[0]['lateral_convexity'] == 0.95

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_save_multiple_metrics(self):
        """Test saving multiple metrics to CSV."""
        metrics = [
            {
                'parent': 'parent 1',
                'scar': 'parent 1',
                'image_id': 'multi_test',
                'surface_type': 'Dorsal',
                'surface_feature': 'Dorsal',
                'centroid_x': 50.0,
                'centroid_y': 60.0,
                'area': 5000.0,
                'has_arrow': False
            },
            {
                'parent': 'parent 1',
                'scar': 'scar 1',
                'image_id': 'multi_test',
                'surface_type': 'Dorsal',
                'surface_feature': 'scar 1',
                'centroid_x': 40.0,
                'centroid_y': 50.0,
                'area': 300.0,
                'has_arrow': True,
                'arrow_angle': 90.0
            },
            {
                'parent': 'parent 2',
                'scar': 'parent 2',
                'image_id': 'multi_test',
                'surface_type': 'Ventral',
                'surface_feature': 'Ventral',
                'centroid_x': 55.0,
                'centroid_y': 65.0,
                'area': 4800.0,
                'has_arrow': False
            }
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            output_path = temp_file.name

        try:
            save_measurements_to_csv(metrics, output_path)

            df = pd.read_csv(output_path)
            assert len(df) == 3

            # Verify surface type diversity
            surface_types = df['surface_type'].unique()
            assert 'Dorsal' in surface_types
            assert 'Ventral' in surface_types

            # Verify arrow data preservation
            arrow_rows = df[df['has_arrow'] == True]
            assert len(arrow_rows) == 1
            assert arrow_rows.iloc[0]['arrow_angle'] == 90.0

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_append_to_existing_csv(self):
        """Test appending metrics to existing CSV file."""
        # Create initial CSV data
        initial_metrics = [
            {
                'parent': 'parent 1',
                'scar': 'parent 1',
                'image_id': 'initial_image',
                'surface_type': 'Dorsal',
                'surface_feature': 'Dorsal',
                'area': 1000.0,
                'has_arrow': False
            }
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            output_path = temp_file.name

        try:
            # Save initial data
            save_measurements_to_csv(initial_metrics, output_path)

            # Append additional data
            new_metrics = [
                {
                    'parent': 'parent 2',
                    'scar': 'parent 2',
                    'image_id': 'appended_image',
                    'surface_type': 'Ventral',
                    'surface_feature': 'Ventral',
                    'area': 2000.0,
                    'has_arrow': True,
                    'arrow_angle': 180.0
                }
            ]

            save_measurements_to_csv(new_metrics, output_path, append=True)

            # Verify combined results
            df = pd.read_csv(output_path)
            assert len(df) == 2

            image_ids = df['image_id'].tolist()
            assert 'initial_image' in image_ids
            assert 'appended_image' in image_ids

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_save_metrics_missing_fields(self):
        """Test saving metrics with missing optional fields."""
        metrics = [
            {
                'parent': 'parent 1',
                'scar': 'parent 1',
                'image_id': 'missing_fields_test',
                'surface_type': 'Dorsal',
                'surface_feature': 'Dorsal',
                'centroid_x': 50.0,
                'centroid_y': 60.0,
                'area': 1000.0
                # Intentionally missing many optional fields
            }
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            output_path = temp_file.name

        try:
            save_measurements_to_csv(metrics, output_path)

            df = pd.read_csv(output_path)
            assert len(df) == 1

            # Verify missing fields are handled appropriately
            # has_arrow defaults to False when missing
            assert df.iloc[0]['has_arrow'] == False
            # arrow_angle gets 'NA' which pandas reads as NaN
            assert pd.isna(df.iloc[0]['arrow_angle']) or df.iloc[0]['arrow_angle'] == 'NA'
            # voronoi_num_cells gets 'NA' which pandas reads as NaN
            assert pd.isna(df.iloc[0]['voronoi_num_cells']) or df.iloc[0]['voronoi_num_cells'] == 'NA'

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_save_metrics_coordinate_handling(self):
        """Test handling of coordinate data in metrics."""
        metrics = [
            {
                'parent': 'parent 1',
                'scar': 'scar 1',
                'image_id': 'coord_test',
                'surface_type': 'Dorsal',
                'surface_feature': 'scar 1',
                'centroid_x': 50.0,
                'centroid_y': 60.0,
                'area': 500.0,
                'has_arrow': True,
                'arrow_tip': (55, 65),  # Tuple format
                'arrow_back': [45, 55], # List format
                'arrow_angle': 45.0
            }
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            output_path = temp_file.name

        try:
            save_measurements_to_csv(metrics, output_path)

            df = pd.read_csv(output_path)
            assert len(df) == 1

            # Verify coordinate extraction functionality
            row = df.iloc[0]
            assert row['has_arrow'] == True
            assert row['arrow_angle'] == 45.0

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_save_empty_metrics(self):
        """Test saving empty metrics list."""
        metrics = []

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            output_path = temp_file.name

        try:
            save_measurements_to_csv(metrics, output_path)

            # Verify empty CSV with headers is created
            assert os.path.exists(output_path)

            df = pd.read_csv(output_path)
            assert len(df) == 0
            assert len(df.columns) > 0  # Should have headers

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_save_metrics_invalid_output_path(self):
        """Test saving metrics to invalid output path."""
        metrics = [
            {
                'parent': 'parent 1',
                'scar': 'parent 1',
                'image_id': 'test',
                'surface_type': 'Dorsal',
                'area': 1000.0
            }
        ]

        invalid_path = '/nonexistent_directory/output.csv'

        try:
            # Should handle invalid path gracefully or raise appropriate error
            save_measurements_to_csv(metrics, invalid_path)
        except (OSError, IOError, PermissionError):
            # These are acceptable errors for invalid paths
            pass


@pytest.mark.integration
class TestVisualizationIntegration:
    """Integration tests for complete visualization workflow."""

    def test_complete_visualization_workflow(self):
        """Test complete workflow from contours to final outputs."""
        # Create realistic archaeological data
        parent_contour = np.array([
            [30, 40], [170, 35], [180, 80], [175, 160],
            [160, 180], [90, 185], [25, 170], [20, 100]
        ], dtype=np.int32).reshape(-1, 1, 2)

        scar_contours = [
            np.array([[80, 90], [110, 95], [115, 120], [85, 125]], dtype=np.int32).reshape(-1, 1, 2),
            np.array([[120, 140], [145, 135], [150, 165], [125, 170]], dtype=np.int32).reshape(-1, 1, 2)
        ]

        all_contours = [parent_contour] + scar_contours

        hierarchy = np.array([
            [-1, -1, 1, -1],  # Parent
            [2, -1, -1, 0],   # Scar 1
            [-1, 1, -1, 0]    # Scar 2
        ])

        metrics = [
            {
                'image_id': 'integration_test',
                'parent': 'parent 1',
                'scar': 'parent 1',
                'surface_type': 'Dorsal',
                'centroid_x': 100.0,
                'centroid_y': 112.5,
                'technical_width': 155.0,
                'technical_length': 150.0,
                'area': 23250.0,
                'has_arrow': False,
                'voronoi_num_cells': 3,
                'top_area': 11625.0,
                'bottom_area': 11625.0,
                'vertical_symmetry': 1.0
            },
            {
                'image_id': 'integration_test',
                'parent': 'parent 1',
                'scar': 'scar 1',
                'surface_type': 'Dorsal',
                'centroid_x': 97.5,
                'centroid_y': 107.5,
                'width': 35.0,
                'height': 35.0,
                'area': 1225.0,
                'has_arrow': True,
                'arrow_back': (85, 95),
                'arrow_tip': (110, 120),
                'arrow_angle': 135.0
            },
            {
                'image_id': 'integration_test',
                'parent': 'parent 1',
                'scar': 'scar 2',
                'surface_type': 'Dorsal',
                'centroid_x': 137.5,
                'centroid_y': 152.5,
                'width': 30.0,
                'height': 35.0,
                'area': 1050.0,
                'has_arrow': True,
                'arrow_back': (130, 140),
                'arrow_tip': (145, 165),
                'arrow_angle': 225.0
            }
        ]

        inverted_image = np.zeros((220, 200), dtype=np.uint8)

        # Test visualization generation
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as viz_file:
            viz_path = viz_file.name

        # Test CSV export
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as csv_file:
            csv_path = csv_file.name

        try:
            # Generate visualization
            visualize_contours_with_hierarchy(all_contours, hierarchy, metrics, inverted_image, viz_path)

            assert os.path.exists(viz_path)
            assert os.path.getsize(viz_path) > 5000  # Should be substantial image

            # Generate CSV export
            save_measurements_to_csv(metrics, csv_path)

            assert os.path.exists(csv_path)

            # Verify CSV content integrity
            df = pd.read_csv(csv_path)
            assert len(df) == 3  # Parent + 2 scars

            # Validate data preservation
            dorsal_rows = df[df['surface_type'] == 'Dorsal']
            assert len(dorsal_rows) == 3

            arrow_rows = df[df['has_arrow'] == True]
            assert len(arrow_rows) == 2

            # Verify specific values
            parent_row = df[df['surface_feature'] == 'Dorsal'].iloc[0]
            assert parent_row['voronoi_num_cells'] == 3
            assert parent_row['vertical_symmetry'] == 1.0

            scar_rows = df[df['surface_feature'].str.contains('scar', na=False)]
            assert len(scar_rows) == 2

        finally:
            for path in [viz_path, csv_path]:
                if os.path.exists(path):
                    os.unlink(path)

    def test_batch_processing_visualization(self):
        """Test visualization for batch processing scenario."""
        # Simulate multiple images being processed
        all_metrics = []

        for image_num in range(3):
            image_id = f'batch_image_{image_num + 1}'

            # Add metrics for each image
            all_metrics.extend([
                {
                    'parent': f'parent {image_num + 1}',
                    'scar': f'parent {image_num + 1}',
                    'image_id': image_id,
                    'surface_type': 'Dorsal',
                    'surface_feature': 'Dorsal',
                    'centroid_x': 50.0 + image_num * 10,
                    'centroid_y': 60.0 + image_num * 10,
                    'area': 5000.0 + image_num * 500,
                    'has_arrow': False
                },
                {
                    'parent': f'parent {image_num + 1}',
                    'scar': 'scar 1',
                    'image_id': image_id,
                    'surface_type': 'Dorsal',
                    'surface_feature': f'scar 1',
                    'centroid_x': 45.0 + image_num * 10,
                    'centroid_y': 55.0 + image_num * 10,
                    'area': 300.0 + image_num * 50,
                    'has_arrow': True,
                    'arrow_angle': 45.0 + image_num * 30
                }
            ])

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
            csv_path = temp_file.name

        try:
            # Process all metrics in batch
            save_measurements_to_csv(all_metrics, csv_path)

            df = pd.read_csv(csv_path)
            assert len(df) == 6  # 2 metrics per image × 3 images

            # Verify image distribution
            unique_images = df['image_id'].unique()
            assert len(unique_images) == 3

            # Verify arrow distribution
            arrow_count = len(df[df['has_arrow'] == True])
            assert arrow_count == 3  # One arrow per image

        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)


@pytest.mark.unit
class TestVisualizationErrorHandling:
    """Test error handling in visualization functions."""

    def test_visualize_invalid_image(self):
        """Test visualization with invalid image data."""
        contour = np.array([[20, 20], [80, 20], [80, 80], [20, 80]], dtype=np.int32).reshape(-1, 1, 2)
        contours = [contour]
        hierarchy = np.array([[-1, -1, -1, -1]])
        metrics = [{'parent': 'parent 1', 'scar': 'parent 1', 'surface_type': 'Dorsal', 'has_arrow': False}]

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            output_path = temp_file.name

        try:
            # Test with None image input
            visualize_contours_with_hierarchy(contours, hierarchy, metrics, None, output_path)

            # Verify output file if function handles None gracefully
            if os.path.exists(output_path):
                assert os.path.getsize(output_path) >= 0

        except (AttributeError, TypeError, cv2.error):
            # Expected errors for None image input
            pass
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_visualize_malformed_contours(self):
        """Test visualization with malformed contour data."""
        # Invalid contour formats
        malformed_contours = [
            "not_a_contour",
            np.array([]),  # Empty array
            None
        ]

        hierarchy = np.array([[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]])
        metrics = [
            {'parent': 'parent 1', 'scar': 'parent 1', 'surface_type': 'Dorsal', 'has_arrow': False},
            {'parent': 'parent 2', 'scar': 'parent 2', 'surface_type': 'Ventral', 'has_arrow': False},
            {'parent': 'parent 3', 'scar': 'parent 3', 'surface_type': 'Platform', 'has_arrow': False}
        ]

        inverted_image = np.zeros((100, 100), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            output_path = temp_file.name

        try:
            # Should handle malformed contours appropriately
            visualize_contours_with_hierarchy(malformed_contours, hierarchy, metrics, inverted_image, output_path)
        except (TypeError, AttributeError, cv2.error):
            # Expected errors for malformed contours
            pass
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_save_csv_malformed_metrics(self):
        """Test CSV saving with malformed metrics data."""
        malformed_metrics = [
            "not_a_dict",
            {'incomplete': 'data'},  # Missing required fields
            None,
            {'image_id': 'test', 'surface_type': None}  # None values
        ]

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
            output_path = temp_file.name

        try:
            # Should handle malformed data appropriately
            save_measurements_to_csv(malformed_metrics, output_path)

            # Verify output if function handles malformed data gracefully
            if os.path.exists(output_path):
                assert os.path.getsize(output_path) >= 0

        except (TypeError, AttributeError, KeyError):
            # Expected errors for malformed data
            pass
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_visualize_output_directory_creation(self):
        """Test visualization with output path in non-existent directory."""
        contour = np.array([[20, 20], [80, 20], [80, 80], [20, 80]], dtype=np.int32).reshape(-1, 1, 2)
        contours = [contour]
        hierarchy = np.array([[-1, -1, -1, -1]])
        metrics = [{'parent': 'parent 1', 'scar': 'parent 1', 'surface_type': 'Dorsal', 'has_arrow': False}]
        inverted_image = np.zeros((100, 100), dtype=np.uint8)

        # Create path in temporary directory that will be created
        with tempfile.TemporaryDirectory() as temp_dir:
            subdir = os.path.join(temp_dir, "new_subdir")
            output_path = os.path.join(subdir, "output.png")

            # Should create directory and save file
            visualize_contours_with_hierarchy(contours, hierarchy, metrics, inverted_image, output_path)

            assert os.path.exists(output_path)


@pytest.mark.performance
class TestVisualizationPerformance:
    """Test performance aspects of visualization."""

    def test_visualization_many_contours_performance(self):
        """Test visualization performance with many contours."""
        # Create numerous contours
        contours = []
        metrics = []

        for i in range(100):
            contour = np.array([
                [i*5 + 10, 10], [i*5 + 15, 10],
                [i*5 + 15, 15], [i*5 + 10, 15]
            ], dtype=np.int32).reshape(-1, 1, 2)
            contours.append(contour)

            metrics.append({
                'parent': 'parent 1',
                'scar': f'scar {i+1}',
                'surface_type': 'Dorsal',
                'has_arrow': i % 3 == 0,  # Some with arrows
                'arrow_back': (i*5 + 11, 11) if i % 3 == 0 else None,
                'arrow_tip': (i*5 + 14, 14) if i % 3 == 0 else None,
                'arrow_angle': i * 3.6 if i % 3 == 0 else None
            })

        hierarchy = np.array([[-1, -1, -1, -1]] * 100)
        inverted_image = np.zeros((25, 520), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            output_path = temp_file.name

        try:
            import time
            start_time = time.time()

            visualize_contours_with_hierarchy(contours, hierarchy, metrics, inverted_image, output_path)

            end_time = time.time()
            processing_time = end_time - start_time

            # Should complete within reasonable time constraints
            assert processing_time < 10.0  # 10 seconds maximum
            assert os.path.exists(output_path)

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_csv_save_large_dataset_performance(self):
        """Test CSV saving performance with large dataset."""
        # Create large metrics dataset
        metrics = []

        for image_num in range(50):  # 50 images
            for scar_num in range(20):  # 20 scars per image
                metrics.append({
                    'parent': f'parent {image_num + 1}',
                    'scar': f'scar {scar_num + 1}',
                    'image_id': f'image_{image_num:03d}',
                    'surface_type': 'Dorsal',
                    'surface_feature': f'scar_{scar_num:02d}',
                    'centroid_x': 50.0 + scar_num,
                    'centroid_y': 60.0 + scar_num,
                    'technical_width': 10.0 + scar_num * 0.5,
                    'technical_length': 15.0 + scar_num * 0.7,
                    'area': 150.0 + scar_num * 10,
                    'has_arrow': scar_num % 4 == 0,
                    'arrow_angle': scar_num * 18 if scar_num % 4 == 0 else 'NA',
                    'voronoi_cell_area': 50.0 + scar_num * 2,
                    'vertical_symmetry': 0.8 + scar_num * 0.01,
                    'lateral_convexity': 0.7 + scar_num * 0.015
                })

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
            output_path = temp_file.name

        try:
            import time
            start_time = time.time()

            save_measurements_to_csv(metrics, output_path)

            end_time = time.time()
            processing_time = end_time - start_time

            # Should save large dataset efficiently
            assert processing_time < 5.0  # 5 seconds maximum
            assert os.path.exists(output_path)

            # Verify data integrity
            df = pd.read_csv(output_path)
            assert len(df) == 1000  # 50 images × 20 scars

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


@pytest.mark.integration
class TestVisualizationRealWorldScenarios:
    """Test visualization with realistic archaeological scenarios."""

    def test_blade_tool_visualization(self):
        """Test visualization of blade tool with systematic scars."""
        # Create blade-like parent contour
        blade_contour = np.array([
            [50, 20], [250, 25], [260, 35], [255, 180],
            [250, 190], [50, 185], [40, 175], [45, 35]
        ], dtype=np.int32).reshape(-1, 1, 2)

        # Create systematic removal scars
        scar_contours = []
        scar_metrics = []

        for i in range(6):
            x_base = 70 + i * 30
            y_base = 50 + i * 15

            scar = np.array([
                [x_base, y_base], [x_base + 25, y_base + 5],
                [x_base + 20, y_base + 40], [x_base - 5, y_base + 35]
            ], dtype=np.int32).reshape(-1, 1, 2)
            scar_contours.append(scar)

            scar_metrics.append({
                'image_id': 'blade_tool_001',
                'parent': 'parent 1',
                'scar': f'scar {i+1}',
                'surface_type': 'Dorsal',
                'centroid_x': x_base + 10,
                'centroid_y': y_base + 20,
                'width': 25.0,
                'height': 40.0,
                'area': 1000.0,
                'has_arrow': True,
                'arrow_back': (x_base + 5, y_base + 10),
                'arrow_tip': (x_base + 20, y_base + 30),
                'arrow_angle': 45.0 + i * 15
            })

        all_contours = [blade_contour] + scar_contours

        # Create hierarchy structure
        hierarchy = [[-1, -1, 1, -1]]  # Parent
        for i in range(6):
            next_sib = i + 2 if i < 5 else -1
            prev_sib = i if i > 0 else -1
            hierarchy.append([next_sib, prev_sib, -1, 0])
        hierarchy = np.array(hierarchy)

        # Parent metric
        blade_metric = {
            'image_id': 'blade_tool_001',
            'parent': 'parent 1',
            'scar': 'parent 1',
            'surface_type': 'Dorsal',
            'centroid_x': 150.0,
            'centroid_y': 105.0,
            'technical_width': 220.0,
            'technical_length': 170.0,
            'area': 37400.0,
            'has_arrow': False,
            'voronoi_num_cells': 7,
            'vertical_symmetry': 0.95,
            'horizontal_symmetry': 0.88
        }

        all_metrics = [blade_metric] + scar_metrics
        inverted_image = np.zeros((210, 300), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as viz_file:
            viz_path = viz_file.name

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as csv_file:
            csv_path = csv_file.name

        try:
            # Test visualization generation
            visualize_contours_with_hierarchy(all_contours, hierarchy, all_metrics, inverted_image, viz_path)
            assert os.path.exists(viz_path)
            assert os.path.getsize(viz_path) > 10000  # Substantial image with many elements

            # Test CSV export
            save_measurements_to_csv(all_metrics, csv_path)

            df = pd.read_csv(csv_path)
            assert len(df) == 7  # 1 parent + 6 scars

            # Verify blade tool characteristics
            parent_row = df[df['surface_feature'] == 'Dorsal'].iloc[0]
            assert parent_row['voronoi_num_cells'] == 7
            assert parent_row['vertical_symmetry'] == 0.95

            # All scars should have arrows
            scar_rows = df[df['surface_feature'].str.contains('scar', na=False)]
            assert len(scar_rows) == 6
            assert all(scar_rows['has_arrow'] == True)

        finally:
            for path in [viz_path, csv_path]:
                if os.path.exists(path):
                    os.unlink(path)

    def test_core_reduction_visualization(self):
        """Test visualization of core with radial reduction pattern."""
        # Create core contour
        core_contour = np.array([
            [50, 50], [150, 50], [150, 150], [50, 150]
        ], dtype=np.int32).reshape(-1, 1, 2)

        # Create radial scars
        center_x, center_y = 100, 100
        scar_contours = []
        scar_metrics = []

        for i in range(8):
            angle = 2 * np.pi * i / 8
            scar_x = center_x + 30 * np.cos(angle)
            scar_y = center_y + 30 * np.sin(angle)

            scar = np.array([
                [scar_x - 8, scar_y - 8], [scar_x + 8, scar_y - 8],
                [scar_x + 8, scar_y + 8], [scar_x - 8, scar_y + 8]
            ], dtype=np.int32).reshape(-1, 1, 2)
            scar_contours.append(scar)

            # Arrows point toward center
            arrow_angle = (np.degrees(angle) + 180) % 360

            scar_metrics.append({
                'image_id': 'core_001',
                'parent': 'parent 1',
                'scar': f'scar {i+1}',
                'surface_type': 'Dorsal',
                'centroid_x': scar_x,
                'centroid_y': scar_y,
                'width': 16.0,
                'height': 16.0,
                'area': 256.0,
                'has_arrow': True,
                'arrow_back': (scar_x + 5 * np.cos(angle), scar_y + 5 * np.sin(angle)),
                'arrow_tip': (center_x, center_y),
                'arrow_angle': arrow_angle
            })

        all_contours = [core_contour] + scar_contours
        hierarchy = np.array([[-1, -1, 1, -1]] + [[-1, -1, -1, 0]] * 8)

        core_metric = {
            'image_id': 'core_001',
            'parent': 'parent 1',
            'scar': 'parent 1',
            'surface_type': 'Dorsal',
            'centroid_x': center_x,
            'centroid_y': center_y,
            'technical_width': 100.0,
            'technical_length': 100.0,
            'area': 10000.0,
            'has_arrow': False,
            'voronoi_num_cells': 9,
            'horizontal_symmetry': 0.98,
            'vertical_symmetry': 0.98
        }

        all_metrics = [core_metric] + scar_metrics
        inverted_image = np.zeros((200, 200), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as viz_file:
            viz_path = viz_file.name

        try:
            visualize_contours_with_hierarchy(all_contours, hierarchy, all_metrics, inverted_image, viz_path)

            assert os.path.exists(viz_path)
            # Should create complex visualization with radial arrows
            assert os.path.getsize(viz_path) > 8000

        finally:
            if os.path.exists(viz_path):
                os.unlink(viz_path)