"""
PyLithics Voronoi Analysis Tests
================================

Tests for Voronoi diagram generation and spatial analysis including convex hull calculations,
cell area computations, and visualization generation for archaeological spatial analysis.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
from shapely.geometry import Point, Polygon, MultiPoint

from pylithics.image_processing.modules.voronoi_analysis import (
    calculate_voronoi_points,
    visualize_voronoi_diagram
)


@pytest.mark.unit
class TestCalculateVoronoiPoints:
    """Test the main Voronoi calculation function."""

    def test_calculate_voronoi_no_dorsal_surface(self):
        """Test Voronoi calculation when no dorsal surface is found."""
        metrics = [
            {
                'surface_type': 'Ventral',
                'parent': 'parent 1',
                'scar': 'parent 1',
                'centroid_x': 50.0,
                'centroid_y': 60.0
            }
        ]

        inverted_image = np.zeros((100, 100), dtype=np.uint8)

        result = calculate_voronoi_points(metrics, inverted_image)

        assert result is None

    def test_calculate_voronoi_dorsal_only(self):
        """Test Voronoi calculation with only dorsal surface (no scars)."""
        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'parent 1',
                'centroid_x': 50.0,
                'centroid_y': 60.0,
                'contour': [[[30, 40]], [[70, 40]], [[70, 80]], [[30, 80]]]  # Rectangle
            }
        ]

        inverted_image = np.zeros((100, 100), dtype=np.uint8)

        result = calculate_voronoi_points(metrics, inverted_image, padding_factor=0.02)

        assert result is not None
        assert 'voronoi_diagram' in result
        assert 'voronoi_cells' in result
        assert 'voronoi_metrics' in result
        assert 'convex_hull' in result
        assert 'convex_hull_metrics' in result
        assert 'points' in result
        assert 'bounding_box' in result
        assert 'dorsal_contour' in result

        # Should have one cell for the dorsal surface
        assert result['voronoi_metrics']['num_cells'] == 1
        assert len(result['voronoi_cells']) == 1

    def test_calculate_voronoi_dorsal_with_scars(self):
        """Test Voronoi calculation with dorsal surface and scars."""
        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'parent 1',
                'centroid_x': 50.0,
                'centroid_y': 60.0,
                'contour': [[[20, 30]], [[80, 30]], [[80, 90]], [[20, 90]]]
            },
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'scar 1',
                'centroid_x': 40.0,
                'centroid_y': 50.0
            },
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'scar 2',
                'centroid_x': 60.0,
                'centroid_y': 70.0
            }
        ]

        inverted_image = np.zeros((120, 120), dtype=np.uint8)

        result = calculate_voronoi_points(metrics, inverted_image)

        assert result is not None

        # Should have three cells (dorsal + 2 scars)
        assert result['voronoi_metrics']['num_cells'] == 3
        assert len(result['voronoi_cells']) == 3

        # Check that each cell has required properties
        for cell in result['voronoi_cells']:
            assert 'polygon' in cell
            assert 'area' in cell
            assert 'shared_edges' in cell
            assert 'metric_index' in cell
            assert cell['area'] >= 0

    def test_calculate_voronoi_convex_hull_metrics(self):
        """Test convex hull metric calculations."""
        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'parent 1',
                'centroid_x': 25.0,
                'centroid_y': 25.0,
                'contour': [[[10, 10]], [[40, 10]], [[40, 40]], [[10, 40]]]
            },
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'scar 1',
                'centroid_x': 35.0,
                'centroid_y': 15.0
            }
        ]

        inverted_image = np.zeros((50, 50), dtype=np.uint8)

        result = calculate_voronoi_points(metrics, inverted_image)

        assert result is not None

        # Check convex hull metrics
        ch_metrics = result['convex_hull_metrics']
        assert 'width' in ch_metrics
        assert 'height' in ch_metrics
        assert 'area' in ch_metrics

        assert ch_metrics['width'] > 0
        assert ch_metrics['height'] > 0
        assert ch_metrics['area'] > 0

        # For two points, convex hull should be a line (degenerate case)
        # Width should be the distance between points
        expected_width = abs(35.0 - 25.0)  # 10.0
        expected_height = abs(15.0 - 25.0)  # 10.0

        assert abs(ch_metrics['width'] - expected_width) < 1.0
        assert abs(ch_metrics['height'] - expected_height) < 1.0

    def test_calculate_voronoi_bounding_box_padding(self):
        """Test bounding box calculation with padding."""
        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'parent 1',
                'centroid_x': 50.0,
                'centroid_y': 50.0,
                'contour': [[[0, 0]], [[100, 0]], [[100, 100]], [[0, 100]]]  # 100x100 square
            }
        ]

        inverted_image = np.zeros((100, 100), dtype=np.uint8)
        padding_factor = 0.1  # 10% padding

        result = calculate_voronoi_points(metrics, inverted_image, padding_factor=padding_factor)

        assert result is not None

        bbox = result['bounding_box']

        # Original bounds: 0-100, with 10% padding should be -10 to 110
        expected_padding = 100 * 0.1  # 10

        assert abs(bbox['x_min'] - (-expected_padding)) < 1.0
        assert abs(bbox['x_max'] - (100 + expected_padding)) < 1.0
        assert abs(bbox['y_min'] - (-expected_padding)) < 1.0
        assert abs(bbox['y_max'] - (100 + expected_padding)) < 1.0

    def test_calculate_voronoi_invalid_contour_format(self):
        """Test handling of invalid contour format."""
        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'parent 1',
                'centroid_x': 50.0,
                'centroid_y': 50.0,
                'contour': "invalid_contour_format"  # Wrong format
            }
        ]

        inverted_image = np.zeros((100, 100), dtype=np.uint8)

        try:
            result = calculate_voronoi_points(metrics, inverted_image)
            # If it doesn't crash, should return None or handle gracefully
            if result is not None:
                # If it succeeds despite invalid format, that's also acceptable
                assert isinstance(result, dict)
        except (ValueError, TypeError, AttributeError):
            # These are acceptable errors for invalid contour format
            pass

    def test_calculate_voronoi_missing_contour_data(self):
        """Test handling when dorsal surface lacks contour data."""
        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'parent 1',
                'centroid_x': 50.0,
                'centroid_y': 50.0
                # Missing 'contour' field
            }
        ]

        inverted_image = np.zeros((100, 100), dtype=np.uint8)

        with patch('pylithics.image_processing.modules.voronoi_analysis.logging') as mock_logging:
            result = calculate_voronoi_points(metrics, inverted_image)

            assert result is None
            mock_logging.warning.assert_called()

    def test_calculate_voronoi_shared_edges_calculation(self):
        """Test calculation of shared edges between Voronoi cells."""
        # Create three points in a triangle to ensure shared edges
        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'parent 1',
                'centroid_x': 50.0,
                'centroid_y': 30.0,
                'contour': [[[20, 10]], [[80, 10]], [[80, 70]], [[20, 70]]]
            },
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'scar 1',
                'centroid_x': 35.0,
                'centroid_y': 55.0
            },
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'scar 2',
                'centroid_x': 65.0,
                'centroid_y': 55.0
            }
        ]

        inverted_image = np.zeros((80, 100), dtype=np.uint8)

        result = calculate_voronoi_points(metrics, inverted_image)

        assert result is not None
        assert len(result['voronoi_cells']) == 3

        # Check that some cells have shared edges
        shared_edge_counts = [cell['shared_edges'] for cell in result['voronoi_cells']]

        # In a triangle arrangement, each cell should share edges with others
        assert any(count > 0 for count in shared_edge_counts)

    def test_calculate_voronoi_metric_index_mapping(self):
        """Test that metric indices are correctly mapped to Voronoi cells."""
        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'parent 1',
                'centroid_x': 25.0,
                'centroid_y': 25.0,
                'contour': [[[10, 10]], [[40, 10]], [[40, 40]], [[10, 40]]]
            },
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'scar 1',
                'centroid_x': 35.0,
                'centroid_y': 35.0
            }
        ]

        inverted_image = np.zeros((50, 50), dtype=np.uint8)

        result = calculate_voronoi_points(metrics, inverted_image)

        assert result is not None

        # Check metric index mapping
        metric_indices = [cell['metric_index'] for cell in result['voronoi_cells']]

        # Should have valid indices
        assert all(idx >= 0 for idx in metric_indices)
        assert all(idx < len(metrics) for idx in metric_indices)

        # Should map to different metrics
        assert len(set(metric_indices)) == len(metrics)


@pytest.mark.unit
class TestVisualizeVoronoiDiagram:
    """Test Voronoi diagram visualization functionality."""

    def test_visualize_voronoi_basic(self):
        """Test basic Voronoi diagram visualization."""
        # Create mock voronoi_data
        from shapely.geometry import Polygon

        # Create simple polygon cells
        cell1 = Polygon([(10, 10), (30, 10), (30, 30), (10, 30)])
        cell2 = Polygon([(30, 10), (50, 10), (50, 30), (30, 30)])

        voronoi_data = {
            'voronoi_cells': [
                {'polygon': cell1, 'area': 400.0, 'shared_edges': 1, 'metric_index': 0},
                {'polygon': cell2, 'area': 400.0, 'shared_edges': 1, 'metric_index': 1}
            ],
            'convex_hull': Polygon([(10, 10), (50, 10), (50, 30), (10, 30)]),
            'points': MultiPoint([Point(20, 20), Point(40, 20)]),
            'bounding_box': {'x_min': 5, 'x_max': 55, 'y_min': 5, 'y_max': 35}
        }

        inverted_image = np.zeros((60, 60), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            output_path = temp_file.name

        try:
            # Should not crash
            visualize_voronoi_diagram(voronoi_data, inverted_image, output_path)

            # Check that file was created
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_visualize_voronoi_complex_geometry(self):
        """Test visualization with complex geometry."""
        from shapely.geometry import Polygon, LinearRing

        # Create irregular polygons
        cell1 = Polygon([(10, 10), (25, 5), (35, 20), (25, 35), (10, 30)])
        cell2 = Polygon([(35, 20), (50, 15), (55, 35), (40, 40), (25, 35)])

        # Create complex convex hull
        convex_hull = Polygon([(10, 5), (55, 5), (55, 40), (10, 40)])

        voronoi_data = {
            'voronoi_cells': [
                {'polygon': cell1, 'area': 375.0, 'shared_edges': 1, 'metric_index': 0},
                {'polygon': cell2, 'area': 425.0, 'shared_edges': 1, 'metric_index': 1}
            ],
            'convex_hull': convex_hull,
            'points': MultiPoint([Point(22, 20), Point(42, 25)]),
            'bounding_box': {'x_min': 0, 'x_max': 65, 'y_min': 0, 'y_max': 50}
        }

        inverted_image = np.zeros((70, 70), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            output_path = temp_file.name

        try:
            visualize_voronoi_diagram(voronoi_data, inverted_image, output_path)

            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_visualize_voronoi_multipolygon_cells(self):
        """Test visualization with MultiPolygon cells."""
        from shapely.geometry import Polygon, MultiPolygon

        # Create MultiPolygon cell (disconnected regions)
        poly1 = Polygon([(10, 10), (20, 10), (20, 20), (10, 20)])
        poly2 = Polygon([(30, 30), (40, 30), (40, 40), (30, 40)])
        multi_cell = MultiPolygon([poly1, poly2])

        simple_cell = Polygon([(50, 10), (60, 10), (60, 30), (50, 30)])

        voronoi_data = {
            'voronoi_cells': [
                {'polygon': multi_cell, 'area': 300.0, 'shared_edges': 0, 'metric_index': 0},
                {'polygon': simple_cell, 'area': 200.0, 'shared_edges': 0, 'metric_index': 1}
            ],
            'convex_hull': Polygon([(10, 10), (60, 10), (60, 40), (10, 40)]),
            'points': MultiPoint([Point(25, 25), Point(55, 20)]),
            'bounding_box': {'x_min': 5, 'x_max': 65, 'y_min': 5, 'y_max': 45}
        }

        inverted_image = np.zeros((70, 70), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            output_path = temp_file.name

        try:
            with patch('pylithics.image_processing.modules.voronoi_analysis.logging') as mock_logging:
                visualize_voronoi_diagram(voronoi_data, inverted_image, output_path)

                # Should handle MultiPolygon gracefully
                assert os.path.exists(output_path)

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_visualize_voronoi_degenerate_convex_hull(self):
        """Test visualization with degenerate convex hull geometries."""
        from shapely.geometry import Polygon, Point, LineString

        cell1 = Polygon([(10, 10), (30, 10), (30, 30), (10, 30)])

        voronoi_data = {
            'voronoi_cells': [
                {'polygon': cell1, 'area': 400.0, 'shared_edges': 0, 'metric_index': 0}
            ],
            'convex_hull': Point(20, 20),  # Degenerate: single point
            'points': MultiPoint([Point(20, 20)]),
            'bounding_box': {'x_min': 5, 'x_max': 35, 'y_min': 5, 'y_max': 35}
        }

        inverted_image = np.zeros((40, 40), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            output_path = temp_file.name

        try:
            visualize_voronoi_diagram(voronoi_data, inverted_image, output_path)

            assert os.path.exists(output_path)

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

        # Test with LineString convex hull
        voronoi_data['convex_hull'] = LineString([(10, 20), (30, 20)])

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            output_path = temp_file.name

        try:
            visualize_voronoi_diagram(voronoi_data, inverted_image, output_path)

            assert os.path.exists(output_path)

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_visualize_voronoi_invalid_output_path(self):
        """Test visualization with invalid output path."""
        from shapely.geometry import Polygon

        voronoi_data = {
            'voronoi_cells': [
                {'polygon': Polygon([(10, 10), (30, 10), (30, 30), (10, 30)]),
                 'area': 400.0, 'shared_edges': 0, 'metric_index': 0}
            ],
            'convex_hull': Polygon([(10, 10), (30, 10), (30, 30), (10, 30)]),
            'points': MultiPoint([Point(20, 20)]),
            'bounding_box': {'x_min': 5, 'x_max': 35, 'y_min': 5, 'y_max': 35}
        }

        inverted_image = np.zeros((40, 40), dtype=np.uint8)
        invalid_path = '/nonexistent_directory/output.png'

        try:
            # Should handle invalid path gracefully
            visualize_voronoi_diagram(voronoi_data, inverted_image, invalid_path)
        except (OSError, IOError, PermissionError):
            # These are acceptable errors for invalid paths
            pass


@pytest.mark.integration
class TestVoronoiAnalysisIntegration:
    """Integration tests for complete Voronoi analysis workflow."""

    def test_complete_voronoi_workflow(self):
        """Test complete Voronoi analysis workflow from metrics to visualization."""
        # Create realistic archaeological artifact metrics
        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'parent 1',
                'centroid_x': 100.0,
                'centroid_y': 150.0,
                'contour': [[[50, 100]], [[150, 100]], [[150, 200]], [[50, 200]]]  # Main surface
            },
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'scar 1',
                'centroid_x': 80.0,
                'centroid_y': 130.0  # Removal scar 1
            },
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'scar 2',
                'centroid_x': 120.0,
                'centroid_y': 170.0  # Removal scar 2
            },
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'scar 3',
                'centroid_x': 90.0,
                'centroid_y': 160.0  # Removal scar 3
            }
        ]

        inverted_image = np.zeros((250, 200), dtype=np.uint8)

        # Step 1: Calculate Voronoi
        voronoi_result = calculate_voronoi_points(metrics, inverted_image, padding_factor=0.05)

        assert voronoi_result is not None
        assert voronoi_result['voronoi_metrics']['num_cells'] == 4

        # Step 2: Verify cell properties
        total_area = sum(cell['area'] for cell in voronoi_result['voronoi_cells'])
        assert total_area > 0

        # Each cell should have valid metric mapping
        metric_indices = [cell['metric_index'] for cell in voronoi_result['voronoi_cells']]
        assert all(0 <= idx < len(metrics) for idx in metric_indices)

        # Step 3: Test visualization
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            viz_path = temp_file.name

        try:
            visualize_voronoi_diagram(voronoi_result, inverted_image, viz_path)

            assert os.path.exists(viz_path)
            assert os.path.getsize(viz_path) > 1000  # Should be substantial image file

        finally:
            if os.path.exists(viz_path):
                os.unlink(viz_path)

        # Step 4: Verify convex hull encompasses all points
        ch_metrics = voronoi_result['convex_hull_metrics']
        assert ch_metrics['width'] >= 40.0  # Should span multiple scars
        assert ch_metrics['height'] >= 40.0
        assert ch_metrics['area'] > 0

    def test_voronoi_with_clustered_scars(self):
        """Test Voronoi analysis with clustered removal scars."""
        # Create metrics with scars clustered in one area
        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'parent 1',
                'centroid_x': 100.0,
                'centroid_y': 100.0,
                'contour': [[[50, 50]], [[150, 50]], [[150, 150]], [[50, 150]]]
            },
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'scar 1',
                'centroid_x': 80.0,
                'centroid_y': 80.0  # Clustered group
            },
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'scar 2',
                'centroid_x': 85.0,
                'centroid_y': 85.0  # Close to scar 1
            },
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'scar 3',
                'centroid_x': 82.0,
                'centroid_y': 88.0  # Close to scars 1&2
            }
        ]

        inverted_image = np.zeros((200, 200), dtype=np.uint8)

        result = calculate_voronoi_points(metrics, inverted_image)

        assert result is not None
        assert result['voronoi_metrics']['num_cells'] == 4

        # Clustered scars should have more shared edges
        shared_edges = [cell['shared_edges'] for cell in result['voronoi_cells']]

        # Some cells should share edges due to clustering
        assert sum(shared_edges) > 0

    def test_voronoi_with_edge_case_geometries(self):
        """Test Voronoi analysis with edge case geometric configurations."""
        # Create collinear points (edge case for Voronoi)
        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'parent 1',
                'centroid_x': 50.0,
                'centroid_y': 50.0,
                'contour': [[[25, 25]], [[75, 25]], [[75, 75]], [[25, 75]]]
            },
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'scar 1',
                'centroid_x': 40.0,
                'centroid_y': 50.0  # Collinear arrangement
            },
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'scar 2',
                'centroid_x': 60.0,
                'centroid_y': 50.0  # Collinear with above
            }
        ]

        inverted_image = np.zeros((100, 100), dtype=np.uint8)

        result = calculate_voronoi_points(metrics, inverted_image)

        # Should handle collinear points gracefully
        if result is not None:
            assert result['voronoi_metrics']['num_cells'] >= 1
            assert len(result['voronoi_cells']) >= 1

            # Verify all cells have valid areas
            for cell in result['voronoi_cells']:
                assert cell['area'] >= 0


@pytest.mark.unit
class TestVoronoiAnalysisErrorHandling:
    """Test error handling in Voronoi analysis."""

    def test_voronoi_empty_metrics(self):
        """Test Voronoi analysis with empty metrics."""
        result = calculate_voronoi_points([], np.zeros((100, 100), dtype=np.uint8))
        assert result is None

    def test_voronoi_malformed_contour(self):
        """Test Voronoi analysis with malformed contour data."""
        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'parent 1',
                'centroid_x': 50.0,
                'centroid_y': 50.0,
                'contour': []  # Empty contour
            }
        ]

        inverted_image = np.zeros((100, 100), dtype=np.uint8)

        try:
            result = calculate_voronoi_points(metrics, inverted_image)
            # Should handle gracefully
        except (ValueError, TypeError, IndexError):
            # These are acceptable errors for malformed data
            pass

    def test_voronoi_invalid_coordinates(self):
        """Test Voronoi analysis with invalid coordinate values."""
        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'parent 1',
                'centroid_x': float('inf'),  # Invalid coordinate
                'centroid_y': 50.0,
                'contour': [[[25, 25]], [[75, 25]], [[75, 75]], [[25, 75]]]
            }
        ]

        inverted_image = np.zeros((100, 100), dtype=np.uint8)

        try:
            result = calculate_voronoi_points(metrics, inverted_image)
            # Should handle invalid coordinates gracefully
        except (ValueError, TypeError):
            # These are acceptable errors for invalid coordinates
            pass

    def test_voronoi_nan_coordinates(self):
        """Test Voronoi analysis with NaN coordinates."""
        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'parent 1',
                'centroid_x': 50.0,
                'centroid_y': float('nan'),  # NaN coordinate
                'contour': [[[25, 25]], [[75, 25]], [[75, 75]], [[25, 75]]]
            }
        ]

        inverted_image = np.zeros((100, 100), dtype=np.uint8)

        try:
            result = calculate_voronoi_points(metrics, inverted_image)
            # Should handle NaN coordinates gracefully
        except (ValueError, TypeError):
            # These are acceptable errors for NaN coordinates
            pass

    def test_voronoi_duplicate_points(self):
        """Test Voronoi analysis with duplicate point coordinates."""
        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'parent 1',
                'centroid_x': 50.0,
                'centroid_y': 50.0,
                'contour': [[[25, 25]], [[75, 25]], [[75, 75]], [[25, 75]]]
            },
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'scar 1',
                'centroid_x': 50.0,
                'centroid_y': 50.0  # Exact duplicate
            }
        ]

        inverted_image = np.zeros((100, 100), dtype=np.uint8)

        try:
            result = calculate_voronoi_points(metrics, inverted_image)
            # Shapely/scipy should handle duplicate points
            if result is not None:
                # Should have valid result despite duplicates
                assert result['voronoi_metrics']['num_cells'] >= 1
        except (ValueError, RuntimeError):
            # These are acceptable errors for duplicate points
            pass

    def test_visualize_voronoi_missing_data(self):
        """Test visualization with missing required data."""
        # Missing required fields
        incomplete_voronoi_data = {
            'voronoi_cells': [],
            # Missing other required fields
        }

        inverted_image = np.zeros((100, 100), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            output_path = temp_file.name

        try:
            # Should handle missing data gracefully
            visualize_voronoi_diagram(incomplete_voronoi_data, inverted_image, output_path)
        except (KeyError, AttributeError, TypeError):
            # These are acceptable errors for incomplete data
            pass
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_voronoi_very_large_coordinates(self):
        """Test Voronoi analysis with very large coordinate values."""
        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'parent 1',
                'centroid_x': 1e10,  # Very large coordinate
                'centroid_y': 1e10,
                'contour': [[[1e10-50, 1e10-50]], [[1e10+50, 1e10-50]],
                           [[1e10+50, 1e10+50]], [[1e10-50, 1e10+50]]]
            }
        ]

        inverted_image = np.zeros((100, 100), dtype=np.uint8)

        try:
            result = calculate_voronoi_points(metrics, inverted_image)
            # Should handle large coordinates or fail gracefully
        except (ValueError, OverflowError, MemoryError):
            # These are acceptable errors for extreme coordinates
            pass


@pytest.mark.performance
class TestVoronoiAnalysisPerformance:
    """Test performance aspects of Voronoi analysis."""

    def test_voronoi_many_points_performance(self):
        """Test Voronoi analysis performance with many points."""
        # Create many scars on dorsal surface
        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'parent 1',
                'centroid_x': 250.0,
                'centroid_y': 250.0,
                'contour': [[[50, 50]], [[450, 50]], [[450, 450]], [[50, 450]]]
            }
        ]

        # Add many scars
        for i in range(100):
            angle = 2 * np.pi * i / 100
            x = 250 + 150 * np.cos(angle)
            y = 250 + 150 * np.sin(angle)

            metrics.append({
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': f'scar_{i}',
                'centroid_x': x,
                'centroid_y': y
            })

        inverted_image = np.zeros((500, 500), dtype=np.uint8)

        import time
        start_time = time.time()

        result = calculate_voronoi_points(metrics, inverted_image)

        end_time = time.time()
        processing_time = end_time - start_time

        # Should complete in reasonable time even with many points
        assert processing_time < 30.0  # 30 seconds max

        if result is not None:
            assert result['voronoi_metrics']['num_cells'] == len(metrics)
            assert len(result['voronoi_cells']) == len(metrics)

    def test_voronoi_large_image_performance(self):
        """Test Voronoi analysis with large image dimensions."""
        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'parent 1',
                'centroid_x': 1000.0,
                'centroid_y': 1000.0,
                'contour': [[[500, 500]], [[1500, 500]], [[1500, 1500]], [[500, 1500]]]
            },
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'scar 1',
                'centroid_x': 800.0,
                'centroid_y': 900.0
            },
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'scar 2',
                'centroid_x': 1200.0,
                'centroid_y': 1100.0
            }
        ]

        # Large image
        large_image = np.zeros((2000, 2000), dtype=np.uint8)

        import time
        start_time = time.time()

        result = calculate_voronoi_points(metrics, large_image)

        end_time = time.time()
        processing_time = end_time - start_time

        # Should handle large images efficiently
        assert processing_time < 15.0  # 15 seconds max

        if result is not None:
            assert result['voronoi_metrics']['num_cells'] == 3

    def test_visualization_performance(self):
        """Test visualization performance with complex Voronoi diagram."""
        from shapely.geometry import Polygon

        # Create many complex cells
        voronoi_cells = []
        for i in range(50):
            # Create irregular polygon
            angles = np.linspace(0, 2*np.pi, 12)
            points = []
            for angle in angles:
                radius = 20 + 10 * np.sin(3 * angle)
                x = 100 + (i % 10) * 50 + radius * np.cos(angle)
                y = 100 + (i // 10) * 50 + radius * np.sin(angle)
                points.append((x, y))

            cell_polygon = Polygon(points)
            voronoi_cells.append({
                'polygon': cell_polygon,
                'area': cell_polygon.area,
                'shared_edges': np.random.randint(0, 4),
                'metric_index': i
            })

        # Create complex convex hull
        all_points = []
        for cell in voronoi_cells:
            all_points.extend(list(cell['polygon'].exterior.coords))

        from shapely.geometry import MultiPoint
        convex_hull = MultiPoint(all_points).convex_hull

        voronoi_data = {
            'voronoi_cells': voronoi_cells,
            'convex_hull': convex_hull,
            'points': MultiPoint([Point(i*10, i*10) for i in range(50)]),
            'bounding_box': {'x_min': 0, 'x_max': 600, 'y_min': 0, 'y_max': 600}
        }

        large_image = np.zeros((700, 700), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            output_path = temp_file.name

        try:
            import time
            start_time = time.time()

            visualize_voronoi_diagram(voronoi_data, large_image, output_path)

            end_time = time.time()
            viz_time = end_time - start_time

            # Visualization should complete quickly
            assert viz_time < 10.0  # 10 seconds max
            assert os.path.exists(output_path)

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


@pytest.mark.integration
class TestVoronoiRealWorldScenarios:
    """Test Voronoi analysis with realistic archaeological scenarios."""

    def test_voronoi_blade_tool_analysis(self):
        """Test Voronoi analysis on a blade tool with multiple removal scars."""
        # Simulate blade tool metrics
        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'parent 1',
                'centroid_x': 150.0,
                'centroid_y': 100.0,
                'contour': [[[50, 30]], [[250, 30]], [[280, 50]], [[270, 170]],
                           [[250, 180]], [[50, 180]], [[20, 160]], [[30, 50]]]  # Blade shape
            },
            # Longitudinal removal scars
            {'surface_type': 'Dorsal', 'parent': 'parent 1', 'scar': 'scar 1', 'centroid_x': 80.0, 'centroid_y': 70.0},
            {'surface_type': 'Dorsal', 'parent': 'parent 1', 'scar': 'scar 2', 'centroid_x': 120.0, 'centroid_y': 65.0},
            {'surface_type': 'Dorsal', 'parent': 'parent 1', 'scar': 'scar 3', 'centroid_x': 180.0, 'centroid_y': 80.0},
            {'surface_type': 'Dorsal', 'parent': 'parent 1', 'scar': 'scar 4', 'centroid_x': 220.0, 'centroid_y': 75.0},
            # Transverse removal scars
            {'surface_type': 'Dorsal', 'parent': 'parent 1', 'scar': 'scar 5', 'centroid_x': 100.0, 'centroid_y': 120.0},
            {'surface_type': 'Dorsal', 'parent': 'parent 1', 'scar': 'scar 6', 'centroid_x': 200.0, 'centroid_y': 130.0}
        ]

        inverted_image = np.zeros((220, 320), dtype=np.uint8)

        result = calculate_voronoi_points(metrics, inverted_image)

        assert result is not None
        assert result['voronoi_metrics']['num_cells'] == 7  # 1 dorsal + 6 scars

        # Blade tools should have elongated convex hull
        ch_metrics = result['convex_hull_metrics']
        aspect_ratio = ch_metrics['width'] / ch_metrics['height']

        # Should be elongated (width > height for horizontal blade)
        assert aspect_ratio > 1.5 or aspect_ratio < 0.67  # Allow either orientation

        # Check that cells have reasonable shared edges (scars should be connected)
        shared_edges = [cell['shared_edges'] for cell in result['voronoi_cells']]
        assert sum(shared_edges) > 4  # Should have good connectivity

    def test_voronoi_core_reduction_analysis(self):
        """Test Voronoi analysis on a core with radial scar pattern."""
        # Simulate core with radial reduction pattern
        center_x, center_y = 150.0, 150.0

        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'parent 1',
                'centroid_x': center_x,
                'centroid_y': center_y,
                'contour': [[[50, 50]], [[250, 50]], [[250, 250]], [[50, 250]]]  # Square core
            }
        ]

        # Add radial scars around the center
        for i in range(8):
            angle = 2 * np.pi * i / 8
            radius = 60
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)

            metrics.append({
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': f'scar_{i+1}',
                'centroid_x': x,
                'centroid_y': y
            })

        inverted_image = np.zeros((300, 300), dtype=np.uint8)

        result = calculate_voronoi_points(metrics, inverted_image)

        assert result is not None
        assert result['voronoi_metrics']['num_cells'] == 9  # 1 core + 8 scars

        # Radial pattern should create high connectivity
        shared_edges = [cell['shared_edges'] for cell in result['voronoi_cells']]
        avg_shared_edges = sum(shared_edges) / len(shared_edges)

        # Each scar should share edges with neighbors and center
        assert avg_shared_edges >= 1.5

        # Convex hull should be roughly circular (similar width and height)
        ch_metrics = result['convex_hull_metrics']
        aspect_ratio = ch_metrics['width'] / ch_metrics['height']
        assert 0.8 <= aspect_ratio <= 1.2  # Roughly square/circular

    def test_voronoi_bifacial_tool_analysis(self):
        """Test Voronoi analysis on bifacial tool with systematic thinning."""
        # Simulate bifacial tool with alternating scar pattern
        metrics = [
            {
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': 'parent 1',
                'centroid_x': 100.0,
                'centroid_y': 120.0,
                'contour': [[[30, 50]], [[170, 50]], [[190, 70]], [[180, 190]],
                           [[170, 200]], [[30, 200]], [[10, 180]], [[20, 70]]]  # Biface shape
            }
        ]

        # Add alternating edge scars (systematic thinning pattern)
        left_edge_x = 45
        right_edge_x = 155

        for i in range(6):
            y_pos = 80 + i * 20

            # Left edge scar
            metrics.append({
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': f'left_scar_{i+1}',
                'centroid_x': left_edge_x,
                'centroid_y': y_pos
            })

            # Right edge scar (offset vertically for alternating pattern)
            metrics.append({
                'surface_type': 'Dorsal',
                'parent': 'parent 1',
                'scar': f'right_scar_{i+1}',
                'centroid_x': right_edge_x,
                'centroid_y': y_pos + 10
            })

        inverted_image = np.zeros((250, 220), dtype=np.uint8)

        result = calculate_voronoi_points(metrics, inverted_image)

        assert result is not None
        assert result['voronoi_metrics']['num_cells'] == 13  # 1 biface + 12 edge scars

        # Bifacial tools should show linear organization
        ch_metrics = result['convex_hull_metrics']
        assert ch_metrics['width'] > 0
        assert ch_metrics['height'] > 0

        # Each scar should have some cell area
        for cell in result['voronoi_cells']:
            assert cell['area'] > 0

        # Test visualization with complex bifacial pattern
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            viz_path = temp_file.name

        try:
            visualize_voronoi_diagram(result, inverted_image, viz_path)
            assert os.path.exists(viz_path)
            assert os.path.getsize(viz_path) > 1000

        finally:
            if os.path.exists(viz_path):
                os.unlink(viz_path)