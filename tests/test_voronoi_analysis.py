"""Tests for Voronoi diagram generation and visualization."""

import os
import tempfile
import time

import cv2
import numpy as np
import pytest
from shapely.geometry import MultiPoint, Point, Polygon

from pylithics.image_processing.modules.voronoi_analysis import (
    calculate_voronoi_points,
    visualize_voronoi_diagram,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rect_contour(x0, y0, x1, y1):
    """Shapely-compatible contour format: list of [[[x, y]], ...]."""
    return [[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]]


def _dorsal_parent(centroid, contour):
    cx, cy = centroid
    return {
        "surface_type": "Dorsal",
        "parent": "parent 1",
        "scar": "parent 1",
        "centroid_x": cx,
        "centroid_y": cy,
        "contour": contour,
    }


def _dorsal_scar(name, centroid):
    cx, cy = centroid
    return {
        "surface_type": "Dorsal",
        "parent": "parent 1",
        "scar": name,
        "centroid_x": cx,
        "centroid_y": cy,
    }


# Stock 100x100 dorsal parent used as baseline in most tests
BASE_PARENT = _dorsal_parent(
    centroid=(50, 50), contour=_rect_contour(0, 0, 100, 100)
)


# ---------------------------------------------------------------------------
# calculate_voronoi_points
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCalculateVoronoiPoints:
    """Core behaviors of calculate_voronoi_points."""

    def test_returns_none_when_no_dorsal_parent(self):
        metrics = [
            {
                "surface_type": "Ventral",
                "parent": "parent 1", "scar": "parent 1",
                "centroid_x": 50.0, "centroid_y": 50.0,
            }
        ]
        assert calculate_voronoi_points(
            metrics, np.zeros((100, 100), dtype=np.uint8)
        ) is None

    def test_returns_none_on_empty_metrics(self):
        assert calculate_voronoi_points(
            [], np.zeros((100, 100), dtype=np.uint8)
        ) is None

    def test_dorsal_without_contour_returns_none(self):
        metrics = [{
            "surface_type": "Dorsal",
            "parent": "parent 1", "scar": "parent 1",
            "centroid_x": 50.0, "centroid_y": 50.0,
            # no 'contour' key
        }]
        assert calculate_voronoi_points(
            metrics, np.zeros((100, 100), dtype=np.uint8)
        ) is None

    def test_dorsal_only_yields_single_cell(self):
        result = calculate_voronoi_points(
            [BASE_PARENT], np.zeros((100, 100), dtype=np.uint8)
        )

        assert result is not None
        expected_keys = {
            "voronoi_diagram", "voronoi_cells", "voronoi_metrics",
            "convex_hull", "convex_hull_metrics", "points",
            "bounding_box", "dorsal_contour",
        }
        assert set(result.keys()) == expected_keys
        assert result["voronoi_metrics"]["num_cells"] == 1
        # One-cell area should equal the dorsal polygon area (10000).
        assert result["voronoi_cells"][0]["area"] == pytest.approx(10000.0, rel=0.01)

    def test_dorsal_with_scars_yields_one_cell_per_point(self):
        metrics = [
            BASE_PARENT,
            _dorsal_scar("scar 1", (30, 30)),
            _dorsal_scar("scar 2", (70, 70)),
        ]
        result = calculate_voronoi_points(
            metrics, np.zeros((100, 100), dtype=np.uint8)
        )

        assert result["voronoi_metrics"]["num_cells"] == 3
        assert len(result["voronoi_cells"]) == 3

        for cell in result["voronoi_cells"]:
            assert cell["area"] > 0
            assert cell["metric_index"] >= 0
            assert cell["shared_edges"] >= 0

    def test_cells_partition_dorsal_area(self):
        """The Voronoi cells clipped to the dorsal contour should tile it."""
        metrics = [
            BASE_PARENT,
            _dorsal_scar("scar 1", (25, 25)),
            _dorsal_scar("scar 2", (75, 75)),
        ]
        result = calculate_voronoi_points(
            metrics, np.zeros((100, 100), dtype=np.uint8)
        )

        total_cell_area = sum(c["area"] for c in result["voronoi_cells"])
        # Cells tile the dorsal polygon (area 10000) exactly.
        assert total_cell_area == pytest.approx(10000.0, rel=0.01)

    def test_metric_indices_map_back_to_source_metrics(self):
        metrics = [
            BASE_PARENT,
            _dorsal_scar("scar 1", (30, 30)),
            _dorsal_scar("scar 2", (70, 70)),
        ]
        result = calculate_voronoi_points(
            metrics, np.zeros((100, 100), dtype=np.uint8)
        )

        metric_indices = [c["metric_index"] for c in result["voronoi_cells"]]
        # Every cell points to a distinct metric in the input list.
        assert sorted(metric_indices) == [0, 1, 2]

    def test_shared_edges_are_symmetric(self):
        """If cell A shares an edge with B, then B shares with A — count must be balanced."""
        metrics = [
            BASE_PARENT,
            _dorsal_scar("scar 1", (30, 50)),
            _dorsal_scar("scar 2", (70, 50)),
        ]
        result = calculate_voronoi_points(
            metrics, np.zeros((100, 100), dtype=np.uint8)
        )

        edge_sum = sum(c["shared_edges"] for c in result["voronoi_cells"])
        # Every shared edge contributes 2 to the total. Must be even and > 0.
        assert edge_sum > 0
        assert edge_sum % 2 == 0

    def test_convex_hull_matches_span_of_points(self):
        metrics = [
            _dorsal_parent(centroid=(25, 25), contour=_rect_contour(10, 10, 40, 40)),
            _dorsal_scar("scar 1", (35, 15)),
        ]
        result = calculate_voronoi_points(
            metrics, np.zeros((50, 50), dtype=np.uint8)
        )

        hull = result["convex_hull_metrics"]
        # Points at (25, 25) and (35, 15) span width 10, height 10
        assert hull["width"] == pytest.approx(10.0, abs=0.5)
        assert hull["height"] == pytest.approx(10.0, abs=0.5)

    def test_bounding_box_applies_padding_factor(self):
        """Padding factor expands each side by (span * factor)."""
        metrics = [_dorsal_parent(
            centroid=(50, 50), contour=_rect_contour(0, 0, 100, 100)
        )]
        padding_factor = 0.1

        result = calculate_voronoi_points(
            metrics, np.zeros((100, 100), dtype=np.uint8),
            padding_factor=padding_factor,
        )

        bbox = result["bounding_box"]
        expected_pad = 100 * padding_factor
        assert bbox["x_min"] == pytest.approx(-expected_pad, abs=0.1)
        assert bbox["x_max"] == pytest.approx(100 + expected_pad, abs=0.1)
        assert bbox["y_min"] == pytest.approx(-expected_pad, abs=0.1)
        assert bbox["y_max"] == pytest.approx(100 + expected_pad, abs=0.1)


# ---------------------------------------------------------------------------
# visualize_voronoi_diagram
# ---------------------------------------------------------------------------


def _fake_voronoi_data(cells, hull=None, points=None, bbox=None):
    default_hull = Polygon([(0, 0), (60, 0), (60, 40), (0, 40)])
    default_points = MultiPoint([Point(20, 20), Point(40, 20)])
    default_bbox = {"x_min": 0, "x_max": 60, "y_min": 0, "y_max": 40}
    return {
        "voronoi_cells": cells,
        "convex_hull": hull if hull is not None else default_hull,
        "points": points if points is not None else default_points,
        "bounding_box": bbox if bbox is not None else default_bbox,
    }


@pytest.mark.unit
class TestVisualizeVoronoiDiagram:
    """Rendering of the Voronoi diagram to an image file."""

    def test_writes_valid_png_with_nonzero_size(self):
        cell = Polygon([(10, 10), (30, 10), (30, 30), (10, 30)])
        data = _fake_voronoi_data([
            {"polygon": cell, "area": 400.0,
             "shared_edges": 1, "metric_index": 0},
        ])

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "voronoi.png")
            visualize_voronoi_diagram(
                data, np.zeros((60, 60), dtype=np.uint8), output_path
            )

            assert os.path.exists(output_path)
            # Matplotlib savefig always produces >1KB for a real plot
            assert os.path.getsize(output_path) > 1000

            # Confirm it's a readable image, not just bytes on disk
            rendered = cv2.imread(output_path)
            assert rendered is not None
            assert rendered.ndim == 3

    def test_creates_missing_output_directory(self):
        cell = Polygon([(10, 10), (30, 10), (30, 30), (10, 30)])
        data = _fake_voronoi_data([
            {"polygon": cell, "area": 400.0,
             "shared_edges": 0, "metric_index": 0},
        ])

        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = os.path.join(temp_dir, "a", "b", "out.png")
            visualize_voronoi_diagram(
                data, np.zeros((50, 50), dtype=np.uint8), nested_path
            )
            assert os.path.exists(nested_path)


# ---------------------------------------------------------------------------
# Integration: end-to-end calculate + visualize
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_full_voronoi_pipeline_produces_matching_metrics_and_image():
    metrics = [
        _dorsal_parent(centroid=(100, 150), contour=_rect_contour(50, 100, 150, 200)),
        _dorsal_scar("scar 1", (80, 130)),
        _dorsal_scar("scar 2", (120, 170)),
        _dorsal_scar("scar 3", (90, 160)),
    ]
    inverted = np.zeros((250, 200), dtype=np.uint8)

    result = calculate_voronoi_points(metrics, inverted, padding_factor=0.05)

    assert result["voronoi_metrics"]["num_cells"] == 4
    total_area = sum(c["area"] for c in result["voronoi_cells"])
    # Cells should partition the 100x100 dorsal polygon
    assert total_area == pytest.approx(10000.0, rel=0.02)

    with tempfile.TemporaryDirectory() as temp_dir:
        viz_path = os.path.join(temp_dir, "voronoi.png")
        visualize_voronoi_diagram(result, inverted, viz_path)
        assert os.path.getsize(viz_path) > 1000


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


@pytest.mark.performance
def test_voronoi_with_fifty_points_completes_under_ten_seconds():
    metrics = [_dorsal_parent(
        centroid=(250, 250), contour=_rect_contour(50, 50, 450, 450)
    )]
    # Place 50 points on a circle for a deterministic, non-degenerate layout
    for i in range(50):
        angle = 2 * np.pi * i / 50
        metrics.append(_dorsal_scar(
            f"scar_{i}",
            (250 + 150 * np.cos(angle), 250 + 150 * np.sin(angle)),
        ))

    start = time.time()
    result = calculate_voronoi_points(
        metrics, np.zeros((500, 500), dtype=np.uint8)
    )
    elapsed = time.time() - start

    assert result["voronoi_metrics"]["num_cells"] == 51
    assert elapsed < 10.0, f"took {elapsed:.1f}s (limit 10s)"
