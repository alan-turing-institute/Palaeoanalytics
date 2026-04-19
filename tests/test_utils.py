"""Tests for metadata, config, and contour filtering utilities."""

import csv
import json
import os
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from pylithics.image_processing.utils import (
    filter_contours_by_min_area,
    load_config,
    read_metadata,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _square_contour(origin=(0, 0), side=10):
    """Create a square contour in OpenCV's (N, 1, 2) int32 format."""
    x, y = origin
    pts = np.array(
        [[x, y], [x + side, y], [x + side, y + side], [x, y + side]],
        dtype=np.int32,
    )
    return pts.reshape(-1, 1, 2)


def _flat_hierarchy(n):
    return np.array([[-1, -1, -1, -1]] * n)


# ---------------------------------------------------------------------------
# read_metadata
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReadMetadata:
    """read_metadata parses metadata CSVs into a list of dicts."""

    def test_parses_valid_csv(self, sample_metadata_file):
        metadata = read_metadata(sample_metadata_file)

        assert len(metadata) == 3
        assert metadata[0] == {
            "image_id": "test_image_1.png",
            "scale_id": "scale_1",
            "scale": "10.0",
        }

    def test_preserves_additional_columns(self, tmp_path):
        path = tmp_path / "custom.csv"
        _write_csv(
            path,
            [{
                "image_id": "blade_01.png",
                "scale_id": "scale_A",
                "scale": "12.5",
                "artifact_type": "blade",
                "site": "Sector-A",
            }],
            fieldnames=["image_id", "scale_id", "scale", "artifact_type", "site"],
        )

        metadata = read_metadata(str(path))

        assert len(metadata) == 1
        entry = metadata[0]
        assert entry["artifact_type"] == "blade"
        assert entry["site"] == "Sector-A"
        assert entry["scale"] == "12.5"

    def test_handles_unicode_content(self, tmp_path):
        path = tmp_path / "unicode.csv"
        _write_csv(
            path,
            [{
                "image_id": "français.png",
                "scale_id": "échelle_1",
                "scale": "10.0",
                "notes": "Artéfact trouvé à 深圳",
            }],
            fieldnames=["image_id", "scale_id", "scale", "notes"],
        )

        metadata = read_metadata(str(path))

        assert metadata[0]["image_id"] == "français.png"
        assert metadata[0]["notes"] == "Artéfact trouvé à 深圳"

    def test_returns_empty_for_empty_file(self, tmp_path):
        empty = tmp_path / "empty.csv"
        empty.touch()
        assert read_metadata(str(empty)) == []

    @pytest.mark.parametrize(
        "side_effect",
        [FileNotFoundError, PermissionError("denied")],
    )
    def test_returns_empty_and_logs_on_io_error(self, side_effect):
        with patch("builtins.open", side_effect=side_effect), \
             patch("pylithics.image_processing.utils.logging") as mock_log:
            assert read_metadata("/unused/path.csv") == []
        mock_log.error.assert_called()


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoadConfig:
    """load_config reads JSON configs, returns None on failure."""

    def test_loads_valid_json(self, tmp_path):
        config_data = {"thresholding": {"method": "otsu", "threshold_value": 150}}
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config_data))

        loaded = load_config(str(path))
        assert loaded == config_data

    def test_loads_nested_structure(self, tmp_path):
        nested = {"a": {"b": {"c": [1, 2, 3], "enabled": True}}}
        path = tmp_path / "nested.json"
        path.write_text(json.dumps(nested))

        assert load_config(str(path)) == nested

    def test_returns_none_for_invalid_json(self, tmp_path):
        path = tmp_path / "invalid.json"
        path.write_text('{"bad": json content}')

        with patch("pylithics.image_processing.utils.logging") as mock_log:
            assert load_config(str(path)) is None
        mock_log.error.assert_called()

    @pytest.mark.parametrize(
        "side_effect",
        [FileNotFoundError, PermissionError("denied")],
    )
    def test_returns_none_and_logs_on_io_error(self, side_effect):
        with patch("builtins.open", side_effect=side_effect), \
             patch("pylithics.image_processing.utils.logging") as mock_log:
            assert load_config("/unused/path.json") is None
        mock_log.error.assert_called()


# ---------------------------------------------------------------------------
# filter_contours_by_min_area
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFilterContoursByMinArea:
    """filter_contours_by_min_area drops contours below the threshold."""

    @pytest.mark.parametrize(
        "min_area, expected_sides",
        [
            (0.0, [100, 10, 1]),     # zero threshold keeps everything
            (-10.0, [100, 10, 1]),   # negative threshold keeps everything
            (50.0, [100, 10]),       # drops only the 1x1
            (500.0, [100]),          # drops everything except the 100x100
            (20000.0, []),           # nothing meets threshold
        ],
    )
    def test_threshold_filtering(self, min_area, expected_sides):
        contours = [
            _square_contour(side=100),  # area 10000
            _square_contour(side=10),   # area 100
            _square_contour(side=1),    # area 1
        ]
        hierarchy = _flat_hierarchy(3)

        filtered, filtered_hier = filter_contours_by_min_area(
            contours, hierarchy, min_area=min_area
        )

        actual_sides = [
            int(round(np.sqrt(cv2.contourArea(c)))) for c in filtered
        ]
        assert actual_sides == expected_sides
        if expected_sides:
            assert len(filtered_hier) == len(expected_sides)
        else:
            assert filtered_hier is None

    def test_default_threshold_is_one(self):
        # Contour with area == 1.0 should pass the default threshold
        contours = [_square_contour(side=1)]
        filtered, _ = filter_contours_by_min_area(contours, _flat_hierarchy(1))
        assert len(filtered) == 1

    def test_empty_input_short_circuits(self):
        filtered, filtered_hier = filter_contours_by_min_area([], None)
        assert filtered == []
        assert filtered_hier is None

    def test_none_hierarchy_returns_empty(self):
        # Per the contract, hierarchy=None always yields empty output
        filtered, filtered_hier = filter_contours_by_min_area(
            [_square_contour(side=20)], None
        )
        assert filtered == []
        assert filtered_hier is None

    def test_preserves_hierarchy_for_retained_contours(self):
        parent = _square_contour(side=100)
        child = _square_contour(origin=(20, 20), side=60)
        noise = _square_contour(origin=(5, 5), side=2)

        hierarchy = np.array([
            [-1, -1, 1, -1],  # parent -> child at index 1
            [-1, -1, -1, 0],  # child -> parent at index 0
            [-1, -1, -1, -1],
        ])

        filtered, filtered_hier = filter_contours_by_min_area(
            [parent, child, noise], hierarchy, min_area=1000.0
        )

        assert len(filtered) == 2
        # Hierarchy rows for retained contours preserved in order
        assert np.array_equal(filtered_hier[0], hierarchy[0])
        assert np.array_equal(filtered_hier[1], hierarchy[1])


# ---------------------------------------------------------------------------
# Integration: config-driven metadata + contour filtering
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_metadata_config_and_filter_workflow(tmp_path):
    """Read metadata, read JSON config, use it to drive filter_contours."""
    metadata_path = tmp_path / "metadata.csv"
    _write_csv(
        metadata_path,
        [
            {"image_id": "artifact_001.png", "scale_id": "s1", "scale": "10.5"},
            {"image_id": "artifact_002.png", "scale_id": "s2", "scale": "12.0"},
        ],
        fieldnames=["image_id", "scale_id", "scale"],
    )

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "contour_filtering": {"min_area": 200.0},
    }))

    metadata = read_metadata(str(metadata_path))
    config = load_config(str(config_path))

    assert [row["image_id"] for row in metadata] == [
        "artifact_001.png", "artifact_002.png",
    ]
    min_area = config["contour_filtering"]["min_area"]

    contours = [_square_contour(side=50), _square_contour(side=5)]
    filtered, _ = filter_contours_by_min_area(
        contours, _flat_hierarchy(2), min_area=min_area
    )

    # Only the 50x50 (area 2500) survives a min_area of 200
    assert len(filtered) == 1
    assert cv2.contourArea(filtered[0]) == pytest.approx(2500.0)
