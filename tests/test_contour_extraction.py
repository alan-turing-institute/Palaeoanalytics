"""Tests for contour extraction and hierarchy sorting."""

from unittest.mock import patch

import cv2
import numpy as np
import pytest

from pylithics.image_processing.modules.contour_extraction import (
    extract_contours_with_hierarchy,
    hide_nested_child_contours,
    sort_contours_by_hierarchy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _poly(*pts):
    return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)


def _patch_filtering_config(min_area):
    """Patch the config getter used inside extract_contours_with_hierarchy."""
    return patch(
        "pylithics.image_processing.modules.contour_extraction."
        "get_contour_filtering_config",
        return_value={"min_area": min_area},
    )


# ---------------------------------------------------------------------------
# extract_contours_with_hierarchy: border filter, area filter, shape of output
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractContoursWithHierarchy:

    def test_single_rectangle_is_extracted_and_indexed(self, tmp_path):
        image = np.zeros((200, 300), dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (120, 120), 255, thickness=-1)

        with _patch_filtering_config(100.0):
            contours, hierarchy = extract_contours_with_hierarchy(
                image, "shape", str(tmp_path)
            )

        assert len(contours) == 1
        assert hierarchy is not None
        assert hierarchy.shape == (1, 4)
        # Parent marker is -1 for a root contour
        assert hierarchy[0][3] == -1
        # Exact bounding box area = 100 * 100
        assert cv2.contourArea(contours[0]) == pytest.approx(10000.0, abs=10.0)

    def test_empty_image_returns_empty_list_and_none_hierarchy(self, tmp_path):
        image = np.zeros((100, 100), dtype=np.uint8)
        with _patch_filtering_config(50.0):
            contours, hierarchy = extract_contours_with_hierarchy(
                image, "empty", str(tmp_path)
            )
        assert contours == []
        assert hierarchy is None

    def test_border_touching_shapes_are_removed(self, tmp_path):
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(image, (0, 20), (30, 80), 255, thickness=-1)   # left border
        cv2.rectangle(image, (50, 50), (80, 80), 255, thickness=-1)  # away from border

        with _patch_filtering_config(50.0):
            contours, _ = extract_contours_with_hierarchy(
                image, "border", str(tmp_path)
            )

        assert len(contours) == 1
        x, y, w, h = cv2.boundingRect(contours[0])
        assert x > 0 and y > 0 and x + w < 100 and y + h < 100

    def test_contours_below_min_area_are_filtered_out(self, tmp_path):
        image = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (150, 150), 255, -1)   # area 10000
        cv2.rectangle(image, (10, 10), (20, 20), 255, -1)     # area 100

        with _patch_filtering_config(500.0):
            contours, _ = extract_contours_with_hierarchy(
                image, "area", str(tmp_path)
            )

        assert len(contours) == 1
        assert cv2.contourArea(contours[0]) >= 500.0

    def test_hierarchy_parent_indices_remain_valid_after_filtering(self, tmp_path):
        """
        After border + area filtering, every parent index in hierarchy must
        either be -1 or point to a row that still exists.
        """
        image = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (150, 150), 255, -1)  # outer shape
        cv2.rectangle(image, (70, 70), (130, 130), 0, -1)    # inner hole
        cv2.rectangle(image, (90, 90), (110, 110), 255, -1)  # inner shape

        with _patch_filtering_config(50.0):
            contours, hierarchy = extract_contours_with_hierarchy(
                image, "hierarchy", str(tmp_path)
            )

        for row in hierarchy:
            parent_idx = row[3]
            assert parent_idx == -1 or 0 <= parent_idx < len(hierarchy)


# ---------------------------------------------------------------------------
# sort_contours_by_hierarchy: dispatch into parents/children/nested_children
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSortContoursByHierarchy:

    def test_parent_with_one_child(self, sample_contours, sample_hierarchy):
        result = sort_contours_by_hierarchy(sample_contours, sample_hierarchy)
        assert len(result["parents"]) == 1
        assert len(result["children"]) == 1
        assert len(result["nested_children"]) == 0

    def test_empty_inputs_return_empty_buckets(self):
        assert sort_contours_by_hierarchy([], None) == {
            "parents": [], "children": [], "nested_children": [],
        }

    def test_parents_without_children(self):
        contours = [_poly([10, 10], [20, 10], [20, 20], [10, 20]),
                    _poly([30, 30], [40, 30], [40, 40], [30, 40])]
        hierarchy = np.array([[-1, -1, -1, -1], [-1, -1, -1, -1]])

        result = sort_contours_by_hierarchy(contours, hierarchy)
        assert len(result["parents"]) == 2
        assert result["children"] == []
        assert result["nested_children"] == []

    def test_nested_children_go_to_nested_bucket(self):
        """Grandchildren (depth ≥ 2) should land in nested_children."""
        contours = [
            _poly([0, 0], [100, 0], [100, 100], [0, 100]),    # parent
            _poly([20, 20], [80, 20], [80, 80], [20, 80]),    # child
            _poly([40, 40], [60, 40], [60, 60], [40, 60]),    # grandchild
        ]
        hierarchy = np.array([
            [-1, -1, 1, -1],   # parent at index 0, child at index 1
            [-1, -1, 2, 0],    # child at index 1, parent=0, grandchild at 2
            [-1, -1, -1, 1],   # grandchild at index 2, parent=1
        ])

        result = sort_contours_by_hierarchy(contours, hierarchy)
        assert len(result["parents"]) == 1
        assert len(result["children"]) == 1
        assert len(result["nested_children"]) == 1

    def test_exclude_flags_skip_matching_contours(
        self, sample_contours, sample_hierarchy
    ):
        # Flag the first contour (parent) for exclusion
        flags = [True, False]
        result = sort_contours_by_hierarchy(sample_contours, sample_hierarchy, flags)
        # The excluded parent should not appear in parents
        assert result["parents"] == []

    def test_exclude_flags_of_wrong_length_fall_back_to_defaults(
        self, sample_contours, sample_hierarchy
    ):
        # Wrong-length flags must not bleed into the output
        result = sort_contours_by_hierarchy(
            sample_contours, sample_hierarchy,
            exclude_nested_flags=[True, True, True],  # 3 flags, 2 contours
        )
        # Defaults are used — parent+child both included
        assert len(result["parents"]) == 1
        assert len(result["children"]) == 1

    def test_hierarchy_indices_beyond_contours_are_ignored(self):
        """Hierarchy entries with no matching contour must be skipped safely."""
        contours = [_poly([10, 10], [20, 10], [20, 20], [10, 20])]
        hierarchy = np.array([
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],   # no contour at index 1
        ])
        result = sort_contours_by_hierarchy(contours, hierarchy)
        # Only the real contour can end up in a bucket
        assert len(result["parents"]) == 1
        assert len(result["children"]) == 0


# ---------------------------------------------------------------------------
# hide_nested_child_contours: feature is intentionally disabled (always False)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHideNestedChildContours:
    """The function currently returns an all-False mask — feature is disabled."""

    def test_all_flags_false_for_single_child(self):
        contours = [
            _poly([0, 0], [100, 0], [100, 100], [0, 100]),
            _poly([20, 20], [40, 20], [40, 40], [20, 40]),
        ]
        hierarchy = np.array([[-1, -1, 1, -1], [-1, -1, -1, 0]])
        flags = hide_nested_child_contours(contours, hierarchy)
        assert flags == [False, False]

    def test_all_flags_false_for_multiple_children(self):
        contours = [
            _poly([0, 0], [100, 0], [100, 100], [0, 100]),
            _poly([10, 10], [20, 10], [20, 20], [10, 20]),
            _poly([30, 30], [40, 30], [40, 40], [30, 40]),
        ]
        hierarchy = np.array([
            [-1, -1, 1, -1],
            [2, -1, -1, 0],
            [-1, 1, -1, 0],
        ])
        flags = hide_nested_child_contours(contours, hierarchy)
        assert flags == [False, False, False]

    def test_empty_inputs_return_empty_flag_list(self):
        assert hide_nested_child_contours([], None) == []


# ---------------------------------------------------------------------------
# Integration: extract -> sort round trip
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_extract_then_sort_finds_one_parent_and_one_child(tmp_path):
    image = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (150, 150), 255, -1)  # outer
    cv2.rectangle(image, (70, 70), (130, 130), 0, -1)    # inner hole (child)

    with _patch_filtering_config(50.0):
        contours, hierarchy = extract_contours_with_hierarchy(
            image, "integration", str(tmp_path)
        )

    sorted_contours = sort_contours_by_hierarchy(contours, hierarchy)
    assert len(sorted_contours["parents"]) == 1
    assert len(sorted_contours["children"]) == 1
