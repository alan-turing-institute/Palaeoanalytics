"""Tests for the arrow-integration pipeline: the marks inside recognised scars."""

from unittest.mock import patch

import cv2
import numpy as np
import pytest

from pylithics.image_processing.modules.arrow_integration import (
    integrate_arrows,
    marks_in_scar,
    recognised_scars,
)

DETECT = "pylithics.image_processing.modules.arrow_integration.analyze_child_contour_for_arrow"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _drawing(with_loose=False):
    """
    A surface outline holding one scar, a mark inside the scar, and the contours found.

    Returns the binary, the surface contour, the scar contour and the
    source box of the mark. With ``with_loose`` a second mark sits on the
    surface outside the scar.
    """
    canvas = np.zeros((200, 200), np.uint8)
    cv2.rectangle(canvas, (10, 10), (190, 190), 255, 1)
    cv2.polylines(canvas, [np.array([[10, 40], [120, 44], [116, 120], [10, 116]], np.int32)],
                  True, 255, 1)   # the scar's outline joins the surface's, as drawn
    cv2.line(canvas, (60, 80), (90, 80), 255, 2)
    if with_loose:
        cv2.line(canvas, (140, 150), (170, 150), 255, 2)
    return canvas, *_surface_and_scar(canvas)


def _surface_and_scar(canvas):
    """Return the surface contour and the scar contour: the one holding the mark at (60, 80)."""
    contours, hierarchy = cv2.findContours(canvas, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    holding = [
        i for i, c in enumerate(contours)
        if cv2.pointPolygonTest(c, (75.0, 80.0), False) >= 0
    ]
    mark = min(holding, key=lambda i: cv2.contourArea(contours[i]))
    return max(contours, key=cv2.contourArea), contours[hierarchy[0][mark][3]]


def _metrics(surface, scar, **scar_fields):
    base = {"parent": "parent 1", "surface_type": "Dorsal"}
    return [
        {**base, "scar": "parent 1", "area": cv2.contourArea(surface)},
        {**base, "scar": "scar 1", "area": cv2.contourArea(scar), **scar_fields},
    ]


def _result(stage, score=None, back=(60, 80), tip=(90, 80)):
    result = {
        "arrow_back": back, "arrow_tip": tip, "angle_rad": 0.0, "angle_deg": 0.0,
        "compass_angle": 270.0, "stage": stage,
    }
    if score is not None:
        result["score"] = score
    return result


# ---------------------------------------------------------------------------
# marks_in_scar
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMarksInScar:

    def test_a_mark_inside_the_scar_is_found_and_the_outline_is_not(self):
        canvas, _, scar = _drawing()
        marks = marks_in_scar(scar, canvas, 1)
        assert len(marks.contours) == 1
        x, y, w, h = cv2.boundingRect(marks.contours[0])
        assert marks.to_source((x, y)) == (59, 79)
        assert (w, h) == (33, 3)

    def test_a_piece_joined_to_the_outline_is_not_a_mark(self):
        canvas, _, scar = _drawing()
        cv2.line(canvas, (10, 60), (70, 60), 255, 1)   # starts on the outline
        _, scar = _surface_and_scar(canvas)
        marks = marks_in_scar(scar, canvas, 1)
        assert len(marks.contours) == 1   # the free mark only

    @pytest.mark.parametrize("factor", [2, 3, 4])
    def test_marks_are_cut_from_the_working_copy_and_mapped_back(self, factor):
        canvas, _, scar = _drawing()
        working = cv2.resize(canvas, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
        marks = marks_in_scar(scar, working, factor)
        assert marks.factor == factor and len(marks.contours) == 1
        x, y, w, h = cv2.boundingRect(marks.contours[0])
        assert marks.to_source((x, y)) == (59, 79)
        assert (w, h) == (33 * factor, 3 * factor)
        assert marks.ink.shape[0] < working.shape[0]   # a crop, not the whole image

    def test_an_empty_scar_has_no_marks(self):
        canvas = np.zeros((100, 100), np.uint8)
        cv2.rectangle(canvas, (20, 20), (80, 80), 255, 1)
        contours, _ = cv2.findContours(canvas, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        scar = min(contours, key=cv2.contourArea)
        assert marks_in_scar(scar, canvas, 1).contours == []


# ---------------------------------------------------------------------------
# integrate_arrows
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIntegrateArrows:

    def test_only_a_mark_inside_a_recognised_scar_is_offered(self):
        canvas, surface, scar = _drawing(with_loose=True)
        metrics = _metrics(surface, scar)
        offered = []

        def record(contour, entry, ink, *_args):
            offered.append((cv2.boundingRect(contour), entry["scar"]))
            return None

        with patch(DETECT, side_effect=record):
            integrate_arrows({"parents": [surface], "children": [scar]}, metrics, canvas, 300.0)
        assert len(offered) == 1
        assert offered[0][1] == "scar 1 mark 1"

    def test_the_result_is_applied_in_source_pixels(self):
        canvas, surface, scar = _drawing()
        working = cv2.resize(canvas, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        metrics = _metrics(surface, scar)
        seen = {}

        def detect(contour, entry, ink, image_dpi, factor):
            seen["factor"], seen["shape"] = factor, ink.shape
            return _result("model", 0.8, back=(100, 40), tip=(220, 40))

        with patch(DETECT, side_effect=detect):
            integrate_arrows(
                {"parents": [surface], "children": [scar]}, metrics, canvas, 300.0,
                working=working, factor=4,
            )
        assert seen["factor"] == 4 and seen["shape"][0] < working.shape[0]
        row = metrics[1]
        assert row["has_arrow"] is True and row["arrow_score"] == 0.8
        assert row["arrow_angle"] == 270.0
        marks = marks_in_scar(scar, working, 4)
        assert row["arrow_back"] == marks.to_source((100, 40))
        assert row["arrow_tip"] == marks.to_source((220, 40))
        assert row["arrow_tip"][0] - row["arrow_back"][0] == 30
        # the mark's outline comes along in source pixels, for the overlay
        expected = marks.contour_to_source(marks.contours[0])
        assert np.array_equal(row["arrow_contour"], expected)
        assert expected.reshape(-1, 2).max(axis=0).tolist() <= [canvas.shape[1], canvas.shape[0]]

    def test_a_cortex_scar_is_not_tried(self):
        canvas, surface, scar = _drawing()
        metrics = _metrics(surface, scar, is_cortex=True)
        with patch(DETECT, return_value=_result("triangle")) as detect:
            integrate_arrows({"parents": [surface], "children": [scar]}, metrics, canvas, 300.0)
        assert not detect.called and "has_arrow" not in metrics[1]

    def test_the_notch_method_beats_the_model_when_both_find_an_arrow(self):
        canvas, surface, scar = _drawing()
        cv2.line(canvas, (60, 100), (90, 100), 255, 2)   # a second mark
        metrics = _metrics(surface, scar)
        answers = iter([_result("model", 0.9, tip=(70, 80)), _result("triangle", tip=(95, 80))])
        with patch(DETECT, side_effect=lambda *a: next(answers)):
            integrate_arrows({"parents": [surface], "children": [scar]}, metrics, canvas, 300.0)
        marks = marks_in_scar(scar, canvas, 1)
        assert metrics[1]["arrow_tip"] == marks.to_source((95, 80))
        assert "arrow_score" not in metrics[1]

    def test_no_arrow_leaves_the_scar_unchanged(self):
        canvas, surface, scar = _drawing()
        metrics = _metrics(surface, scar)
        with patch(DETECT, return_value=None):
            integrate_arrows({"parents": [surface], "children": [scar]}, metrics, canvas, 300.0)
        assert "has_arrow" not in metrics[1]

    def test_empty_input_returns_empty_list(self):
        result = integrate_arrows(
            {"parents": [], "children": []}, [], np.zeros((100, 100), dtype=np.uint8), 300.0,
        )
        assert result == []


@pytest.mark.unit
class TestRecognisedScars:

    def test_scars_are_matched_to_their_contours_by_area(self):
        canvas, surface, scar = _drawing()
        metrics = _metrics(surface, scar)
        pairs = recognised_scars(metrics, [scar], 1.0)
        assert len(pairs) == 1 and pairs[0][1]["scar"] == "scar 1"
        assert np.array_equal(pairs[0][0], scar)

    def test_a_scar_without_a_contour_is_dropped(self):
        canvas, surface, scar = _drawing()
        metrics = _metrics(surface, scar)
        assert recognised_scars(metrics, [], 1.0) == []

    def test_the_integration_config_has_only_the_match_tolerance(self):
        from pylithics.image_processing.config import get_arrow_integration_config
        assert set(get_arrow_integration_config({})) == {'area_match_tolerance'}
