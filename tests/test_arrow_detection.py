"""Tests for arrow detection: the detector's decision, with and without a model."""

import os

import cv2
import numpy as np
import pytest

from pylithics.forest import Model, Tree
from pylithics.image_processing.modules import arrow_profile as ap
from pylithics.image_processing.modules.arrow_detection import (
    ArrowDetector,
    analyze_child_contour_for_arrow,
)


def _poly(*pts):
    return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)


def _drawn_arrow():
    """An arrow pointing up as an illustrator draws it: a stroke with a filled head."""
    canvas = np.zeros((120, 120), np.uint8)
    cv2.line(canvas, (60, 90), (60, 35), 255, 3)
    cv2.fillPoly(canvas, [np.array([[60, 15], [72, 40], [48, 40]], np.int32)], 255)
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea)


_ARROW = _drawn_arrow()
_BAR = _poly([10, 10], [70, 10], [70, 18], [10, 18])
_NO_MODEL = {'classifier': {'enabled': False}}


def _model_on(feature: str, split: float, below: float, above: float) -> Model:
    """A one-tree forest that scores ``above`` when the feature exceeds the split."""
    index = ap.FEATURES.index(feature)
    tree = Tree(
        left=np.array([1, -1, -1]), right=np.array([2, -1, -1]),
        feature=np.array([index, 0, 0]), threshold=np.array([split, 0.0, 0.0]),
        value=np.array([0.0, below, above]),
    )
    return Model(features=list(ap.FEATURES), trees=[tree])


@pytest.mark.unit
class TestRuleDetector:
    def test_an_arrow_is_found_and_points_up(self):
        result = ArrowDetector(_NO_MODEL).analyze_contour_for_arrow(_ARROW, {}, None)
        assert result is not None
        assert result['compass_angle'] == pytest.approx(180, abs=3)
        assert result['arrow_tip'][1] < result['arrow_back'][1]
        assert result['score'] is None

    def test_a_bar_is_not_an_arrow(self):
        assert ArrowDetector(_NO_MODEL).analyze_contour_for_arrow(_BAR, {}, None) is None

    def test_a_degenerate_contour_gives_none(self):
        detector = ArrowDetector(_NO_MODEL)
        speck = np.zeros((2, 1, 2), dtype=np.int32)
        assert detector.analyze_contour_for_arrow(speck, {}, None) is None

    def test_assess_reports_both_orientations(self):
        assessment = ArrowDetector(_NO_MODEL).assess(_ARROW)
        assert set(assessment.forward) == set(ap.FEATURES)
        assert assessment.scores == (None, None)
        assert assessment.accepted and assessment.rule[1 if assessment.reversed else 0]


@pytest.mark.unit
class TestModelDetector:
    def test_the_model_decides_and_scores(self):
        model = _model_on('head_ratio', 1.5, 0.1, 0.9)
        detector = ArrowDetector({'classifier': {'threshold': 0.5}}, model=model)
        result = detector.analyze_contour_for_arrow(_ARROW, {}, None)
        assert result is not None and result['score'] == 0.9
        assert detector.analyze_contour_for_arrow(_BAR, {}, None) is None

    def test_the_threshold_is_respected(self):
        model = _model_on('head_ratio', 1.5, 0.1, 0.6)
        detector = ArrowDetector({'classifier': {'threshold': 0.7}}, model=model)
        assert detector.analyze_contour_for_arrow(_ARROW, {}, None) is None

    def test_the_higher_score_gives_the_direction(self):
        # tail_end is ~1 with the tail first and small with the head first;
        # rewarding a small tail_end picks the orientation with the head first.
        model = _model_on('tail_end', 0.6, 0.9, 0.1)
        detector = ArrowDetector({'classifier': {'threshold': 0.5}}, model=model)
        result = detector.analyze_contour_for_arrow(_ARROW, {}, None)
        assert result is not None and result['compass_angle'] == pytest.approx(0, abs=3)

    def test_the_bundled_model_knows_an_arrow_from_a_bar(self):
        detector = ArrowDetector({'classifier': {'enabled': True, 'threshold': 0.5}})
        assert detector.model is not None
        arrow = detector.analyze_contour_for_arrow(_ARROW, {}, None)
        assert arrow is not None and arrow['compass_angle'] == pytest.approx(180, abs=3)
        assert detector.analyze_contour_for_arrow(_BAR, {}, None) is None

    def test_a_missing_model_falls_back_to_the_rule(self, caplog):
        detector = ArrowDetector({'classifier': {'enabled': True, 'model': 'absent.json'}})
        assert detector.model is None
        assert detector.analyze_contour_for_arrow(_ARROW, {}, None) is not None


@pytest.mark.unit
class TestDebugOutput:
    def test_debug_files_are_written(self, tmp_path):
        detector = ArrowDetector({**_NO_MODEL, 'debug_enabled': True})
        image = np.full((80, 100), 255, dtype=np.uint8)
        entry = {'scar': 'scar 1', 'debug_dir': str(tmp_path / 'lithic')}
        detector.analyze_contour_for_arrow(_ARROW, entry, image)
        assert os.path.isfile(tmp_path / 'lithic' / 'scar_1.png')
        text = (tmp_path / 'lithic' / 'scar_1.txt').read_text()
        assert text.startswith('arrow: yes') and 'head_ratio=' in text


@pytest.mark.unit
def test_module_function_follows_the_configuration(tmp_path):
    from pylithics.image_processing.config import get_config_manager
    manager = get_config_manager()
    manager.update_value('arrow_detection', 'debug_enabled', True)
    try:
        entry = {'scar': 'scar 1', 'debug_dir': str(tmp_path / 'lithic')}
        result = analyze_child_contour_for_arrow(_ARROW, entry, np.full((120, 120), 255, np.uint8))
    finally:
        manager.update_value('arrow_detection', 'debug_enabled', False)
    assert result is not None and os.path.isfile(tmp_path / 'lithic' / 'scar_1.png')
