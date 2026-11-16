"""Tests for arrow detection: the notch method first, the classifier on what it rejects."""

import os
import tempfile

import cv2
import numpy as np
import pytest

from pylithics.forest import Model, Tree
from pylithics.image_processing.modules import arrow_profile as ap
from pylithics.image_processing.modules.arrow_detection import (
    REFERENCE_THRESHOLDS,
    ArrowDetector,
    analyze_child_contour_for_arrow,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _poly(*pts):
    return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)


def _drawn_arrow(scale=1.0, hollow=False):
    """
    An arrow pointing up as an illustrator draws it, and the binary it was drawn on.

    At scale 1 the head is 24 px wide with notches 10 px deep, which the
    notch method accepts. At scale 0.15 the arrow is 12 px long and the
    notches are under a pixel, which it rejects.
    """
    size = int(120 * scale)
    canvas = np.zeros((size, size), np.uint8)

    def at(x, y):
        return (int(round(x * scale)), int(round(y * scale)))

    if hollow:
        outline = [at(60, 15), at(72, 40), at(63, 40), at(63, 90), at(57, 90), at(57, 40),
                   at(48, 40)]
        cv2.polylines(canvas, [np.array(outline, np.int32)], True, 255, 1)
    else:
        cv2.line(canvas, at(60, 90), at(60, 35), 255, max(1, int(round(3 * scale))))
        cv2.fillPoly(canvas, [np.array([at(60, 15), at(72, 40), at(48, 40)], np.int32)], 255)
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea), canvas


def _ring(with_mark=False):
    """A closed outline like a scar, empty or with a stroke drawn inside it."""
    canvas = np.zeros((60, 60), np.uint8)
    cv2.polylines(canvas, [np.array([[10, 10], [50, 14], [46, 50], [12, 46]], np.int32)],
                  True, 255, 1)
    if with_mark:
        cv2.line(canvas, (25, 30), (35, 30), 255, 1)
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea), canvas


def _model_on(feature: str, split: float, below: float, above: float) -> Model:
    """A one-tree forest that scores ``above`` when the feature exceeds the split."""
    index = ap.FEATURES.index(feature)
    tree = Tree(
        left=np.array([1, -1, -1]), right=np.array([2, -1, -1]),
        feature=np.array([index, 0, 0]), threshold=np.array([split, 0.0, 0.0]),
        value=np.array([0.0, below, above]),
    )
    return Model(features=list(ap.FEATURES), trees=[tree])


def _constant_model(score: float) -> Model:
    return _model_on('length', -1.0, score, score)


_BIG, _BIG_INK = _drawn_arrow()
_SMALL, _SMALL_INK = _drawn_arrow(0.15)
_BAR = _poly([10, 10], [70, 10], [70, 18], [10, 18])
_NO_MODEL = {'classifier': {'enabled': False}}
_HEAD_MODEL = _model_on('head_ratio', 1.5, 0.1, 0.9)

# The notch method's own fixture: a tall triangle head on a narrow rectangular shaft.
_ARROW_CONTOUR = _poly(
    [50, 10],   # tip
    [70, 30],   # right head
    [60, 30],   # right shoulder
    [60, 60],   # right shaft
    [40, 60],   # left shaft
    [40, 30],   # left shoulder
    [30, 30],   # left head
)


def _with_model(model: Model, threshold: float = 0.5) -> ArrowDetector:
    return ArrowDetector({'classifier': {'threshold': threshold}}, model=model)


# ---------------------------------------------------------------------------
# ArrowDetector initialization
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestArrowDetectorInit:

    def test_default_config_loads_reference_values(self):
        detector = ArrowDetector(_NO_MODEL)
        assert detector.reference_dpi == 300.0
        assert detector.debug_enabled is False
        assert set(detector.ref_thresholds) >= {
            "min_area", "min_defect_depth", "solidity_bounds",
            "min_triangle_height", "min_significant_defects",
        }
        assert detector.ref_thresholds == REFERENCE_THRESHOLDS

    def test_custom_config_overrides_reference_dpi_and_debug(self):
        detector = ArrowDetector({
            "reference_dpi": 150.0,
            "debug_enabled": True,
            **_NO_MODEL,
        })
        assert detector.reference_dpi == 150.0
        assert detector.debug_enabled is True


# ---------------------------------------------------------------------------
# scale_parameters_for_dpi: quadratic area scaling, linear depth scaling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestScaleParametersForDpi:

    def test_returns_reference_thresholds_for_matching_dpi(self):
        detector = ArrowDetector(_NO_MODEL)
        params = detector.scale_parameters_for_dpi(300.0)
        # At reference DPI, min_area is ref * 1.0 * safety_factor (0.7)
        assert params["min_area"] == pytest.approx(
            detector.ref_thresholds["min_area"] * 0.7, abs=1e-6
        )
        assert params["solidity_bounds"] == detector.ref_thresholds["solidity_bounds"]

    def test_area_scales_quadratically_with_dpi(self):
        detector = ArrowDetector(_NO_MODEL)
        params_300 = detector.scale_parameters_for_dpi(300.0)
        params_600 = detector.scale_parameters_for_dpi(600.0)
        # Doubling DPI should quadruple the min_area threshold
        assert params_600["min_area"] == pytest.approx(
            params_300["min_area"] * 4.0, rel=0.01
        )

    def test_defect_depth_scales_linearly_with_dpi(self):
        detector = ArrowDetector(_NO_MODEL)
        params_300 = detector.scale_parameters_for_dpi(300.0)
        params_600 = detector.scale_parameters_for_dpi(600.0)
        assert params_600["min_defect_depth"] == pytest.approx(
            params_300["min_defect_depth"] * 2.0, rel=0.01
        )

    @pytest.mark.parametrize("invalid_dpi", [None, 0.0, -100.0])
    def test_invalid_dpi_returns_copy_of_reference_thresholds(self, invalid_dpi):
        detector = ArrowDetector(_NO_MODEL)
        params = detector.scale_parameters_for_dpi(invalid_dpi)
        assert params == detector.ref_thresholds


# ---------------------------------------------------------------------------
# Basic filtering: tiny or non-arrow contours are rejected
# ---------------------------------------------------------------------------


    def test_the_working_copy_factor_scales_like_a_higher_dpi(self):
        detector = ArrowDetector(_NO_MODEL)
        with_factor = detector.scale_parameters_for_dpi(300.0, 2)
        assert with_factor == detector.scale_parameters_for_dpi(600.0)

    def test_without_a_dpi_the_factor_still_scales_from_the_reference(self):
        detector = ArrowDetector(_NO_MODEL)
        with_factor = detector.scale_parameters_for_dpi(None, 4)
        assert with_factor == detector.scale_parameters_for_dpi(1200.0)


@pytest.mark.unit
class TestValidateBasicProperties:

    def test_tiny_contour_fails_area_check(self):
        detector = ArrowDetector(_NO_MODEL)
        params = detector.scale_parameters_for_dpi(300.0)
        tiny = _poly([0, 0], [1, 0], [0, 1])
        assert detector._validate_basic_properties(tiny, params, None) is False

    def test_arrow_contour_passes_basic_validation(self):
        detector = ArrowDetector(_NO_MODEL)
        params = detector.scale_parameters_for_dpi(300.0)
        assert detector._validate_basic_properties(_ARROW_CONTOUR, params, None) is True


# ---------------------------------------------------------------------------
# _calculate_arrow_properties: verify the angle math directly
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCalculateArrowProperties:
    """The angle math here is load-bearing for downstream archaeological outputs."""

    def _properties_for_tip_and_base(self, tip, base_p1, base_p2):
        detector = ArrowDetector(_NO_MODEL)
        base_midpoint = ((base_p1[0] + base_p2[0]) // 2,
                         (base_p1[1] + base_p2[1]) // 2)
        return detector._calculate_arrow_properties({
            "base_p1": base_p1,
            "base_p2": base_p2,
            "base_midpoint": base_midpoint,
            "triangle_tip": tip,
            "triangle_height": 20.0,
            "significant_defects": [],
        }, None)

    def test_arrow_back_is_triangle_tip_arrow_tip_is_base_midpoint(self):
        """The *named* arrow tip is the midpoint of the scar's base edge."""
        result = self._properties_for_tip_and_base((20, 10), (10, 30), (30, 30))
        assert result["arrow_back"] == (20, 10)
        assert result["arrow_tip"] == (20, 30)

    def test_downward_arrow_yields_compass_angle_zero(self):
        """
        Arrow from tip (20,10) to base midpoint (20,30) points +y in image
        coords. The code maps that to compass angle 0 (north in its rotated
        frame).
        """
        result = self._properties_for_tip_and_base((20, 10), (10, 30), (30, 30))
        assert result["angle_deg"] == pytest.approx(90.0, abs=1.0)
        assert result["compass_angle"] == pytest.approx(0.0, abs=1.0)

    def test_rightward_arrow_maps_to_compass_two_seventy(self):
        """
        Rightward in image coords (dx>0, dy=0) maps to compass 270 because the
        code rotates the frame by 270°. Downward = 0 and rightward = 270 fully
        pin the rotation direction.
        """
        result = self._properties_for_tip_and_base((10, 20), (30, 10), (30, 30))
        assert result["angle_deg"] == pytest.approx(0.0, abs=1.0)
        assert result["compass_angle"] == pytest.approx(270.0, abs=1.0)


# ---------------------------------------------------------------------------
# Stage 1 decides first; its acceptance is never overruled, its direction can be
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStageOne:

    def test_a_notched_arrow_is_accepted_whatever_the_model_says(self):
        result = _with_model(_constant_model(0.0)).analyze_contour_for_arrow(_BIG, {}, _BIG_INK)
        assert result is not None
        assert result['stage'] == 'triangle'
        assert 'score' not in result

    def test_the_classifier_turns_the_arrow_when_it_is_sure_of_the_other_direction(self):
        # The notch method points the drawn up-arrow down (compass 0); the head model
        # scores the upward reading 0.9 against 0.1, so the arrow is turned.
        plain = ArrowDetector(_NO_MODEL).assess(_BIG, _BIG_INK, 300.0)
        turned = _with_model(_HEAD_MODEL).assess(_BIG, _BIG_INK, 300.0)
        assert plain.stage == 'triangle' and plain.geometry['compass_angle'] == 0.0
        assert turned.stage == 'triangle turned' and turned.accepted
        assert turned.geometry['compass_angle'] == pytest.approx(180.0, abs=5.0)
        assert 'score' not in turned.result()

    def test_a_doubtful_classifier_leaves_the_direction_alone(self):
        both_equal = _with_model(_constant_model(0.9)).assess(_BIG, _BIG_INK, 300.0)
        too_low = ArrowDetector({'classifier': {'direction_threshold': 0.95}},
                                model=_HEAD_MODEL).assess(_BIG, _BIG_INK, 300.0)
        assert both_equal.stage == 'triangle' and too_low.stage == 'triangle'
        assert too_low.geometry['compass_angle'] == 0.0

    def test_the_same_arrow_is_accepted_without_a_model(self):
        result = ArrowDetector(_NO_MODEL).analyze_contour_for_arrow(_BIG, {}, _BIG_INK)
        assert result is not None and result['stage'] == 'triangle'

    def test_rectangle_is_not_detected_as_arrow(self):
        detector = ArrowDetector(_NO_MODEL)
        rectangle = _poly([10, 10], [50, 10], [50, 40], [10, 40])
        result = detector.analyze_contour_for_arrow(
            rectangle, {"scar": "test"},
            np.zeros((100, 100), dtype=np.uint8), 300.0,
        )
        assert result is None

    @pytest.mark.parametrize("degenerate", [
        np.array([], dtype=np.int32).reshape(0, 1, 2),
        np.array([[10, 10]], dtype=np.int32).reshape(-1, 1, 2),
        np.zeros((2, 1, 2), dtype=np.int32),
    ])
    def test_degenerate_contours_return_none(self, degenerate):
        detector = ArrowDetector(_NO_MODEL)
        result = detector.analyze_contour_for_arrow(
            degenerate, {"scar": "x"},
            np.zeros((100, 100), dtype=np.uint8), 300.0,
        )
        assert result is None

    def test_assess_carries_the_structure_features_both_ways(self):
        assessment = _with_model(_HEAD_MODEL).assess(_BIG, _BIG_INK)
        assert set(assessment.forward) == set(ap.FEATURES)
        assert assessment.forward['notches'] == 2
        assert assessment.forward['notches'] == assessment.reverse['notches']
        assert assessment.forward['triangle_height'] > 0


# ---------------------------------------------------------------------------
# Stage 2: the classifier on what stage 1 rejects
# ---------------------------------------------------------------------------


    @pytest.mark.parametrize("factor", [2, 4])
    def test_on_a_working_copy_the_lengths_stay_in_source_pixels(self, factor):
        detector = ArrowDetector(_NO_MODEL)
        at_source = detector.assess(_SMALL, _SMALL_INK, 300.0)
        big = cv2.resize(_SMALL_INK, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        on_copy = detector.assess(max(contours, key=cv2.contourArea), big, 300.0, factor)
        assert abs(on_copy.forward['length'] - at_source.forward['length']) < 0.5
        assert abs(on_copy.forward['aspect'] - at_source.forward['aspect']) < 0.5
        tip = on_copy.geometry['arrow_tip']
        assert abs(tip[0] / factor - at_source.geometry['arrow_tip'][0]) <= 1


@pytest.mark.unit
class TestStageTwo:

    def test_a_small_arrow_is_rejected_by_stage_one_and_found_by_the_model(self):
        result = _with_model(_HEAD_MODEL).analyze_contour_for_arrow(_SMALL, {}, _SMALL_INK)
        assert result is not None
        assert result['stage'] == 'model' and result['score'] == 0.9
        assert result['compass_angle'] == pytest.approx(180, abs=5)

    def test_the_threshold_is_respected(self):
        detector = _with_model(_model_on('head_ratio', 1.5, 0.1, 0.6), threshold=0.7)
        assert detector.analyze_contour_for_arrow(_SMALL, {}, _SMALL_INK) is None

    def test_the_higher_score_gives_the_direction(self):
        # tail_end is ~1 with the tail first and small with the head first;
        # rewarding a large tail_end picks the orientation with the tail first.
        detector = _with_model(_model_on('tail_end', 0.6, 0.1, 0.9))
        result = detector.analyze_contour_for_arrow(_SMALL, {}, _SMALL_INK)
        assert result is not None and result['compass_angle'] == pytest.approx(180, abs=5)

    def test_a_bar_is_not_an_arrow(self):
        assert _with_model(_HEAD_MODEL).analyze_contour_for_arrow(_BAR, {}, None) is None

    def test_without_a_model_stage_one_alone_decides(self):
        detector = ArrowDetector(_NO_MODEL)
        assert detector.analyze_contour_for_arrow(_SMALL, {}, _SMALL_INK) is None
        assessment = detector.assess(_SMALL, _SMALL_INK)
        assert assessment.stage == 'no model' and assessment.scores == (None, None)

    def test_a_missing_model_file_is_reported_and_stage_one_decides(self, caplog):
        detector = ArrowDetector({'classifier': {'enabled': True, 'model': 'absent.json'}})
        assert detector.model is None
        assert 'notch method alone' in caplog.text
        assert detector.analyze_contour_for_arrow(_BIG, {}, _BIG_INK) is not None
        assert detector.analyze_contour_for_arrow(_SMALL, {}, _SMALL_INK) is None

    def test_the_bundled_model_knows_a_small_arrow_from_a_bar(self):
        detector = ArrowDetector({'classifier': {'enabled': True, 'threshold': 0.5}})
        assert detector.model is not None
        arrow = detector.analyze_contour_for_arrow(_SMALL, {}, _SMALL_INK)
        assert arrow is not None and arrow['compass_angle'] == pytest.approx(180, abs=5)
        assert detector.analyze_contour_for_arrow(_BAR, {}, None) is None


# ---------------------------------------------------------------------------
# The vetoes before scoring
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVetoes:

    def test_a_mark_with_ink_inside_is_refused(self):
        contour, ink = _ring(with_mark=True)
        assessment = _with_model(_constant_model(0.9)).assess(contour, ink)
        assert assessment.stage == 'ink inside' and not assessment.accepted

    def test_an_empty_outline_reaches_the_model(self):
        contour, ink = _ring()
        assessment = _with_model(_constant_model(0.9)).assess(contour, ink)
        assert assessment.stage == 'model'

    def test_a_hollow_arrow_is_not_refused(self):
        contour, ink = _drawn_arrow(0.2, hollow=True)
        assessment = _with_model(_constant_model(0.9)).assess(contour, ink)
        assert assessment.stage == 'model' and assessment.accepted

    def test_a_speck_is_too_short(self):
        canvas = np.zeros((20, 20), np.uint8)
        cv2.rectangle(canvas, (8, 8), (11, 10), 255, -1)
        contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        assessment = _with_model(_constant_model(0.9)).assess(contours[0], canvas)
        assert assessment.stage == 'too short' and not assessment.accepted

    def test_without_the_image_the_ink_inside_veto_cannot_apply(self):
        contour, _ = _ring(with_mark=True)
        assessment = _with_model(_constant_model(0.9)).assess(contour)
        assert assessment.stage == 'model'


# ---------------------------------------------------------------------------
# Wrapper and debug output
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_backward_compat_wrapper_delegates_to_class_method():
    """analyze_child_contour_for_arrow is a thin wrapper around the class."""
    via_wrapper = analyze_child_contour_for_arrow(_BIG, {"scar": "test"}, _BIG_INK, 300.0)
    via_class = ArrowDetector().analyze_contour_for_arrow(_BIG, {"scar": "test"}, _BIG_INK, 300.0)
    assert via_wrapper == via_class


@pytest.mark.unit
class TestDebugOutput:

    def test_debug_files_are_written_for_both_stages(self, tmp_path):
        detector = ArrowDetector({**_NO_MODEL, 'debug_enabled': True})
        for name, contour, ink in (('big', _BIG, _BIG_INK), ('small', _SMALL, _SMALL_INK)):
            entry = {'scar': f'scar {name}', 'debug_dir': str(tmp_path / 'lithic')}
            detector.analyze_contour_for_arrow(contour, entry, ink)
            assert os.path.isfile(tmp_path / 'lithic' / f'scar_{name}.png')
            text = (tmp_path / 'lithic' / f'scar_{name}.txt').read_text()
            assert 'Scaled parameters' in text and 'head_ratio=' in text
        big = (tmp_path / 'lithic' / 'scar_big.txt').read_text()
        small = (tmp_path / 'lithic' / 'scar_small.txt').read_text()
        assert 'arrow: yes; stage triangle' in big
        assert 'arrow: no; stage no model' in small

    def test_debug_visualization_writes_file_when_successful(self):
        detector = ArrowDetector({"debug_enabled": True, **_NO_MODEL})
        with tempfile.TemporaryDirectory() as temp_dir:
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            triangle_data = {
                "base_p1": (10, 30),
                "base_p2": (30, 30),
                "base_midpoint": (20, 30),
                "triangle_tip": (20, 10),
                "significant_defects": [
                    ((10, 30), (30, 30), (15, 25), 5.0),
                ],
            }
            arrow_data = {
                "arrow_back": (20, 10),
                "arrow_tip": (20, 30),
                "compass_angle": 0.0,
            }
            prefix = os.path.join(temp_dir, "flake", "scar_3")
            detector._create_debug_visualizations(
                _ARROW_CONTOUR, triangle_data, arrow_data, image, prefix,
            )
            assert os.path.exists(os.path.join(temp_dir, "flake", "scar_3.png"))


@pytest.mark.unit
def test_module_function_follows_the_configuration(tmp_path):
    from pylithics.image_processing.config import get_config_manager
    manager = get_config_manager()
    manager.update_value('arrow_detection', 'debug_enabled', True)
    try:
        entry = {'scar': 'scar 1', 'debug_dir': str(tmp_path / 'lithic')}
        result = analyze_child_contour_for_arrow(_BIG, entry, _BIG_INK)
    finally:
        manager.update_value('arrow_detection', 'debug_enabled', False)
    assert result is not None and os.path.isfile(tmp_path / 'lithic' / 'scar_1.png')
