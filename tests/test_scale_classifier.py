"""Tests for the scale bar classifier: features, scoring, candidates and the model file."""

import json
import os

import numpy as np
import pytest

from pylithics.page_segmentation import grouping, scale_classifier as sc
from pylithics.page_segmentation.detection import Component
from pylithics.page_segmentation.export import (
    SCALE_UNCERTAIN_FLAG, ManifestRow, _metadata_row,
)

PAGE = (1000, 800)


def _component(box, mask=None):
    x0, y0, x1, y1 = box
    if mask is None:
        mask = np.ones((y1 - y0, x1 - x0), dtype=bool)
    return Component(box=list(box), width=x1 - x0, height=y1 - y0,
                     ink=int(mask.sum()), mask=mask)


def _page(components, width=PAGE[0], height=PAGE[1]):
    """A raw mask and a grey page holding the components' ink."""
    raw = np.zeros((height, width), dtype=np.uint8)
    gray = np.full((height, width), 255, dtype=np.uint8)
    for component in components:
        x0, y0, x1, y1 = component.box
        raw[y0:y1, x0:x1][component.mask] = 255
        gray[y0:y1, x0:x1][component.mask] = 0
    return raw, gray


def _segmented_bar(box, blocks=5):
    """A bar of alternating filled and open blocks."""
    x0, y0, x1, y1 = box
    mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    block = (x1 - x0) // blocks
    for k in range(0, blocks, 2):
        mask[:, k * block:(k + 1) * block] = True
    return _component(box, mask)


def _ruled_scale(box, ticks=4):
    """A rule along the bottom with short ticks standing up from it."""
    x0, y0, x1, y1 = box
    mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    mask[-2:, :] = True
    for k in range(ticks):
        column = int((x1 - x0 - 1) * k / (ticks - 1))
        mask[:, max(0, column - 1):column + 2] = True
    return _component(box, mask)


def _stump(feature=None, threshold=0.0, below=0.5, above=0.5):
    """A one-split tree on ``feature``, or a single leaf worth ``below``."""
    if feature is None:
        return sc.Tree(left=np.array([-1]), right=np.array([-1]), feature=np.array([0]),
                       threshold=np.array([0.0]), value=np.array([below]))
    return sc.Tree(
        left=np.array([1, -1, -1]), right=np.array([2, -1, -1]),
        feature=np.array([sc.FEATURES.index(feature), 0, 0]),
        threshold=np.array([threshold, 0.0, 0.0]), value=np.array([0.0, below, above]),
    )


def _model(leaf=0.5, **split):
    """A forest of one tree: a single leaf, or one split given as feature=(threshold, lo, hi)."""
    if split:
        (feature, (threshold, lo, hi)), = split.items()
        return sc.Model(features=list(sc.FEATURES), trees=[_stump(feature, threshold, lo, hi)])
    return sc.Model(features=list(sc.FEATURES), trees=[_stump(below=leaf)])


@pytest.mark.unit
class TestFeatures:
    """Each feature measures what its name says."""

    def test_a_segmented_bar_has_regular_runs_and_no_ticks(self):
        bar = _segmented_bar((100, 100, 400, 112))
        raw, gray = _page([bar])
        candidate = sc.Candidate(indices=[0], box=list(bar.box))
        features = sc.candidate_features(candidate, [bar], raw, gray, *PAGE)
        assert features['runs'] == 3
        assert features['run_regularity'] < 0.1
        assert features['ticks'] == 0
        assert 0.5 < features['density'] < 0.7
        assert features['aspect'] == 25.0
        assert features['tone_variance'] == 0.0

    def test_a_ruled_scale_has_ticks_and_a_spine(self):
        rule = _ruled_scale((100, 100, 400, 120))
        raw, gray = _page([rule])
        candidate = sc.Candidate(indices=[0], box=list(rule.box))
        features = sc.candidate_features(candidate, [rule], raw, gray, *PAGE)
        assert features['ticks'] >= 3 and features['end_ticks'] == 2
        assert features['spine'] == 1.0 and features['plain_line'] == 0.0

    def test_a_plain_line_with_no_caption_is_marked(self):
        line = _component((100, 100, 400, 104))
        raw, gray = _page([line])
        candidate = sc.Candidate(indices=[0], box=list(line.box))
        features = sc.candidate_features(candidate, [line], raw, gray, *PAGE)
        assert features['plain_line'] == 1.0 and features['plain_no_caption'] == 1.0
        caption = _component((150, 108, 162, 122))
        raw, gray = _page([line, caption])
        features = sc.candidate_features(candidate, [line, caption], raw, gray, *PAGE)
        assert features['plain_no_caption'] == 0.0

    def test_a_mark_between_two_views_is_flanked(self):
        left, right = _component((100, 100, 300, 400)), _component((400, 100, 600, 400))
        dash = _component((320, 248, 380, 252))
        raw, gray = _page([left, right, dash])
        candidate = sc.Candidate(indices=[2], box=list(dash.box))
        features = sc.candidate_features(candidate, [left, right, dash], raw, gray, *PAGE)
        assert features['flanked'] == 1.0
        alone = sc.Candidate(indices=[2], box=list(dash.box))
        raw, gray = _page([dash])
        assert sc.candidate_features(alone, [dash], raw, gray, *PAGE)['flanked'] == 0.0

    def test_a_line_of_text_has_many_ink_pieces(self):
        letters = [_component((100 + 14 * k, 100, 110 + 14 * k, 116)) for k in range(12)]
        raw, gray = _page(letters)
        box = [100, 100, 264, 116]
        candidate = sc.Candidate(indices=list(range(12)), box=box)
        features = sc.candidate_features(candidate, letters, raw, gray, *PAGE)
        assert features['ink_pieces'] > np.log1p(10)

    def test_an_upright_bar_is_measured_along_its_length(self):
        bar = _segmented_bar((100, 100, 400, 112))
        upright = _component((100, 100, 112, 400), bar.mask.T.copy())
        raw, gray = _page([upright])
        candidate = sc.Candidate(indices=[0], box=list(upright.box), upright=True)
        features = sc.candidate_features(candidate, [upright], raw, gray, *PAGE)
        assert features['runs'] == 3 and features['aspect'] == 25.0

    def test_caption_marks_count_small_neighbours_only(self):
        bar = _segmented_bar((100, 100, 400, 112))
        caption = _component((150, 116, 162, 130))
        far = _component((700, 116, 712, 130))
        raw, gray = _page([bar, caption, far])
        candidate = sc.Candidate(indices=[0], box=list(bar.box))
        features = sc.candidate_features(candidate, [bar, caption, far], raw, gray, *PAGE)
        assert features['caption_marks'] == 1

    def test_a_photograph_strip_has_tone_variance(self):
        strip = _component((100, 100, 400, 140))
        raw, gray = _page([strip])
        gray[100:140, 100:400] = np.random.RandomState(0).randint(0, 255, (40, 300))
        candidate = sc.Candidate(indices=[0], box=list(strip.box))
        features = sc.candidate_features(candidate, [strip], raw, gray, *PAGE)
        assert features['tone_variance'] > 0.2


@pytest.mark.unit
class TestScoring:
    def test_a_single_leaf_gives_its_value(self):
        assert sc.score({name: 3.0 for name in sc.FEATURES}, _model(leaf=0.7)) == 0.7

    def test_a_split_sends_each_side_to_its_leaf(self):
        model = _model(runs=(2.5, 0.1, 0.9))
        assert sc.score({'runs': 5.0}, model) == 0.9 and sc.score({'runs': 1.0}, model) == 0.1

    def test_the_forest_averages_its_trees(self):
        model = sc.Model(features=list(sc.FEATURES), trees=[_stump(below=0.2), _stump(below=0.6)])
        assert sc.score({}, model) == pytest.approx(0.4)

    def test_score_candidates_marks_those_at_the_threshold(self):
        bar = _segmented_bar((100, 100, 400, 112))
        raw, gray = _page([bar])
        candidates = [sc.Candidate(indices=[0], box=list(bar.box))]
        sc.score_candidates(candidates, [bar], raw, gray, *PAGE, _model(leaf=0.5), threshold=0.5)
        assert candidates[0].accepted and candidates[0].score == pytest.approx(0.5)
        sc.score_candidates(candidates, [bar], raw, gray, *PAGE, _model(leaf=0.5), threshold=0.51)
        assert not candidates[0].accepted


@pytest.mark.unit
class TestModelFile:
    def test_the_bundled_model_knows_a_bar_from_a_dash(self):
        """A segmented bar is accepted; a short plain dash between two views is not."""
        model = sc.load_model()
        bar = _segmented_bar((100, 700, 400, 712))
        raw, gray = _page([bar])
        accepted = [sc.Candidate([0], list(bar.box))]
        sc.score_candidates(accepted, [bar], raw, gray, *PAGE, model, 0.4)
        assert accepted[0].accepted and accepted[0].score >= 0.4
        left, right = _component((100, 100, 300, 400)), _component((400, 100, 600, 400))
        dash = _component((320, 248, 380, 252))
        raw, gray = _page([left, right, dash])
        rejected = [sc.Candidate([2], list(dash.box))]
        sc.score_candidates(rejected, [left, right, dash], raw, gray, *PAGE, model, 0.4)
        assert not rejected[0].accepted and rejected[0].rejected_by == 'rule between views'

    def test_round_trip(self, tmp_path):
        path = tmp_path / 'model.json'
        path.write_text(json.dumps({
            'features': sc.FEATURES,
            'trees': [{'left': [1, -1, -1], 'right': [2, -1, -1], 'feature': [2, 0, 0],
                       'threshold': [5.0, 0.0, 0.0], 'value': [0.0, 0.1, 0.9]}],
        }))
        model = sc.load_model(str(path))
        assert len(model.trees) == 1 and sc.score({'aspect': 10.0}, model) == 0.9

    def test_a_missing_file_is_reported(self, tmp_path):
        with pytest.raises(sc.ModelUnavailable):
            sc.load_model(str(tmp_path / 'absent.json'))

    def test_an_inconsistent_file_is_reported(self, tmp_path):
        path = tmp_path / 'model.json'
        path.write_text(json.dumps({'features': sc.FEATURES, 'trees': [
            {'left': [1, -1], 'right': [2, -1, -1], 'feature': [0, 0, 0],
             'threshold': [0.0, 0.0, 0.0], 'value': [0.0, 0.1, 0.9]}]}))
        with pytest.raises(sc.ModelUnavailable):
            sc.load_model(str(path))

    def test_the_bundled_model_loads(self):
        assert os.path.isfile(os.path.join(sc.MODELS_DIR, sc.DEFAULT_MODEL))
        model = sc.load_model()
        assert model.features == sc.FEATURES and len(model.trees) >= 10


@pytest.mark.unit
class TestCandidates:
    """The candidate step is wider than the rules but never admits a drawing."""

    def test_a_thin_mark_is_a_candidate_and_a_square_is_not(self):
        bar = _component((100, 100, 300, 110))
        square = _component((500, 100, 600, 200))
        found = grouping.find_scale_candidates([bar, square], *PAGE, set())
        assert [c.indices for c in found] == [[0]]

    def test_an_upright_mark_is_a_candidate(self):
        bar = _component((100, 100, 110, 300))
        found = grouping.find_scale_candidates([bar], *PAGE, set())
        assert len(found) == 1 and found[0].upright

    def test_collinear_blocks_form_one_candidate(self):
        blocks = [_component((100 + 60 * k, 100, 130 + 60 * k, 108)) for k in range(4)]
        found = grouping.find_scale_candidates(blocks, *PAGE, set())
        assert len(found) == 1 and found[0].indices == [0, 1, 2, 3]
        assert found[0].box == [100, 100, 310, 108]

    def test_what_the_rules_accept_is_a_candidate_too(self):
        odd = _component((100, 100, 140, 120))      # aspect 2, below the limit
        found = grouping.find_scale_candidates([odd], *PAGE, {0})
        assert len(found) == 1 and found[0].by_rules

    def test_a_mark_touching_a_drawing_is_not_a_candidate(self):
        drawing = _component((100, 100, 400, 400))
        mark = _component((380, 390, 600, 400))
        assert grouping.find_scale_candidates([drawing, mark], *PAGE, set()) == []


@pytest.mark.unit
class TestGroupingWithTheClassifier:
    """With a context, the classifier decides the bars and the context keeps the results."""

    CONFIG = {'grouping': {'min_area': 0.0004}, 'scale_bars': {'enabled': True}}

    def _run(self, leaf):
        bar = _segmented_bar((100, 700, 400, 712))
        lithic = _component((100, 100, 400, 400))
        raw, gray = _page([bar, lithic])
        context = grouping.ScaleContext(raw=raw, gray=gray, model=_model(leaf=leaf), threshold=0.5)
        boxes, bars = grouping.group_components([bar, lithic], PAGE, self.CONFIG, context)
        return boxes, bars, context

    def test_an_accepting_model_exports_the_bar_with_its_score(self):
        boxes, bars, context = self._run(leaf=0.95)
        assert bars == [[100, 700, 400, 712]] and len(boxes) == 1
        assert len(context.candidates) == 1 and context.bar_scores[0] > 0.9

    def test_a_rejecting_model_leaves_no_bar(self):
        boxes, bars, context = self._run(leaf=0.05)
        assert bars == [] and not context.candidates[0].accepted

    def test_without_a_context_the_rules_decide(self):
        bar = _segmented_bar((100, 700, 400, 712))
        boxes, bars = grouping.group_components([bar, _component((100, 100, 400, 400))],
                                                PAGE, self.CONFIG)
        assert len(bars) <= 1


@pytest.mark.unit
class TestOutputs:
    def test_an_uncertain_scale_flags_its_artefacts(self):
        row = ManifestRow('p_box_01.png', 'artefact', 'p.png', '1', 0, 0, 10, 10, 10, 10,
                          '300', 'RGB', 1, '')
        plain = _metadata_row(row, ['p_scale_bar.png'], set())
        flagged = _metadata_row(row, ['p_scale_bar.png'], {'p_scale_bar.png'})
        assert plain['flag'] == ''
        assert flagged['flag'] == SCALE_UNCERTAIN_FLAG


def _raw(components, width=PAGE[0], height=PAGE[1]):
    """Paint the components' masks onto a blank page as the raw ink mask."""
    raw = np.zeros((height, width), dtype=np.uint8)
    for component in components:
        x0, y0, x1, y1 = component.box
        raw[y0:y1, x0:x1][component.mask] = 255
    return raw


def _hollow(box):
    """A drawing whose ink is only its outline, so its box spans empty paper."""
    x0, y0, x1, y1 = box
    mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    mask[[0, -1], :] = True
    mask[:, [0, -1]] = True
    return _component(box, mask)


@pytest.mark.unit
class TestCandidatesByInk:
    """With the raw ink, a bar is judged clear by its ink, not by the boxes around it."""

    def test_a_bar_under_a_profile_within_its_box_is_a_candidate(self):
        drawing = _hollow((100, 100, 400, 400))    # an L-shaped outline spans the bar
        bar = _component((150, 360, 350, 370))
        raw = _raw([drawing, bar])
        raw[360:370, 150:350] = 255
        raw[101:399, 101:399] = 0                   # the outline's interior is paper
        raw[360:370, 150:350] = 255
        found = grouping.find_scale_candidates([drawing, bar], *PAGE, set(), None, raw)
        assert [c.indices for c in found] == [[1]] and found[0].enclosed

    def test_without_the_ink_the_box_rule_still_holds(self):
        drawing = _hollow((100, 100, 400, 400))
        bar = _component((150, 360, 350, 370))
        assert grouping.find_scale_candidates([drawing, bar], *PAGE, set()) == []

    def test_a_bar_whose_ink_meets_the_drawing_is_not_a_candidate(self):
        drawing = _component((100, 100, 400, 400))
        mark = _component((380, 390, 600, 400))
        raw = _raw([drawing, mark])
        assert grouping.find_scale_candidates([drawing, mark], *PAGE, set(), None, raw) == []

    def test_a_clear_bar_is_not_enclosed(self):
        bar = _component((100, 100, 300, 110))
        found = grouping.find_scale_candidates([bar], *PAGE, set(), None, _raw([bar]))
        assert len(found) == 1 and not found[0].enclosed

    def test_upright_dashes_on_one_line_form_one_candidate(self):
        dashes = [_component((100, 100 + 40 * k, 104, 125 + 40 * k)) for k in range(4)]
        found = grouping.find_scale_candidates(dashes, *PAGE, set())
        assert len(found) == 1 and found[0].upright and found[0].indices == [0, 1, 2, 3]
        assert found[0].box == [100, 100, 104, 245]

    def test_a_lone_short_dash_is_not_a_candidate(self):
        dash = _component((100, 100, 104, 120))     # 20 px: a quarter of the length limit
        assert grouping.find_scale_candidates([dash], *PAGE, set()) == []

    def test_the_short_side_limit_follows_the_long_side_of_a_strip(self):
        bar = _component((100, 10, 500, 60))        # 50 px thick on a 1200x200 strip
        found = grouping.find_scale_candidates([bar], 1200, 200, set())
        assert len(found) == 1


@pytest.mark.unit
class TestShapeAlong:
    """Solid, straight and taper describe a bar's cross-section along its length."""

    def test_a_solid_bar_is_solid_straight_and_square_ended(self):
        ink = np.zeros((10, 100), dtype=bool)
        ink[3:7, :] = True
        assert sc._shape_along(ink) == (1.0, 1.0, 1.0)

    def test_an_outline_is_hollow(self):
        ink = np.zeros((10, 100), dtype=bool)
        ink[[2, 7], :] = True
        solid, straight, taper = sc._shape_along(ink)
        assert solid < 0.1 and straight == 1.0

    def test_a_tapered_outline_is_neither_straight_nor_square_ended(self):
        ink = np.zeros((40, 100), dtype=bool)
        for column in range(100):
            half = 1 + 18 * min(column, 99 - column) // 50
            ink[20 - half, column] = ink[20 + half, column] = True
        solid, straight, taper = sc._shape_along(ink)
        assert straight < 0.7 and taper < 0.5

    def test_too_little_ink_measures_nothing(self):
        assert sc._shape_along(np.zeros((5, 5), dtype=bool)) == (0.0, 0.0, 0.0)


@pytest.mark.unit
class TestConnectorRule:
    """A plain uncaptioned stroke at a drawing's edge or between two views is never scored."""

    PLAIN = {'ticks': 0.0, 'caption_marks': 0.0, 'runs': 1.0, 'enclosed': 0.0, 'flanked': 0.0}

    def test_a_tick_at_a_drawing(self):
        assert sc._connector_rule({**self.PLAIN, 'enclosed': 1.0}) == 'tick at a drawing'

    def test_a_rule_between_views(self):
        features = {**self.PLAIN, 'flanked': 1.0, 'runs': 2.0}
        assert sc._connector_rule(features) == 'rule between views'

    def test_a_segmented_bar_at_a_drawing_is_kept(self):
        assert sc._connector_rule({**self.PLAIN, 'enclosed': 1.0, 'runs': 2.0}) == ''

    def test_a_captioned_or_ticked_mark_is_kept(self):
        assert sc._connector_rule({**self.PLAIN, 'enclosed': 1.0, 'caption_marks': 1.0}) == ''
        assert sc._connector_rule({**self.PLAIN, 'flanked': 1.0, 'ticks': 2.0}) == ''

    def test_a_rejected_candidate_scores_zero(self):
        drawing = _component((100, 100, 400, 400))
        tick = _component((398, 150, 402, 250))      # a plain stroke at the drawing's edge
        raw = _raw([drawing, tick])
        raw[100:400, 100:400] = 0
        raw[150:250, 398:402] = 255
        raw[100:400, 100:101] = 255                  # the drawing keeps a far edge of ink
        found = grouping.find_scale_candidates([drawing, tick], *PAGE, set(), None, raw)
        assert len(found) == 1
        sc.score_candidates(found, [drawing, tick], raw, np.full((PAGE[1], PAGE[0]), 255, np.uint8),
                            *PAGE, _model(leaf=0.9), 0.4)
        assert found[0].rejected_by == 'tick at a drawing' and found[0].score == 0.0
        assert not found[0].accepted


@pytest.mark.unit
class TestCaptionsAmongBars:
    """A weaker accepted mark on a stronger bar's caption line is that bar's caption."""

    def test_a_numeral_above_a_ruler_is_demoted(self):
        ruler = sc.Candidate([0], [100, 500, 600, 516], score=0.9, accepted=True)
        one = sc.Candidate([1], [200, 470, 207, 495], score=0.45, accepted=True)
        grouping._demote_captions([ruler, one], *PAGE)
        assert ruler.accepted and not one.accepted and one.rejected_by == 'caption of a bar'

    def test_two_bars_apart_are_both_kept(self):
        first = sc.Candidate([0], [100, 500, 600, 516], score=0.9, accepted=True)
        second = sc.Candidate([1], [100, 700, 600, 716], score=0.5, accepted=True)
        grouping._demote_captions([first, second], *PAGE)
        assert first.accepted and second.accepted

    def test_a_tall_mark_is_never_a_caption(self):
        bar = sc.Candidate([0], [100, 500, 600, 516], score=0.9, accepted=True)
        tall = sc.Candidate([1], [620, 300, 660, 520], score=0.5, accepted=True, upright=True)
        grouping._demote_captions([bar, tall], *PAGE)
        assert tall.accepted

    def test_a_glyph_sized_drawing_on_the_line_becomes_a_caption(self):
        result = grouping.Classified()
        result.bars = [[100, 500, 600, 516]]
        digit = _component((200, 468, 215, 495))        # 27 px tall: over the label limit
        view = _component((200, 300, 300, 420))         # a small surface view, well above
        result.drawings = [digit, view]
        grouping._captions_among_drawings(result, 800, 200)
        assert result.bar_captions == [digit.box] and result.drawings == [view]

    def test_a_unit_drawing_follows_a_digit_caption(self):
        result = grouping.Classified()
        result.bars = [[100, 500, 600, 516]]
        digit = _component((585, 468, 600, 495))
        unit = _component((610, 470, 656, 495))         # "cm": past the end, at a corner
        result.drawings = [unit, digit]                  # order must not matter
        grouping._captions_among_drawings(result, 800, 200)
        assert sorted(result.bar_captions) == sorted([digit.box, unit.box])
        assert result.drawings == []


@pytest.mark.unit
class TestRunsAndFusedBars:
    """Runs are made of blocks; a bar fused with its caption is cut from the raw ink."""

    def test_outlined_surfaces_in_a_row_do_not_form_a_run(self):
        views = [_hollow((100 + 150 * k, 100, 220 + 150 * k, 140)) for k in range(4)]
        found = grouping.find_scale_candidates(views, *PAGE, set())
        assert sorted(len(c.indices) for c in found) == [1, 1, 1, 1]

    def test_solid_blocks_in_a_row_still_do(self):
        blocks = [_component((100 + 60 * k, 100, 130 + 60 * k, 108)) for k in range(4)]
        found = grouping.find_scale_candidates(blocks, *PAGE, set())
        assert len(found) == 1 and found[0].indices == [0, 1, 2, 3]

    def test_a_bar_fused_with_its_caption_is_a_candidate(self):
        mask = np.zeros((150, 70), dtype=bool)
        mask[5:146, 0:24] = True                     # the upright bar, 24 x 141
        mask[0:16, 30:40] = mask[0:16, 44:54] = True  # "2" and "c" beside its top
        mask[138:148, 28:40] = True                   # "0" at its foot
        blob = _component((1000, 1600, 1070, 1750), mask)
        raw = _raw([blob], 1200, 1800)
        found = grouping.find_scale_candidates([blob], 1200, 1800, set(), None, raw)
        assert len(found) == 1
        assert found[0].box == [1000, 1605, 1024, 1746] and found[0].upright
        assert found[0].crop == [1000, 1600, 1070, 1750]

    def test_a_blob_without_the_raw_ink_stays_out(self):
        blob = _component((1000, 1600, 1070, 1750))
        assert grouping.find_scale_candidates([blob], 1200, 1800, set()) == []

    def test_a_wide_drawing_on_the_line_is_not_a_caption(self):
        """Spain 5: the small surface views of lithic 16 sat just above the bar."""
        result = grouping.Classified()
        result.bars = [[305, 470, 394, 475]]
        view = _component((354, 413, 466, 450))        # 112 x 37 on a 700 x 1000 page
        result.drawings = [view]
        grouping._captions_among_drawings(result, 700, 1000)
        assert result.drawings == [view] and result.bar_captions == []
