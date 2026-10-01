"""
The identifier rules: count decides whether, position decides where.

Pages are synthetic: boxes, ink components and reads placed by hand,
so each test states one rule and one refusal.
"""

import pytest

from pylithics.page_segmentation import regroup
from pylithics.page_segmentation.detection import Component
from pylithics.page_segmentation.identifiers import PageReads


PAGE = (1000, 1000)
CONFIG = {
    'grouping': {'min_area': 0.0004},               # 400 px on a 1000x1000 page
    'identifiers': {'regroup': {'enabled': True, 'join_gap': 0.12, 'max_cuts': 4}},
}


def _ink(box):
    """A drawing-sized component filling a box."""
    w, h = box[2] - box[0], box[3] - box[1]
    return Component(box=list(box), width=w, height=h, ink=w * h)


def _reads(*items, seen=()):
    """Reads as ``(text, glyph box)``, all confident, plus unread glyphs."""
    reads = [(text, 0.99, list(glyph)) for text, glyph in items]
    return PageReads(reads=reads, seen=[list(g) for g in seen] + [r[2] for r in reads],
                     attempted=True)


def _refine(boxes, reads, components):
    return regroup.refine_boxes(boxes, reads, components, PAGE, CONFIG)


@pytest.mark.unit
class TestSplit:
    """Rule 1: a box holding N identifiers becomes N boxes, or stays."""

    def test_two_lithics_side_by_side_are_cut_apart(self):
        ink = [_ink([100, 100, 300, 400]), _ink([500, 100, 700, 400])]
        reads = _reads(('1', [280, 410, 296, 430]), ('2', [680, 410, 696, 430]))
        boxes, tags = _refine([[100, 100, 700, 430]], reads, ink)
        assert boxes == [[100, 100, 300, 430], [500, 100, 700, 430]]
        assert tags == [regroup.SPLIT, regroup.SPLIT]

    def test_two_lithics_stacked_are_cut_apart(self):
        ink = [_ink([100, 100, 300, 300]), _ink([100, 500, 300, 700])]
        reads = _reads(('4', [280, 310, 296, 330]), ('9', [280, 710, 296, 730]))
        boxes, tags = _refine([[100, 100, 300, 730]], reads, ink)
        assert boxes == [[100, 100, 300, 330], [100, 500, 300, 730]]
        assert tags == [regroup.SPLIT, regroup.SPLIT]

    def test_a_grid_of_six_becomes_six(self):
        ink, items = [], []
        for row, y in enumerate((100, 500)):
            for col, x in enumerate((100, 400, 700)):
                ink.append(_ink([x, y, x + 200, y + 250]))
                items.append((str(row * 3 + col + 7), [x + 180, y + 260, x + 196, y + 280]))
        boxes, tags = _refine([[100, 100, 900, 780]], _reads(*items), ink)
        assert len(boxes) == 6 and set(tags) == {regroup.SPLIT}

    def test_a_loose_numeral_is_not_a_lithic(self):
        """Archaic Oldowan figure 9, box 4: label 1 sits above lithic 4."""
        ink = [_ink([100, 200, 300, 500])]
        reads = _reads(('1', [180, 110, 196, 130]), ('4', [280, 510, 296, 530]))
        boxes, tags = _refine([[100, 110, 300, 530]], reads, ink)
        assert boxes == [[100, 110, 300, 530]] and tags == ['']

    def test_no_empty_run_means_no_cut(self):
        ink = [_ink([100, 100, 700, 400])]           # one blob spans both labels
        reads = _reads(('1', [280, 410, 296, 430]), ('2', [680, 410, 696, 430]))
        boxes, tags = _refine([[100, 100, 700, 430]], reads, ink)
        assert boxes == [[100, 100, 700, 430]] and tags == ['']

    def test_a_box_with_one_identifier_is_untouched(self):
        ink = [_ink([100, 100, 300, 400]), _ink([500, 100, 700, 400])]
        reads = _reads(('3', [280, 410, 296, 430]))
        boxes, tags = _refine([[100, 100, 700, 430]], reads, ink)
        assert boxes == [[100, 100, 700, 430]] and tags == ['']


@pytest.mark.unit
class TestJoin:
    """Rule 2: an unlabelled box joins its one labelled neighbour, or stays."""

    def test_an_unlabelled_profile_joins_the_view_beside_it(self):
        ink = [_ink([100, 100, 300, 400]), _ink([340, 100, 380, 400])]
        reads = _reads(('5', [280, 410, 296, 430]))
        boxes, tags = _refine([[100, 100, 300, 430], [340, 100, 380, 400]], reads, ink)
        assert boxes == [[100, 100, 380, 430]] and tags == [regroup.JOIN]

    def test_an_unlabelled_section_joins_the_view_above_it(self):
        ink = [_ink([100, 100, 300, 400]), _ink([120, 470, 280, 520])]
        reads = _reads(('2', [280, 410, 296, 430]))
        boxes, tags = _refine([[100, 100, 300, 430], [120, 470, 280, 520]], reads, ink)
        assert boxes == [[100, 100, 300, 520]] and tags == [regroup.JOIN]

    def test_a_chain_of_unlabelled_views_collapses(self):
        ink = [_ink([100, 100, 300, 400]), _ink([340, 100, 380, 400]),
               _ink([420, 100, 600, 400])]
        reads = _reads(('9', [580, 410, 596, 430]))
        boxes, tags = _refine(
            [[100, 100, 300, 400], [340, 100, 380, 400], [420, 100, 600, 430]], reads, ink
        )
        assert boxes == [[100, 100, 600, 430]] and tags == [regroup.JOIN]

    def test_a_box_holding_an_unread_glyph_is_left_alone(self):
        """1980 page 29, box 2: its "2" is printed but was not read."""
        ink = [_ink([100, 100, 300, 400]), _ink([340, 100, 540, 400])]
        reads = _reads(('1', [280, 410, 296, 430]), seen=[[520, 410, 536, 430]])
        boxes, tags = _refine([[100, 100, 300, 430], [340, 100, 540, 430]], reads, ink)
        assert len(boxes) == 2 and tags == ['', '']

    def test_two_equidistant_hosts_mean_no_join(self):
        ink = [_ink([100, 100, 300, 400]), _ink([340, 100, 380, 400]),
               _ink([420, 100, 620, 400])]
        reads = _reads(('1', [280, 410, 296, 430]), ('2', [600, 410, 616, 430]))
        boxes, tags = _refine(
            [[100, 100, 300, 430], [340, 100, 380, 400], [420, 100, 620, 430]], reads, ink
        )
        assert len(boxes) == 3 and tags == ['', '', '']

    def test_a_box_in_between_blocks_the_join(self):
        """An unlabelled box does not reach past its neighbour to a host."""
        ink = [_ink([100, 100, 300, 400]), _ink([340, 100, 380, 400]),
               _ink([420, 100, 620, 400])]
        reads = _reads(('2', [600, 410, 616, 430]), seen=[[360, 410, 376, 430]])
        boxes, tags = _refine(
            [[100, 100, 300, 400], [340, 100, 380, 430], [420, 100, 620, 430]], reads, ink
        )
        assert len(boxes) == 3 and tags == ['', '', '']

    def test_a_join_that_would_hold_two_identifiers_is_refused(self):
        """The unlabelled box sits under a second lithic's loose label."""
        ink = [_ink([100, 100, 300, 400]), _ink([340, 100, 380, 400])]
        reads = _reads(('1', [280, 410, 296, 430]), ('2', [352, 60, 368, 80]))
        boxes, tags = _refine([[100, 100, 300, 430], [340, 100, 380, 400]], reads, ink)
        assert len(boxes) == 2 and tags == ['', '']

    def test_a_box_many_times_larger_does_not_join(self):
        """Homo erectus figure 29: a whole row of lithics joined into one small labelled box."""
        ink = [_ink([100, 100, 300, 300]), _ink([100, 340, 900, 600])]   # 5.5 times the host
        reads = _reads(('a', [280, 310, 296, 330]))
        boxes, tags = _refine([[100, 100, 300, 330], [100, 340, 900, 600]], reads, ink)
        assert len(boxes) == 2 and tags == ['', '']

    def test_a_somewhat_larger_view_still_joins(self):
        """Fauresmith figure 16, G: the labelled box is the smaller view of the lithic."""
        ink = [_ink([100, 100, 250, 400]), _ink([280, 100, 600, 400])]   # 2.1 times the host
        reads = _reads(('G', [230, 410, 246, 430]))
        boxes, tags = _refine([[100, 100, 250, 430], [280, 100, 600, 400]], reads, ink)
        assert boxes == [[100, 100, 600, 430]] and tags == [regroup.JOIN]

    def test_a_host_takes_every_view_of_its_lithic(self):
        """Vallonnet figure 8: three unlabelled views beside one label all join."""
        ink = [_ink([100, 100, 400, 400]), _ink([430, 100, 470, 400]),
               _ink([500, 100, 540, 400]), _ink([570, 100, 610, 400])]
        reads = _reads(('5', [380, 410, 396, 430]))
        boxes, tags = _refine(
            [[100, 100, 400, 430], [430, 100, 470, 400], [500, 100, 540, 400],
             [570, 100, 610, 400]], reads, ink
        )
        assert boxes == [[100, 100, 610, 430]] and tags == [regroup.JOIN]

    def test_beyond_the_join_gap_nothing_happens(self):
        ink = [_ink([100, 100, 300, 400]), _ink([500, 100, 540, 400])]
        reads = _reads(('5', [280, 410, 296, 430]))
        boxes, tags = _refine([[100, 100, 300, 430], [500, 100, 540, 400]], reads, ink)
        assert len(boxes) == 2 and tags == ['', '']


@pytest.mark.unit
class TestSeededSplit:
    """A dense plate with staggered rows has no empty run across the box."""

    INK = [[100, 100, 300, 300], [350, 150, 550, 350],
           [100, 330, 380, 530], [350, 380, 550, 580]]
    LABELS = [('1', [305, 280, 321, 300]), ('2', [555, 330, 571, 350]),
              ('3', [385, 510, 401, 530]), ('4', [555, 560, 571, 580])]

    def test_each_lithic_goes_to_the_label_on_the_plates_side(self):
        """Labels sit to the right of every lithic; each drawing takes its own."""
        ink = [_ink(b) for b in self.INK]
        boxes, tags = _refine([[100, 100, 571, 580]], _reads(*self.LABELS), ink)
        assert len(boxes) == 4 and set(tags) == {regroup.SPLIT}
        assert [100, 100, 321, 300] in boxes and [350, 380, 571, 580] in boxes

    def test_a_label_with_no_drawing_refuses_the_split(self):
        ink = [_ink(b) for b in self.INK[:3]]           # lithic 4 is not drawn
        boxes, tags = _refine([[100, 100, 571, 580]], _reads(*self.LABELS), ink)
        assert boxes == [[100, 100, 571, 580]] and tags == ['']

    def test_two_lithics_that_arrived_as_one_piece_stay_together(self):
        """Overlapping hulls are one flagged piece; the rest of the box still splits."""
        ink = [_ink([100, 100, 300, 300]), _ink([350, 150, 550, 350]),
               _ink([100, 330, 400, 580]), _ink([230, 380, 550, 580])]   # 3 and 4 overlap
        labels = [('1', [305, 280, 321, 300]), ('2', [555, 330, 571, 350]),
                  ('3', [80, 510, 96, 530]), ('4', [555, 560, 571, 580])]
        boxes, tags = _refine([[80, 100, 571, 580]], _reads(*labels), ink)
        assert len(boxes) == 3 and set(tags) == {regroup.SPLIT}
        assert [80, 330, 571, 580] in boxes

    def test_a_label_off_the_plates_side_refuses_the_split(self):
        """Two labels right of the lithics and one left: the left one gets nothing."""
        ink = [_ink([100, 100, 500, 400]), _ink([100, 420, 500, 700])]
        reads = _reads(('1', [80, 100, 96, 120]), ('2', [505, 380, 521, 400]),
                       ('3', [505, 680, 521, 700]))
        boxes, tags = _refine([[80, 100, 521, 700]], reads, ink)
        assert len(boxes) == 1 and tags == ['']


@pytest.mark.unit
class TestLabelsBelowTheLithic:
    """Archaic Oldowan plates: each number sits below its lithic, in the box beneath."""

    def test_a_stray_label_at_the_top_edge_does_not_split(self):
        """Figure 5, box 5: one lithic of four views under three loose labels."""
        ink = [_ink([100, 400, 300, 700]), _ink([340, 400, 380, 700]),
               _ink([420, 400, 620, 700])]
        upper = [[100, 100, 300, 350], [420, 100, 620, 350]]
        reads = _reads(('3', [180, 360, 196, 380]), ('4', [500, 360, 516, 380]))
        boxes, tags = _refine(upper + [[100, 360, 620, 700]], reads, ink)
        assert [100, 360, 620, 700] in boxes and set(tags) == {''}

    def test_an_unlabelled_lithic_with_its_label_beneath_is_not_joined(self):
        """Figure 2, box 2: its own "2" lands in the box below."""
        ink = [_ink([100, 100, 300, 400]), _ink([340, 100, 540, 400]),
               _ink([340, 460, 540, 700])]
        reads = _reads(('1', [180, 410, 196, 430]), ('2', [420, 435, 436, 455]))
        boxes, tags = _refine(
            [[100, 100, 300, 430], [340, 100, 540, 400], [340, 435, 540, 700]], reads, ink
        )
        assert len(boxes) == 3 and tags == ['', '', '']


@pytest.mark.unit
class TestNothingToDo:
    def test_no_reads_means_no_change(self):
        ink = [_ink([100, 100, 300, 400]), _ink([340, 100, 380, 400])]
        boxes = [[100, 100, 300, 400], [340, 100, 380, 400]]
        assert _refine(boxes, PageReads(attempted=True), ink) == (boxes, ['', ''])

    def test_not_attempted_means_no_change(self):
        boxes = [[100, 100, 700, 430]]
        assert _refine(boxes, PageReads(), []) == (boxes, [''])

    def test_switched_off_in_config(self):
        ink = [_ink([100, 100, 300, 400]), _ink([340, 100, 380, 400])]
        reads = _reads(('5', [280, 410, 296, 430]))
        off = {**CONFIG, 'identifiers': {'regroup': {'enabled': False}}}
        boxes = [[100, 100, 300, 430], [340, 100, 380, 400]]
        assert regroup.refine_boxes(boxes, reads, ink, PAGE, off) == (boxes, ['', ''])

    def test_one_box_page_with_one_read_is_unchanged(self):
        ink = [_ink([100, 100, 300, 400])]
        reads = _reads(('1', [280, 410, 296, 430]))
        assert _refine([[100, 100, 300, 430]], reads, ink) == ([[100, 100, 300, 430]], [''])
