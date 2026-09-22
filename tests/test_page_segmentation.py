"""
Tests for the page segmentation workflow (``pylithics-pages``).

Grouping correctness is validated against ``sample_plate.png``, a
composite of the five real lithic drawings shipped with PyLithics. Each
of those drawings already shows two to four surface views, so the plate
carries far more ink blobs than artefacts. Synthetic rectangles would
not exercise the profile, platform, or caption rules at all, which is
why the real plate is used for the grouping assertions.
"""

import csv
import os
import sys

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'fixtures'))
import draw_scales  # noqa: E402

from pylithics.image_processing.config import (
    clear_config_cache,
    get_config_manager,
    get_page_segmentation_config,
)
from pylithics.page_segmentation import export as export_module
from pylithics.page_segmentation import grouping as grouping_module
from pylithics.page_segmentation import identifiers as identifiers_module
from pylithics.page_segmentation import overrides as overrides_module
from pylithics.page_segmentation.cli import main
from pylithics.page_segmentation.detection import (
    build_detection_mask,
    find_components,
    load_page,
)
from pylithics.page_segmentation.export import (
    ManifestRow,
    MANIFEST_COLUMNS,
    prepare_output_dir,
    remove_page_output,
    write_metadata,
)
from pylithics.page_segmentation.geometry import (
    box_distance,
    boxes_intersect,
    clamp_box,
    horizontal_gap,
    union,
    vertical_overlap,
)
from pylithics.page_segmentation.grouping import (
    group_components,
    merge_bar_segments,
    reading_order,
)

# The sample plate shows five lithics and one shared scale bar.
EXPECTED_ARTEFACTS = 5
EXPECTED_SCALE_BARS = 1
EXPECTED_DPI = 300.0


### FIXTURES ###

@pytest.fixture(scope="module")
def plate_path():
    """Path to the committed sample plate."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    path = os.path.join(
        root, "pylithics", "data", "pages", "sample_plate.png"
    )
    if not os.path.isfile(path):
        pytest.skip(
            "sample_plate.png missing; run "
            "tests/fixtures/generate_sample_plate.py"
        )
    return path


@pytest.fixture(scope="module")
def plate_page(plate_path):
    """The loaded sample plate."""
    return load_page(plate_path)


@pytest.fixture(scope="module")
def plate_components(plate_page):
    """Ink blobs detected on the sample plate."""
    closed, raw = build_detection_mask(
        plate_page, get_config_manager().config
    )
    return find_components(closed, raw)


@pytest.fixture(scope="module")
def segmentation_config():
    """Page segmentation configuration with shipped defaults."""
    return get_page_segmentation_config(get_config_manager().config)


@pytest.fixture
def pages_dir(tmp_path, plate_path):
    """An input directory holding a copy of the sample plate."""
    directory = tmp_path / "pages"
    directory.mkdir()
    Image.open(plate_path).save(
        directory / "sample_plate.png", dpi=(EXPECTED_DPI, EXPECTED_DPI)
    )
    return str(directory)


def run_prep(pages, output, *extra):
    """Invoke the CLI and return its exit code."""
    return main(
        ["--data_dir", str(pages), "--output_dir", str(output), *extra]
    )


def read_manifest(output_dir):
    """Read the manifest written into an output directory."""
    path = os.path.join(str(output_dir), "pages_manifest.csv")
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


### GEOMETRY ###

@pytest.mark.unit
class TestGeometry:
    """Bounding-box primitives."""

    def test_vertical_overlap_is_negative_when_separated(self):
        assert vertical_overlap([0, 0, 10, 10], [0, 20, 10, 30]) == -10

    def test_horizontal_gap_measures_separation(self):
        assert horizontal_gap([0, 0, 10, 10], [30, 0, 40, 10]) == 20

    def test_union_encloses_both_boxes(self):
        assert union([5, 5, 10, 10], [0, 8, 7, 20]) == [0, 5, 10, 20]

    def test_box_distance_is_zero_when_overlapping(self):
        assert box_distance([0, 0, 10, 10], [5, 5, 15, 15]) == 0.0

    def test_box_distance_is_diagonal_when_offset(self):
        assert box_distance([0, 0, 10, 10], [13, 14, 20, 20]) == 5.0

    def test_boxes_intersect_requires_both_axes(self):
        assert not boxes_intersect([0, 0, 10, 10], [20, 0, 30, 10])
        assert boxes_intersect([0, 0, 10, 10], [5, 5, 15, 15])

    def test_clamp_box_pads_and_clips_to_page(self):
        assert clamp_box([5, 5, 50, 50], 20, 60, 60) == [0, 0, 60, 60]


### GROUPING ###

@pytest.mark.archaeological
class TestArtefactGrouping:
    """A lithic drawn with several surfaces is one artefact."""

    def test_plate_yields_one_crop_per_lithic(
        self, plate_components, plate_page, segmentation_config
    ):
        """Five lithics, not one crop per surface view."""
        boxes, _ = group_components(
            plate_components,
            (plate_page.width, plate_page.height),
            segmentation_config,
        )
        assert len(boxes) == EXPECTED_ARTEFACTS

    def test_grouping_consumes_many_blobs_per_artefact(
        self, plate_components, plate_page, segmentation_config
    ):
        """
        Each artefact groups several ink blobs.

        Guards against a regression where grouping degrades into one
        crop per blob while still, by chance, returning five boxes.
        """
        boxes, _ = group_components(
            plate_components,
            (plate_page.width, plate_page.height),
            segmentation_config,
        )
        assert len(plate_components) > 3 * len(boxes)

    def test_artefact_boxes_do_not_overlap(
        self, plate_components, plate_page, segmentation_config
    ):
        """Overlapping crops would duplicate ink between artefacts."""
        boxes, _ = group_components(
            plate_components,
            (plate_page.width, plate_page.height),
            segmentation_config,
        )
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                assert not boxes_intersect(boxes[i], boxes[j])

    def test_scale_bar_detected_once(
        self, plate_components, plate_page, segmentation_config
    ):
        """A segmented bar is one bar, not one per filled block."""
        _, bars = group_components(
            plate_components,
            (plate_page.width, plate_page.height),
            segmentation_config,
        )
        assert len(bars) == EXPECTED_SCALE_BARS

    def test_scale_bar_excluded_from_artefacts(
        self, plate_components, plate_page, segmentation_config
    ):
        """A page scale bar belongs to no single artefact."""
        boxes, bars = group_components(
            plate_components,
            (plate_page.width, plate_page.height),
            segmentation_config,
        )
        for box in boxes:
            assert not boxes_intersect(box, bars[0])


@pytest.mark.unit
class TestReadingOrder:
    """Crops are numbered the way a plate is read."""

    def test_rows_run_top_to_bottom_then_left_to_right(self):
        boxes = [
            [500, 0, 600, 100],
            [0, 0, 100, 100],
            [0, 400, 100, 500],
        ]
        assert reading_order(boxes) == [
            [0, 0, 100, 100],
            [500, 0, 600, 100],
            [0, 400, 100, 500],
        ]

    def test_slight_vertical_offset_stays_one_row(self):
        boxes = [[300, 12, 400, 112], [0, 0, 100, 100]]
        assert reading_order(boxes)[0][0] == 0


@pytest.mark.unit
class TestScaleBarAssembly:
    """Segmented bars are rejoined; unrelated marks are not."""

    def test_collinear_segments_merge(self):
        segments = [
            [100, 500, 200, 515],
            [220, 500, 320, 515],
            [340, 500, 440, 515],
        ]
        assert merge_bar_segments(segments, 2480) == [[100, 500, 440, 515]]

    def test_distant_segments_stay_separate(self):
        segments = [[100, 500, 200, 515], [1800, 500, 1900, 515]]
        assert len(merge_bar_segments(segments, 2480)) == 2

    def test_segments_on_different_lines_stay_separate(self):
        segments = [[100, 500, 200, 515], [220, 900, 320, 915]]
        assert len(merge_bar_segments(segments, 2480)) == 2


@pytest.mark.unit
class TestCaptionRejection:
    """Captions are discarded; lone drawing elements are not."""

    def test_lone_wide_mark_is_not_a_caption(self):
        """
        A single blob is never a line of text.

        A platform view is a wide, isolated mark. Treating it as a
        caption would drop a real surface from its artefact.
        """
        platform = [[300, 300, 460, 360]]
        assert grouping_module.reject_caption_lines(platform, 2480) == platform

    def test_run_of_small_blobs_is_rejected(self):
        caption = [
            [100, 900, 130, 940],
            [140, 900, 170, 940],
            [180, 900, 210, 940],
            [220, 900, 320, 940],
        ]
        assert grouping_module.reject_caption_lines(caption, 2480) == []


@pytest.mark.unit
class TestComponentClassification:
    """Roles are assigned from shape and ink coverage."""

    def test_hollow_outline_is_a_drawing_not_a_label(
        self, plate_components, plate_page, segmentation_config
    ):
        """
        Sparse ink alone must not demote a surface view to a label.

        Lithic drawings are hollow outlines, so they always carry little
        ink for their size.
        """
        classified = grouping_module.classify_components(
            plate_components,
            plate_page.width,
            plate_page.height,
            segmentation_config['grouping']['min_area'],
            segmentation_config['scale_bars'],
        )
        hollow = [
            d for d in classified.drawings
            if d.density < 0.15 and d.height < 0.04 * plate_page.height
        ]
        assert hollow, "expected platform views among the drawings"

    def test_identifier_letters_become_labels(
        self, plate_components, plate_page, segmentation_config
    ):
        classified = grouping_module.classify_components(
            plate_components,
            plate_page.width,
            plate_page.height,
            segmentation_config['grouping']['min_area'],
            segmentation_config['scale_bars'],
        )
        assert len(classified.labels) >= EXPECTED_ARTEFACTS


### SCALE BAR STYLES ###

@pytest.mark.archaeological
class TestScaleBarStyles:
    """
    Every scale style seen in the source literature is recognised.

    Scales vary far more than lithic drawings do, and ink coverage is a
    poor guide: across measured examples it runs from 0.22 to 1.00,
    because the numerals printed beside a bar pull whitespace into its
    bounding box. What every style shares is a long straight edge — the
    frame of a chequered bar, the solid top of a block bar, the rule of
    a graduated one — and proportions no surface view has.

    Styles are drawn rather than shipped as images, so the detector's
    routes stay covered without committing scanned figures.
    """

    STYLES = sorted(draw_scales.STYLES)

    def _page_with_scale(self, tmp_path, name):
        """Place a drawn scale alone on a realistic blank page."""
        pages = tmp_path / 'pages'
        pages.mkdir(exist_ok=True)
        draw_scales.page_with_scale(name).save(
            pages / f'{name}.png', dpi=(300, 300)
        )
        return pages

    @pytest.mark.parametrize('name', STYLES)
    def test_style_is_detected(self, tmp_path, name):
        """Each scale style yields exactly one exported scale."""
        pages = self._page_with_scale(tmp_path, name)
        out = tmp_path / f'out_{name}'
        assert run_prep(pages, out) == 0
        assert len(os.listdir(out / 'scales')) == 1

    @pytest.mark.parametrize('name', STYLES)
    def test_style_yields_no_artefacts(self, tmp_path, name):
        """
        A scale is never mistaken for a lithic.

        The chequered style previously fired on the numeral "1" beside
        it, a digit being structurally a spine with two ticks attached.
        """
        pages = self._page_with_scale(tmp_path, name)
        out = tmp_path / f'out_{name}'
        run_prep(pages, out)
        assert os.listdir(out / 'images') == []


@pytest.mark.unit
class TestTextRejection:
    """
    Captions, legend words and legend keys are discarded before linking.

    A word reaches the detector as one blob, its letters fused by
    closing, but its raw ink still breaks into a row of letter-sized
    pieces. A lithic view is one outline holding nearly all its ink.
    """

    PAGE = (2000, 1500)

    def _classify(self, mask, x0=500, y0=500, text_config=None):
        height, width = mask.shape
        component = grouping_module.Component(
            box=[x0, y0, x0 + width, y0 + height],
            width=width, height=height,
            ink=int(mask.sum()), mask=mask,
        )
        return grouping_module.classify_components(
            [component], *self.PAGE, 0.0004, {}, text_config
        )

    @staticmethod
    def _word(letters, letter_w, letter_h, gap):
        """Solid letter blocks in a row, as a closed word's raw ink looks."""
        width = letters * letter_w + (letters - 1) * gap
        mask = np.zeros((letter_h, width), dtype=bool)
        for index in range(letters):
            left = index * (letter_w + gap)
            mask[:, left:left + letter_w] = True
        return mask

    @staticmethod
    def _hollow(width, height, border=3):
        mask = np.zeros((height, width), dtype=bool)
        mask[:border, :] = mask[-border:, :] = True
        mask[:, :border] = mask[:, -border:] = True
        return mask

    def test_legend_word_is_discarded(self):
        classified = self._classify(self._word(6, 14, 40, 6))
        assert classified.drawings == []
        assert classified.labels == []

    def test_caption_line_is_discarded(self):
        classified = self._classify(self._word(40, 8, 14, 4))
        assert classified.drawings == []

    def test_hatched_section_is_kept(self):
        """Hatching joins the outline, so the ink stays one piece."""
        mask = self._hollow(300, 60)
        for start in range(0, 300, 20):
            for step in range(60):
                column = start + step
                if column < 300:
                    mask[step, column] = True
        assert len(self._classify(mask).drawings) == 1

    def test_tall_outline_is_kept(self):
        assert len(self._classify(self._hollow(200, 400)).drawings) == 1

    def test_row_of_views_is_kept(self):
        """
        Three views side by side break into three pieces like a word,
        but stand far taller than any line of text on the page.
        """
        mask = np.zeros((180, 280), dtype=bool)
        for left in (0, 100, 200):
            mask[:, left:left + 80] = self._hollow(80, 180)
        assert len(self._classify(mask).drawings) == 1

    def test_legend_swatch_is_discarded(self):
        classified = self._classify(np.ones((60, 80), dtype=bool))
        assert classified.drawings == []
        assert classified.labels == []

    def test_identifier_numeral_is_kept_as_label(self):
        classified = self._classify(np.ones((40, 30), dtype=bool))
        assert len(classified.labels) == 1

    def test_running_head_is_furniture(self):
        classified = self._classify(np.ones((80, 1800), dtype=bool), x0=100)
        assert classified.drawings == []
        assert classified.labels == []
        assert classified.bars == []

    def test_can_be_disabled(self):
        classified = self._classify(
            self._word(6, 14, 40, 6), text_config={'enabled': False}
        )
        assert len(classified.drawings) == 1

    def test_word_with_descender_is_discarded(self):
        """
        One letter in five reaching below the baseline, as the g of
        "negat" does, must not rescue a word: real words have descenders.
        """
        mask = np.zeros((50, 114), dtype=bool)
        mask[:40, :] = self._word(6, 14, 40, 6)
        mask[40:, 20:34] = True          # the second letter descends
        assert self._classify(mask).drawings == []

    def test_remnants_of_a_rejected_line_are_dropped(self):
        """
        A bracket or a swatch beside a discarded word is label-sized, and
        a label attaches to the nearest artefact, dragging the crop back
        over the text. Anything sharing a baseline with rejected text
        goes with it.
        """
        word = self._word(6, 14, 40, 6)
        remnant = np.ones((30, 12), dtype=bool)
        components = [
            grouping_module.Component(
                box=[500, 500, 614, 540], width=114, height=40,
                ink=int(word.sum()), mask=word,
            ),
            grouping_module.Component(
                box=[630, 508, 642, 538], width=12, height=30,
                ink=360, mask=remnant,
            ),
        ]
        classified = grouping_module.classify_components(
            components, *self.PAGE, 0.0004, {}
        )
        assert classified.drawings == []
        assert classified.labels == []

    def test_wrapped_text_block_is_discarded(self):
        """
        Two lines fused into one block leave no letter 40% of the
        block's height. Each line is tested on its own instead.
        """
        line = self._word(8, 12, 14, 4)
        block = np.zeros((14 + 5 + 14, line.shape[1]), dtype=bool)
        block[:14] = line
        block[19:] = line
        assert self._classify(block).drawings == []

    def test_accented_word_is_still_discarded(self):
        """An accent forms a second band; it must not rescue the word."""
        word = self._word(6, 14, 40, 6)
        mask = np.zeros((50, word.shape[1]), dtype=bool)
        mask[10:] = word
        mask[0:5, 22:30] = True
        assert self._classify(mask).drawings == []

    def test_dash_in_a_rejected_line_is_dropped(self):
        """
        The em-dash of "Fig. 10 — Racloirs" reads as a connector mark
        and was attached to the lithic above the caption.
        """
        word = self._word(6, 14, 40, 6)
        dash = np.ones((3, 30), dtype=bool)
        components = [
            grouping_module.Component(
                box=[500, 500, 614, 540], width=114, height=40,
                ink=int(word.sum()), mask=word,
            ),
            grouping_module.Component(
                box=[460, 520, 490, 523], width=30, height=3,
                ink=90, mask=dash,
            ),
        ]
        classified = grouping_module.classify_components(
            components, *self.PAGE, 0.0004, {}
        )
        assert classified.drawings == []
        assert classified.dashes == []


@pytest.mark.unit
class TestIdentifierStages:
    """
    Each stage of identifier reading, without an OCR engine.

    The engine is the one part that cannot be tested deterministically,
    so the candidate search, the filters, the page-level validation and
    the assignment to crops are exercised on their own.
    """

    PAGE = (2000, 1500)

    @staticmethod
    def _component(mask, x0, y0):
        height, width = mask.shape
        return grouping_module.Component(
            box=[x0, y0, x0 + width, y0 + height],
            width=width, height=height, ink=int(mask.sum()), mask=mask,
        )

    def test_validate_corrects_a_letter_among_digits(self):
        reads = [('1', 0.9, [0, 0, 1, 1]), ('2', 0.9, [0, 0, 1, 1]),
                 ('E', 0.8, [0, 0, 1, 1]), ('4', 0.9, [0, 0, 1, 1])]
        assert [r[0] for r in identifiers_module._validate(reads)] == ['1', '2', '3', '4']

    def test_validate_keeps_a_lettered_page(self):
        reads = [('a', 0.9, [0, 0, 1, 1]), ('b', 0.9, [0, 0, 1, 1]),
                 ('5', 0.8, [0, 0, 1, 1])]
        assert [r[0] for r in identifiers_module._validate(reads)] == ['a', 'b', 'S']

    def test_assign_names_one_identifier_per_crop(self):
        boxes = [[0, 0, 100, 100], [200, 0, 300, 100]]
        found = [identifiers_module.Identifier() for _ in boxes]
        identifiers_module._assign(
            [('7', 0.9, [80, 80, 90, 95]), ('8', 0.9, [280, 80, 290, 95])],
            boxes, found,
        )
        assert [f.label for f in found] == ['7', '8']
        assert all(f.source == 'read' and not f.flag for f in found)

    def test_assign_flags_a_crop_holding_two_identifiers(self):
        """Two identifiers means two lithics: flagged, never named."""
        boxes = [[0, 0, 300, 100]]
        found = [identifiers_module.Identifier()]
        identifiers_module._assign(
            [('4', 0.9, [80, 80, 90, 95]), ('9', 0.9, [280, 80, 290, 95])],
            boxes, found,
        )
        assert found[0].flag == 'several_identifiers'
        assert found[0].source == 'index' and found[0].label == ''

    def test_type_size_comes_from_confident_reads(self):
        """
        On Archaic Oldowan Figure 1 eleven edge strokes read as "1" at
        low confidence outnumbered five real numerals read at 0.98 or
        better; a plain median sized the type from the strokes and threw
        the numerals away.
        """
        real = [(str(n), 0.99, [0, 0, 20, 34]) for n in range(1, 6)]
        junk = [('1', 0.7, [0, 0, 3, 12]) for _ in range(11)]
        kept = identifiers_module._consistent_size(real + junk)
        assert [r[0] for r in kept] == ['1', '2', '3', '4', '5']

    def test_lone_one_needs_high_confidence(self):
        """A connector rule reads as "1" at 0.7; a printed "1" at 0.97+."""
        class Engine:
            def __init__(self, conf): self.conf = conf
            def text_rec(self, imgs): return [('1', self.conf)], None
        page = type('P', (), {'gray': np.full((100, 100), 255, np.uint8)})()
        weak = identifiers_module._read([[40, 40, 44, 60]], page, Engine(0.7), {})
        sure = identifiers_module._read([[40, 40, 44, 60]], page, Engine(0.97), {})
        assert weak == [] and [r[0] for r in sure] == ['1']

    def test_crop_with_nothing_read_says_so(self):
        found = [identifiers_module.Identifier()]
        identifiers_module._assign([], [[0, 0, 100, 100]], found)
        assert found[0].flag == 'no_identifier' and not found[0].named

    def test_readings_in_a_flagged_crop_are_kept(self):
        found = [identifiers_module.Identifier()]
        identifiers_module._assign(
            [('4', 0.9, [80, 80, 90, 95]), ('9', 0.9, [280, 80, 290, 95])],
            [[0, 0, 300, 100]], found,
        )
        assert found[0].candidate_text == '4;9'

    def test_assign_flags_duplicates_across_crops(self):
        boxes = [[0, 0, 100, 100], [200, 0, 300, 100]]
        found = [identifiers_module.Identifier() for _ in boxes]
        identifiers_module._assign(
            [('3', 0.9, [80, 80, 90, 95]), ('3', 0.7, [280, 80, 290, 95])],
            boxes, found,
        )
        assert [f.flag for f in found] == ['duplicate', 'duplicate']
        assert not any(f.named for f in found)

    def test_glyph_inside_an_outline_is_not_a_candidate(self):
        """Text never overlaps the illustration."""
        raw = np.zeros((300, 300), dtype=np.uint8)
        raw[50:250, 50:250] = 255          # a filled shape
        raw[100:200, 100:200] = 0          # hollowed out
        raw[50:52, :] = raw[248:250, :] = 0
        raw[140:160, 145:155] = 255        # an arrow-like mark inside it
        page = type('P', (), {'height': 300, 'width': 300})()
        outlines = identifiers_module._lithic_outlines(raw, page)
        assert identifiers_module._on_outline([145, 140, 155, 160], outlines)
        assert not identifiers_module._on_outline([260, 260, 270, 275], outlines)

    def test_candidate_inside_a_scale_bar_is_dropped(self):
        raw = np.zeros((200, 400), dtype=np.uint8)
        page = type('P', (), {'height': 200, 'width': 400})()
        raw[22:33, 22:28] = 255            # ink in each glyph
        raw[152:163, 122:128] = 255
        kept = identifiers_module._filter_glyphs(
            [[20, 20, 30, 35], [120, 150, 130, 165]], page,
            [[100, 140, 300, 170]], raw, raw, {},
        )
        assert kept == [[20, 20, 30, 35]]

    def test_hatching_deep_in_a_gappy_outline_is_rejected(self):
        """
        An outline drawn with gaps cannot be filled from the raw ink, but
        the closed mask still gives a silhouette; a stroke a glyph-height
        or more inside it is on the drawing. A numeral fused to the
        corner is not deep: its own edge is the silhouette's edge.
        """
        closed = np.zeros((1000, 1000), dtype=np.uint8)
        closed[50:350, 50:350] = 255                 # a lithic, closed
        closed[350:380, 330:350] = 255               # a numeral fused to its corner
        page = type('P', (), {'height': 1000, 'width': 1000})()
        depth = identifiers_module._silhouette_depth(closed, page)
        assert identifiers_module._deep_inside([190, 190, 200, 215], depth)
        assert not identifiers_module._deep_inside([330, 350, 350, 380], depth)
        assert not identifiers_module._deep_inside([360, 20, 370, 45], depth)

    def test_glyph_just_outside_a_box_belongs_to_it(self):
        """Some plates set the letter above the drawing, clear of its box."""
        boxes = [[100, 100, 400, 400], [600, 100, 900, 400]]
        assert identifiers_module._home_box([110, 40, 140, 80], boxes) == 0
        assert identifiers_module._home_box([110, 500, 140, 540], boxes) is None

    def test_glyph_between_two_boxes_belongs_to_neither(self):
        boxes = [[100, 100, 400, 400], [430, 100, 700, 400]]
        assert identifiers_module._home_box([405, 200, 425, 230], boxes) is None

    def test_read_name_and_index_name_differ(self):
        read = identifiers_module.Identifier(label='7', source='read', confidence=0.9)
        assert export_module._artefact_name('plate', 3, read) == 'plate_figure_7.png'
        name = export_module._artefact_name('plate', 3, identifiers_module.Identifier())
        assert name == 'plate_box_03.png'


@pytest.mark.integration
class TestIdentifierReading:
    """Reading real glyphs end to end, when the OCR engine is installed."""

    @staticmethod
    def _page(tmp_path):
        """Two hollow lithics with a printed numeral beneath each."""
        try:
            font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 44)
        except OSError:
            pytest.skip('no scalable font available for the fixture')
        image = Image.new('RGB', (2000, 1500), 'white')
        draw = ImageDraw.Draw(image)
        for index, left in enumerate((300, 1100), start=1):
            draw.rectangle([left, 300, left + 400, 800], outline='black', width=4)
            draw.line([left + 60, 340, left + 340, 760], fill='black', width=3)
            draw.text((left + 330, 830), str(index), fill='black', font=font)
        draw.rectangle([600, 1200, 1200, 1230], fill='black')
        for x in (800, 1000):
            draw.rectangle([x, 1204, x + 100, 1226], fill='white')
        pages = tmp_path / 'pages'
        pages.mkdir()
        image.save(pages / 'plate.png', dpi=(300, 300))
        return pages

    def test_crops_are_named_from_the_plate(self, tmp_path):
        pytest.importorskip('rapidocr_onnxruntime')
        pages = self._page(tmp_path)
        out = tmp_path / 'out'
        assert run_prep(pages, out, '--debug') == 0
        names = sorted(os.listdir(out / 'images'))
        assert names == ['plate_figure_1.png', 'plate_figure_2.png']
        rows = read_manifest(out)
        assert {r['label_source'] for r in rows if r['image_type'] == 'artefact'} == {'read'}
        assert os.path.isfile(out / 'pages_debug' / 'plate.png')

    def test_without_the_engine_crops_keep_index_names(self, tmp_path, monkeypatch):
        monkeypatch.setattr(identifiers_module, '_load_engine', lambda: None)
        pages = self._page(tmp_path)
        out = tmp_path / 'out'
        assert run_prep(pages, out) == 0
        assert sorted(os.listdir(out / 'images')) == ['plate_box_01.png', 'plate_box_02.png']
        rows = [r for r in read_manifest(out) if r['image_type'] == 'artefact']
        assert {r['label_source'] for r in rows} == {'index'}


@pytest.mark.unit
class TestArrowIsNotRuledScale:
    """A legend arrow is a rule with a head at one end; a scale has ticks at both."""

    def _rule_with_ticks(self, at):
        mask = np.zeros((20, 300), dtype=bool)
        mask[10:12, 10:290] = True
        for x in at:
            mask[4:10, x:x + 2] = True
        return mask

    def test_ticks_at_both_ends_pass(self):
        mask = self._rule_with_ticks([12, 100, 190, 286])
        assert grouping_module._count_ticks(mask, 10, 10, 280, {}) >= 2

    def test_head_at_one_end_fails(self):
        mask = self._rule_with_ticks([270, 280, 286])
        assert grouping_module._count_ticks(mask, 10, 10, 280, {}) == 0


@pytest.mark.unit
class TestBarCaptions:
    """A caption sits on the bar's line; an identifier beside it does not."""

    BAR = [400, 1000, 800, 1020]

    def test_numeral_above_the_bar_is_a_caption(self):
        assert grouping_module._is_bar_caption([500, 970, 512, 990], self.BAR)

    def test_unit_past_the_end_is_a_caption(self):
        assert grouping_module._is_bar_caption([810, 998, 840, 1018], self.BAR)

    def test_identifier_beside_and_above_is_not(self):
        """Continuité's "18" sat 45px above and left of its scale."""
        assert not grouping_module._is_bar_caption([340, 940, 364, 962], self.BAR)

    def test_lithic_mark_well_above_is_not(self):
        assert not grouping_module._is_bar_caption([600, 900, 612, 920], self.BAR)


@pytest.mark.unit
class TestLegendKeys:
    """An arrow beside a legend sentence is a key, not a scale."""

    def _classified(self, text_box):
        result = grouping_module.Classified()
        result.bars.append([100, 500, 420, 560])
        result.text.append(text_box)
        return result

    def test_bar_beside_a_text_line_is_discarded(self):
        result = self._classified([440, 505, 1200, 555])
        grouping_module._drop_legend_keys(result)
        assert result.bars == []

    def test_bar_with_text_below_it_is_kept(self):
        """Revue 380's caption sits under its scale, not beside it."""
        result = self._classified([100, 620, 900, 660])
        grouping_module._drop_legend_keys(result)
        assert len(result.bars) == 1


@pytest.mark.unit
class TestBrokenRule:
    """
    A thin rule the scan has cut into segments is still a scale's edge.

    The Revue de Comminges page 380 scale reached the detector as a
    555x13 blob whose longest unbroken run was 32% of its width, under
    the 45% floor, and was exported as an artefact with its numerals.
    """

    @staticmethod
    def _component(mask):
        height, width = mask.shape
        return grouping_module.Component(
            box=[100, 100, 100 + width, 100 + height],
            width=width, height=height, ink=int(mask.sum()), mask=mask,
        )

    @staticmethod
    def _rule(width=400, segments=4, gap=12, thickness=3):
        """A thin rule in segments, a short tick rising at each cut."""
        mask = np.zeros((8, width), dtype=bool)
        step = width // segments
        for index in range(segments):
            left = index * step + (gap // 2 if index else 0)
            right = (index + 1) * step - gap // 2
            mask[5:5 + thickness, left:right] = True
            mask[0:5, left] = True
        return mask

    def test_segmented_thin_rule_passes(self):
        assert grouping_module._has_straight_edge(self._component(self._rule()))

    def test_caption_line_does_not_pass(self):
        """Letters are a few percent of a line's width each."""
        mask = np.zeros((14, 400), dtype=bool)
        for left in range(0, 400, 10):
            mask[:, left:left + 7] = True
        assert not grouping_module._has_straight_edge(self._component(mask))

    def test_squat_blob_keeps_the_single_run_test(self):
        """
        An arrow legend has one long run (the shaft) among text, and is
        about 5:1. Only blobs far wider than tall may sum segments.
        """
        mask = np.zeros((100, 500), dtype=bool)
        mask[48:52, 20:180] = True             # the arrow shaft, 32%
        mask[45:55, 200:260] = True            # a word
        mask[45:55, 300:420] = True            # another word
        assert not grouping_module._has_straight_edge(self._component(mask))


@pytest.mark.unit
class TestVerticalScales:
    """
    A scale drawn down the page is found; a tall drawing is not.

    Upright scales sat inside artefact crops on five plates, and on two
    of them bridged separate rows of lithics into one crop.
    """

    PAGE = (2000, 1500)

    def _component(self, mask):
        height, width = mask.shape
        return grouping_module.Component(
            box=[900, 300, 900 + width, 300 + height],
            width=width, height=height,
            ink=int(mask.sum()), mask=mask,
        )

    def _zigzag(self, blocks=5, block=90, bar=34, step=26):
        """Blocks stepping left and right, overlapping through the centre."""
        mask = np.zeros((blocks * block, bar + step), dtype=bool)
        for index in range(blocks):
            top = index * block
            left = 0 if index % 2 == 0 else step
            mask[top:top + block, left:left + bar] = True
        return mask

    def _hollow(self, width=60, height=450):
        """A tall narrow outline, as a thin profile view is drawn."""
        mask = np.zeros((height, width), dtype=bool)
        mask[:3, :] = mask[-3:, :] = True
        mask[:, :3] = mask[:, -3:] = True
        return mask

    def test_upright_bar_is_detected(self):
        found = grouping_module._scale_bar_indices(
            [self._component(self._zigzag())], *self.PAGE, {}
        )
        assert found == {0}

    def _mark(self, width, height, y):
        """A small solid mark, as a stray rule or tick appears."""
        mask = np.ones((height, width), dtype=bool)
        return grouping_module.Component(
            box=[400, y, 400 + width, y + height],
            width=width, height=height, ink=int(mask.sum()), mask=mask,
        )

    def test_two_small_marks_are_not_a_scale(self):
        """
        Two hairlines, one above the other, spanned 8.8% of a short page
        and were both exported as scales from Saint-Marcel figure 10.
        Combining blocks belongs to the horizontal route only.
        """
        marks = [self._mark(3, 21, y=164), self._mark(2, 11, y=189)]
        found = grouping_module._scale_bar_indices(marks, 895, 407, {})
        assert found == set()

    def test_tall_outline_is_not_a_scale(self):
        """
        Shape alone is not enough: a thin outline has a straight edge
        running its whole length too. Ink coverage separates them.
        """
        found = grouping_module._scale_bar_indices(
            [self._component(self._hollow())], *self.PAGE, {}
        )
        assert found == set()


@pytest.mark.unit
class TestBarSpan:
    """A scale spans a plausible share of the page width."""

    PAGE_WIDTH = 2000

    def _component(self, x0, x1):
        return grouping_module.Component(
            box=[x0, 100, x1, 115], width=x1 - x0, height=15, ink=10
        )

    def test_running_head_rule_is_rejected(self):
        """A rule spanning 90% of the page is a header separator."""
        components = [self._component(30, 1830)]
        assert not grouping_module._plausible_bar_span(
            components, [0], self.PAGE_WIDTH
        )

    def test_short_dashed_line_is_rejected(self):
        """A pair of short dashes is too short to carry a measure."""
        components = [self._component(200, 222), self._component(240, 262)]
        assert not grouping_module._plausible_bar_span(
            components, [0, 1], self.PAGE_WIDTH
        )

    def test_typical_scale_is_accepted(self):
        components = [self._component(400, 800)]
        assert grouping_module._plausible_bar_span(
            components, [0], self.PAGE_WIDTH
        )


@pytest.mark.archaeological
class TestCrossSectionAttachment:
    """A bare outline beneath a lithic is its section, whatever its height."""

    PAGE = 2000

    def _component(self, box, hatched):
        """A rectangular view: bare outline, or with interior hatching."""
        x0, y0, x1, y1 = box
        mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        mask[:2, :] = mask[-2:, :] = True
        mask[:, :2] = mask[:, -2:] = True
        if hatched:
            mask[8:-8:10, 8:-8] = True
        return grouping_module.Component(
            box=list(box), width=x1 - x0, height=y1 - y0,
            ink=int(mask.sum()), mask=mask,
        )

    def _linked(self, section_hatched):
        """Whether a view 90% of the lithic's height joins it."""
        lithic = self._component((100, 100, 400, 400), hatched=True)
        section = self._component((100, 430, 400, 700), section_hatched)
        sets = grouping_module._UnionFind(2)
        grouping_module.link_cross_sections(
            [lithic, section], sets, self.PAGE, 0.06, self.PAGE
        )
        return sets.find(0) == sets.find(1)

    def test_bare_outline_is_recognised(self):
        view = self._component((0, 0, 300, 270), hatched=False)
        assert grouping_module._is_plain_outline(view, self.PAGE)

    def test_hatched_surface_is_not_an_outline(self):
        view = self._component((0, 0, 300, 300), hatched=True)
        assert not grouping_module._is_plain_outline(view, self.PAGE)

    def test_tall_bare_section_attaches(self):
        """
        A section 90% of its lithic's height still joins it.

        On a published plate, sections ran from 20% to 96% of their
        lithic's height, so height cannot be what decides.
        """
        assert self._linked(section_hatched=False)

    def test_tall_detailed_view_does_not_attach(self):
        """Without the outline signal, the height limit still applies."""
        assert not self._linked(section_hatched=True)

    def test_profile_shaped_outline_does_not_attach_downward(self):
        """
        A profile is a bare outline too, but belongs to the lithic beside
        it, not the one above. Attaching it upward chains one row of
        lithics to the next, which merged whole plates on the
        Saint-Marcel and Tabun pages.
        """
        lithic = self._component((100, 100, 400, 400), hatched=True)
        profile = self._component((100, 430, 190, 700), hatched=False)
        assert grouping_module._is_profile_shaped(profile)

        sets = grouping_module._UnionFind(2)
        grouping_module.link_cross_sections(
            [lithic, profile], sets, self.PAGE, 0.06, self.PAGE
        )
        assert sets.find(0) != sets.find(1)


@pytest.mark.unit
class TestPageFurniture:
    """A page-wide header rule is neither a scale nor a drawing."""

    PAGE = (2000, 2500)

    def _classify(self, component):
        return grouping_module.classify_components(
            [component], *self.PAGE, 0.0004, {}
        )

    def test_header_rule_is_discarded(self):
        """
        A rule this wide would otherwise fuse the row of artefacts
        beneath it into one crop.
        """
        rule = grouping_module.Component(
            box=[30, 30, 1830, 48], width=1800, height=18, ink=3600
        )
        classified = self._classify(rule)
        assert classified.drawings == []
        assert classified.bars == []
        assert classified.labels == []

    def test_ordinary_drawing_is_kept(self):
        drawing = grouping_module.Component(
            box=[300, 300, 700, 800], width=400, height=500, ink=20000
        )
        assert len(self._classify(drawing).drawings) == 1


@pytest.mark.unit
class TestRuledScaleSpan:
    """A ruled mark spanning most of the page is a header rule."""

    PAGE = (2000, 1500)

    def _ruled(self, length):
        """A rule with three ticks above it, as a ruled scale is drawn."""
        mask = np.zeros((21, length), dtype=bool)
        mask[3:5, 3:length - 3] = True
        for x in (3, length // 2, length - 5):
            mask[0:3, x:x + 2] = True
        return grouping_module.Component(
            box=[100, 100, 100 + length, 121], width=length, height=21,
            ink=int(mask.sum()), mask=mask,
        )

    def test_scale_sized_rule_is_detected(self):
        found = grouping_module._find_tick_scale_components(
            [self._ruled(400)], *self.PAGE, {}
        )
        assert found == {0}

    def test_page_wide_rule_is_rejected(self):
        """The rule under a running head is not a scale."""
        found = grouping_module._find_tick_scale_components(
            [self._ruled(1800)], *self.PAGE, {}
        )
        assert found == set()


### DETECTION ###

@pytest.mark.unit
class TestDetection:
    """Page loading preserves what the crops must carry."""

    def test_dpi_read_from_page(self, plate_page):
        assert plate_page.dpi is not None
        assert round(plate_page.dpi[0]) == EXPECTED_DPI

    def test_colour_mode_preserved(self, plate_page):
        assert plate_page.mode == "RGB"

    def test_greyscale_page_is_named_in_the_manifest(self, tmp_path, plate_path):
        """The manifest spells the mode out; Pillow's "L" means nothing to a reader."""
        pages = tmp_path / "pages"
        pages.mkdir()
        Image.open(plate_path).convert("L").save(pages / "grey.png", dpi=(300, 300))
        out = tmp_path / "out"
        assert run_prep(pages, out) == 0
        assert {r["colour_mode"] for r in read_manifest(out)} == {"greyscale"}

    def test_missing_dpi_is_not_guessed(self, tmp_path):
        """
        An untagged page yields untagged crops.

        Defaulting to 300 would fabricate provenance and silently
        corrupt any downstream calibration.
        """
        path = tmp_path / "no_dpi.png"
        Image.new("RGB", (400, 400), "white").save(path)
        assert load_page(str(path)).dpi is None

    def test_unreadable_file_returns_none(self, tmp_path):
        path = tmp_path / "broken.png"
        path.write_bytes(b"not an image")
        assert load_page(str(path)) is None

    def test_speckle_is_discarded(self, tmp_path):
        """Isolated scanner noise is not an artefact."""
        array = np.full((400, 400), 255, dtype=np.uint8)
        array[10, 10] = 0
        array[100:200, 100:200] = 0
        path = tmp_path / "speckle.png"
        Image.fromarray(array).save(path, dpi=(300, 300))

        page = load_page(str(path))
        closed, raw = build_detection_mask(
            page, get_config_manager().config
        )
        boxes = [c.box for c in find_components(closed, raw)]
        assert all(c[2] - c[0] > 5 for c in boxes)


### OVERRIDES ###

@pytest.mark.unit
class TestOverrides:
    """Per-page corrections keep a fixed run reproducible."""

    def _write(self, tmp_path, body):
        path = tmp_path / "corrections.csv"
        path.write_text("page_id,expect,join,split\n" + body)
        return str(path)

    def test_parses_all_three_correction_kinds(self, tmp_path):
        path = self._write(
            tmp_path,
            "a.png,5,,\nb.png,,3+4 7+8,\nc.png,,,2 5\n",
        )
        loaded = overrides_module.load_overrides(path)
        assert loaded["a.png"].expect == 5
        assert loaded["b.png"].joins == [[3, 4], [7, 8]]
        assert loaded["c.png"].splits == [2, 5]

    def test_empty_rows_are_dropped(self, tmp_path):
        path = self._write(tmp_path, "a.png,,,\n")
        assert overrides_module.load_overrides(path) == {}

    def test_missing_file_is_an_error(self):
        """A typo in the path must not silently skip corrections."""
        with pytest.raises(FileNotFoundError):
            overrides_module.load_overrides("/nonexistent/fixes.csv")

    def test_no_path_means_no_corrections(self):
        assert overrides_module.load_overrides(None) == {}

    def test_unmatched_page_warns(self, caplog):
        overrides = {"typo.png": overrides_module.PageOverride(expect=3)}
        overrides_module.warn_unmatched(overrides, ["real.png"])
        assert "typo.png" in caplog.text

    def test_join_merges_named_boxes(self):
        boxes = [[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]]
        merged = overrides_module.apply_joins(boxes, [[1, 2]])
        assert len(merged) == 2
        assert [0, 0, 30, 10] in merged

    def test_join_out_of_range_is_ignored(self, caplog):
        boxes = [[0, 0, 10, 10]]
        assert overrides_module.apply_joins(boxes, [[1, 9]]) == boxes

    def test_split_cuts_at_widest_empty_column(self):
        array = np.full((100, 200), 255, dtype=np.uint8)
        array[20:80, 10:40] = 0
        array[20:80, 160:190] = 0
        result = overrides_module.apply_splits(
            [[0, 0, 200, 100]], [1], array
        )
        assert len(result) == 2
        assert result[0][2] == result[1][0]

    def test_split_without_gap_is_ignored(self, caplog):
        array = np.zeros((100, 200), dtype=np.uint8)
        result = overrides_module.apply_splits(
            [[0, 0, 200, 100]], [1], array
        )
        assert len(result) == 1


### OUTPUT SAFETY ###

def _write_previous_run(out):
    """Leave the files of an earlier pylithics-pages run in ``out``."""
    (out / "images" / "plate_01.png").write_bytes(b"")
    (out / "scales" / "plate_scale_bar.png").write_bytes(b"")
    (out / "meta_data.csv").write_text("image_id,scale_id,scale,flag\n")
    (out / "pages_manifest.csv").write_text(
        "output_crop_id,image_type,input_page_id\n"
        "plate_01.png,artefact,plate.png\n"
        "plate_scale_bar.png,scale_bar,plate.png\n"
    )


def _fill_scales(out, value="50"):
    """Type a scale value into every row, as the user would, and return the text."""
    path = out / "meta_data.csv"
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames, rows = reader.fieldnames, list(reader)
    for row in rows:
        row["scale"] = value
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path.read_text()


@pytest.mark.unit
class TestOutputSafety:
    """A run replaces its own previous output and nothing else."""

    def test_creates_expected_layout(self, tmp_path):
        assert prepare_output_dir(str(tmp_path / "out")) == []
        assert (tmp_path / "out" / "images").is_dir()
        assert (tmp_path / "out" / "scales").is_dir()

    def test_a_previous_run_is_returned(self, tmp_path):
        out = tmp_path / "out"
        prepare_output_dir(str(out))
        _write_previous_run(out)
        previous = prepare_output_dir(str(out))
        assert [row["input_page_id"] for row in previous] == ["plate.png"] * 2

    def test_remove_page_output_removes_only_that_pages_crops(self, tmp_path):
        out = tmp_path / "out"
        previous = prepare_output_dir(str(out))
        _write_previous_run(out)
        (out / "images" / "my_flake.png").write_bytes(b"")
        previous = prepare_output_dir(str(out))

        removed = remove_page_output(str(out), previous)

        assert removed == {"plate_01.png", "plate_scale_bar.png"}
        assert not (out / "images" / "plate_01.png").exists()
        assert not (out / "scales" / "plate_scale_bar.png").exists()
        assert (out / "images" / "my_flake.png").exists()

    def test_a_second_run_replaces_its_crops_and_keeps_typed_scales(
        self, tmp_path, pages_dir
    ):
        out = tmp_path / "out"
        assert run_prep(pages_dir, out) == 0
        before = sorted(os.listdir(out / "images"))
        manifest_before = read_manifest(out)
        meta = _fill_scales(out)

        assert run_prep(pages_dir, out) == 0

        assert sorted(os.listdir(out / "images")) == before
        assert read_manifest(out) == manifest_before
        assert (out / "meta_data.csv").read_text() == meta

    def test_added_pages_are_cut_and_appended(self, tmp_path, pages_dir, plate_path):
        """Adding a plate adds its crops. The rest is cut again, unchanged."""
        out = tmp_path / "out"
        assert run_prep(pages_dir, out) == 0
        meta = _fill_scales(out)
        Image.open(plate_path).save(
            os.path.join(pages_dir, "second_plate.png"),
            dpi=(EXPECTED_DPI, EXPECTED_DPI),
        )

        assert run_prep(pages_dir, out) == 0

        pages = {row["input_page_id"] for row in read_manifest(out)}
        assert pages == {"sample_plate.png", "second_plate.png"}
        assert (out / "meta_data.csv").read_text().startswith(meta)
        rows = _read_metadata(out)
        assert sum(name.startswith("second_plate") for name in rows) > 0

    def test_a_stale_crop_of_a_recut_page_is_removed(self, tmp_path, pages_dir):
        """A crop the new run does not produce does not linger."""
        out = tmp_path / "out"
        assert run_prep(pages_dir, out) == 0
        stale = out / "images" / "sample_plate_99.png"
        stale.write_bytes(b"")
        with open(out / "pages_manifest.csv", "a", newline="") as handle:
            handle.write("sample_plate_99.png,artefact,sample_plate.png\n")
        with open(out / "meta_data.csv", "a", newline="") as handle:
            handle.write("sample_plate_99.png,,,no_scale\n")

        assert run_prep(pages_dir, out) == 0

        assert not stale.exists()
        assert "sample_plate_99.png" not in _read_metadata(out)
        assert all(
            row["output_crop_id"] != "sample_plate_99.png"
            for row in read_manifest(out)
        )

    def test_the_users_own_files_and_rows_survive_a_rerun(self, tmp_path, pages_dir):
        out = tmp_path / "out"
        prepare_output_dir(str(out))
        (out / "images" / "my_flake.png").write_bytes(b"mine")
        (out / "meta_data.csv").write_text(
            "image_id,scale_id,scale\nmy_flake.png,sc_001,50\n"
        )
        assert run_prep(pages_dir, out) == 0
        assert run_prep(pages_dir, out) == 0
        assert (out / "images" / "my_flake.png").read_bytes() == b"mine"
        assert _read_metadata(out)["my_flake.png"]["scale"] == "50"

    def test_a_crop_never_replaces_an_existing_image(self, tmp_path, pages_dir):
        """A page whose crop name is taken is refused whole, before any write."""
        out = tmp_path / "out"
        prepare_output_dir(str(out))
        taken = out / "images" / "sample_plate_box_01.png"
        taken.write_bytes(b"mine")
        assert run_prep(pages_dir, out) == 0
        assert taken.read_bytes() == b"mine"
        assert os.listdir(out / "images") == ["sample_plate_box_01.png"]
        assert _read_metadata(out) == {}


### METADATA FOR THE ANALYSIS ###

def _row(crop_id, image_type, page, label_flag=''):
    """A minimal manifest row."""
    return ManifestRow(
        output_crop_id=crop_id, image_type=image_type, input_page_id=page,
        crop_id_index='1', x0=0, y0=0, x1=1, y1=1, width_px=1, height_px=1,
        dpi='300', colour_mode='RGB', n_components=1, correction_applied='',
        label_flag=label_flag,
    )


def _read_metadata(out):
    with open(os.path.join(str(out), "meta_data.csv"), newline="") as handle:
        return {row["image_id"]: row for row in csv.DictReader(handle)}


@pytest.mark.unit
class TestMetadataFile:
    """meta_data.csv links each crop to its scale and flags the rest."""

    def test_columns_match_what_pylithics_reads(self, tmp_path):
        write_metadata([_row("p_01.png", "artefact", "p.png")], str(tmp_path))
        with open(tmp_path / "meta_data.csv", newline="") as handle:
            assert next(csv.reader(handle)) == [
                "image_id", "scale_id", "scale", "flag"
            ]

    def test_one_scale_on_the_page_is_linked_with_no_flag(self, tmp_path):
        rows = [
            _row("p_01.png", "artefact", "p.png"),
            _row("p_02.png", "artefact", "p.png"),
            _row("p_scale_bar.png", "scale_bar", "p.png"),
        ]
        write_metadata(rows, str(tmp_path))
        meta = _read_metadata(tmp_path)
        assert set(meta) == {"p_01.png", "p_02.png"}
        assert meta["p_01.png"]["scale_id"] == "p_scale_bar.png"
        assert meta["p_01.png"]["scale"] == ""
        assert meta["p_01.png"]["flag"] == ""

    def test_no_scale_on_the_page_is_flagged(self, tmp_path):
        write_metadata([_row("p_01.png", "artefact", "p.png")], str(tmp_path))
        meta = _read_metadata(tmp_path)
        assert meta["p_01.png"]["scale_id"] == ""
        assert meta["p_01.png"]["flag"] == "no_scale"

    def test_several_scales_on_the_page_are_left_for_the_user(self, tmp_path):
        rows = [
            _row("p_01.png", "artefact", "p.png"),
            _row("p_scale_bar_01.png", "scale_bar", "p.png"),
            _row("p_scale_bar_02.png", "scale_bar", "p.png"),
        ]
        write_metadata(rows, str(tmp_path))
        meta = _read_metadata(tmp_path)
        assert meta["p_01.png"]["scale_id"] == ""
        assert meta["p_01.png"]["flag"] == "several_scales"

    def test_identifier_flag_is_carried_over(self, tmp_path):
        rows = [
            _row("p_01.png", "artefact", "p.png", label_flag="no_identifier"),
            _row("p_scale_bar.png", "scale_bar", "p.png"),
        ]
        write_metadata(rows, str(tmp_path))
        assert _read_metadata(tmp_path)["p_01.png"]["flag"] == "no_identifier"

    def test_flags_from_both_sources_are_joined(self, tmp_path):
        rows = [_row("p_01.png", "artefact", "p.png", label_flag="duplicate")]
        write_metadata(rows, str(tmp_path))
        assert _read_metadata(tmp_path)["p_01.png"]["flag"] == "no_scale;duplicate"

    def test_rows_are_added_after_the_users_rows(self, tmp_path):
        """A hand-written meta_data.csv gains a flag column and the new rows."""
        (tmp_path / "meta_data.csv").write_text(
            "image_id,scale_id,scale\nmy_flake.png,sc_001,50\n"
        )
        rows = [
            _row("p_01.png", "artefact", "p.png"),
            _row("p_scale_bar.png", "scale_bar", "p.png"),
        ]
        write_metadata(rows, str(tmp_path))
        with open(tmp_path / "meta_data.csv", newline="") as handle:
            table = list(csv.reader(handle))
        assert table == [
            ["image_id", "scale_id", "scale", "flag"],
            ["my_flake.png", "sc_001", "50", ""],
            ["p_01.png", "p_scale_bar.png", "", ""],
        ]

    def test_a_row_cut_again_keeps_its_typed_scale(self, tmp_path):
        (tmp_path / "meta_data.csv").write_text(
            "image_id,scale_id,scale,flag\np_01.png,old_bar.png,50,\n"
        )
        rows = [
            _row("p_01.png", "artefact", "p.png"),
            _row("p_scale_bar.png", "scale_bar", "p.png"),
        ]
        write_metadata(rows, str(tmp_path))
        meta = _read_metadata(tmp_path)
        assert meta["p_01.png"]["scale"] == "50"
        assert meta["p_01.png"]["scale_id"] == "p_scale_bar.png"
        assert len(meta) == 1

    def test_a_removed_crop_loses_its_row(self, tmp_path):
        (tmp_path / "meta_data.csv").write_text(
            "image_id,scale_id,scale,flag\np_07.png,bar.png,50,\n"
        )
        write_metadata([_row("p_01.png", "artefact", "p.png")], str(tmp_path),
                       removed={"p_07.png"})
        assert set(_read_metadata(tmp_path)) == {"p_01.png"}

    def test_scales_from_another_page_are_not_linked(self, tmp_path):
        rows = [
            _row("a_01.png", "artefact", "a.png"),
            _row("b_scale_bar.png", "scale_bar", "b.png"),
        ]
        write_metadata(rows, str(tmp_path))
        assert _read_metadata(tmp_path)["a_01.png"]["flag"] == "no_scale"


### END TO END ###

@pytest.mark.integration
class TestPipeline:
    """The full page-to-crops workflow."""

    def test_plate_produces_expected_outputs(self, tmp_path, pages_dir):
        out = tmp_path / "out"
        assert run_prep(pages_dir, out) == 0

        images = sorted(os.listdir(out / "images"))
        scales = sorted(os.listdir(out / "scales"))
        assert len(images) == EXPECTED_ARTEFACTS
        assert scales == ["sample_plate_scale_bar.png"]
        assert images[0] == "sample_plate_box_01.png"

        meta = _read_metadata(out)
        assert set(meta) == set(images)
        assert meta["sample_plate_box_01.png"]["scale_id"] == "sample_plate_scale_bar.png"

    def test_default_output_is_the_project_folder(self, tmp_path, pages_dir):
        """pages/ in, images/ and scales/ out, in the same folder."""
        project = os.path.dirname(pages_dir)
        assert main(["--data_dir", project]) == 0
        assert (tmp_path / "images").is_dir()
        assert (tmp_path / "scales").is_dir()
        assert (tmp_path / "pages_manifest.csv").is_file()
        assert (tmp_path / "meta_data.csv").is_file()

    def test_crop_pixels_are_identical_to_the_page(
        self, tmp_path, pages_dir, plate_path
    ):
        """
        Crops are cut, never enhanced.

        The main pipeline owns all denoising, contrast, and
        thresholding. Doing any of it here would make a measurement
        from a crop non-comparable to one from a single-artefact scan,
        so this asserts exact equality rather than a tolerance.
        """
        out = tmp_path / "out"
        run_prep(pages_dir, out)
        page = np.array(Image.open(plate_path))

        for row in read_manifest(out):
            subdir = (
                "images" if row["image_type"] == "artefact" else "scales"
            )
            crop = np.array(
                Image.open(out / subdir / row["output_crop_id"])
            )
            expected = page[
                int(row["y0"]):int(row["y1"]),
                int(row["x0"]):int(row["x1"]),
            ]
            assert np.array_equal(crop, expected), row["output_crop_id"]

    def test_dpi_and_mode_preserved(self, tmp_path, pages_dir):
        out = tmp_path / "out"
        run_prep(pages_dir, out)

        for name in os.listdir(out / "images"):
            with Image.open(out / "images" / name) as crop:
                assert round(crop.info["dpi"][0]) == EXPECTED_DPI
                assert crop.mode == "RGB"

    def test_manifest_records_every_crop(self, tmp_path, pages_dir):
        out = tmp_path / "out"
        run_prep(pages_dir, out)
        rows = read_manifest(out)

        assert list(rows[0]) == MANIFEST_COLUMNS
        assert len(rows) == EXPECTED_ARTEFACTS + EXPECTED_SCALE_BARS
        artefacts = [r for r in rows if r["image_type"] == "artefact"]
        assert [r["crop_id_index"] for r in artefacts] == ["1", "2", "3", "4", "5"]
        assert all(r["input_page_id"] == "sample_plate.png" for r in rows)

    def test_manifest_counts_grouped_components(self, tmp_path, pages_dir):
        """Each artefact groups several surface views, not one blob."""
        out = tmp_path / "out"
        run_prep(pages_dir, out)
        artefacts = [
            r for r in read_manifest(out) if r["image_type"] == "artefact"
        ]
        assert all(int(r["n_components"]) > 1 for r in artefacts)

    def test_debug_overlay_written_on_request(self, tmp_path, pages_dir, caplog):
        out = tmp_path / "out"
        run_prep(pages_dir, out, "--debug")
        assert (out / "pages_debug" / "sample_plate.png").is_file()
        assert "Debug overlays:" in caplog.text

    def test_scale_bar_export_can_be_disabled(self, tmp_path, pages_dir):
        out = tmp_path / "out"
        run_prep(pages_dir, out, "--disable_scale_bars")
        assert os.listdir(out / "scales") == []

    def test_run_is_reproducible(self, tmp_path, pages_dir):
        """Re-running a folder reproduces the same output."""
        first, second = tmp_path / "a", tmp_path / "b"
        run_prep(pages_dir, first)
        run_prep(pages_dir, second)

        assert read_manifest(first) == read_manifest(second)
        for name in os.listdir(first / "images"):
            assert (first / "images" / name).read_bytes() == (
                second / "images" / name
            ).read_bytes()

    def test_cli_override_changes_grouping(self, tmp_path, pages_dir):
        """A CLI value takes precedence over the YAML default."""
        out = tmp_path / "out"
        run_prep(pages_dir, out, "--gap", "0.4", "--narrow", "0.4")
        assert len(os.listdir(out / "images")) < EXPECTED_ARTEFACTS


### EDGE CASES ###

@pytest.mark.error_scenarios
class TestEdgeCases:
    """Degenerate input fails cleanly rather than silently."""

    def test_blank_page_yields_no_crops(self, tmp_path):
        pages = tmp_path / "pages"
        pages.mkdir()
        Image.new("RGB", (1000, 1000), "white").save(
            pages / "blank.png", dpi=(300, 300)
        )
        out = tmp_path / "out"

        assert run_prep(pages, out) == 0
        assert os.listdir(out / "images") == []

    def test_single_artefact_page_yields_one_crop(self, tmp_path):
        pages = tmp_path / "pages"
        pages.mkdir()
        array = np.full((1000, 1000), 255, dtype=np.uint8)
        array[300:700, 300:700] = 0
        Image.fromarray(array).save(pages / "one.png", dpi=(300, 300))
        out = tmp_path / "out"

        assert run_prep(pages, out) == 0
        assert len(os.listdir(out / "images")) == 1

    def test_grayscale_page_stays_grayscale(self, tmp_path):
        pages = tmp_path / "pages"
        pages.mkdir()
        array = np.full((1000, 1000), 255, dtype=np.uint8)
        array[300:700, 300:700] = 0
        Image.fromarray(array, mode="L").save(
            pages / "gray.png", dpi=(300, 300)
        )
        out = tmp_path / "out"
        run_prep(pages, out)

        name = os.listdir(out / "images")[0]
        with Image.open(out / "images" / name) as crop:
            assert crop.mode == "L"

    def test_unreadable_page_does_not_stop_the_batch(
        self, tmp_path, plate_path
    ):
        """One corrupt file must not cost the whole folder."""
        pages = tmp_path / "pages"
        pages.mkdir()
        (pages / "broken.png").write_bytes(b"not an image")
        Image.open(plate_path).save(
            pages / "sample_plate.png", dpi=(300, 300)
        )
        out = tmp_path / "out"

        assert run_prep(pages, out) == 0
        assert len(os.listdir(out / "images")) == EXPECTED_ARTEFACTS

    def test_pdf_is_reported_as_unsupported(self, tmp_path, caplog):
        """
        A PDF names itself in the log rather than vanishing.

        Supplying a paper as a PDF is the commonest mistake, and it looks
        like valid input to whoever supplies it.
        """
        pages = tmp_path / "pages"
        pages.mkdir()
        (pages / "Smith 2019.pdf").write_bytes(b"%PDF-1.4")

        with caplog.at_level("WARNING"):
            assert run_prep(pages, tmp_path / "out") == 0

        assert "PDF" in caplog.text
        assert "Smith 2019.pdf" in caplog.text

    def test_unreadable_format_names_the_file(self, tmp_path, caplog):
        """Any unsupported extension is named, not silently skipped."""
        pages = tmp_path / "pages"
        pages.mkdir()
        (pages / "scan.webp").write_bytes(b"x")

        with caplog.at_level("WARNING"):
            run_prep(pages, tmp_path / "out")

        assert "scan.webp" in caplog.text
        assert "not supported" in caplog.text.lower()

    def test_supported_pages_still_process_alongside_skipped(
        self, tmp_path, plate_path
    ):
        """An unreadable file does not stop the rest of the batch."""
        pages = tmp_path / "pages"
        pages.mkdir()
        (pages / "paper.pdf").write_bytes(b"%PDF-1.4")
        Image.open(plate_path).save(
            pages / "sample_plate.png", dpi=(300, 300)
        )
        out = tmp_path / "out"

        assert run_prep(pages, out) == 0
        assert len(os.listdir(out / "images")) == EXPECTED_ARTEFACTS

    def test_missing_input_directory_is_reported(self, tmp_path):
        assert run_prep(tmp_path / "absent", tmp_path / "out") == 1

    def test_empty_input_directory_is_not_an_error(self, tmp_path):
        pages = tmp_path / "pages"
        pages.mkdir()
        assert run_prep(pages, tmp_path / "out") == 0


### CONFIGURATION ###

@pytest.mark.unit
class TestConfiguration:
    """Configuration follows the established PyLithics pattern."""

    def test_defaults_available_without_yaml(self):
        config = get_page_segmentation_config({})
        assert config['enabled'] is True
        assert config['grouping']['gap'] == 0.025
        assert config['scale_bars']['include_caption'] is True

    def test_section_present_in_shipped_yaml(self):
        clear_config_cache()
        config = get_page_segmentation_config(
            get_config_manager().config
        )
        assert set(config) >= {
            'enabled', 'grouping', 'export', 'scale_bars', 'debug'
        }

    def test_yaml_values_used_when_present(self):
        config = get_page_segmentation_config(
            {'page_segmentation': {'grouping': {'gap': 0.99}}}
        )
        assert config['grouping']['gap'] == 0.99
