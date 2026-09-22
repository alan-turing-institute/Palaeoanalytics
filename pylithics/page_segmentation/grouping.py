"""
Grouping of ink blobs into whole artefacts.

A lithic is conventionally drawn as a set of adjacent surface views —
platform, dorsal, ventral, lateral — sometimes tied together by a dash
or rule mark, and usually captioned with an identifier. The views of one
artefact must end up in one crop, so this module links blobs before any
cropping happens.

Four linking rules run in sequence, each capturing a drawing convention:

- adjacent views      side-by-side surfaces separated by a small gap
- cross sections      a short view directly beneath a taller one
- profiles            a thin outline drawn beside the surface it belongs to
- dash connectors     views explicitly tied by a rule mark

Blobs too small to be artefacts (identifier letters, flaking arrows) are
attached to the nearest artefact rather than becoming artefacts, while
caption and legend text is discarded. Scale bars are separated out: they
belong to the page, not to any one artefact.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .detection import Component
from .geometry import (
    BBox,
    box_distance,
    box_height,
    box_width,
    horizontal_gap,
    horizontal_overlap,
    union,
    vertical_gap,
    vertical_overlap,
)

# A dash connector is at most this fraction of page width or height.
# Sized from the reference plates, where connector rules run to ~9% of
# page width; at the previous 0.06 they were missed entirely and the
# surfaces they tie together were cropped as separate lithics.
_MAX_DASH_EXTENT = 0.10
# Minimum dash length, as a fraction of the page, to be a real connector.
_MIN_DASH_LENGTH = 0.007
# Dash thickness ceiling, as a fraction of the page.
_DASH_THICKNESS = 0.006
# A label blob is smaller than this fraction of page height or width.
_LABEL_MAX_HEIGHT = 0.03
_LABEL_MAX_WIDTH = 0.05
# Ink coverage separating annotation marks from hollow outline drawings.
# Measured across the reference plates: outline drawings 0.02-0.11,
# letters and flaking arrows 0.24-0.54. Open glyphs sit between — the
# "A" of a plate signature measures 0.17 — so the floor is placed in the
# observed gap. Above 0.15 that glyph is read as a drawing and becomes a
# spurious artefact; below 0.11 real surface views start to be lost.
_MIN_ANNOTATION_DENSITY = 0.15
# Segments of one scale bar sit within this fraction of page width.
_BAR_SEGMENT_GAP = 0.05
# A scale bar carries evidence of measurement: alternating filled and
# open blocks, tick marks, or numbers. A mark that is completely solid
# shows none of these and is a connector rule, drawn between two views
# to say they are the same lithic. Measured on the reference plates:
# real bars 0.40 and 0.77, connector rules 1.00.
_MAX_SOLID_DENSITY = 0.85
# Scale bar blocks are filled, unlike the hollow outlines of a drawing.
# A bar may arrive as separate filled blocks (density ~1.0) or, where its
# blocks are joined by a drawn border, as one component whose box includes
# the open blocks (density ~0.4). Both must be recognised, so the floor
# sits below the whole-bar case while staying clear of hatched outlines,
# which measure 0.02-0.24 on the sample pages.
_MIN_BAR_DENSITY = 0.35
# A bar's straight edge spans at least this fraction of its own width.
# The lowest across the reference scales is 0.50 (the Bulletin plate).
_MIN_BAR_EDGE_SPAN = 0.45
# A rule the scan has broken into segments is still a rule when the blob
# is this many times wider than tall. The Revue 380 scale is 43:1; the
# arrow legend that would otherwise qualify on the same terms is 5:1.
_MIN_SEGMENTED_ASPECT = 20
# A segment counts toward a broken rule's edge when at least this
# fraction of the blob's width. Letters are a few percent of a caption.
_MIN_BAR_SEGMENT = 0.10
# Two boxes are one artefact only when they genuinely contain one another.
# Diagonally adjacent artefacts on a dense plate have bounding boxes that
# clip at the corners, so a bare intersection test fuses unrelated lithics.
# Set from visual inspection of the reference plates, not from artefact
# counts: at 0.05 the Mobility plate merges two lithics in its left column
# while separately over-splitting a pair on the right, which happens to
# yield the correct total from two cancelling errors.
_MIN_CONTAINMENT = 0.10
# A scale's caption sits within this many glyph heights above or below
# the bar, and within this many past its end ("cm" after the last
# numeral). A lithic's identifier beside a scale sits further off.
_CAPTION_LINE_GAP = 1.5
_CAPTION_END_GAP = 2.0
# A caption line spanning more than this fraction of page width is text.
_CAPTION_LINE_WIDTH = 0.06
# Annotations further than this fraction of page width from any artefact
# are left out rather than dragged into a distant crop.
_ANNOTATION_REACH = 0.06
# A view overlapping vertically by less than this fraction is not a
# side-by-side neighbour.
_SIDE_BY_SIDE_OVERLAP = 0.3
# The cross-section rule asks two different questions, which need
# different answers:
#
#   _SECTION_ATTACHED_RATIO - "is this view already subordinate to a
#       taller one in its own group?" If so it is part of that artefact
#       and must not drag the group elsewhere.
#   _SECTION_HOST_RATIO - "is the candidate host taller enough than this
#       view for it to read as a section of it?"
#
# They are kept separate because they are genuinely different questions,
# but both currently sit at 0.4. Raising the host ratio to 0.6 makes the
# Flake plate return exactly 13 artefacts while being visibly wrong: whole
# columns zip together as each section attaches upward and the next flake
# then attaches to the enlarged group. Counts are not a quality measure.
_SECTION_ATTACHED_RATIO = 0.4
_SECTION_HOST_RATIO = 0.4
# A profile is narrower than this fraction of its own height. The same
# test keeps a profile from being taken for a lone cross-section: both
# are bare outlines, but a section is drawn wide and flat beneath its
# lithic while a profile stands tall beside it. Measured across three
# plates: sections 1.15-2.98 wide to tall, profiles 0.10-0.49.
_PROFILE_ASPECT = 0.5
# A ruled scale's spine spans at least this fraction of the component.
# Measured on the Mobility plate, where the rule runs 61px of an 88px
# component (0.69); a hand-drawn rule may fall short of its own extent.
_MIN_SPINE_SPAN = 0.5
# A scale spans a meaningful part of the page, measured along its own
# axis. Without this, a letter reads as a ruled scale: the stem of an
# "E" is a spine and its three bars are ticks. Real scales on the
# reference plates run from 7% to 45% of the page dimension they lie
# along; letters and digits fall well below.
_MIN_SPINE_PAGE_SPAN = 0.05
# A drawn rule is thin. When the run of near-solid rows keeps growing,
# the "spine" is a filled region rather than a rule — a stippled lithic
# has hundreds of near-full rows, and removing them all leaves only
# fragments, which would otherwise pass the residual test.
_MAX_SPINE_THICKNESS = 0.2
# A lone block bar shows its measure as alternating filled and open
# blocks. One unbroken run of ink is a plain rule — a running head
# separator or a frame edge — and carries no measure at all.
_MIN_BAR_ALTERNATIONS = 2
# A block bar spans a plausible share of the page width. Measured across
# 27 real scales in a 42-page run: 10.2% to 37.3%. Running-head rules
# under a page header span 90% or more; short dashed lines under 3%.
_MIN_BAR_PAGE_SPAN = 0.05
_MAX_BAR_PAGE_SPAN = 0.6
# A page-wide mark shorter than this fraction of page height is a rule
# under a running head, or the running head itself, discarded before
# grouping. Raised from 0.03 for strip-shaped pages: the Saint-Marcel
# running head is 90% of the page wide and 5.2% tall. No illustration is
# 60% of a page wide and 6% tall.
_FURNITURE_MAX_HEIGHT = 0.06
# Text is a row of letter-sized pieces of raw ink. A piece counts as a
# letter when at least this fraction of the component's height, which
# excludes stipple and punctuation but keeps x-height letters.
_TEXT_PIECE_HEIGHT = 0.4
# Letter bottoms sit on one baseline: within this fraction of the
# component's height of the median bottom. Descenders fall inside it.
_TEXT_BASELINE_TOLERANCE = 0.25
# A band of ink shorter than this is an accent or a dot, not a line.
_MIN_BAND_ROWS = 4
# A spine shorter than this is too small to measure anything with.
_MIN_SPINE_PIXELS = 20
# A tick is no wider than this fraction of the spine it sits on.
_MAX_TICK_WIDTH = 0.15
# A row belongs to the drawn rule itself when this fraction of the
# spine's span is inked, rather than being a tick attached to it.
_SPINE_THICKNESS_COVER = 0.8
# A ruled scale's outermost ticks lie within this fraction of the rule's
# length from each end. An arrow's head sits at one end only.
_TICK_END_FRACTION = 0.3
# Minimum ticks for a ruled scale: one at each end of the measure.
_MIN_TICK_COUNT = 2
# With the rule deleted, no surviving piece may exceed this share of the
# component's ink. A scale falls apart into ticks and numerals; an
# artefact outline with a straight edge stays largely whole.
_MAX_SPINE_RESIDUAL = 0.35
# Upper bound on section links per page. Links are made one at a time,
# so this is a runaway guard rather than a tuning value.
_MAX_SECTION_LINKS = 200
# Bare outlines - cross-sections and platform views - keep little ink
# away from their edge; lithic surfaces carry scar ridges, hatching or
# stipple inside it. Measured across three reference plates: outlines
# 0.00-0.26; surfaces and hatched sections 0.48-0.82.
_MAX_OUTLINE_INTERIOR = 0.35
# Ink lying this far inside a silhouette, as a fraction of the smaller
# page dimension, counts as interior detail rather than the edge itself.
_INTERIOR_MARGIN = 0.003
# Background added around a mask before measuring its silhouette. Wider
# than the closing kernel's reach, so closing cannot fill it back in.
_SILHOUETTE_PAD = 10


@dataclass
class Classified:
    """
    Page components sorted by the role they play in the illustration.

    Attributes
    ----------
    drawings : list of Component
        Candidate artefact views.
    labels : list of BBox
        Identifier letters and flaking arrows, attached to artefacts.
    dashes : list of tuple
        ``(orientation, box)`` connector marks, orientation ``h`` or ``v``.
    bars : list of BBox
        Scale bars and rule lines belonging to the page.
    bar_captions : list of BBox
        Small text sitting beside a bar, such as ``5 cm``.
    text : list of BBox
        Captions, legend words and keys, discarded before linking. Kept
        so the small remnants of the same line can be discarded too.
    """

    drawings: List[Component] = field(default_factory=list)
    labels: List[BBox] = field(default_factory=list)
    dashes: List[Tuple[str, BBox]] = field(default_factory=list)
    bars: List[BBox] = field(default_factory=list)
    bar_captions: List[BBox] = field(default_factory=list)
    text: List[BBox] = field(default_factory=list)


class _UnionFind:
    """Disjoint-set forest over drawing indices, with path compression."""

    def __init__(self, size: int):
        self._parent = list(range(size))

    def find(self, item: int) -> int:
        """Return the representative of an item's set."""
        while self._parent[item] != item:
            self._parent[item] = self._parent[self._parent[item]]
            item = self._parent[item]
        return item

    def link(self, left: int, right: int) -> None:
        """Merge the sets containing two items."""
        self._parent[self.find(left)] = self.find(right)


def group_components(
    components: Sequence[Component],
    page_size: Tuple[int, int],
    config: Dict,
) -> Tuple[List[BBox], List[BBox]]:
    """
    Group ink blobs into one box per artefact.

    Parameters
    ----------
    components : sequence of Component
        Ink blobs found on the page.
    page_size : tuple of int
        Page ``(width, height)`` in pixels.
    config : dict
        Page segmentation configuration.

    Returns
    -------
    tuple of list
        ``(artefact_boxes, scale_bar_boxes)``. Artefact boxes are in
        reading order; scale bar boxes are ordered top to bottom.
    """
    width, height = page_size
    grouping = config.get('grouping', {})
    bar_config = config.get('scale_bars', {})

    classified = classify_components(
        components, width, height,
        grouping.get('min_area', 0.0004),
        bar_config,
        config.get('text_rejection', {}),
    )
    classified.labels = reject_caption_lines(classified.labels, width)

    boxes = _link_drawings(classified, width, height, grouping)
    boxes = merge_overlapping(boxes)
    boxes = attach_annotations(classified, boxes, width)

    bars = build_scale_bars(classified, bar_config, width)

    logging.debug(
        "Grouped %d components into %d artefacts and %d scale bars",
        len(components), len(boxes), len(bars),
    )
    return reading_order(boxes), sorted(bars, key=lambda b: b[1])


def _link_drawings(
    classified: Classified,
    width: int,
    height: int,
    grouping: Dict,
) -> List[BBox]:
    """Apply every linking rule, then collapse each set into one box."""
    drawings = classified.drawings
    sets = _UnionFind(len(drawings))

    link_adjacent_views(drawings, sets, width, grouping.get('gap', 0.025))
    link_cross_sections(
        drawings, sets, height, grouping.get('vertical_gap', 0.06), width,
    )
    link_profiles(drawings, sets, width, grouping.get('narrow', 0.07))
    link_dash_connectors(
        classified, sets, width, height, grouping.get('bridge', 0.06)
    )

    return _collapse_sets(drawings, sets)


def _collapse_sets(
    drawings: Sequence[Component], sets: _UnionFind
) -> List[BBox]:
    """Reduce each disjoint set to its enclosing box."""
    groups: Dict[int, BBox] = {}
    for index, drawing in enumerate(drawings):
        root = sets.find(index)
        groups[root] = (
            union(groups[root], drawing.box) if root in groups
            else list(drawing.box)
        )
    return list(groups.values())


### COMPONENT CLASSIFICATION ###

def classify_components(
    components: Sequence[Component],
    width: int,
    height: int,
    min_area_frac: float,
    bar_config: Dict,
    text_config: Optional[Dict] = None,
) -> Classified:
    """
    Sort components into drawings, labels, dashes, and scale bars.

    Text and legend keys are recognised here and discarded before linking.

    Parameters
    ----------
    components : sequence of Component
        Ink blobs found on the page.
    width, height : int
        Page dimensions in pixels.
    min_area_frac : float
        Blobs with less ink than this fraction of page area are labels.
    bar_config : dict
        Scale bar configuration.
    text_config : dict, optional
        Text rejection configuration. Defaults apply when omitted.

    Returns
    -------
    Classified
        Components sorted by role.
    """
    result = Classified()
    bar_indices = _scale_bar_indices(components, width, height, bar_config)
    min_ink = min_area_frac * width * height

    for position, component in enumerate(components):
        if position in bar_indices:
            result.bars.append(component.box)
            continue
        role = _component_role(
            component, width, height, min_ink, text_config or {}
        )
        _assign_role(result, role, component)

    _drop_text_line_remnants(result)
    _drop_legend_keys(result)
    result.bar_captions, result.labels = _split_bar_captions(
        result.labels, result.bars
    )
    return result


def _scale_bar_indices(
    components: Sequence[Component], width: int, height: int, bar_config: Dict
) -> set:
    """
    Identify every component belonging to a scale bar, of either style.

    Scale bars are found before any other role is assigned: a segmented
    bar breaks into short solid blocks at threshold, and a lone block is
    indistinguishable from a connector dash until its neighbours are
    taken into account.
    """
    return (
        _find_bar_components(components, width, height, bar_config)
        | _find_vertical_bar_components(components, width, height, bar_config)
        | _find_tick_scale_components(components, width, height, bar_config)
    )


def _assign_role(
    result: Classified, role: str, component: Component
) -> None:
    """File a classified component under its role."""
    if role == 'furniture':
        return
    if role == 'text':
        result.text.append(component.box)
    elif role == 'dash_h':
        result.dashes.append(('h', component.box))
    elif role == 'dash_v':
        result.dashes.append(('v', component.box))
    elif role == 'label':
        result.labels.append(component.box)
    else:
        result.drawings.append(component)


def _component_role(
    component: Component,
    width: int,
    height: int,
    min_ink: float,
    text_config: Dict,
) -> str:
    """
    Classify a single component by shape and ink coverage.

    Text is tested last, on what would otherwise become a drawing, so an
    identifier numeral is filed as a label before the text rules can
    see it.
    """
    box_w, box_h = component.width, component.height

    if _is_page_furniture(box_w, box_h, width, height):
        return 'furniture'
    if _is_horizontal_dash(box_w, box_h, width):
        return 'dash_h'
    if _is_vertical_dash(box_w, box_h, height):
        return 'dash_v'
    if _is_label(component, width, height, min_ink):
        return 'label'
    if _is_text(component, width, height, text_config):
        return 'text'
    return 'drawing'


def _is_page_furniture(
    box_w: int, box_h: int, width: int, height: int
) -> bool:
    """
    Report whether a mark is page furniture rather than illustration.

    The rule beneath a running head spans most of the page and is only a
    few pixels tall. It is not a scale, and it must not become a drawing
    either: a mark that wide touches every artefact in the row below and
    fuses them into a single crop.
    """
    return (
        box_w > _MAX_BAR_PAGE_SPAN * width
        and box_h < _FURNITURE_MAX_HEIGHT * height
    )


def _is_horizontal_dash(box_w: int, box_h: int, width: int) -> bool:
    """Report whether a blob is a horizontal connector mark."""
    return (
        box_h <= max(3, int(_DASH_THICKNESS * width))
        and box_w >= max(5, 3 * box_h, _MIN_DASH_LENGTH * width)
        and box_w <= _MAX_DASH_EXTENT * width
    )


def _is_vertical_dash(box_w: int, box_h: int, height: int) -> bool:
    """Report whether a blob is a vertical connector mark."""
    return (
        box_w <= max(3, int(_DASH_THICKNESS * height))
        and box_h >= max(5, 3 * box_w, _MIN_DASH_LENGTH * height)
        and box_h <= _MAX_DASH_EXTENT * height
    )


def _is_label(
    component: Component, width: int, height: int, min_ink: float
) -> bool:
    """
    Report whether a blob is an identifier letter or arrow.

    A blob small in both dimensions is an annotation. A blob carrying
    little ink is only an annotation when it is also reasonably solid:
    a lithic drawing is a hollow outline and so always carries little
    ink for its size, which would otherwise misclassify small surface
    views such as platforms as labels and drop them from their artefact.
    """
    small = (
        component.height < _LABEL_MAX_HEIGHT * height
        and component.width < _LABEL_MAX_WIDTH * width
    )
    sparse = (
        component.ink < min_ink
        and component.height < 0.04 * height
        and component.density >= _MIN_ANNOTATION_DENSITY
    )
    return small or sparse


def _is_text(
    component: Component, width: int, height: int, text_config: Dict
) -> bool:
    """
    Report whether a drawing-sized component is text or a legend key.

    A word, caption or legend line arrives as one component: closing
    fuses its letters together. Its raw ink, though, still breaks into a
    row of letter-sized pieces with none of them dominant, whereas a
    lithic view is one outline holding nearly all of its ink, with any
    hatching attached to it. Measured on ten hand-labelled plates: text
    had a median of 8 letter-height pieces, the largest holding 20% of
    the ink; drawings a median of 1 piece holding 97%.

    Two guards remove the drawings that happen to break into pieces. A
    line of text is short relative to its page — under 5.2% of page
    height on every one of 45 pages, against 6-46% for hatched surfaces
    and large outlines — and text is never as sparse as a hollow outline.

    A legend key is the opposite case: a solid block beside a word,
    which no hollow drawing resembles.

    Each rejection is logged at DEBUG with the rule that fired.
    """
    if not text_config.get('enabled', True) or component.mask is None:
        return False
    rule = _text_rule(component, width, height, text_config)
    if rule is None:
        return False
    logging.debug(
        "Discarded %dx%d component at %s as %s",
        component.width, component.height, component.box, rule,
    )
    return True


def _text_rule(
    component: Component, width: int, height: int, text_config: Dict
) -> Optional[str]:
    """Name the text rule a component satisfies, or None."""
    if (
        component.density >= text_config.get('swatch_density', 0.9)
        and max(component.width, component.height)
        <= text_config.get('swatch_max_size', 0.06) * min(width, height)
    ):
        return 'legend swatch'

    if component.density < text_config.get('min_density', 0.12):
        return None

    max_height = text_config.get('max_height', 0.05) * height
    bands = _ink_bands(component.mask)
    if len(bands) >= 2 and all(
        band.shape[0] <= max_height and _is_text_line(band, text_config)
        for band in bands
    ):
        return 'wrapped text'
    if (
        component.height <= max_height
        and _is_text_line(component.mask, text_config)
    ):
        return 'text structure'
    return None


def _is_text_line(mask: np.ndarray, text_config: Dict) -> bool:
    """Report whether a mask's raw ink reads as one line of text."""
    pieces, largest, aligned = _ink_structure(mask, mask.shape[0])
    return (
        pieces >= text_config.get('min_pieces', 3)
        and largest <= text_config.get('max_piece_share', 0.5)
        and aligned >= text_config.get('min_aligned', 0.75)
    )


def _ink_bands(mask: np.ndarray) -> List[np.ndarray]:
    """
    Split a mask into bands of ink separated by blank rows.

    Wrapped text arrives as one component when its lines sit close
    enough for closing to fuse them, and then no letter stands 40% of
    the block's height, so the single-line test sees nothing. The raw
    ink still keeps a blank row between the lines; a drawing has no
    blank row across its whole width. Each band is then tested as a
    line of its own. Bands too short to be a line — accents, the dots
    of i's — are dropped rather than allowed to veto the word beneath.
    """
    inked = mask.any(axis=1)
    bands, start = [], None
    for row, has_ink in enumerate(inked):
        if has_ink and start is None:
            start = row
        elif not has_ink and start is not None:
            bands.append(mask[start:row])
            start = None
    if start is not None:
        bands.append(mask[start:])
    return [band for band in bands if band.shape[0] >= _MIN_BAND_ROWS]


def _ink_structure(mask: np.ndarray, height: int) -> Tuple[int, float, float]:
    """
    Describe how a component's raw ink breaks into pieces.

    Returns
    -------
    tuple
        ``(pieces, largest_share, aligned_share)``: the number of pieces
        at least ``_TEXT_PIECE_HEIGHT`` of the component's height, the
        share of all ink held by the single largest piece, and the share
        of counted pieces whose bottoms sit on a common baseline.
    """
    _, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    stats = stats[1:]  # drop the background
    if len(stats) == 0:
        return 0, 1.0, 0.0

    ink = float(stats[:, cv2.CC_STAT_AREA].sum()) or 1.0
    largest = float(stats[:, cv2.CC_STAT_AREA].max()) / ink
    tall = stats[stats[:, cv2.CC_STAT_HEIGHT] >= _TEXT_PIECE_HEIGHT * height]
    if len(tall) == 0:
        return 0, largest, 0.0

    bottoms = tall[:, cv2.CC_STAT_TOP] + tall[:, cv2.CC_STAT_HEIGHT]
    tolerance = _TEXT_BASELINE_TOLERANCE * height
    aligned = np.abs(bottoms - np.median(bottoms)) <= tolerance
    return int(len(tall)), largest, float(aligned.mean())


def _bar_candidates(
    components: Sequence[Component],
    width: int,
    height: int,
    bar_config: Dict,
) -> List[int]:
    """Indices of isolated blobs shaped like one block of a bar."""
    return [
        index for index, component in enumerate(components)
        if _is_bar_segment(component, height, bar_config)
        and _is_isolated(index, components, width, height)
    ]


def _find_bar_components(
    components: Sequence[Component],
    width: int,
    height: int,
    bar_config: Dict,
    allow_runs: bool = True,
) -> set:
    """
    Identify the components making up the page's scale bars.

    Scale bars are drawn as a row of alternating filled and open
    blocks, and thresholding keeps only the filled ones, so a bar
    arrives as several solid blocks rather than one long mark. A lone
    block then looks just like a connector dash, so context decides:
    collinear blocks form a segmented bar, a lone one must be longer.

    Parameters
    ----------
    components : sequence of Component
        Ink blobs found on the page.
    width, height : int
        Page dimensions in pixels.
    bar_config : dict
        Scale bar configuration.
    allow_runs : bool
        Whether collinear blocks may combine into one bar. False makes
        every candidate stand alone, facing the lone-block tests.

    Returns
    -------
    set
        Indices into ``components`` belonging to a scale bar.
    """
    if not bar_config.get('enabled', True):
        return set()

    candidates = _bar_candidates(components, width, height, bar_config)
    if not candidates:
        return set()

    runs = (
        _collinear_runs(components, candidates, width) if allow_runs
        else [[index] for index in candidates]
    )
    identified = set()
    for run in runs:
        if _run_is_scale_bar(components, run, width):
            identified.update(run)
    return identified


def _run_is_scale_bar(
    components: Sequence[Component], run: Sequence[int], width: int
) -> bool:
    """
    Decide whether a run of collinear bar-shaped blocks is a scale.

    Several collinear blocks are the filled half of an alternating bar,
    its open blocks lost at threshold. A single component can hold a
    whole bar only if it is not solid, since its open blocks fall inside
    its box; a solid mark of the same proportions is a connector rule
    joining two surfaces of one lithic.
    """
    if not _plausible_bar_span(components, run, width):
        return False
    if len(run) > 1:
        return True
    lone = components[run[0]]
    return (
        lone.density <= _MAX_SOLID_DENSITY
        and box_width(lone.box) > _MAX_DASH_EXTENT * width
        and _shows_alternation(lone)
    )


def _transposed(component: Component) -> Component:
    """
    Mirror a component across the diagonal, swapping its two axes.

    Every block-bar test is written for a bar lying across the page.
    Measuring transposed components against a transposed page finds
    upright bars with the same rules, rather than a second set of rules
    to keep in step with the first.
    """
    x0, y0, x1, y1 = component.box
    return Component(
        box=[y0, x0, y1, x1],
        width=component.height,
        height=component.width,
        ink=component.ink,
        mask=None if component.mask is None else component.mask.T,
    )


def _find_vertical_bar_components(
    components: Sequence[Component],
    width: int,
    height: int,
    bar_config: Dict,
) -> set:
    """
    Identify block bars drawn down the page rather than across it.

    A plate of tall artefacts often carries its scale down the side,
    drawn as blocks stepping left and right so the filled parts overlap
    through the centre line. Transposing the page and its components
    turns that into the horizontal case, and indices map straight back.

    Blocks are never combined here: all eleven upright scales measured
    arrived as one component, so a run of separate blocks belongs to
    the horizontal route. Allowing runs let two hairline marks span
    enough of a short page to qualify, and both were exported.

    Coverage cannot be shared either. A horizontal bar prints numerals
    inside its own box, dragging coverage as low as 0.22, while an
    upright bar keeps them outside: measured upright bars ran 0.51 to
    0.72, against 0.06 to 0.11 for the tall narrow drawings among which
    they sit.

    Parameters
    ----------
    components : sequence of Component
        Ink blobs found on the page.
    width, height : int
        Page dimensions in pixels.
    bar_config : dict
        Scale bar configuration.

    Returns
    -------
    set
        Indices into ``components`` belonging to an upright scale bar.
    """
    flipped = [_transposed(component) for component in components]
    found = _find_bar_components(
        flipped, height, width, bar_config, allow_runs=False
    )
    return {
        index for index in found
        if components[index].density >= _MIN_BAR_DENSITY
    }


def _find_tick_scale_components(
    components: Sequence[Component],
    width: int,
    height: int,
    bar_config: Dict,
) -> set:
    """
    Identify scale bars drawn as a ruled line with tick marks.

    The other common scale style is a plain rule carrying short ticks at
    the measured intervals, usually with a numeral at each end. It has
    none of the proportions of a block bar — the numerals sit inside its
    bounding box, dragging ink coverage down to a level indistinguishable
    from a hollow surface view — so it is found by structure instead.

    Two features identify it, both robust to hand-drawn work:

    - a **spine**: an unbroken straight run of ink spanning much of the
      component, in either orientation
    - **ticks**: two or more short, narrow strokes attached to that spine

    No angle is measured. A tick is simply a small stroke touching the
    spine, so a hand-drawn mark leaning off the perpendicular still
    counts, as does a scale drawn vertically.

    Parameters
    ----------
    components : sequence of Component
        Ink blobs found on the page.
    width, height : int
        Page dimensions in pixels.
    bar_config : dict
        Scale bar configuration.

    Returns
    -------
    set
        Indices into ``components`` forming a ruled scale.
    """
    if not bar_config.get('enabled', True):
        return set()

    return {
        index for index in range(len(components))
        if _is_ruled_scale(index, components, width, height, bar_config)
    }


def _is_ruled_scale(
    index: int,
    components: Sequence[Component],
    width: int,
    height: int,
    bar_config: Dict,
) -> bool:
    """Apply the ruled-scale tests to one component."""
    component = components[index]
    if component.mask is None:
        return False
    if (
        component.width > _MAX_BAR_PAGE_SPAN * width
        or component.height > _MAX_BAR_PAGE_SPAN * height
    ):
        # Spans the page like the rule beneath a running head. Real
        # ruled scales on the reference run span 12-15% of the page.
        return False
    if not _is_isolated(index, components, width, height):
        return False
    return _has_ruled_scale_structure(
        component.mask, bar_config, width, height,
    )


def _has_ruled_scale_structure(
    mask, bar_config: Dict, page_width: int, page_height: int
) -> bool:
    """
    Report whether a blob is a ruled line carrying tick marks.

    Each orientation is measured against the page dimension the spine
    would lie along, so a tall narrow mark is not credited for spanning
    a wide page.
    """
    across = int(_MIN_SPINE_PAGE_SPAN * page_width)
    down = int(_MIN_SPINE_PAGE_SPAN * page_height)
    if _spine_with_ticks(mask, bar_config, across):
        return True
    return bool(_spine_with_ticks(mask.T, bar_config, down))


def _spine_with_ticks(mask, bar_config: Dict, min_spine: int) -> bool:
    """
    Look for a horizontal spine with ticks attached, in one orientation.

    Called twice, the second time on the transposed mask, so a vertical
    scale is found by the same code.
    """
    min_span = bar_config.get('spine_span', _MIN_SPINE_SPAN)
    min_ticks = bar_config.get('min_ticks', _MIN_TICK_COUNT)
    rows, columns = mask.shape
    if columns < max(_MIN_SPINE_PIXELS, min_spine):
        return False

    spine = _longest_horizontal_run(mask)
    if spine is None:
        return False
    row, start, length = spine
    if length < min_span * columns or length < min_spine:
        return False

    top, bottom = _spine_thickness(mask, row, start, length)
    if (bottom - top + 1) > _MAX_SPINE_THICKNESS * rows:
        # A filled region, not a drawn rule.
        return False
    if _count_ticks(mask, row, start, length, bar_config) < min_ticks:
        return False
    return _spine_dominates(mask, row, start, length, bar_config)


def _spine_dominates(
    mask, spine_row: int, start: int, length: int, bar_config: Dict
) -> bool:
    """
    Check that the rule is the component's main structure.

    A spine with ticks is not enough on its own. A flake with a straight
    snapped edge and a scar line meeting it satisfies both, and would be
    exported as a scale. The two are separated by deleting the rule and
    measuring what survives: a scale collapses into ticks and numerals,
    all small, while an artefact outline remains largely intact.

    Parameters
    ----------
    mask : np.ndarray
        Boolean ink mask for the component, spine horizontal.
    spine_row, start, length : int
        Location and span of the detected rule.
    bar_config : dict
        Scale bar configuration.

    Returns
    -------
    bool
        True when nothing substantial survives removal of the rule.
    """
    limit = bar_config.get('max_residual', _MAX_SPINE_RESIDUAL)
    total = int(mask.sum())
    if not total:
        return False

    remainder = mask.copy()
    top, bottom = _spine_thickness(mask, spine_row, start, length)
    remainder[top:bottom + 1, :] = False

    count, _, stats, _ = cv2.connectedComponentsWithStats(
        remainder.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return True

    largest = max(stats[index, cv2.CC_STAT_AREA] for index in range(1, count))
    return largest <= limit * total


def _longest_horizontal_run(mask):
    """
    Find the row holding the longest unbroken run of ink.

    Returns
    -------
    tuple or None
        ``(row, start_column, length)`` for the best run found.
    """
    best = None
    for row_index, row in enumerate(mask):
        start = run = 0
        for column, value in enumerate(row):
            if value:
                if run == 0:
                    start = column
                run += 1
                if best is None or run > best[2]:
                    best = (row_index, start, run)
            else:
                run = 0
    return best


def _count_ticks(
    mask, spine_row: int, start: int, length: int, bar_config: Dict
) -> int:
    """
    Count short strokes attached to the spine.

    A drawn rule is usually two or three pixels thick, so its full
    thickness is measured and excluded first — otherwise the spine's own
    second row reads as one enormous tick spanning every mark.

    Ticks are then collected by walking outward from the spine while
    there is ink, stopping at the first clear row. A tick touches the
    rule it marks; the numerals printed beside a scale sit clear of it
    and are never reached.
    """
    max_tick_width = max(
        2, int(bar_config.get('max_tick_width', _MAX_TICK_WIDTH) * length)
    )
    top, bottom = _spine_thickness(mask, spine_row, start, length)

    columns = set()
    columns.update(_ticks_beyond(mask, top, -1, start, length))
    columns.update(_ticks_beyond(mask, bottom, 1, start, length))
    columns = sorted(columns)
    if not _ticks_at_both_ends(columns, start, length):
        return 0
    return _count_clusters(columns, max_tick_width)


def _ticks_at_both_ends(columns: Sequence[int], start: int, length: int) -> bool:
    """
    Report whether ticks stand near both ends of the rule.

    A scale marks its measure from one end to the other, so its rule
    carries a tick near each end. An arrow in a legend is a rule with a
    head at one end only, and read as a spine with ticks until this
    was required.
    """
    if not columns:
        return False
    reach = _TICK_END_FRACTION * length
    return columns[0] <= start + reach and columns[-1] >= start + length - reach


def _spine_thickness(
    mask, spine_row: int, start: int, length: int
) -> Tuple[int, int]:
    """
    Measure how many rows the drawn rule occupies.

    Returns
    -------
    tuple of int
        First and last row of the spine itself.
    """
    rows = mask.shape[0]
    columns = range(start, start + length)
    solid = _SPINE_THICKNESS_COVER * length

    top = bottom = spine_row
    while top - 1 >= 0 and sum(
        1 for c in columns if mask[top - 1][c]
    ) >= solid:
        top -= 1
    while bottom + 1 < rows and sum(
        1 for c in columns if mask[bottom + 1][c]
    ) >= solid:
        bottom += 1
    return top, bottom


def _ticks_beyond(
    mask, edge: int, direction: int, start: int, length: int
) -> set:
    """Collect ink columns adjoining the spine on one side."""
    rows = mask.shape[0]
    columns = set()
    row = edge + direction
    while 0 <= row < rows:
        hits = {
            column for column in range(start, start + length)
            if mask[row][column]
        }
        if not hits:
            break
        columns.update(hits)
        row += direction
    return columns


def _count_clusters(columns: Sequence[int], max_width: int) -> int:
    """Count runs of adjacent columns no wider than a tick."""
    if not columns:
        return 0

    clusters = 0
    run_start = previous = columns[0]
    for column in columns[1:] + [None]:
        if column is not None and column - previous <= 1:
            previous = column
            continue
        if previous - run_start + 1 <= max_width:
            clusters += 1
        if column is None:
            break
        run_start = previous = column
    return clusters


def _is_isolated(
    index: int,
    components: Sequence[Component],
    width: int,
    height: int,
) -> bool:
    """
    Report whether a component stands clear of every drawing.

    A scale bar is always drawn in open space: it never touches or
    overlaps an artefact. A thin mark that does touch one is part of the
    illustration — a connector rule tying two surfaces together, or an
    interior detail — and must not be exported as a scale.

    Parameters
    ----------
    index : int
        Component under test.
    components : sequence of Component
        All ink blobs found on the page.
    width, height : int
        Page dimensions in pixels.

    Returns
    -------
    bool
        True when the component touches no drawing-sized neighbour.
    """
    box = components[index].box
    for other, component in enumerate(components):
        if other == index:
            continue
        if not _is_drawing_sized(component, width, height):
            continue
        if (
            horizontal_overlap(box, component.box) > 0
            and vertical_overlap(box, component.box) > 0
        ):
            return False
    return True


def _is_drawing_sized(
    component: Component, width: int, height: int
) -> bool:
    """Report whether a component is too large to be an annotation."""
    return (
        component.height >= _LABEL_MAX_HEIGHT * height
        or component.width >= _LABEL_MAX_WIDTH * width
    )


def _is_bar_segment(
    component: Component, height: int, bar_config: Dict
) -> bool:
    """
    Report whether a blob could be one block of a scale bar.

    Bars are recognised by a long straight edge rather than by ink
    coverage. Coverage varies far too widely to gate on: across the
    reference scales it runs from 0.22 to 1.00, because the numerals
    printed beside a bar pull whitespace into its bounding box. A ruled
    edge is present in every style — the frame of a chequered bar, the
    solid top of a block bar, the rule of a graduated one.

    Proportion does the work of excluding drawings: every reference
    scale has an aspect ratio of 6.1 or more, every surface view 4.1 or
    less.
    """
    min_ratio = bar_config.get('min_aspect_ratio', 5.0)
    max_height = bar_config.get('max_height', 0.06)
    if component.height <= 0:
        return False
    if not (
        component.height < max_height * height
        and component.width > min_ratio * component.height
    ):
        return False
    return _has_straight_edge(component)


def _has_straight_edge(component: Component) -> bool:
    """
    Report whether a blob contains a long straight run.

    A drawn rule can reach the detector broken into segments where the
    scan lost ink. A thin blob is still a rule, so for one far wider
    than tall the segments of at least a tenth of its width are summed
    along the row holding most of them. A caption never passes that:
    its pieces are letters, each a few percent of the width. A squatter
    blob keeps the single-run test, so an arrow legend whose shaft is
    one long run among lines of text is not credited.
    """
    if component.mask is None:
        return component.density >= _MIN_BAR_DENSITY
    needed = _MIN_BAR_EDGE_SPAN * component.width
    run = _longest_horizontal_run(component.mask)
    if run is not None and run[2] >= needed:
        return True
    if component.width < _MIN_SEGMENTED_ASPECT * component.height:
        return False
    return _segmented_run(component.mask) >= needed


def _segmented_run(mask) -> int:
    """
    Length of the longest broken straight run, in pixels.

    On each row, runs shorter than ``_MIN_BAR_SEGMENT`` of the width are
    ignored and the rest summed; the best row's total is returned.
    """
    width = mask.shape[1]
    minimum = _MIN_BAR_SEGMENT * width
    best = 0
    for row in mask:
        total = run = 0
        for value in row:
            if value:
                run += 1
                continue
            if run >= minimum:
                total += run
            run = 0
        if run >= minimum:
            total += run
        best = max(best, total)
    return best


def _plausible_bar_span(
    components: Sequence[Component], run: Sequence[int], width: int
) -> bool:
    """
    Report whether a bar candidate spans a believable share of the page.

    A scale is sized to the artefacts it measures, so it never runs the
    full width of a page. A rule that does is the separator beneath a
    running head. At the other extreme, a few short collinear dashes are
    a dotted line, too short to carry a measure.

    Parameters
    ----------
    components : sequence of Component
        Ink blobs found on the page.
    run : sequence of int
        Indices of the collinear components forming the candidate.
    width : int
        Page width in pixels.

    Returns
    -------
    bool
        True when the candidate's total span lies within the limits.
    """
    left = min(components[index].box[0] for index in run)
    right = max(components[index].box[2] for index in run)
    span = right - left
    return _MIN_BAR_PAGE_SPAN * width <= span <= _MAX_BAR_PAGE_SPAN * width


def _shows_alternation(component: Component) -> bool:
    """
    Report whether a bar carries alternating filled and open blocks.

    A scale states its measure through repetition: filled blocks
    separated by open ones. A single unbroken run of ink states nothing
    — it is a plain rule, such as the separator under a running head or
    the edge of a frame — and must not be exported as a scale.

    Counted on the bar's densest row, so a drawn border around the
    blocks does not mask the pattern inside it.

    Parameters
    ----------
    component : Component
        Candidate bar.

    Returns
    -------
    bool
        True when the bar breaks into two or more filled runs.
    """
    if component.mask is None:
        return True

    best = 0
    for row in component.mask:
        runs, in_run = 0, False
        for value in row:
            if value and not in_run:
                runs, in_run = runs + 1, True
            elif not value:
                in_run = False
        best = max(best, runs)
        if best >= _MIN_BAR_ALTERNATIONS:
            return True
    return False


def _collinear_runs(
    components: Sequence[Component],
    candidates: Sequence[int],
    width: int,
) -> List[List[int]]:
    """Group bar-segment candidates into runs sharing a baseline."""
    ordered = sorted(candidates, key=lambda i: components[i].box[0])
    limit = _BAR_SEGMENT_GAP * width
    runs: List[List[int]] = []

    for index in ordered:
        box = components[index].box
        for run in runs:
            if _is_same_bar(components[run[-1]].box, box, limit):
                run.append(index)
                break
        else:
            runs.append([index])
    return runs


def _drop_text_line_remnants(result: Classified) -> None:
    """
    Discard the small pieces of a line that was rejected as text.

    A bracket, a numeral in a key, a legend swatch or an arrow icon is
    small enough to be a label, and a label attaches to the nearest
    artefact. The dash in "Fig. 10 — Racloirs" reads as a connector
    mark, which is attached likewise. Left behind after their words are
    discarded, they drag the crop back over the space the text occupied.
    Anything sharing a baseline with rejected text is part of that line
    and goes with it. An identifier numeral sits beside its lithic, not
    on a legend's baseline, so it is untouched.
    """
    if not result.text:
        return
    before = len(result.labels) + len(result.dashes)
    result.labels = [
        box for box in result.labels
        if not _on_rejected_line(box, result.text)
    ]
    result.dashes = [
        (orientation, box) for orientation, box in result.dashes
        if not _on_rejected_line(box, result.text)
    ]
    dropped = before - len(result.labels) - len(result.dashes)
    if dropped:
        logging.debug(
            "Discarded %d remnant(s) of rejected text lines", dropped
        )


def _drop_legend_keys(result: Classified) -> None:
    """
    Discard a bar that is the key of a legend rather than a scale.

    The arrow in "→ Direction of flake-scars" is a solid rule with a
    head, and the block-bar tests accept it. What no scale has is a
    sentence set beside it on its own baseline: a scale's caption is a
    numeral or a unit. A bar sharing a baseline with rejected text is a
    legend key and goes with the text.
    """
    if not result.text or not result.bars:
        return
    keys = [bar for bar in result.bars if _on_rejected_line(bar, result.text)]
    if keys:
        result.bars = [bar for bar in result.bars if bar not in keys]
        result.text.extend(keys)
        logging.debug("Discarded %d legend key(s) read as scale bars", len(keys))


def _on_rejected_line(box: BBox, text: Sequence[BBox]) -> bool:
    """Report whether a mark shares a baseline with rejected text."""
    return any(_shares_baseline(box, line) for line in text)


def _split_bar_captions(
    labels: Sequence[BBox], bars: Sequence[BBox]
) -> Tuple[List[BBox], List[BBox]]:
    """
    Separate text sitting beside a scale bar from artefact labels.

    Captions such as ``0`` or ``5 cm`` belong to the bar, not to any
    artefact. They are kept so the exported bar crop is self-describing,
    and withheld from artefact grouping.

    Returns
    -------
    tuple of list
        ``(bar_captions, artefact_labels)``.
    """
    captions, remaining = [], []
    for box in labels:
        if any(_is_bar_caption(box, bar) for bar in bars):
            captions.append(box)
        else:
            remaining.append(box)
    return captions, remaining


def _is_bar_caption(box: BBox, bar: BBox) -> bool:
    """
    Report whether a label sits on a scale bar as its caption does.

    A caption is set on the bar's own line: numerals directly above or
    below it within its span, or a unit just past its end. Anything
    within three glyph heights in any direction was taken before, which
    swallowed the identifiers of the lithics beside a scale and the
    marks of the lithics above it into the scale's crop.
    """
    height = max(1, box_height(box))
    if vertical_gap(box, bar) >= _CAPTION_LINE_GAP * height:
        return False
    return horizontal_gap(box, bar) < _CAPTION_END_GAP * height


def reject_caption_lines(
    labels: Sequence[BBox], width: int
) -> List[BBox]:
    """
    Discard figure captions and legend text.

    Identifier labels are isolated marks; captions are runs of small
    blobs sitting side by side on a line. Runs that span a wide stretch
    of the page, or contain a distinctly wide blob, are treated as text
    and dropped so they are never attached to an artefact.

    Parameters
    ----------
    labels : sequence of BBox
        Candidate label boxes.
    width : int
        Page width in pixels.

    Returns
    -------
    list of BBox
        Labels with caption text removed.
    """
    ordered = sorted(labels, key=lambda b: (b[1], b[0]))
    consumed = [False] * len(ordered)
    kept: List[BBox] = []

    for index, box in enumerate(ordered):
        if consumed[index]:
            continue
        line = _collect_text_line(ordered, consumed, index, box)
        if not _is_caption_line(line, width):
            kept.extend(line)

    dropped = len(ordered) - len(kept)
    if dropped:
        logging.debug("Discarded %d caption/legend blobs", dropped)
    return kept


def _collect_text_line(
    ordered: Sequence[BBox],
    consumed: List[bool],
    start: int,
    box: BBox,
) -> List[BBox]:
    """Grow a run of blobs sharing a baseline with the starting blob."""
    line = [box]
    consumed[start] = True
    changed = True
    while changed:
        changed = False
        for index, candidate in enumerate(ordered):
            if consumed[index]:
                continue
            if any(_shares_baseline(candidate, member) for member in line):
                line.append(candidate)
                consumed[index] = True
                changed = True
    return line


def _shares_baseline(candidate: BBox, member: BBox) -> bool:
    """Report whether two blobs sit on the same line of text."""
    heights = (box_height(candidate), box_height(member))
    return (
        vertical_overlap(candidate, member) > 0.5 * min(heights)
        and horizontal_gap(candidate, member) < 4 * max(heights)
    )


def _is_caption_line(line: Sequence[BBox], width: int) -> bool:
    """
    Report whether a run of blobs reads as caption text.

    A lone mark is never a line of text, however wide it is. Requiring
    more than one blob keeps an isolated drawing element from being
    discarded as a caption.
    """
    if len(line) < 2:
        return False

    span = max(b[2] for b in line) - min(b[0] for b in line)
    if span > _CAPTION_LINE_WIDTH * width:
        return True
    return any(
        box_width(b) > 2.5 * box_height(b) and box_width(b) > 0.02 * width
        for b in line
    )


### LINKING RULES ###

def link_adjacent_views(
    drawings: Sequence[Component],
    sets: _UnionFind,
    width: int,
    gap: float,
) -> None:
    """
    Link surface views drawn side by side.

    Two views belong to one artefact when one substantially contains the
    other, or when they sit side by side on the same baseline separated
    by less than the configured gap.

    Containment is measured as a fraction of the smaller box rather than
    as a bare intersection. On a densely packed plate, two diagonally
    adjacent artefacts have bounding boxes that clip at the corners, and
    treating that clip as containment fuses unrelated lithics — and,
    through union-find, can collapse a whole page into one crop.
    """
    for i in range(len(drawings)):
        for j in range(i + 1, len(drawings)):
            a, b = drawings[i].box, drawings[j].box
            if overlap_fraction(a, b) >= _MIN_CONTAINMENT:
                sets.link(i, j)
                continue
            # Only boxes that are actually separated can be side by side;
            # for overlapping boxes the gap is negative and would always
            # satisfy the test.
            separation = horizontal_gap(a, b)
            shared = _SIDE_BY_SIDE_OVERLAP * min(
                box_height(a), box_height(b)
            )
            if (
                separation >= 0
                and vertical_overlap(a, b) > shared
                and separation < gap * width
            ):
                sets.link(i, j)


def link_cross_sections(
    drawings: Sequence[Component],
    sets: _UnionFind,
    height: int,
    vertical_limit: float,
    width: int = 0,
) -> None:
    """
    Link short views sitting directly above or below a taller group.

    A cross-section is drawn as a slice beneath the surface it cuts
    through, and a platform view as a strip above it.

    Links are made one at a time, closest pair first, with the groups
    recomputed after each. Order matters because merging is
    irreversible: a link made early against half-assembled groups cannot
    be reconsidered once a later link reveals it was wrong.

    Parameters
    ----------
    drawings : sequence of Component
        Candidate artefact views.
    sets : _UnionFind
        Grouping built so far, updated in place.
    height : int
        Page height in pixels.
    vertical_limit : float
        Largest gap, as a fraction of page height, a view may bridge.
    width : int, optional
        Page width in pixels, used with ``height`` to scale the test
        for bare outlines. Defaults to the height when omitted.
    """
    page_size = min(width, height) if width else height
    outlines = [
        _is_plain_outline(d, page_size) and not _is_profile_shaped(d)
        for d in drawings
    ]

    for _ in range(_MAX_SECTION_LINKS):
        candidates = _section_candidates(
            drawings, sets, height, vertical_limit, outlines
        )
        if not candidates:
            break
        _, index, member = candidates[0]
        if sets.find(index) == sets.find(member):
            break
        sets.link(index, member)


def _section_candidates(
    drawings: Sequence[Component],
    sets: _UnionFind,
    height: int,
    vertical_limit: float,
    outlines: Sequence[bool],
) -> List[Tuple[int, int, int]]:
    """
    Collect every permissible section link, closest first.

    Returns
    -------
    list of tuple
        ``(gap, drawing_index, host_member_index)`` sorted by gap.
    """
    bounds, tallest = _group_bounds(drawings, sets)
    sizes: Dict[int, int] = {}
    for index in range(len(drawings)):
        root = sets.find(index)
        sizes[root] = sizes.get(root, 0) + 1

    found = []
    for index, drawing in enumerate(drawings):
        lone_outline = outlines[index] and sizes[sets.find(index)] == 1
        host = _nearest_section_host(
            drawing, index, sets, bounds, tallest, height, vertical_limit,
            lone_outline,
        )
        if host is None:
            continue
        found.append((
            vertical_gap(drawing.box, bounds[sets.find(host)]),
            index,
            host,
        ))
    return sorted(found)


def _group_bounds(
    drawings: Sequence[Component], sets: _UnionFind
) -> Tuple[Dict[int, BBox], Dict[int, int]]:
    """Compute each set's enclosing box and its tallest single view."""
    bounds: Dict[int, BBox] = {}
    tallest: Dict[int, int] = {}
    for index, drawing in enumerate(drawings):
        root = sets.find(index)
        bounds[root] = (
            union(bounds[root], drawing.box) if root in bounds
            else list(drawing.box)
        )
        tallest[root] = max(tallest.get(root, 0), drawing.height)
    return bounds, tallest


def _nearest_section_host(
    drawing: Component,
    index: int,
    sets: _UnionFind,
    bounds: Dict[int, BBox],
    tallest: Dict[int, int],
    height: int,
    vertical_limit: float,
    lone_outline: bool = False,
) -> int:
    """
    Find the group a short view should join, if any.

    Candidate groups must overlap the view horizontally, which is how a
    platform or cross-section is drawn: directly over or under the
    surfaces it belongs to.

    A bare outline standing alone is taken to be a section or platform
    view and joins the nearest taller group, whatever its relative
    height. Profile-shaped outlines are excluded: a profile is drawn
    beside its surface, not under it, and is linked sideways by
    ``link_profiles``. Without that exclusion a profile joins a lithic in
    the row above or below and zips whole rows together.

    Any other view must also be much shorter than its host, so a lithic
    is never absorbed as the section of its neighbour.

    Returns the index of a drawing in the winning group, or None when
    the view is already part of a tall group or nothing is close enough.
    """
    own_root = sets.find(index)
    if drawing.height < _SECTION_ATTACHED_RATIO * tallest[own_root]:
        # Already sits alongside a much taller view in its own group.
        return None
    if _is_entangled(drawing.box, own_root, bounds):
        # Boxed in by a different artefact on both axes, as a scale bar
        # between two lithics is: a real section sits clear of all but
        # its own parent.
        return None
    return _closest_vertical_host(
        drawing, own_root, bounds, tallest, vertical_limit * height,
        lone_outline,
    )


def _closest_vertical_host(
    drawing: Component,
    own_root: int,
    bounds: Dict[int, BBox],
    tallest: Dict[int, int],
    max_gap: float,
    lone_outline: bool,
) -> int:
    """
    Pick the nearest group lying directly above or below a view.

    Returns the winning group's root index, or None when no group lies
    within ``max_gap`` with horizontal overlap and a suitable height.
    """
    best_gap, best_root = None, None
    for root, bound in bounds.items():
        if root == own_root or vertical_overlap(drawing.box, bound) > 0:
            continue
        if horizontal_overlap(drawing.box, bound) <= 0:
            continue
        if lone_outline:
            if tallest[root] <= drawing.height:
                continue  # a section is never taller than its lithic
        elif drawing.height >= _SECTION_HOST_RATIO * tallest[root]:
            continue
        gap = vertical_gap(drawing.box, bound)
        if gap < max_gap and (best_gap is None or gap < best_gap):
            best_gap, best_root = gap, root
    return best_root


def _is_plain_outline(component: Component, page_size: int) -> bool:
    """
    Report whether a view is an outline with nothing drawn inside it.

    Cross-sections and platform views are drawn as bare outlines, while
    a lithic surface carries scar ridges, hatching or stipple inside its
    edge. This separates the two where relative height cannot: on a
    published plate, sections ran from 20% to 96% of their lithic's
    height.

    Parameters
    ----------
    component : Component
        View under test.
    page_size : int
        Smaller page dimension in pixels, setting how far inside the
        edge ink must lie to count as interior detail.

    Returns
    -------
    bool
        True when the view is a bare outline.
    """
    if component.mask is None or not component.mask.any():
        return False
    margin = max(3.0, _INTERIOR_MARGIN * page_size)
    return _interior_ink_fraction(component.mask, margin) < _MAX_OUTLINE_INTERIOR


def _interior_ink_fraction(mask: np.ndarray, margin: float) -> float:
    """
    Measure the share of ink lying well inside a shape's outer edge.

    The shape is closed and filled to recover its silhouette; ink more
    than ``margin`` pixels inside that silhouette is interior detail.

    The mask is padded with background first. A shape that fills its
    own bounding box, as a rectangular outline does, would otherwise
    leave no background for the distance to be measured from, and every
    pixel would read as deep interior.
    """
    ink = np.pad(mask, _SILHOUETTE_PAD).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    silhouette = np.zeros_like(ink)
    cv2.drawContours(silhouette, contours, -1, 255, -1)
    depth = cv2.distanceTransform(silhouette, cv2.DIST_L2, 5)
    inked = ink > 0
    return float((inked & (depth > margin)).sum()) / max(1, int(inked.sum()))


def _is_entangled(
    box: BBox, own_root: int, bounds: Dict[int, BBox]
) -> bool:
    """Report whether a view overlaps a different artefact on both axes."""
    for root, bound in bounds.items():
        if root == own_root:
            continue
        if (
            vertical_overlap(box, bound) > 0
            and horizontal_overlap(box, bound) > 0
        ):
            return True
    return False


def _is_profile_shaped(component: Component) -> bool:
    """Report whether a view is narrow enough to read as a profile."""
    return component.width < _PROFILE_ASPECT * component.height


def link_profiles(
    drawings: Sequence[Component],
    sets: _UnionFind,
    width: int,
    narrow: float,
) -> None:
    """
    Link thin profile and section views to their neighbouring surface.

    A profile shows the thickness of a flake and is drawn as a narrow
    outline beside the surface it belongs to. Where a profile sits
    between two surfaces at comparable distance, it links to both, since
    an equidistant profile is ambiguous and merging is the safer error.
    """
    for index, drawing in enumerate(drawings):
        if not _is_profile_shaped(drawing):
            continue
        left, right = _flanking_views(drawings, index, width, narrow)
        if left and right:
            near, far = sorted([left, right])
            sets.link(index, near[1])
            if far[0] <= 3 * near[0] + 4:
                sets.link(index, far[1])
        elif left or right:
            sets.link(index, (left or right)[1])


def _flanking_views(
    drawings: Sequence[Component],
    index: int,
    width: int,
    narrow: float,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Find the nearest qualifying view on each side of a profile.

    Returns
    -------
    tuple
        ``(left, right)``, each ``(gap, index)`` or None.
    """
    profile = drawings[index]
    left = right = None

    for other, candidate in enumerate(drawings):
        if other == index:
            continue
        if not _is_profile_host(profile, candidate):
            continue
        gap = horizontal_gap(profile.box, candidate.box)
        if gap >= narrow * width:
            continue
        if candidate.box[2] <= profile.box[0] and (
            left is None or gap < left[0]
        ):
            left = (gap, other)
        if candidate.box[0] >= profile.box[2] and (
            right is None or gap < right[0]
        ):
            right = (gap, other)
    return left, right


def _is_profile_host(profile: Component, candidate: Component) -> bool:
    """Report whether a view is a plausible parent for a profile."""
    shared = _SIDE_BY_SIDE_OVERLAP * min(profile.height, candidate.height)
    if vertical_overlap(profile.box, candidate.box) <= shared:
        return False
    if candidate.height < 0.5 * profile.height:
        return False
    return profile.density < 0.15 or profile.width < 0.4 * candidate.width


def link_dash_connectors(
    classified: Classified,
    sets: _UnionFind,
    width: int,
    height: int,
    bridge: float,
) -> None:
    """
    Link views explicitly tied together by a dash or rule mark.

    A plain rule drawn between two views is the illustrator stating that
    they are two surfaces of one lithic, so it is strong evidence for
    grouping — stronger than the distance between the views, which may
    sit either side of any gap threshold.

    Such a rule is drawn *between* the views and so routinely overlaps
    their bounding boxes, which are generous. The only mark that must be
    ignored is one lying wholly inside a single drawing: that is a tick
    or interior detail, not a link between artefacts.
    """
    drawings = classified.drawings
    for orientation, dash in classified.dashes:
        if any(_contains(d.box, dash) for d in drawings):
            continue
        pair = (
            _dash_neighbours_horizontal(drawings, dash, width, bridge)
            if orientation == 'h'
            else _dash_neighbours_vertical(drawings, dash, height, bridge)
        )
        if pair is not None:
            sets.link(*pair)


def _dash_neighbours_horizontal(
    drawings: Sequence[Component],
    dash: BBox,
    width: int,
    bridge: float,
) -> Tuple[int, int]:
    """
    Find the views a horizontal connector joins.

    The connector need only start within or beside the left view and end
    within or beside the right one; requiring clear space either side
    would miss the common case of a rule drawn under two adjacent
    surfaces, overlapping both of their bounding boxes.
    """
    centre_y = (dash[1] + dash[3]) / 2
    reach = bridge * width

    def spans_row(box):
        return box[1] <= centre_y <= box[3]

    left = [
        k for k, d in enumerate(drawings)
        if spans_row(d.box)
        and d.box[0] < dash[0] and d.box[2] < dash[2]
        and dash[0] - d.box[2] < reach
    ]
    right = [
        k for k, d in enumerate(drawings)
        if spans_row(d.box)
        and d.box[2] > dash[2] and d.box[0] > dash[0]
        and d.box[0] - dash[2] < reach
    ]
    if not (left and right):
        return None

    i = max(left, key=lambda k: drawings[k].box[2])
    j = min(right, key=lambda k: drawings[k].box[0])
    return (i, j) if i != j else None


def _dash_neighbours_vertical(
    drawings: Sequence[Component],
    dash: BBox,
    height: int,
    bridge: float,
) -> Tuple[int, int]:
    """Find the views a vertical connector joins."""
    centre_x = (dash[0] + dash[2]) / 2
    above = [
        k for k, d in enumerate(drawings)
        if d.box[0] <= centre_x <= d.box[2]
        and d.box[1] < dash[1] and d.box[3] < dash[3]
        and dash[1] - d.box[3] < bridge * height
    ]
    below = [
        k for k, d in enumerate(drawings)
        if d.box[0] <= centre_x <= d.box[2]
        and d.box[3] > dash[3] and d.box[1] > dash[1]
        and d.box[1] - dash[3] < bridge * height
    ]
    if not (above and below):
        return None

    i = max(above, key=lambda k: drawings[k].box[3])
    j = min(below, key=lambda k: drawings[k].box[1])
    return (i, j) if i != j else None


def _contains(outer: Sequence[int], inner: Sequence[int]) -> bool:
    """Report whether one box lies wholly within another."""
    return (
        outer[0] <= inner[0] and outer[1] <= inner[1]
        and outer[2] >= inner[2] and outer[3] >= inner[3]
    )


### BOX ASSEMBLY ###

def overlap_fraction(a: Sequence[int], b: Sequence[int]) -> float:
    """
    Measure how much of the smaller box the overlap covers.

    Returns 0.0 when the boxes do not overlap on both axes, and 1.0 when
    the smaller box lies wholly inside the larger. A scar inside a
    surface scores near 1.0; two neighbouring artefacts clipping at the
    corners score a few percent.

    Parameters
    ----------
    a, b : sequence of int
        Boxes as ``[x0, y0, x1, y1]``.

    Returns
    -------
    float
        Overlap area as a fraction of the smaller box's area.
    """
    shared_w = horizontal_overlap(a, b)
    shared_h = vertical_overlap(a, b)
    if shared_w <= 0 or shared_h <= 0:
        return 0.0

    smaller = min(
        box_width(a) * box_height(a), box_width(b) * box_height(b)
    )
    return (shared_w * shared_h) / smaller if smaller else 0.0


def merge_overlapping(boxes: Sequence[BBox]) -> List[BBox]:
    """
    Merge artefact boxes that substantially contain one another.

    Linking can produce enclosing boxes that intersect even when no
    individual view did. A box largely inside another cannot be cropped
    separately without duplicating ink, so the two are one artefact — but
    a corner clip between neighbours is not grounds for merging.
    """
    merged = [list(box) for box in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                if overlap_fraction(merged[i], merged[j]) >= _MIN_CONTAINMENT:
                    merged[i] = union(merged[i], merged[j])
                    merged.pop(j)
                    changed = True
                    break
            if changed:
                break
    return merged


def attach_annotations(
    classified: Classified,
    boxes: Sequence[BBox],
    width: int,
) -> List[BBox]:
    """
    Attach identifier labels and arrows to their nearest artefact.

    Distances are measured against the artefact boxes as they stood
    before any attachment, so a label cannot pull a box towards a second
    label and chain across the page.

    Parameters
    ----------
    classified : Classified
        Sorted page components.
    boxes : sequence of BBox
        Artefact boxes.
    width : int
        Page width in pixels.

    Returns
    -------
    list of BBox
        Artefact boxes grown to include their annotations.
    """
    if not boxes:
        return []

    grown = [list(box) for box in boxes]
    anchors = [list(box) for box in boxes]
    annotations = list(classified.labels) + [
        dash for _, dash in classified.dashes
    ]

    for annotation in annotations:
        nearest = min(
            range(len(anchors)),
            key=lambda k: box_distance(annotation, anchors[k]),
        )
        if box_distance(annotation, anchors[nearest]) < _ANNOTATION_REACH * width:
            grown[nearest] = union(grown[nearest], annotation)
    return grown


def merge_bar_segments(
    bars: Sequence[BBox], page_width: int
) -> List[BBox]:
    """
    Join the blocks of a segmented scale bar into one box.

    Scale bars are conventionally drawn as alternating filled and open
    blocks. The open blocks fall out at threshold, leaving the filled
    ones as separate components, so collinear neighbours are rejoined
    into the single bar they represent.

    Parameters
    ----------
    bars : sequence of BBox
        Candidate bar boxes.
    page_width : int
        Page width in pixels.

    Returns
    -------
    list of BBox
        One box per physical bar.
    """
    merged = [list(bar) for bar in bars]
    limit = _BAR_SEGMENT_GAP * page_width
    changed = True

    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                if _is_same_bar(merged[i], merged[j], limit):
                    merged[i] = union(merged[i], merged[j])
                    merged.pop(j)
                    changed = True
                    break
            if changed:
                break
    return merged


def _is_same_bar(a: BBox, b: BBox, limit: float) -> bool:
    """Report whether two blocks are segments of one scale bar."""
    shared = 0.5 * min(box_height(a), box_height(b))
    return (
        vertical_overlap(a, b) > shared
        and horizontal_gap(a, b) < limit
    )


def build_scale_bars(
    classified: Classified, bar_config: Dict, page_width: int
) -> List[BBox]:
    """
    Assemble scale bar boxes, optionally including their captions.

    The caption states the bar's real-world length, so including it
    makes the exported crop independently readable. Captions are always
    withheld from artefact crops regardless of this setting.

    Parameters
    ----------
    classified : Classified
        Sorted page components.
    bar_config : dict
        Scale bar configuration.
    page_width : int
        Page width in pixels.

    Returns
    -------
    list of BBox
        One box per scale bar.
    """
    if not bar_config.get('enabled', True):
        return []

    bars = merge_bar_segments(classified.bars, page_width)
    if not bar_config.get('include_caption', True) or not bars:
        return bars

    for caption in classified.bar_captions:
        nearest = min(
            range(len(bars)), key=lambda k: box_distance(caption, bars[k])
        )
        bars[nearest] = union(bars[nearest], caption)
    return bars


def reading_order(boxes: Sequence[BBox]) -> List[BBox]:
    """
    Sort artefact boxes into reading order.

    Boxes are banded into rows top to bottom, then ordered left to right
    within each row, matching how a plate is read and how figure numbers
    are conventionally assigned.

    Parameters
    ----------
    boxes : sequence of BBox
        Artefact boxes in any order.

    Returns
    -------
    list of BBox
        Boxes in reading order.
    """
    ordered = sorted(boxes, key=lambda b: (b[1], b[0]))
    rows: List[List[BBox]] = []
    current: List[BBox] = []

    for box in ordered:
        starts_new_row = current and box[1] > (
            max(c[3] for c in current) - 0.3 * box_height(box)
        )
        if starts_new_row:
            rows.append(current)
            current = []
        current.append(box)
    if current:
        rows.append(current)

    return [box for row in rows for box in sorted(row, key=lambda b: b[0])]
