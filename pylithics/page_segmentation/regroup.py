"""
Let the identifiers printed on a plate correct the grouping.

The illustrator set one number or letter beside each lithic, so the
identifiers read from a plate say how many lithics a box holds. Their
count is reliable; their position is only a hint, since the number can
sit above, below or beside its lithic at any distance. So here the
count decides *whether* a box is wrong and the position decides
*where* to change it, and a change is kept only when the two agree.
When they do not, the box is left as it was, with the flag it already
carries.

Two rules, run in this order:

1. **Split** a box that holds several identifiers, along the widest
   empty run across it, until each piece holds exactly one identifier
   and at least one drawing. A piece that is only a loose numeral is
   not a lithic, and refuses the cut. A dense plate with staggered rows
   has no empty run across the whole box; there each drawing goes to
   its nearest identifier instead, and the split is kept only when
   every identifier gets a drawing and the pieces do not overlap.
2. **Join** a box that holds no identifier to its nearest aligned
   neighbour that holds exactly one, when nothing lies between them
   and the joined box still holds exactly one. A box that holds a
   glyph the reader saw but could not read is left alone: that glyph
   is probably its own identifier.

Some plates set the number below its lithic, where it lands at the top
edge of the box beneath. So a read sitting just under a box that does
not hold it is ambiguous and is not evidence for either rule, and a
box with a read just under it is never joined: that read is probably
its own.
"""

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .detection import Component
from .geometry import (
    BBox, box_distance, boxes_intersect, horizontal_overlap, union,
    vertical_overlap,
)
from .grouping import reading_order
from .identifiers import PageReads, Read, assign_reads

SPLIT = 'identifier_split'
JOIN = 'identifier_join'

# A cut needs at least this much empty space, in pixels, to be a gap
# between two lithics rather than the space between two hatch strokes.
_MIN_RUN = 4
# Two boxes are aligned when they overlap by at least this share of
# the shorter one across the axis that separates them.
_MIN_ALIGNMENT = 0.5
# Two lithics made by a seeded split may overlap by at most this share
# of the smaller one's drawings: more means the views were not sorted.
_MAX_HULL_OVERLAP = 0.5
# A piece of raw ink smaller than this is a speck, not part of a drawing.
_MIN_PIECE_PIXELS = 4
# A piece of raw ink spanning more than this fraction of the page in
# both directions is the frame drawn round the plate, not a drawing.
_MAX_FRAME_FRAC = 0.6
# The adjacency score divides the gap by this plus the share of the
# shorter side the two boxes overlap along, so a mark directly below
# or beside a drawing counts as nearer than one off its corner.
_ADJACENCY_FLOOR = 0.1
# A vertical gap counts this much more than a horizontal one: the
# views of a lithic are set side by side in a row, and the next lithic
# down is often nearer than the profile drawn beside the view.
_ROW_WEIGHT = 1.5
# A drawing-sized piece is a section of the drawing above it when it
# is at most this share of that drawing's height, overlaps it by at
# least half its own width, and sits within its own height of it.
_SECTION_HEIGHT = 0.5
# A read this many glyph heights or fewer under a box that is not its
# home may be that box's label, set beneath its lithic. It is not
# evidence for either rule. Measured on the Archaic Oldowan plates:
# such labels sit 1.3-4.2 heights under their lithic. A label under
# its own lithic lies above the next box, never under it, so the
# direction test keeps those.
_UNDER_BOX = 5.0


def refine_boxes(
    boxes: Sequence[BBox],
    reads: PageReads,
    components: Sequence[Component],
    page_size: Tuple[int, int],
    config: Dict,
) -> Tuple[List[BBox], List[str]]:
    """
    Apply both rules and return the boxes in reading order, tagged.

    Parameters
    ----------
    boxes : sequence of BBox
        Artefact boxes from grouping.
    reads : PageReads
        The identifiers read on the page, with their positions.
    components : sequence of Component
        Every ink blob on the page.
    page_size : tuple of int
        Page ``(width, height)`` in pixels.
    config : dict
        The page segmentation configuration.

    Returns
    -------
    tuple of list
        ``(boxes, tags)``: the boxes in reading order and, for each,
        the rule that made it (``identifier_split``,
        ``identifier_join``, or ``''`` when untouched).
    """
    settings = config.get('identifiers', {}).get('regroup', {})
    if not settings.get('enabled', True) or not reads.reads:
        return list(boxes), [''] * len(boxes)
    width, height = page_size
    min_area = config.get('grouping', {}).get('min_area', 0.0004) * width * height

    evidence = _evidence(reads, boxes)
    tagged = [(list(box), '') for box in boxes]
    tagged = split_by_identifiers(
        tagged, evidence, components, min_area, settings.get('max_cuts', 4), page_size
    )
    tagged = join_by_identifiers(
        tagged, evidence, reads, width, settings.get('join_gap', 0.12)
    )
    ordered = reading_order([box for box, _ in tagged])
    tags = {tuple(box): tag for box, tag in tagged}
    return ordered, [tags[tuple(box)] for box in ordered]


def _evidence(reads: PageReads, boxes: Sequence[BBox]) -> PageReads:
    """
    The reads that count: those clearly inside one box and no other.

    Judged against the boxes grouping produced, before either rule
    moves them: that layout is what says where a label could belong.
    """
    clear = [r for r in reads.reads if not _near_another_box(r[2], boxes)]
    return PageReads(reads=clear, seen=reads.seen, reach=reads.reach,
                     attempted=reads.attempted)


def _near_another_box(glyph: BBox, boxes: Sequence[BBox]) -> bool:
    """Whether a glyph sits just under a box that does not hold it."""
    return any(
        not _inside(glyph, box) and _just_under(glyph, box) for box in boxes
    )


def _just_under(glyph: BBox, box: BBox) -> bool:
    """
    Whether a glyph sits directly beneath a box, within reach of it.

    That is where a label set below its lithic lands. A glyph beside
    the box, or off its corner, is set by some other drawing.
    """
    if horizontal_overlap(glyph, box) <= 0 or glyph[1] < box[3]:
        return False
    return glyph[1] - box[3] <= _UNDER_BOX * (glyph[3] - glyph[1])


### RULE 1: SPLIT ###

def split_by_identifiers(
    tagged: Sequence[Tuple[BBox, str]],
    reads: PageReads,
    components: Sequence[Component],
    min_area: float,
    max_cuts: int,
    page_size: Tuple[int, int],
) -> List[Tuple[BBox, str]]:
    """Cut every box holding several identifiers into one box each."""
    boxes = [box for box, _ in tagged]
    identifiers = assign_reads(reads, boxes)
    result = []
    for (box, tag), identifier in zip(tagged, identifiers):
        if identifier.flag != 'several_identifiers':
            result.append((box, tag))
            continue
        inside = _ink_pieces(
            [c for c in components if _inside(c.box, box)], page_size
        )
        pieces = _split_box(box, identifier.candidates, inside, min_area, max_cuts)
        if pieces is None:
            logging.debug("Box %s holds several identifiers but no clean cut", box)
            result.append((box, tag))
        else:
            logging.debug("Box %s split into %d by its identifiers", box, len(pieces))
            result.extend((piece, SPLIT) for piece in pieces)
    return result


def _ink_pieces(
    components: Sequence[Component], page_size: Tuple[int, int]
) -> List[BBox]:
    """
    The raw ink inside a box, as the boxes of its connected pieces.

    Grouping works on closed components, and on a dense plate closing
    fuses neighbouring lithics into one. The raw ink still holds each
    outline as its own piece, so the split works on that. Specks under
    ``_MIN_PIECE_PIXELS`` are scanner noise, and a piece spanning most
    of the page is the frame drawn round the plate; both are dropped.
    """
    frame_w, frame_h = _MAX_FRAME_FRAC * page_size[0], _MAX_FRAME_FRAC * page_size[1]
    pieces = []
    for component in components:
        if component.mask is None:
            pieces.append(list(component.box))
            continue
        x0, y0 = component.box[0], component.box[1]
        _, _, stats, _ = cv2.connectedComponentsWithStats(
            component.mask.astype(np.uint8), connectivity=8
        )
        for left, top, w, h, area in stats[1:]:
            if area >= _MIN_PIECE_PIXELS and (w <= frame_w or h <= frame_h):
                pieces.append([x0 + left, y0 + top, x0 + left + w, y0 + top + h])
    return pieces


def _split_box(
    box: BBox,
    reads: Sequence[Read],
    ink: Sequence[BBox],
    min_area: float,
    max_cuts: int,
) -> Optional[List[BBox]]:
    """
    Cut a box until each piece holds one identifier, or give up.

    The widest cut whose two sides each hold a label and a drawing is
    taken and each side cut in turn. When a side cannot be finished,
    the seeded split takes the whole box instead of the next cut being
    tried: trying every cut at every level is exponential in the
    number of labels, and hangs on a plate of twenty.

    Returns the pieces, or None when no cut satisfies the rule.
    """
    labels = {text for text, _, _ in reads}
    if len(labels) == 1:
        return [box]
    if not labels:
        return None
    marks = list(ink) + [glyph for _, _, glyph in reads]
    for axis, position in _empty_runs(box, marks)[:max_cuts]:
        halves = _halve(box, reads, ink, axis, position, min_area)
        if halves is None:
            continue
        pieces = []
        for piece, piece_reads, piece_ink in halves:
            done = _split_box(piece, piece_reads, piece_ink, min_area, max_cuts)
            if done is None:
                break
            pieces.extend(done)
        else:
            return pieces
        break        # one cut is tried in depth; trying every cut at every level is exponential
    return _seeded_split(reads, ink, min_area)


def _seeded_split(
    reads: Sequence[Read], ink: Sequence[BBox], min_area: float
) -> Optional[List[BBox]]:
    """
    Give each drawing to its identifier, one piece per label.

    For a dense plate whose rows are staggered, so that no empty run
    crosses the whole box. A label set between two lithics is nearer
    to the wrong one about half the time, so the pairing is decided
    for the whole box at once: every label takes one drawing, chosen
    so the sum of label-to-drawing distances is least. Each remaining
    drawing joins the lithic of the nearest label-placed drawing, since
    the views of one lithic sit beside each other, and small marks
    join the nearest drawing. Refused when there are fewer drawings than
    labels, or when two pieces' drawings overlap by more than half.
    """
    glyphs: Dict[str, List[BBox]] = {}
    for text, _, glyph in reads:
        glyphs.setdefault(text, []).append(glyph)
    drawings = _with_sections(_outer_pieces([m for m in ink if _area(m) >= min_area]))
    small = [m for m in ink if _area(m) < min_area]
    if len(drawings) < len(glyphs):
        return None

    owner = _seed_drawings(list(glyphs), glyphs, drawings)
    _grow_pieces(owner, drawings)
    hulls = {text: _union_all([drawings[i] for i, t in owner.items() if t == text])
             for text in glyphs}
    if _hulls_overlap(list(hulls.values())):
        return None
    pieces = {text: [hull] + [list(g) for g in glyphs[text]] for text, hull in hulls.items()}
    for mark in small:
        nearest = min(range(len(drawings)), key=lambda i: _adjacency(mark, drawings[i]))
        pieces[owner[nearest]].append(mark)
    return [_union_all(marks) for marks in pieces.values()]


def _adjacency(a: BBox, b: BBox) -> float:
    """
    How far apart two boxes are, as the illustrator's layout reads it.

    A section is set directly below its view and a profile directly
    beside it, sharing its span. The gap is divided by the share of
    span shared, so a mark that lines up with a drawing is nearer than
    one the same distance off its corner, whose gap is counted along
    both axes.
    """
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = _ROW_WEIGHT * max(0, max(a[1], b[1]) - min(a[3], b[3]))
    shared = max(
        horizontal_overlap(a, b) / max(1, min(a[2] - a[0], b[2] - b[0])),
        vertical_overlap(a, b) / max(1, min(a[3] - a[1], b[3] - b[1])),
    )
    return (dx + dy) / (_ADJACENCY_FLOOR + max(0.0, min(1.0, shared)))


def _outer_pieces(drawings: Sequence[BBox]) -> List[BBox]:
    """
    Drop pieces lying inside another: scars and hatching, not views.

    A large scar inside an outline is drawing-sized on its own, but it
    belongs to the outline around it and must not be paired with a
    label or joined to a neighbour as a section.
    """
    kept: List[BBox] = []
    for piece in drawings:
        if any(_inside(piece, other) and other != piece for other in drawings):
            continue
        if piece not in kept:
            kept.append(list(piece))
    return kept


def _with_sections(drawings: Sequence[BBox]) -> List[BBox]:
    """
    Join each cross-section to the view drawn above it.

    A section is a flat piece set directly under its view. Joined
    first, so the label beside a section names the whole lithic and
    the view above cannot be claimed by a neighbour.
    """
    result = [list(d) for d in drawings]
    changed = True
    while changed:
        changed = False
        for i, piece in enumerate(result):
            host = _view_above(piece, result)
            if host is not None:
                result[host] = union(result[host], piece)
                result.pop(i)
                changed = True
                break
    return result


def _view_above(piece: BBox, drawings: Sequence[BBox]) -> Optional[int]:
    """Index of the taller view a flat piece is the section of, or None."""
    height = piece[3] - piece[1]
    for index, other in enumerate(drawings):
        if other is piece or other[3] > piece[1]:
            continue
        if height > _SECTION_HEIGHT * (other[3] - other[1]):
            continue
        if horizontal_overlap(piece, other) < 0.5 * (piece[2] - piece[0]):
            continue
        if piece[1] - other[3] <= height:
            return index
    return None


def _seed_drawings(
    labels: Sequence[str], glyphs: Dict[str, List[BBox]], drawings: Sequence[BBox]
) -> Dict[int, str]:
    """One drawing per label, the pairing with the least total distance."""
    from scipy.optimize import linear_sum_assignment

    cost = np.array([
        [min(_adjacency(drawing, g) for g in glyphs[text]) for drawing in drawings]
        for text in labels
    ])
    rows, cols = linear_sum_assignment(cost)
    return {int(col): labels[row] for row, col in zip(rows, cols)}


def _grow_pieces(owner: Dict[int, str], drawings: Sequence[BBox]) -> None:
    """
    Attach each unplaced drawing to the lithic of the nearest placed one.

    Only the drawings placed by their label count as anchors. Letting
    a newly attached drawing anchor others chains across a dense
    plate, one lithic swallowing its neighbours.
    """
    anchors = list(owner)
    for index, drawing in enumerate(drawings):
        if index in owner:
            continue
        nearest = min(anchors, key=lambda i: _adjacency(drawing, drawings[i]))
        owner[index] = owner[nearest]


def _hulls_overlap(hulls: Sequence[BBox]) -> bool:
    """Whether two lithics' drawings overlap by more than half the smaller."""
    for i, a in enumerate(hulls):
        for b in hulls[i + 1:]:
            shared = (max(0, min(a[2], b[2]) - max(a[0], b[0]))
                      * max(0, min(a[3], b[3]) - max(a[1], b[1])))
            if shared > _MAX_HULL_OVERLAP * min(_area(a), _area(b)):
                return True
    return False


def _union_all(marks: Sequence[BBox]) -> BBox:
    """The box enclosing every mark."""
    box = list(marks[0])
    for mark in marks[1:]:
        box = union(box, mark)
    return box


def _empty_runs(box: BBox, marks: Sequence[BBox]) -> List[Tuple[int, int]]:
    """
    Every ink-free run across a box, widest first, as ``(axis, cut)``.

    ``axis`` is 0 for a vertical cut at column ``cut`` and 1 for a
    horizontal cut at row ``cut``.
    """
    runs = []
    for axis in (0, 1):
        spans = sorted((m[axis], m[axis + 2]) for m in marks)
        reach = box[axis]
        for start, end in spans:
            if start - reach > _MIN_RUN:
                runs.append((start - reach, axis, (start + reach) // 2))
            reach = max(reach, end)
    runs.sort(key=lambda r: -r[0])
    return [(axis, cut) for _, axis, cut in runs]


def _halve(
    box: BBox,
    reads: Sequence[Read],
    ink: Sequence[BBox],
    axis: int,
    position: int,
    min_area: float,
) -> Optional[List[Tuple[BBox, List[Read], List[BBox]]]]:
    """
    Cut a box at a position, if each side holds a read and a drawing.

    Each side's box is the union of the ink and glyphs on that side,
    so the pieces stay tight and a numeral stays with its lithic.
    """
    halves = []
    for side in (False, True):
        side_reads = [r for r in reads if _on_side(r[2], axis, position) == side]
        side_ink = [m for m in ink if _on_side(m, axis, position) == side]
        if not side_reads or not any(_area(m) >= min_area for m in side_ink):
            return None
        piece = list(side_ink[0])
        for mark in side_ink[1:] + [r[2] for r in side_reads]:
            piece = union(piece, mark)
        halves.append((piece, side_reads, side_ink))
    return halves


def _on_side(mark: BBox, axis: int, position: int) -> bool:
    """Whether a mark's centre lies past a cut along an axis."""
    return (mark[axis] + mark[axis + 2]) / 2 >= position


### RULE 2: JOIN ###

def join_by_identifiers(
    tagged: Sequence[Tuple[BBox, str]],
    evidence: PageReads,
    reads: PageReads,
    width: int,
    join_gap: float,
) -> List[Tuple[BBox, str]]:
    """
    Join every unlabelled box into its labelled neighbour, repeatedly.

    ``evidence`` holds the reads that count; ``reads`` holds every
    read and every glyph seen, which say when a box must be left alone.
    """
    result = [(list(box), tag) for box, tag in tagged]
    limit = join_gap * width
    while True:
        boxes = [box for box, _ in result]
        identifiers = assign_reads(evidence, boxes)
        joined = _join_one(result, identifiers, evidence, reads, limit)
        if joined is None:
            return result
        result = joined


def _join_one(
    tagged: List[Tuple[BBox, str]],
    identifiers: Sequence,
    evidence: PageReads,
    reads: PageReads,
    limit: float,
) -> Optional[List[Tuple[BBox, str]]]:
    """Make the first join the rule allows, or return None."""
    boxes = [box for box, _ in tagged]
    for index, identifier in enumerate(identifiers):
        if identifier.flag != 'no_identifier' or _has_own_label(boxes[index], reads):
            continue
        host = _nearest_host(index, boxes, identifiers, limit)
        if host is None:
            continue
        merged = union(boxes[index], boxes[host])
        trial = [box for i, box in enumerate(boxes) if i not in (index, host)]
        trial.append(merged)
        if not assign_reads(reads, trial)[-1].named:
            continue                     # judged on every read, not only the evidence
        logging.debug("Box %s joined into %s by its identifier", boxes[index], boxes[host])
        kept = [tagged[i] for i in range(len(tagged)) if i not in (index, host)]
        return kept + [(merged, JOIN)]
    return None


def _has_own_label(box: BBox, reads: PageReads) -> bool:
    """
    Whether a box probably has an identifier of its own.

    True when a glyph the reader saw sits inside it, or when a read
    sits just under it: a label set below its lithic lands in the box
    beneath, but it is still that lithic's.
    """
    if any(_inside(glyph, box) for glyph in reads.seen):
        return True
    return any(_just_under(glyph, box) for _, _, glyph in reads.reads)


def _nearest_host(
    index: int, boxes: Sequence[BBox], identifiers: Sequence, limit: float
) -> Optional[int]:
    """
    The one aligned, labelled neighbour nearest a box, or None.

    Two neighbours at the same distance mean the position gives no
    answer, and nothing is joined.
    """
    box = boxes[index]
    hosts = sorted(
        (box_distance(box, other), other_index)
        for other_index, other in enumerate(boxes)
        if other_index != index and identifiers[other_index].named
        and _aligned(box, other) and box_distance(box, other) <= limit
        and not _blocked(box, other, boxes, (index, other_index))
    )
    if not hosts or (len(hosts) > 1 and hosts[0][0] == hosts[1][0]):
        return None
    return hosts[0][1]


def _aligned(a: BBox, b: BBox) -> bool:
    """Whether two boxes sit side by side, or one above the other."""
    beside = vertical_overlap(a, b) >= _MIN_ALIGNMENT * min(a[3] - a[1], b[3] - b[1])
    stacked = horizontal_overlap(a, b) >= _MIN_ALIGNMENT * min(a[2] - a[0], b[2] - b[0])
    return beside or stacked


def _blocked(
    a: BBox, b: BBox, boxes: Sequence[BBox], skip: Tuple[int, int]
) -> bool:
    """Whether a third box lies in the space between two boxes."""
    x0, x1 = _span_between(a[0], a[2], b[0], b[2])
    y0, y1 = _span_between(a[1], a[3], b[1], b[3])
    if x1 <= x0 or y1 <= y0:
        return False
    return any(
        boxes_intersect([x0, y0, x1, y1], other)
        for index, other in enumerate(boxes) if index not in skip
    )


def _span_between(lo_a: int, hi_a: int, lo_b: int, hi_b: int) -> Tuple[int, int]:
    """The gap between two spans on one axis, or their overlap if none."""
    if max(lo_a, lo_b) > min(hi_a, hi_b):
        return min(hi_a, hi_b), max(lo_a, lo_b)
    return max(lo_a, lo_b), min(hi_a, hi_b)


### HELPERS ###

def _inside(inner: Sequence[int], outer: Sequence[int]) -> bool:
    """Whether one box lies wholly within another."""
    return (inner[0] >= outer[0] and inner[1] >= outer[1]
            and inner[2] <= outer[2] and inner[3] <= outer[3])


def _area(box: Sequence[int]) -> int:
    """Pixel area of a box."""
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])
