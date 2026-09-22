"""
Read the identifier printed beside each artefact on a plate.

Every lithic on a published plate carries a number or a letter, and
that is how the publication's text and tables refer to it. Naming a
crop from that identifier keeps the link; naming it by reading order
loses it, and one grouping error shifts every name after it.

The reader works in five steps, each a small function below:

1. **Candidates** — label-sized blobs inside an artefact's box, from
   the components already classified as labels and from the raw ink
   inside the artefact's own component (closing can fuse a numeral to
   its lithic, hiding it from classification).
2. **Filters** — text never overlaps the illustration, so a candidate
   inside a lithic's filled outline is dropped, as is one without clear
   space around it, and one inside a scale bar.
3. **Reading** — RapidOCR on the padded, upscaled glyph.
4. **Validation** — a plate is numbered *or* lettered, never both;
   known confusions (E for 3, S for 5) are corrected within the page's
   alphabet, and the reads are checked for a consecutive run.
5. **Assignment** — one identifier per crop. Several in one crop means
   the crop holds several lithics, and it is flagged rather than named.

RapidOCR is optional. Without it every crop keeps its reading-order
name and the manifest says so.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .detection import Component, PageImage
from .geometry import (
    BBox, box_distance, box_height, box_width, horizontal_gap, vertical_overlap,
)
from .grouping import Classified

# Pieces of raw ink taller than this fraction of the page are drawing,
# not glyphs; the same ceiling the label classifier uses.
_MAX_GLYPH_PAGE_FRAC = 0.05
# A glyph is at least this tall in pixels, or OCR has nothing to read.
_MIN_GLYPH_PIXELS = 6
# Glyphs on one baseline closer than this many glyph heights are one
# string, so "13" is read whole rather than as "1" and "3".
_STRING_GAP = 0.8
# A glyph is at least this wide for its height. A printed "1" is about
# a third as wide as it is tall; a hairline connector dash, which the
# classifier files with the same shape, is a tenth or less.
_MIN_GLYPH_ASPECT = 0.2
# The identifiers on a plate are set in one size. A read this far from
# the page's typical height, in either direction, is stipple that read
# as a digit or a small lithic that did. Digits share one height; the
# lowercase letters of a lettered plate differ by their ascenders, an
# ``a`` standing about 0.72 of a ``d``, which the floor admits.
_MIN_RELATIVE_HEIGHT = 0.65
# The plate's type size is taken from reads at least this confident.
# Real numerals read at 0.98 or better; the edge strokes and stipple
# that read as "1" mostly do not, and on a plate where they outnumber
# the numerals a plain median would size the type from the junk.
_CONFIDENT = 0.9
# A read of a lone "1" must be this confident. It is the one glyph a
# stroke of the drawing can pass for — a connector rule, an edge — and
# those read at 0.6-0.7 where a printed "1" reads at 0.97 or better.
_LONE_STROKE_GLYPHS = {'1', 'l', 'I'}
# An identifier usually sits inside its artefact's box; on some plates
# it is set just outside, above or beside the drawing. A glyph within
# this many glyph heights of one box, and no other, belongs to it.
_NEAR_BOX = 1.5
# A candidate whose box overlaps a lithic's filled outline by more than
# this share sits on the illustration and is not text.
_MAX_OUTLINE_OVERLAP = 0.5
# A mark whose centre lies deeper than this many glyph heights inside a
# drawing's closed silhouette is hatching or stipple, whatever the state
# of the outline. A numeral fused to a corner by closing is never deep:
# its own edge is the silhouette's edge there.
_MAX_DEPTH = 1.0
# A piece of raw ink spanning more than this fraction of the page is a
# frame drawn round the plate, not a lithic; filling its outline would
# cover the whole page (99% on the Spain plates) and reject everything.
_MAX_FRAME_FRAC = 0.6
# Ring around a glyph in which ink is measured, in glyph heights.
_RING = 0.5
# Glyph height the crop is upscaled to before reading, in pixels.
_READ_HEIGHT = 64
# Whitespace added around the upscaled glyph, in pixels.
_READ_PAD = 20
# Single-glyph confusions between the two alphabets, corrected when the
# page's alphabet is known.
_TO_DIGIT = {'O': '0', 'o': '0', 'I': '1', 'l': '1', 'i': '1', 'Z': '2',
             'z': '2', 'E': '3', 'S': '5', 's': '5', 'B': '8', 'g': '9'}
_TO_LETTER = {v: k for k, v in _TO_DIGIT.items() if k.isupper()}

_engine = None
_engine_missing_reported = False


@dataclass
class Identifier:
    """The identifier assigned to one artefact crop."""

    label: str = ''
    source: str = 'index'
    confidence: float = 0.0
    flag: str = ''
    candidates: List[Tuple[str, float, BBox]] = field(default_factory=list)

    @property
    def named(self) -> bool:
        """Whether the crop takes its name from a read identifier."""
        return self.source == 'read' and bool(self.label)

    @property
    def candidate_text(self) -> str:
        """Every distinct reading found in the crop, for the manifest."""
        return ';'.join(sorted({c[0] for c in self.candidates}))


def read_identifiers(
    page: PageImage,
    components: Sequence[Component],
    classified: Classified,
    boxes: Sequence[BBox],
    bars: Sequence[BBox],
    raw: np.ndarray,
    closed: np.ndarray,
    config: Dict,
) -> List[Identifier]:
    """
    Read the identifier printed beside each artefact.

    Parameters
    ----------
    page, components, classified
        The loaded page, its ink blobs, and the blobs sorted by role.
    boxes, bars : sequence of BBox
        Artefact boxes in reading order; scale bars with their captions.
    raw, closed : np.ndarray
        The detection mask before and after morphological closing.
    config : dict
        The ``identifiers`` configuration section.

    Returns
    -------
    list of Identifier
        One per box, in order; an unnamed entry carries its reason in ``flag``.
    """
    identifiers = [Identifier(flag='not_read') for _ in boxes]
    if not config.get('enabled', True) or not boxes:
        return identifiers
    engine = _load_engine()
    if engine is None:
        return identifiers
    for identifier in identifiers:
        identifier.flag = ''

    reach = config.get('reach', _NEAR_BOX)
    glyphs = _candidates(page, components, classified, boxes, reach)
    glyphs = _filter_glyphs(glyphs, page, bars, raw, closed, config)
    strings = _join_strings(_dedupe(glyphs))
    strings = _filter_strings(strings, raw, config)
    reads = _read(strings, page, engine, config)
    reads = _validate(_consistent_size(reads))
    _assign(reads, boxes, identifiers, reach)
    _report(page, identifiers)
    return identifiers


def _report(page: PageImage, identifiers: Sequence[Identifier]) -> None:
    """Say, per page, what identifier reading found."""
    named = sum(i.named for i in identifiers)
    name = page.stem
    if not named and not any(i.candidates for i in identifiers):
        logging.info("%s: no identifiers found on this page", name)
        return
    flagged = sum(1 for i in identifiers if i.flag in ('several_identifiers', 'duplicate'))
    logging.info("%s: %d of %d crop(s) named from the plate, %d flagged",
                 name, named, len(identifiers), flagged)


### ENGINE ###

def _load_engine():
    """Load RapidOCR once per run; report its absence once."""
    global _engine, _engine_missing_reported
    if _engine is not None:
        return _engine
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        if not _engine_missing_reported:
            logging.error(
                "RapidOCR is necessary to read the plate identifiers. Install it with:\n"
                "           pip install 'PyLithics[ocr]'\n"
                "       The crops are numbered in reading order."
            )
            _engine_missing_reported = True
        return None
    _engine = RapidOCR()
    return _engine


### CANDIDATES ###

def _candidates(
    page: PageImage,
    components: Sequence[Component],
    classified: Classified,
    boxes: Sequence[BBox],
    reach: float = _NEAR_BOX,
) -> List[BBox]:
    """
    Collect every glyph-sized box lying inside an artefact box.

    Filtering happens before glyphs are joined into strings: a plate of
    stippled drawings yields over a thousand specks of raw ink, and the
    joining step compares every pair.
    """
    max_h = _MAX_GLYPH_PAGE_FRAC * page.height
    marks = list(classified.labels) + [box for _, box in classified.dashes]
    glyphs = [list(b) for b in marks if _glyph_sized(b, max_h)]
    for component in components:
        if component.mask is None or component.height <= max_h:
            continue
        glyphs.extend(_raw_pieces(component, max_h))
    return [g for g in glyphs if _home_box(g, boxes, reach=reach) is not None]


def _glyph_sized(box: BBox, max_h: float) -> bool:
    """Whether a box is the size of a printed identifier."""
    h, w = box_height(box), box_width(box)
    return (_MIN_GLYPH_PIXELS <= h <= max_h
            and _MIN_GLYPH_ASPECT * h <= w <= 3 * h)


def _raw_pieces(component: Component, max_h: float) -> List[BBox]:
    """Glyph-sized pieces of raw ink inside a drawing-sized component."""
    _, _, stats, _ = cv2.connectedComponentsWithStats(
        component.mask.astype(np.uint8), connectivity=8
    )
    x0, y0 = component.box[0], component.box[1]
    pieces = []
    for left, top, w, h, _ in stats[1:]:
        box = [x0 + left, y0 + top, x0 + left + w, y0 + top + h]
        if _glyph_sized(box, max_h):
            pieces.append(box)
    return pieces


def _dedupe(boxes: Sequence[BBox]) -> List[BBox]:
    """Drop boxes that repeat one already collected from another source."""
    kept: List[BBox] = []
    for box in boxes:
        if not any(_same_box(box, other) for other in kept):
            kept.append(list(box))
    return kept


def _same_box(a: BBox, b: BBox) -> bool:
    """Whether two boxes cover essentially the same glyph."""
    ox = min(a[2], b[2]) - max(a[0], b[0])
    oy = min(a[3], b[3]) - max(a[1], b[1])
    if ox <= 0 or oy <= 0:
        return False
    smaller = min(box_width(a) * box_height(a), box_width(b) * box_height(b))
    return (ox * oy) / max(1, smaller) > 0.6


def _join_strings(glyphs: Sequence[BBox]) -> List[BBox]:
    """Merge glyphs sharing a baseline and nearly touching into one box."""
    parent = list(range(len(glyphs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(glyphs)):
        for j in range(i + 1, len(glyphs)):
            if _same_string(glyphs[i], glyphs[j]):
                parent[find(i)] = find(j)
    groups: Dict[int, List[BBox]] = {}
    for i, box in enumerate(glyphs):
        groups.setdefault(find(i), []).append(box)
    return [
        [min(b[0] for b in g), min(b[1] for b in g),
         max(b[2] for b in g), max(b[3] for b in g)]
        for g in groups.values()
    ]


def _same_string(a: BBox, b: BBox) -> bool:
    """Whether two glyphs are adjacent characters of one identifier."""
    ha, hb = box_height(a), box_height(b)
    if not 0.6 <= ha / max(1, hb) <= 1.6:
        return False
    if vertical_overlap(a, b) < 0.5 * min(ha, hb):
        return False
    return horizontal_gap(a, b) < _STRING_GAP * max(ha, hb)


def _home_box(
    glyph: BBox, boxes: Sequence[BBox], margin: int = 4, reach: float = _NEAR_BOX
) -> Optional[int]:
    """
    Index of the artefact box a glyph belongs to, or None.

    Inside a box, the glyph is that box's. Outside every box, it is the
    nearest box's if that box lies within a glyph height or so and the
    next nearest does not: some plates set the letter above the drawing
    rather than beneath it, clear of the ink and so of the box.
    """
    for index, box in enumerate(boxes):
        if (box[0] - margin <= glyph[0] and box[1] - margin <= glyph[1]
                and box[2] + margin >= glyph[2] and box[3] + margin >= glyph[3]):
            return index
    reach = reach * box_height(glyph)
    near = sorted((box_distance(glyph, box), index) for index, box in enumerate(boxes))
    if near and near[0][0] <= reach and (len(near) == 1 or near[1][0] > reach):
        return near[0][1]
    return None


### FILTERS ###

def _filter_glyphs(
    glyphs: Sequence[BBox],
    page: PageImage,
    bars: Sequence[BBox],
    raw: np.ndarray,
    closed: np.ndarray,
    config: Dict,
) -> List[BBox]:
    """
    Drop glyphs that cannot be part of an identifier.

    Text never overlaps the illustration. That is tested two ways, as
    outlines are not always drawn closed: a glyph inside a lithic's
    filled raw outline is on the drawing, and so is one lying deep
    inside the closed silhouette, which catches hatching within an
    outline the fill could not close. A glyph inside a scale bar is not
    an identifier either, and a box too empty to hold a stroke is a
    fragment of hatching.
    """
    outlines = _lithic_outlines(raw, page)
    depth = _silhouette_depth(closed, page)
    floor = config.get('min_density', 0.2)
    kept = [
        box for box in glyphs
        if _box_density(box, raw) >= floor
        and not _on_outline(box, outlines)
        and not _deep_inside(box, depth)
        and not any(_overlaps(box, bar) for bar in bars)
    ]
    logging.debug("%d of %d glyphs pass the placement filters", len(kept), len(glyphs))
    return kept


def _filter_strings(
    strings: Sequence[BBox], raw: np.ndarray, config: Dict
) -> List[BBox]:
    """
    Drop strings without clear space around them.

    Measured on whole strings, not glyphs, so the second digit of "12"
    is not counted as ink crowding the first.
    """
    limit = config.get('ring_density', 0.04)
    kept = [box for box in strings if _ring_density(box, raw) <= limit]
    logging.debug("%d of %d strings stand in clear space", len(kept), len(strings))
    return kept


def _consistent_size(
    reads: List[Tuple[str, float, BBox]]
) -> List[Tuple[str, float, BBox]]:
    """
    Keep reads set in the plate's one type size.

    The identifiers on a plate share a size; the median height of what
    was read is that size. A read far from it is stipple that happened
    to read as a digit, or a small lithic read as one, and is dropped.
    """
    if len(reads) < 3:
        return list(reads)
    typical = _typical_height(reads)
    kept = [
        r for r in reads
        if _MIN_RELATIVE_HEIGHT * typical <= box_height(r[2]) <= typical / _MIN_RELATIVE_HEIGHT
    ]
    if len(kept) < len(reads):
        logging.debug("Dropped %d read(s) not set in the plate's type size (%dpx)",
                      len(reads) - len(kept), typical)
    return kept


def _lithic_outlines(raw: np.ndarray, page: PageImage) -> np.ndarray:
    """
    Fill the outline of every drawing-sized piece of raw ink.

    Text never overlaps the illustration, so anything inside a filled
    outline — an arrow, stipple, hatching — is not an identifier. The
    raw ink is used rather than the closed mask: closing can join a
    numeral to its lithic, and the closed outline would swallow it.
    """
    ink = (raw > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    filled = np.zeros_like(ink)
    min_h = _MAX_GLYPH_PAGE_FRAC * page.height
    frame_w, frame_h = _MAX_FRAME_FRAC * page.width, _MAX_FRAME_FRAC * page.height
    for index in range(1, count):
        left, top, w, h, _ = stats[index]
        if h <= min_h and w <= min_h:
            continue
        if w > frame_w or h > frame_h:
            continue                 # a plate frame or rule, not a lithic
        piece = (labels[top:top + h, left:left + w] == index).astype(np.uint8)
        contours, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sub = np.zeros_like(piece)
        cv2.drawContours(sub, contours, -1, 1, -1)
        filled[top:top + h, left:left + w] |= sub
    return filled


def _silhouette_depth(closed: np.ndarray, page: PageImage) -> np.ndarray:
    """
    Distance from each pixel to the edge of the drawing it lies in.

    Each drawing-sized piece of the closed mask is covered by its convex
    hull. A contour fill is not enough: a lithic drawn as hatching with
    a broken outline has an "outline" that traces the hatching itself
    and encloses almost nothing (23% of one Revue scraper), while its
    hull covers the whole drawing. Plate frames are skipped as in
    ``_lithic_outlines``.
    """
    ink = (closed > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    filled = np.zeros_like(ink)
    min_h = _MAX_GLYPH_PAGE_FRAC * page.height
    frame_w, frame_h = _MAX_FRAME_FRAC * page.width, _MAX_FRAME_FRAC * page.height
    for index in range(1, count):
        left, top, w, h, _ = stats[index]
        if (h <= min_h and w <= min_h) or w > frame_w or h > frame_h:
            continue
        piece = (labels[top:top + h, left:left + w] == index).astype(np.uint8)
        points = cv2.findNonZero(piece)
        sub = np.zeros_like(piece)
        cv2.fillConvexPoly(sub, cv2.convexHull(points), 1)
        filled[top:top + h, left:left + w] |= sub
    return cv2.distanceTransform(filled, cv2.DIST_L2, 3)


def _deep_inside(box: BBox, depth: np.ndarray) -> bool:
    """Whether a glyph's centre lies deep inside a drawing's silhouette."""
    cy, cx = (box[1] + box[3]) // 2, (box[0] + box[2]) // 2
    cy = min(max(cy, 0), depth.shape[0] - 1)
    cx = min(max(cx, 0), depth.shape[1] - 1)
    return float(depth[cy, cx]) > _MAX_DEPTH * box_height(box)


def _on_outline(box: BBox, outlines: np.ndarray) -> bool:
    """Whether a candidate lies inside a lithic's filled outline."""
    region = outlines[box[1]:box[3], box[0]:box[2]]
    return region.size > 0 and region.mean() > _MAX_OUTLINE_OVERLAP


def _box_density(box: BBox, raw: np.ndarray) -> float:
    """
    Share of a box that carries ink.

    A printed glyph is a stroke filling a fifth to two thirds of its box
    (measured on 137 identifiers: 0.22 to 0.70); a fragment of hatching
    or a stray hairline fills far less.
    """
    region = raw[box[1]:box[3], box[0]:box[2]] > 0
    return float(region.mean()) if region.size else 0.0


def _ring_density(box: BBox, raw: np.ndarray) -> float:
    """Share of the ring around a box that carries ink."""
    pad = max(3, int(_RING * box_height(box)))
    H, W = raw.shape
    X0, Y0 = max(0, box[0] - pad), max(0, box[1] - pad)
    X1, Y1 = min(W, box[2] + pad), min(H, box[3] + pad)
    outer = raw[Y0:Y1, X0:X1] > 0
    inner = np.zeros_like(outer)
    inner[box[1] - Y0:box[3] - Y0, box[0] - X0:box[2] - X0] = True
    ring = ~inner
    return float((outer & ring).sum()) / max(1, int(ring.sum()))


def _overlaps(a: BBox, b: BBox) -> bool:
    """Whether two boxes intersect."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


### READING ###

def _read(
    candidates: Sequence[BBox], page: PageImage, engine, config: Dict
) -> List[Tuple[str, float, BBox]]:
    """Read each candidate, keeping alphanumerics above the confidence floor."""
    floor = config.get('min_confidence', 0.6)
    reads = []
    for box in candidates:
        text, confidence = _read_one(_glyph_image(page, box), engine)
        needed = _CONFIDENT if text in _LONE_STROKE_GLYPHS else floor
        if text and confidence >= needed:
            reads.append((text, confidence, box))
        elif text:
            logging.debug("Discarded low-confidence read %r (%.2f) at %s",
                          text, confidence, box)
    return reads


def _glyph_image(page: PageImage, box: BBox) -> np.ndarray:
    """Cut, pad, upscale and contrast-stretch a glyph for reading."""
    x0, y0, x1, y1 = box
    m = max(4, box_height(box) // 3)
    crop = page.gray[max(0, y0 - m):y1 + m, max(0, x0 - m):x1 + m]
    lo, hi = int(crop.min()), int(crop.max())
    if hi > lo:
        crop = ((crop.astype(np.float32) - lo) * (255.0 / (hi - lo))).astype(np.uint8)
    scale = max(1.0, _READ_HEIGHT / max(1, y1 - y0))
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    padded = np.full((crop.shape[0] + 2 * _READ_PAD, crop.shape[1] + 2 * _READ_PAD),
                     255, np.uint8)
    padded[_READ_PAD:-_READ_PAD, _READ_PAD:-_READ_PAD] = crop
    return cv2.cvtColor(padded, cv2.COLOR_GRAY2RGB)


def _read_one(image: np.ndarray, engine) -> Tuple[str, float]:
    """
    Run the engine on one glyph image.

    The crop already holds exactly one string, so the engine's text
    detector has nothing to find and is skipped: its recogniser alone
    reads the same glyphs nine times faster (38 ms against 346 ms).
    """
    if hasattr(engine, 'text_rec'):
        result, _ = engine.text_rec([image[:, :, ::-1]])     # BGR
        if not result:
            return '', 0.0
        text, score = result[0][0], result[0][1]
        return re.sub(r'[^0-9A-Za-z]', '', text), float(score)
    result, _ = engine(image)
    if not result:
        return '', 0.0
    text = re.sub(r'[^0-9A-Za-z]', '', ''.join(r[1] for r in result))
    return text, min(float(r[2]) for r in result)


### VALIDATION ###

def _typical_height(reads: Sequence[Tuple[str, float, BBox]]) -> int:
    """
    The plate's type size: median height of its confident reads.

    Falls back to all reads when fewer than two are confident.
    """
    sure = [r for r in reads if r[1] >= _CONFIDENT]
    pool = sure if len(sure) >= 2 else list(reads)
    heights = sorted(box_height(r[2]) for r in pool)
    return heights[len(heights) // 2]


def _validate(reads: List[Tuple[str, float, BBox]]) -> List[Tuple[str, float, BBox]]:
    """
    Hold the page to one alphabet and check its identifiers run on.

    A plate is numbered or lettered, never both. The alphabet is decided
    by majority; reads in the other alphabet are corrected where a known
    confusion explains them and dropped otherwise. Holes in the run are
    logged, not filled.
    """
    if not reads:
        return []
    numeric = sum(r[0].isdigit() for r in reads) >= len(reads) / 2
    kept = []
    for text, confidence, box in reads:
        fixed = _to_alphabet(text, numeric)
        if fixed is None:
            logging.debug("Dropped %r: not in the page's alphabet", text)
            continue
        kept.append((fixed, confidence, box))
    _log_run(kept, numeric)
    return kept


def _to_alphabet(text: str, numeric: bool) -> Optional[str]:
    """Correct a read into the page's alphabet, or None if it cannot be."""
    table = _TO_DIGIT if numeric else _TO_LETTER
    want = str.isdigit if numeric else str.isalpha
    out = ''.join(table.get(ch, ch) for ch in text)
    if not numeric and len(out) > 1:
        return None                      # letter identifiers are single
    return out if out and want(out) else None


def _log_run(reads: Sequence[Tuple[str, float, BBox]], numeric: bool) -> None:
    """Log holes in the consecutive run of identifiers on a page."""
    if numeric:
        values = sorted({int(r[0]) for r in reads})
    else:
        values = sorted({ord(r[0].lower()) for r in reads})
    if not values:
        return
    holes = sorted(set(range(values[0], values[-1] + 1)) - set(values))
    if holes:
        shown = holes if numeric else [chr(h) for h in holes]
        logging.debug("Identifier run %s-%s has gaps at %s",
                      values[0] if numeric else chr(values[0]),
                      values[-1] if numeric else chr(values[-1]), shown)


### ASSIGNMENT ###

def _assign(
    reads: Sequence[Tuple[str, float, BBox]],
    boxes: Sequence[BBox],
    identifiers: List[Identifier],
    reach: float = _NEAR_BOX,
) -> None:
    """Give each crop its identifier, flagging crops that cannot take one."""
    for text, confidence, box in reads:
        home = _home_box(box, boxes, reach=reach)
        if home is not None:
            identifiers[home].candidates.append((text, confidence, box))

    for identifier in identifiers:
        labels = {c[0] for c in identifier.candidates}
        if len(labels) == 1:
            best = max(identifier.candidates, key=lambda c: c[1])
            identifier.label, identifier.confidence = best[0], best[1]
            identifier.source = 'read'
        elif len(labels) > 1:
            identifier.flag = 'several_identifiers'
        else:
            identifier.flag = 'no_identifier'

    seen: Dict[str, int] = {}
    for identifier in identifiers:
        if identifier.named:
            seen[identifier.label] = seen.get(identifier.label, 0) + 1
    for identifier in identifiers:
        if identifier.named and seen[identifier.label] > 1:
            identifier.flag = 'duplicate'
            identifier.source, identifier.label = 'index', ''
