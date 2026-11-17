"""
Hide the page's text in an artefact crop.

A crop is the artefact's box with a white margin, cut from the page.
Whatever ink lies in that rectangle comes along: a caption line, a legend
key, the end of a scale bar, a neighbour's identifier. The analysis takes
each of them for a surface or a scar. The page step classified every
piece of ink before it grouped the page, so at export it knows which
pieces are text. This module paints them over with the paper value of
the crop, by each piece's own pixel mask, so a glyph that overlaps the
drawing cannot take lithic ink with it. Nothing inside the artefact's
own box is touched, and a flaking arrow never is. The page is not
changed: the text is hidden in the crop, not erased from the page.

The crop's own identifier stays, unless asked, and its box goes to the
metadata so that the analysis blanks it before it looks for contours.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .detection import Component
from .geometry import BBox, horizontal_overlap, vertical_overlap
from .grouping import Classified
from .identifiers import Identifier, PageReads

LIGHT = 128             # a pixel at or above this grey is paper, not ink
_OWN_GLYPH_SHARE = 0.5  # a component overlapping an identifier box by this much is it

Hider = Callable[[np.ndarray, Tuple[int, int], BBox, Optional[Identifier]],
                 Tuple[np.ndarray, 'Hidden']]


@dataclass
class Hidden:
    """What was hidden in one crop, and what the analysis must ignore."""

    counts: Dict[str, int] = field(default_factory=dict)
    ignore_boxes: List[BBox] = field(default_factory=list)   # crop pixels

    @property
    def text(self) -> str:
        """The manifest text: ``caption 2; scale bar 1``."""
        return '; '.join(f"{kind} {count}" for kind, count in sorted(self.counts.items()))

    @property
    def ignore_text(self) -> str:
        """The metadata text: boxes as ``x0 y0 x1 y1``, separated by ``;``."""
        return ';'.join(' '.join(str(v) for v in box) for box in self.ignore_boxes)


def make_hider(
    components: Sequence[Component],
    classified: Classified,
    reads: PageReads,
    settings: Dict,
) -> Optional[Hider]:
    """
    Return the function that hides the page's text in a crop, or None when hiding is off.

    ``settings`` is the ``export`` section: ``hide_text`` (default on) and
    ``hide_identifiers`` (default off).
    """
    if not settings.get('hide_text', True):
        return None
    hide_identifiers = bool(settings.get('hide_identifiers', False))
    kinds = _hide_kinds(classified, {tuple(b) for b in reads.seen})

    def hide(crop, origin, artefact_box, identifier):
        return hide_in_crop(crop, origin, artefact_box, components, kinds,
                            identifier, hide_identifiers)
    return hide


def _hide_kinds(classified: Classified, seen: Set[Tuple]) -> List[Tuple[str, BBox]]:
    """
    Return the page boxes to hide, each with its kind.

    Captions, legend words and keys, scale bars and their captions are
    always text. Of the labels, only the glyphs the reader read anything
    from: a label the reader never saw is a flaking arrow, and an arrow is
    never hidden.
    """
    kinds = [('caption', tuple(b)) for b in list(classified.text) + list(classified.bar_captions)]
    kinds += [('scale bar', tuple(b)) for b in classified.bars]
    kinds += [('label', tuple(b)) for b in classified.labels if tuple(b) in seen]
    return kinds


def hide_in_crop(
    crop: np.ndarray,
    origin: Tuple[int, int],
    artefact_box: BBox,
    components: Sequence[Component],
    kinds: Sequence[Tuple[str, BBox]],
    identifier: Optional[Identifier],
    hide_identifiers: bool = False,
) -> Tuple[np.ndarray, Hidden]:
    """
    Return a copy of the crop with the page's text painted over, and what was hidden.

    ``origin`` is the crop's top left corner on the page and
    ``artefact_box`` the artefact's own box before the margin; nothing
    inside it is touched.
    """
    result = crop.copy()
    paper = paper_value(crop)
    own_glyphs = [tuple(c[2]) for c in (identifier.candidates if identifier else [])]
    hidden = Hidden()
    left, top = origin
    height, width = crop.shape[:2]
    crop_box = (left, top, left + width, top + height)
    for component in components:
        box = tuple(component.box)
        if component.mask is None or not _touches(box, crop_box):
            continue
        if _is_one_of(box, own_glyphs):
            # The crop's own identifier. The grouping attached it to the
            # artefact's box, so it is looked for before the box rule.
            if not hide_identifiers:
                hidden.ignore_boxes.append(_clip(box, crop_box, origin))
                continue
            kind = 'identifier'
        elif _inside(box, artefact_box):
            continue
        else:
            kind = _kind_of(box, kinds)
            if kind is None:
                continue
        _paint(result, component, origin, paper)
        hidden.counts[kind] = hidden.counts.get(kind, 0) + 1
    return result, hidden


def paper_value(crop: np.ndarray):
    """
    Return the most frequent light value of the crop: its paper.

    Measured for each crop, because a scan has grey paper and a page
    rendered from a PDF has white. For a colour crop the value is found
    for each channel.
    """
    if crop.ndim == 2:
        return _mode_light(crop)
    grey = crop[:, :, :3].mean(axis=2)
    light = grey >= LIGHT
    channels = [_mode_light(crop[:, :, i][light]) if light.any() else 255
                for i in range(crop.shape[2])]
    return np.array(channels, dtype=crop.dtype)


def _mode_light(values: np.ndarray) -> int:
    """Return the most frequent value at or above ``LIGHT``, or 255 when there is none."""
    light = values[values >= LIGHT] if values.ndim else values
    if light.size == 0:
        return 255
    return int(np.bincount(light.astype(np.int64).ravel(), minlength=256).argmax())


### HELPERS ###

def _paint(crop: np.ndarray, component: Component, origin: Tuple[int, int], paper) -> None:
    """Set the component's ink pixels that fall inside the crop to the paper value."""
    x0, y0, x1, y1 = component.box
    left, top = origin
    height, width = crop.shape[:2]
    cx0, cy0 = max(x0 - left, 0), max(y0 - top, 0)
    cx1, cy1 = min(x1 - left, width), min(y1 - top, height)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    mask = component.mask[cy0 + top - y0:cy1 + top - y0, cx0 + left - x0:cx1 + left - x0]
    region = crop[cy0:cy1, cx0:cx1]
    region[mask.astype(bool)] = paper


def _kind_of(box: BBox, kinds: Sequence[Tuple[str, BBox]]) -> Optional[str]:
    """Return the kind of the first hide box that holds the component's box, or None."""
    for kind, hide_box in kinds:
        if _inside(box, hide_box):
            return kind
    return None


def _is_one_of(box: BBox, glyphs: Sequence[BBox]) -> bool:
    """Say whether the box overlaps one of the identifier's glyph boxes by half or more."""
    area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
    for glyph in glyphs:
        shared = max(0, horizontal_overlap(box, glyph)) * max(0, vertical_overlap(box, glyph))
        if shared / area >= _OWN_GLYPH_SHARE:
            return True
    return False


def _inside(inner: BBox, outer: BBox) -> bool:
    """Report whether one box lies wholly within another."""
    return (inner[0] >= outer[0] and inner[1] >= outer[1]
            and inner[2] <= outer[2] and inner[3] <= outer[3])


def _touches(a: BBox, b: BBox) -> bool:
    """Report whether two boxes share any area."""
    return horizontal_overlap(a, b) > 0 and vertical_overlap(a, b) > 0


def _clip(box: BBox, crop_box: BBox, origin: Tuple[int, int]) -> BBox:
    """Return a page box clipped to the crop, in the crop's pixels."""
    left, top = origin
    return (max(box[0], crop_box[0]) - left, max(box[1], crop_box[1]) - top,
            min(box[2], crop_box[2]) - left, min(box[3], crop_box[3]) - top)
