"""
Drop an artefact box that is writing, not a drawing.

Two tests keep text off the artefact list. The shape test
(``text_rejection``, before grouping) catches a word: many letter-sized
pieces of ink on one baseline. A framed specimen code passes it, because
the frame is one piece that holds most of the ink, and a frame with marks
inside has the shape of a drawn surface. This second test runs after
grouping, on each artefact box, and asks the reader. RapidOCR reads the
box as a whole. The box is text when the lines it reads cover nearly all
of the box's ink, a straight frame on the border discounted, and spell at
least ``min_characters``. A lithic with a numeral inside reads the
numeral, but the numeral is a small share of the drawing's ink, so the
lithic stays. Without the engine nothing is dropped.
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..image_processing.resolution import WorkingCopy
from .detection import PageImage
from .geometry import BBox
from .identifiers import _load_engine

DEFAULT_MIN_CONFIDENCE = 0.8    # a line counts from this confidence: print reads near 1,
                                # a drawing misread as a glyph reads at 0.6 to 0.7
DEFAULT_MIN_INK_SHARE = 0.8     # the read lines must cover this share of the box's ink
DEFAULT_MIN_CHARACTERS = 3      # a shorter read is an identifier, not a text box
_FRAME_RUN = 0.85               # a row or column inked over this share of its length is frame
_READ_HEIGHT = 96               # the box is upscaled to at least this many pixels tall
_READ_PAD = 24                  # white margin around the upscaled box, in pixels

# One text line the engine read: its polygon in box pixels, its text, its confidence.
Line = Tuple[np.ndarray, str, float]


@dataclass(frozen=True)
class TextBox:
    """An artefact box that the reader found to be writing."""

    index: int          # position in the page's artefact boxes
    box: BBox
    text: str
    share: float        # share of the box's ink that the read lines cover


def find_text_boxes(
    page: PageImage,
    boxes: Sequence[BBox],
    raw: np.ndarray,
    config: Dict,
    working: Optional[WorkingCopy] = None,
) -> List[TextBox]:
    """
    Return the artefact boxes that are text, by reading each one.

    ``raw`` is the raw ink mask on the source grid. ``config`` is the
    ``text_boxes`` section. Its ``min_confidence`` is higher than the
    identifier reader's: a whole line of print reads near 1, while a
    small drawing misread as a glyph reads at 0.6 to 0.7, and that line
    would cover most of the drawing's ink. Without RapidOCR the list is
    empty and the absence is reported once by the reader.
    """
    if not config.get('enabled', True) or not boxes:
        return []
    engine = _load_engine()
    if engine is None:
        return []
    floor = float(config.get('min_confidence', DEFAULT_MIN_CONFIDENCE))
    min_share = float(config.get('min_ink_share', DEFAULT_MIN_INK_SHARE))
    min_characters = int(config.get('min_characters', DEFAULT_MIN_CHARACTERS))
    found = []
    for index, box in enumerate(boxes):
        lines = read_box(page, box, engine, floor, working)
        text = ' '.join(line[1] for line in lines)
        if len(re.sub(r'[^0-9A-Za-z]', '', text)) < min_characters:
            continue
        share = text_share(ink_without_frame(_box_ink(raw, box)), lines)
        if share >= min_share:
            found.append(TextBox(index, tuple(box), text, round(share, 3)))
            logging.info("Box %d is text, not an artefact: %r (%.0f%% of its ink)",
                         index + 1, text, share * 100)
    return found


### READING ###

def read_box(
    page: PageImage, box: BBox, engine, floor: float,
    working: Optional[WorkingCopy] = None,
) -> List[Line]:
    """Read one box with the engine; return its confident lines in box pixels."""
    image, scale, origin = box_image(page, box, working)
    result, _ = engine(image)
    lines = []
    for points, text, score in result or []:
        clean = re.sub(r'[^0-9A-Za-z]', '', str(text))
        if not clean or float(score) < floor:
            continue
        polygon = (np.array(points, dtype=np.float64) - _READ_PAD) / scale - origin
        lines.append((np.rint(polygon).astype(np.int32), str(text).strip(), float(score)))
    return lines


def box_image(
    page: PageImage, box: BBox, working: Optional[WorkingCopy] = None,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Cut, pad and upscale a box for the reader.

    Returns the RGB image, the scale from box pixels to image pixels
    before padding, and the origin of the cut in box pixels (negative
    when a margin was taken around the box).
    """
    factor = working.factor if working is not None else 1
    gray = working.gray if working is not None else page.gray
    x0, y0, x1, y1 = (int(v) * factor for v in box)
    margin = max(4, (y1 - y0) // 4)
    top, left = max(0, y0 - margin), max(0, x0 - margin)
    crop = gray[top:y1 + margin, left:x1 + margin]
    lo, hi = int(crop.min()), int(crop.max())
    if hi > lo:
        crop = ((crop.astype(np.float32) - lo) * (255.0 / (hi - lo))).astype(np.uint8)
    resize = max(1.0, _READ_HEIGHT / max(1, y1 - y0))
    crop = cv2.resize(crop, None, fx=resize, fy=resize, interpolation=cv2.INTER_LANCZOS4)
    padded = np.full((crop.shape[0] + 2 * _READ_PAD, crop.shape[1] + 2 * _READ_PAD),
                     255, np.uint8)
    padded[_READ_PAD:-_READ_PAD, _READ_PAD:-_READ_PAD] = crop
    scale = resize * factor
    origin = np.array([(x0 - left) / factor, (y0 - top) / factor], dtype=np.float64)
    return cv2.cvtColor(padded, cv2.COLOR_GRAY2RGB), scale, origin


### INK ###

def _box_ink(raw: np.ndarray, box: BBox) -> np.ndarray:
    """Return the raw ink inside the box as a boolean array."""
    x0, y0, x1, y1 = (int(v) for v in box)
    return raw[y0:y1, x0:x1] > 0


def ink_without_frame(ink: np.ndarray) -> np.ndarray:
    """
    Return the ink with a straight frame removed.

    A row or a column inked over ``_FRAME_RUN`` of its length is a line of
    a frame, whatever its thickness, and is cleared. Writing never fills
    a whole row; a solid block does, and loses its rows, but a solid
    block reads as nothing and is not text either way.
    """
    if ink.size == 0:
        return ink
    kept = ink.copy()
    rows = ink.mean(axis=1) >= _FRAME_RUN
    columns = ink.mean(axis=0) >= _FRAME_RUN
    kept[rows, :] = False
    kept[:, columns] = False
    return kept


def text_share(ink: np.ndarray, lines: Sequence[Line]) -> float:
    """Return the share of the ink that lies inside the read lines' polygons."""
    total = int(ink.sum())
    if total == 0 or not lines:
        return 0.0
    covered = np.zeros(ink.shape, dtype=np.uint8)
    for polygon, _, _ in lines:
        cv2.fillPoly(covered, [polygon.reshape(-1, 1, 2)], 1)
    return float(np.logical_and(ink, covered > 0).sum()) / total
