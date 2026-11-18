"""
Remove a thin frame drawn around a letter, before the scars are traced.

Some illustrators draw a thin rectangle around a label such as the C that
marks cortex. The rectangle is a closed outline, so contour finding takes
it for a scar: the letter then marks the frame as cortex, and the frame's
lines cut the real scar it sits in. This module finds such frames on the
binary image and sets their thin pixels to paper. A drawing line that the
frame touches is thicker than the frame and keeps its pixels.

A frame is a hole in the ink that

1. is a rectangle: four corners, sides within ``_AXIS_DEGREES`` of the
   image axes, and at least ``_RECT_FILL`` of its bounding box;
2. is drawn with a thin line: at most ``max_line_ratio`` of the drawing's
   line width;
3. holds one free mark, no larger than ``_MARK_SHARE`` of the frame in
   each direction: a letter.

An empty rectangle, or a rectangle drawn with the drawing's own line
width, is not touched.
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

DEFAULT_FRAMES = {'enabled': True, 'max_line_ratio': 0.5}
_AXIS_DEGREES = 5.0     # a side of a frame is this close to horizontal or vertical
_RECT_FILL = 0.85       # the hole fills this share of its bounding box
_MARK_SHARE = 0.8       # the letter is at most this share of the frame, each way
_MIN_SIDE = 10          # px; a smaller hole is a gap in the ink, not a frame
_INSET = 2              # px inside the frame where the marks are looked for
_RUN_LIMIT = 40         # px; the longest run of ink measured across a frame side

BBox = Tuple[int, int, int, int]


def remove_letter_frames(
    binary: np.ndarray, working: Optional[np.ndarray], factor: int, settings: Dict,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[BBox]]:
    """
    Return the binary image and the working copy with letter frames erased.

    ``binary`` is the image on the source grid, ink white; ``working`` the
    same at ``factor`` times the size, or None. ``settings`` is the
    ``cortex_detection.frames`` section. The third value lists the frames
    taken out, as ``(x, y, w, h)`` on the source grid.
    """
    settings = {**DEFAULT_FRAMES, **(settings or {})}
    if not settings.get('enabled', True):
        return binary, working, []
    stroke = _drawing_stroke(binary)
    erase = np.zeros(binary.shape, bool)
    frames = []
    for box, line in _frames(binary, stroke, float(settings['max_line_ratio'])):
        erase |= _thin_pixels(binary, box, line)
        frames.append(box)
    if not frames:
        return binary, working, []
    logging.debug("Letter frames taken out: %s", frames)
    cleaned = binary.copy()
    cleaned[erase] = 0
    return cleaned, _erase_working(working, erase, factor), frames


def _frames(binary: np.ndarray, stroke: float, max_line_ratio: float):
    """Yield each letter frame as its box and its line width."""
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return
    for contour, (_, _, _, parent) in zip(contours, hierarchy[0]):
        if parent < 0 or not _is_rectangle(contour):
            continue
        box = cv2.boundingRect(contour)
        line = _line_width(binary, box)
        if line <= max_line_ratio * stroke and _holds_one_letter(binary, box):
            yield box, line


def _is_rectangle(hole: np.ndarray) -> bool:
    """Say whether a hole is a rectangle with sides square to the image."""
    x, y, w, h = cv2.boundingRect(hole)
    if w < _MIN_SIDE or h < _MIN_SIDE or cv2.contourArea(hole) < _RECT_FILL * w * h:
        return False
    corners = cv2.approxPolyDP(hole, 0.02 * cv2.arcLength(hole, True), True).reshape(-1, 2)
    if len(corners) != 4:
        return False
    for start, end in zip(corners, np.roll(corners, -1, axis=0)):
        dx, dy = (end - start).astype(float)
        angle = abs(np.degrees(np.arctan2(dy, dx))) % 90
        if min(angle, 90 - angle) > _AXIS_DEGREES:
            return False
    return True


def _line_width(binary: np.ndarray, box: BBox) -> float:
    """
    Return the frame's line width: the 75th percentile of ink runs across its sides.

    Runs are measured outwards from the hole at seven points on each side.
    A drawing line that crosses the frame gives a long run; the percentile
    keeps the frame's own width.
    """
    x, y, w, h = box
    ink = binary > 0
    runs = []
    for t in np.linspace(0.2, 0.8, 7):
        across, down = int(x + t * w), int(y + t * h)
        for px, py, dx, dy in ((x - 1, down, -1, 0), (x + w, down, 1, 0),
                               (across, y - 1, 0, -1), (across, y + h, 0, 1)):
            run = 0
            while (0 <= px < ink.shape[1] and 0 <= py < ink.shape[0]
                   and ink[py, px] and run < _RUN_LIMIT):
                run, px, py = run + 1, px + dx, py + dy
            runs.append(run)
    return float(np.percentile(runs, 75))


def _holds_one_letter(binary: np.ndarray, box: BBox) -> bool:
    """Say whether the frame holds exactly one free mark of letter size."""
    x, y, w, h = box
    inside = binary[y + _INSET:y + h - _INSET, x + _INSET:x + w - _INSET]
    if inside.size == 0:
        return False
    count, _, stats, _ = cv2.connectedComponentsWithStats(inside)
    height, width = inside.shape
    free = [s for s in stats[1:] if s[0] > 0 and s[1] > 0
            and s[0] + s[2] < width and s[1] + s[3] < height]
    return len(free) == 1 and free[0][2] <= _MARK_SHARE * w and free[0][3] <= _MARK_SHARE * h


def _thin_pixels(binary: np.ndarray, box: BBox, line: float) -> np.ndarray:
    """Return the frame's own pixels: ink in a band around the hole that an opening removes."""
    x, y, w, h = box
    size = 2 * max(int(round(line)), 1) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    thick = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel) > 0
    reach = int(round(line)) + 3
    band = np.zeros(binary.shape, np.uint8)
    cv2.rectangle(band, (x - reach, y - reach), (x + w + reach, y + h + reach), 255, -1)
    cv2.rectangle(band, (x + 1, y + 1), (x + w - 2, y + h - 2), 0, -1)
    return (band > 0) & (binary > 0) & ~thick


def _drawing_stroke(binary: np.ndarray) -> float:
    """Return the drawing's line width: twice the 90th percentile of the ink's depth."""
    ink = (binary > 0).astype(np.uint8)
    if not ink.any():
        return 0.0
    depth = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    return 2.0 * float(np.percentile(depth[depth > 0], 90))


def _erase_working(working: Optional[np.ndarray], erase: np.ndarray,
                   factor: int) -> Optional[np.ndarray]:
    """Erase the same frame pixels on the working copy, scaled to its grid."""
    if working is None:
        return None
    if factor <= 1 and working.shape == erase.shape:
        cleaned = working.copy()
        cleaned[erase] = 0
        return cleaned
    scaled = cv2.resize(erase.astype(np.uint8), (working.shape[1], working.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
    scaled = cv2.dilate(scaled, np.ones((factor, factor), np.uint8)) > 0
    cleaned = working.copy()
    cleaned[scaled & (working > 0)] = 0
    return cleaned
