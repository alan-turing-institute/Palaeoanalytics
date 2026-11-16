"""
Width profile of a blob along its long axis, the measurement behind the arrow classifier.

An arrow has a thin, even shaft and a head that widens and then narrows to
a point. Those are properties of the ink width along the axis of the
blob, and they survive at small sizes, where the notches between head and
shaft are a pixel deep and useless. This module takes the ink of a
contour, renders it at a supersampled scale, finds its long axis, cuts
the length into bins and measures the ink in each bin. The features of
the profile are computed for one orientation (tail at the first bin,
head at the last) and again reversed, so a classifier can say which way
the arrow points.

The ink is the real ink of the binary image inside the contour when the
caller gives the image. A scar outline is then measured as the ring it
is, not as the filled shape its contour encloses, and the fill features
tell a drawn stroke from an empty outline. Without the image the filled
contour is measured, which is the same thing for a solid mark.

Every length is in source pixels; every other feature is a ratio.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

SUPERSAMPLE = 4          # the contour is rendered at this scale
BINS = 20                # bins along the axis
_HEAD_SHARE = 0.35       # the head is looked for in this last share of the length
_HEAD_RISE = 1.5         # the head starts where the span exceeds this times the shaft width
_SQUARE_VARIANCE = 2.0   # below this ratio of PCA variances both axes are tried
_SHAFT_SHARE = 0.3       # the shaft width is measured on this first share of the length
_MIN_PIXELS = 8          # fewer rendered pixels than this is no blob
_MIN_WIDTH = 0.5         # floor for a width used as a divisor, in source pixels

PROFILE_POINTS = 10      # the span profile is also given as this many values

# Every feature but ``length`` is a ratio, so the model does not learn the
# scale of the drawings it was trained on.
PROFILE_FEATURES: List[str] = [
    'length', 'head_ratio', 'head_pos', 'tip_taper', 'tail_even', 'tail_end',
    'head_fill', 'shaft_fill', 'head_len', 'asymmetry', 'aspect', 'solidity',
    'head_skew', 'tail_skew', *[f'span_{i}' for i in range(PROFILE_POINTS)],
    *[f'shape_{i}' for i in range(PROFILE_POINTS)],
]

# The measurements of the notch method (``arrow_detection``), given to the
# classifier as ratios so it knows what that method knows. The same in
# both orientations.
STRUCTURE_FEATURES: List[str] = [
    'notches', 'notch_depth_1', 'notch_depth_2', 'triangle_height',
    'shaft_solidity', 'head_solidity',
]

FEATURES: List[str] = PROFILE_FEATURES + STRUCTURE_FEATURES

Point = Tuple[float, float]


@dataclass(frozen=True)
class Profile:
    """
    The ink of a blob measured along its long axis.

    The lengths are in source pixels. ``ends`` are in the pixels of the
    image the blob was measured on, so a caller can draw them on it.
    """

    length: float
    span: np.ndarray
    fill: np.ndarray
    offset: np.ndarray       # per bin, how far the ink's centre sits off the axis
    area: float
    solidity: float
    ends: Tuple[Point, Point]


### MEASUREMENT ###

def measure_profile(
    contour: np.ndarray, ink: Optional[np.ndarray] = None, scale: float = 1.0,
) -> Optional[Profile]:
    """
    Measure the width profile of a contour, or return None for a degenerate one.

    With ``ink``, the binary image the contour came from (ink white), the
    ink pixels inside the contour are measured. Without it the filled
    contour is. Either is rendered at ``SUPERSAMPLE`` times its size, so a
    one-pixel stroke is measured with four pixels. ``scale`` is the number
    of pixels of ``ink`` per source pixel: the lengths come out in source
    pixels whatever the working resolution. ``ends`` are the extreme ink
    points along the axis, in the pixels of ``ink``: the first at the
    first bin.
    """
    points = contour.reshape(-1, 2).astype(np.float64) * SUPERSAMPLE
    if len(points) < 3:
        return None
    coords = _render(points) if ink is None else _render_ink(contour, ink)
    if coords is None:
        return None
    hull_area = cv2.contourArea(cv2.convexHull(points.astype(np.float32)))
    area = float(len(coords))
    solidity = area / hull_area if hull_area > 0 else 0.0
    unit = SUPERSAMPLE * scale
    return min(
        (_profile_along(coords, axis, area, solidity, unit) for axis in _axes(coords)),
        key=_skewness,
    )


def _profile_along(
    coords: np.ndarray, axis: np.ndarray, area: float, solidity: float, unit: float,
) -> Profile:
    """Measure the ink along one unit axis; ``unit`` is the rendered pixels per source pixel."""
    centre = coords.mean(axis=0)
    along = (coords - centre) @ axis
    across = (coords - centre) @ np.array([-axis[1], axis[0]])
    span, fill, offset, extent = _bin_profile(along, across)
    first, last = _axis_ends(coords, axis, along)
    return Profile(
        length=extent / unit,
        span=span / unit,
        fill=fill,
        offset=offset / unit,
        area=area / unit ** 2,
        solidity=solidity,
        ends=(_image_point(first), _image_point(last)),
    )


def _skewness(profile: Profile) -> float:
    """Return how far the ink sits off the axis on average, relative to its span."""
    present = profile.span > 0
    if not present.any():
        return 0.0
    return float((np.abs(profile.offset[present]) / profile.span[present]).mean())


def _render(points: np.ndarray) -> Optional[np.ndarray]:
    """Fill the polygon on a canvas; return the ink coordinates (x, y), or None."""
    origin = np.floor(points.min(axis=0)).astype(int) - 2
    size = np.ceil(points.max(axis=0)).astype(int) - origin + 3
    canvas = np.zeros((int(size[1]), int(size[0])), dtype=np.uint8)
    cv2.fillPoly(canvas, [np.round(points - origin).astype(np.int32)], 255)
    ys, xs = np.nonzero(canvas)
    if len(xs) < _MIN_PIXELS:
        return None
    return np.column_stack((xs + origin[0], ys + origin[1])).astype(np.float64)


def _render_ink(contour: np.ndarray, ink: np.ndarray) -> Optional[np.ndarray]:
    """
    Return the supersampled coordinates of the ink pixels inside the contour, or None.

    Each source pixel becomes a ``SUPERSAMPLE`` square block, so the
    measurement has the same scale as the polygon render.
    """
    pixels, (left, top) = ink_in_contour(contour, ink)
    if int(pixels.sum()) * SUPERSAMPLE ** 2 < _MIN_PIXELS:
        return None
    block = np.ones((SUPERSAMPLE, SUPERSAMPLE), dtype=bool)
    ys, xs = np.nonzero(np.kron(pixels, block))
    return np.column_stack(
        (xs + left * SUPERSAMPLE, ys + top * SUPERSAMPLE)
    ).astype(np.float64)


def ink_in_contour(contour: np.ndarray, ink: np.ndarray) -> Tuple[np.ndarray, Point]:
    """
    Return the ink pixels inside the filled contour as a boolean crop, and the crop's origin.

    The crop is the contour's box with a one-pixel border, clipped to the
    image.
    """
    x, y, w, h = cv2.boundingRect(contour)
    left, top = max(x - 1, 0), max(y - 1, 0)
    right, bottom = min(x + w + 1, ink.shape[1]), min(y + h + 1, ink.shape[0])
    mask = np.zeros((bottom - top, right - left), dtype=np.uint8)
    cv2.drawContours(mask, [contour - np.array([left, top])], 0, 255, cv2.FILLED)
    pixels = (mask > 0) & (ink[top:bottom, left:right] > 0)
    return pixels, (left, top)


def ink_inside(contour: np.ndarray, ink: np.ndarray) -> bool:
    """
    Say whether the contour holds ink that is not joined to its own boundary.

    A scar outline holds the arrow drawn in it; a hollow arrow holds
    nothing. The ink inside the filled contour is split into connected
    pieces, and a piece that touches no pixel of the contour's own rim
    is other ink.
    """
    pixels, _ = ink_in_contour(contour, ink)
    mask = np.zeros(pixels.shape, dtype=np.uint8)
    x, y, _, _ = cv2.boundingRect(contour)
    cv2.drawContours(mask, [contour - np.array([max(x - 1, 0), max(y - 1, 0)])], 0, 255,
                     cv2.FILLED)
    rim = (mask > 0) & (cv2.erode(mask, np.ones((3, 3), dtype=np.uint8)) == 0)
    count, labels = cv2.connectedComponents(pixels.astype(np.uint8), connectivity=8)
    for piece in range(1, count):
        if not rim[labels == piece].any():
            return True
    return False


def _axes(coords: np.ndarray) -> List[np.ndarray]:
    """
    Return the axes to measure along: the PCA long axis, and the other one when squarish.

    An arrow whose head is wider than the arrow is long has its long axis
    across the head; the profile along the shaft is the symmetric one, and
    the caller keeps the less skewed of the two.
    """
    _, vectors, values = cv2.PCACompute2(coords.astype(np.float32), mean=None)
    axes = [vectors[0].astype(np.float64)]
    if values[0, 0] < _SQUARE_VARIANCE * max(float(values[1, 0]), 1e-6):
        axes.append(vectors[1].astype(np.float64))
    return axes


def _axis_ends(
    coords: np.ndarray, axis: np.ndarray, along: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the two ends of the blob on the axis, through the extreme ink pixels."""
    centre = coords.mean(axis=0)
    first = centre + axis * along.min()
    last = centre + axis * along.max()
    return first, last


def _bin_profile(
    along: np.ndarray, across: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Per-bin extent, fill and centre offset across the axis; also the total extent."""
    low, high = along.min(), along.max()
    extent = high - low + 1.0
    index = np.minimum((((along - low) / extent) * BINS).astype(int), BINS - 1)
    counts = np.bincount(index, minlength=BINS).astype(np.float64)
    lowest = np.full(BINS, np.inf)
    highest = np.full(BINS, -np.inf)
    np.minimum.at(lowest, index, across)
    np.maximum.at(highest, index, across)
    span = np.where(counts > 0, highest - lowest + 1.0, 0.0)
    sums = np.bincount(index, weights=across, minlength=BINS)
    offset = np.where(counts > 0, sums / np.maximum(counts, 1.0), 0.0)
    bin_length = extent / BINS
    fill = np.where(counts > 0, counts / (bin_length * np.maximum(span, 1.0)), 0.0)
    return span, np.minimum(fill, 1.0), offset, extent


def _image_point(coord: np.ndarray) -> Point:
    """Return a supersampled coordinate in the pixels of the measured image."""
    return (float(coord[0]) / SUPERSAMPLE, float(coord[1]) / SUPERSAMPLE)


### FEATURES ###

def profile_features(profile: Profile, reverse: bool = False) -> Dict[str, float]:
    """
    Features of the profile with the tail at the first bin and the head at the last.

    With ``reverse`` the profile is read from the other end, so the two
    calls describe the two possible directions of the same blob.
    """
    span = profile.span[::-1] if reverse else profile.span
    fill = profile.fill[::-1] if reverse else profile.fill
    offset = profile.offset[::-1] if reverse else profile.offset
    half = BINS // 2
    shaft = span[:half]
    tail = span[:int(BINS * _SHAFT_SHARE)]
    shaft_width = float(np.median(tail[tail > 0])) if (tail > 0).any() else 0.0
    width_floor = max(shaft_width, _MIN_WIDTH)
    head_start = int(BINS * (1.0 - _HEAD_SHARE))
    head_width = float(span[head_start:].max())
    features = {
        'length': profile.length,
        'head_ratio': head_width / width_floor,
        'head_pos': float(np.argmax(span)) / (BINS - 1),
        'tip_taper': float(span[-1]) / max(head_width, _MIN_WIDTH),
        'tail_even': _variation(shaft),
        'tail_end': float(span[0]) / width_floor,
        'head_fill': _mean_where(fill[head_start:], span[head_start:] > 0),
        'shaft_fill': _mean_where(fill[:half], shaft > 0),
        'head_len': _head_length(span, shaft_width),
        'asymmetry': float(span[half:].sum()) / max(float(shaft.sum()), 1e-6),
        'aspect': profile.length / width_floor,
        'solidity': profile.solidity,
        'head_skew': _skew(offset[head_start:], offset[:half], span[head_start:], width_floor),
        'tail_skew': _skew(offset[:half], offset[half:], span[:half], width_floor),
    }
    points = span.reshape(PROFILE_POINTS, BINS // PROFILE_POINTS).mean(axis=1)
    features.update({f'span_{i}': float(v) / width_floor for i, v in enumerate(points)})
    widest = max(float(points.max()), _MIN_WIDTH)
    features.update({f'shape_{i}': float(v) / widest for i, v in enumerate(points)})
    return features


def _skew(part: np.ndarray, rest: np.ndarray, span: np.ndarray, width: float) -> float:
    """
    Return how far the ink of one part sits off the line of the rest, in shaft widths.

    An arrowhead is centred on its shaft; the flag of a ``1`` or the bar of
    a ``7`` sits to one side of the stem.
    """
    present = span > 0
    if not present.any():
        return 0.0
    baseline = float(np.median(rest[rest != 0])) if (rest != 0).any() else 0.0
    return float(np.abs(part[present] - baseline).mean()) / width


def _variation(values: np.ndarray) -> float:
    """Return the standard deviation over the mean of the non-empty values, 0 when none."""
    present = values[values > 0]
    if len(present) < 2 or present.mean() <= 0:
        return 0.0
    return float(present.std() / present.mean())


def _mean_where(values: np.ndarray, mask: np.ndarray) -> float:
    """Mean of the selected values, 0 when none is selected."""
    return float(values[mask].mean()) if mask.any() else 0.0


def _head_length(span: np.ndarray, shaft_width: float) -> float:
    """Share of the second half of the length where the span exceeds the shaft's."""
    limit = _HEAD_RISE * max(shaft_width, _MIN_WIDTH)
    head = span[BINS // 2:]
    return float((head > limit).sum()) / len(head)


### GEOMETRY ###

def arrow_geometry(profile: Profile, reverse: bool = False) -> Dict[str, object]:
    """
    Return the arrow's back, tip and angles for the given orientation.

    The frame is the one of the CSV: ``angle_deg`` is the image angle from
    the back to the tip, ``compass_angle`` has an arrow pointing down at 0
    and one pointing right at 270.
    """
    back, tip = (profile.ends[1], profile.ends[0]) if reverse else profile.ends
    angle_rad = math.atan2(tip[1] - back[1], tip[0] - back[0])
    angle_deg = (math.degrees(angle_rad) + 360.0) % 360.0
    return {
        'arrow_back': (int(round(back[0])), int(round(back[1]))),
        'arrow_tip': (int(round(tip[0])), int(round(tip[1]))),
        'angle_rad': round(angle_rad, 1),
        'angle_deg': round(angle_deg, 1),
        'compass_angle': round((270.0 + angle_deg) % 360.0, 1),
    }


### INTERIM RULE ###

_RULE_HEAD_RATIO = 1.6
_RULE_ASPECT = 2.0
_RULE_TIP_TAPER = 0.7
_RULE_HEAD_POS = 0.5
_RULE_ASYMMETRY = 1.2
_HEAD_LEN_FLOOR = 0.05


def rule_says_arrow(features: Dict[str, float]) -> bool:
    """Decide by a hand rule on the features, where no trained model is available."""
    return (
        features['head_ratio'] >= _RULE_HEAD_RATIO
        and features['aspect'] >= _RULE_ASPECT
        and features['tip_taper'] <= _RULE_TIP_TAPER
        and features['head_pos'] >= _RULE_HEAD_POS
        and features['asymmetry'] >= _RULE_ASYMMETRY
    )


def rule_direction_score(features: Dict[str, float]) -> float:
    """
    Score how much the features look like a head at the last bin; higher wins.

    A head is wide, tapers to a point and spans several bins, with its
    widest part some way back from the end. A dot at the tail is wide
    too, but round and short, with its widest part at the very end. The
    head length and the position of the maximum tell them apart.
    """
    return (
        features['head_ratio'] * features['asymmetry']
        * (1.0 - features['tip_taper']) * (features['head_len'] + _HEAD_LEN_FLOOR)
        * (1.0 - features['head_pos'] + _HEAD_LEN_FLOOR)
    )
