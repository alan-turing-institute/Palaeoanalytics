"""
Score candidate scale bars with a small trained model.

A candidate is a group of ink components shaped like a bar, found by
``grouping``. This module measures each candidate, scores it with a
random forest trained on the project's own plates, and loads that
forest from the JSON file bundled with the package. Scoring uses numpy
only; training lives in ``tools/train_scale_classifier.py``.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from .detection import Component
from .geometry import BBox, box_height, box_width

FEATURES = [
    'long_frac', 'short_frac', 'aspect', 'density', 'runs', 'run_regularity',
    'ticks', 'end_ticks', 'spine', 'tone_variance', 'caption_marks', 'y_frac',
    'pieces', 'ink_pieces', 'plain_line', 'plain_no_caption', 'flanked',
    'solid', 'straight', 'taper', 'enclosed',
]

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
DEFAULT_MODEL = 'scale_bar_classifier.json'

# A caption sits within this many bar heights of the bar, above or
# below, and within this many past either end; never less than this
# fraction of the smaller page dimension, since a bar may be a hairline.
_CAPTION_REACH = 3.0
_CAPTION_REACH_FLOOR = 0.03
# A caption glyph is at most this fraction of the page tall and wide.
_CAPTION_MAX_HEIGHT = 0.06
_CAPTION_MAX_WIDTH = 0.1
# A gap along a row shorter than this fraction of the length is not a
# break between runs: a scanned edge is ragged, not segmented.
_RUN_GAP = 0.03
# More tick clusters than this is text, not a ruled scale; the count is
# capped so that a caption line does not swamp the feature.
_MAX_TICKS = 12
# A tick is a column whose ink stands this many times taller than the
# typical inked column of the bar.
_TICK_RISE = 1.5
# A tick cluster within this fraction of the length from either end is
# an end tick: a ruled scale has one at each end, an arrow has its
# head at one.
_END_ZONE = 0.15
# A drawing flanks a mark when it lies within this many mark lengths
# of either end, level with it.
_FLANK_REACH = 1.0
# A column is solid when ink fills this much of its span; a bar is solid
# or made of solid blocks, an outline is two edges with paper between.
_SOLID_FILL = 0.9
# A column keeps the bar's width when its span is within this fraction
# of the median span; an outline tapers and wavers.
_WIDTH_TOLERANCE = 0.25
# The ends of a mark are this fraction of its length at each end.
_END_SHARE = 0.1


@dataclass
class Candidate:
    """
    A group of components that may be one scale bar.

    ``enclosed`` marks a candidate that a drawing's box spans, admitted
    because its ink stands clear: a bar set under a lithic's profile, or
    a tick at a lithic's edge. ``crop`` is the box to export when it is
    wider than ``box``: a bar fused with its caption by closing is
    scored on the bar alone and exported with the caption.
    """

    indices: List[int]
    box: BBox
    upright: bool = False
    by_rules: bool = False
    enclosed: bool = False
    crop: Optional[BBox] = None
    rejected_by: str = ''
    score: float = 0.0
    accepted: bool = False
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class Tree:
    """One decision tree as parallel node arrays; a leaf has ``left == -1``."""

    left: np.ndarray
    right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    value: np.ndarray


@dataclass
class Model:
    """A random forest over the named features: the mean of its trees' leaf values."""

    features: List[str]
    trees: List[Tree]


class ModelUnavailable(RuntimeError):
    """The model file is missing or unreadable."""


def load_model(path: Optional[str] = None) -> Model:
    """
    Read a model from JSON, by default the file bundled with the package.

    Raises
    ------
    ModelUnavailable
        When the file is missing or does not hold a complete model.
    """
    path = path or os.path.join(MODELS_DIR, DEFAULT_MODEL)
    if not os.path.isabs(path) and not os.path.isfile(path):
        path = os.path.join(MODELS_DIR, path)
    try:
        with open(path, encoding='utf-8') as handle:
            data = json.load(handle)
        model = Model(
            features=list(data['features']),
            trees=[_tree_from(t) for t in data['trees']],
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ModelUnavailable(f"Scale bar model not read from {path}: {exc}") from exc
    if not model.trees or any(
        (t.feature >= len(model.features)).any() for t in model.trees
    ):
        raise ModelUnavailable(f"Scale bar model at {path} is not consistent")
    return model


def _tree_from(data: Dict) -> Tree:
    """Build a tree from its JSON node arrays, checking they agree in length."""
    arrays = {k: np.asarray(data[k]) for k in ('left', 'right', 'feature', 'threshold', 'value')}
    if len({len(a) for a in arrays.values()}) != 1:
        raise ValueError('tree arrays differ in length')
    return Tree(
        left=arrays['left'].astype(int), right=arrays['right'].astype(int),
        feature=arrays['feature'].astype(int), threshold=arrays['threshold'].astype(float),
        value=arrays['value'].astype(float),
    )


def score(features: Dict[str, float], model: Model) -> float:
    """Return the probability that a candidate with these features is a scale bar."""
    values = np.array([features.get(name, 0.0) for name in model.features], dtype=float)
    return float(np.mean([_leaf_value(tree, values) for tree in model.trees]))


def _leaf_value(tree: Tree, values: np.ndarray) -> float:
    """Follow one tree down to its leaf for these feature values."""
    node = 0
    while tree.left[node] != -1:
        if values[tree.feature[node]] <= tree.threshold[node]:
            node = tree.left[node]
        else:
            node = tree.right[node]
    return float(tree.value[node])


### FEATURES ###

def candidate_features(
    candidate: Candidate,
    components: Sequence[Component],
    raw: np.ndarray,
    gray: np.ndarray,
    width: int,
    height: int,
) -> Dict[str, float]:
    """
    Measure one candidate on the raw ink mask and the grey page.

    Every measurement is made along the candidate's long axis, so an
    upright bar is measured as a horizontal one.
    """
    x0, y0, x1, y1 = candidate.box
    ink = raw[y0:y1, x0:x1] > 0
    tone = gray[y0:y1, x0:x1]
    if candidate.upright:
        ink, tone = ink.T, tone.T
    long_px, short_px = ink.shape[1], ink.shape[0]
    long_page = height if candidate.upright else width
    runs, regularity = _runs_on_best_row(ink)
    ticks, end_ticks = _count_ticks(ink)
    captions = _caption_marks(candidate, components, width, height)
    solid, straight, taper = _shape_along(ink)
    plain = float(runs <= 1 and ticks == 0)
    return {
        'long_frac': long_px / float(long_page),
        'short_frac': short_px / float(min(width, height)),
        'aspect': long_px / float(max(1, short_px)),
        'density': float(ink.mean()) if ink.size else 0.0,
        'runs': float(runs),
        'run_regularity': regularity,
        'ticks': float(ticks),
        'end_ticks': float(end_ticks),
        'spine': _longest_run(ink) / float(max(1, long_px)),
        'tone_variance': float(tone[ink].std()) / 255.0 if ink.any() else 0.0,
        'caption_marks': float(captions),
        'y_frac': ((y0 + y1) / 2.0) / float(height),
        'pieces': float(len(candidate.indices)),
        'ink_pieces': float(np.log1p(_ink_pieces(ink))),
        'plain_line': plain,
        'plain_no_caption': float(plain and captions == 0),
        'flanked': float(_flanked(candidate, components, width, height)),
        'solid': solid,
        'straight': straight,
        'taper': taper,
        'enclosed': float(candidate.enclosed),
    }


def _shape_along(ink: np.ndarray) -> "tuple":
    """
    Describe the mark's cross-section along its length.

    Returns the share of inked columns that are solid from edge to edge,
    the share whose span keeps the median width, and how wide the two
    ends are against the middle. A bar is solid (or solid blocks),
    constant in width and square-ended; the outline of a lithic surface
    is hollow, wavering and tapered.
    """
    spans, fills = [], []
    for column in range(ink.shape[1]):
        inked = np.where(ink[:, column])[0]
        if len(inked):
            span = inked[-1] - inked[0] + 1
            spans.append(span)
            fills.append(len(inked) / float(span))
    if len(spans) < 3:
        return 0.0, 0.0, 0.0
    spans, fills = np.array(spans, dtype=float), np.array(fills)
    median = max(1.0, float(np.median(spans)))
    straight = float((np.abs(spans - median) <= max(1.0, _WIDTH_TOLERANCE * median)).mean())
    end = max(1, int(_END_SHARE * len(spans)))
    taper = float((spans[:end].mean() + spans[-end:].mean()) / (2.0 * median))
    return float((fills >= _SOLID_FILL).mean()), straight, taper


def _runs_on_best_row(ink: np.ndarray) -> "tuple":
    """Return the most ink runs on one row, and how regular their lengths are."""
    best_count, best_lengths = 0, []
    for row in ink:
        lengths = _run_lengths(row)
        if len(lengths) > best_count:
            best_count, best_lengths = len(lengths), lengths
    if best_count < 2:
        return best_count, 0.0
    lengths = np.array(best_lengths, dtype=float)
    return best_count, float(lengths.std() / max(1.0, lengths.mean()))


def _run_lengths(row: np.ndarray) -> List[int]:
    """Lengths of the ink runs along one row, ignoring gaps too short to be a break."""
    tolerance = max(1, int(_RUN_GAP * len(row)))
    lengths, run, gap = [], 0, 0
    for value in row:
        if value:
            if gap and gap <= tolerance and run:
                run += gap
            elif run and gap:
                lengths.append(run)
                run = 0
            run += 1
            gap = 0
        else:
            gap += 1
    if run:
        lengths.append(run)
    return lengths


def _longest_run(ink: np.ndarray) -> int:
    """Length of the longest unbroken ink run on any row."""
    best = 0
    for row in ink:
        lengths = _run_lengths(row)
        if lengths:
            best = max(best, max(lengths))
    return best


def _count_ticks(ink: np.ndarray) -> "tuple":
    """
    Count the clusters of columns whose ink stands well above the bar's band.

    A ruled scale carries short strokes across its spine; a block bar
    has none. Columns are measured by the span of their ink. Returns
    the count and how many of the clusters sit at the two ends, where
    a ruled scale has one each and an arrow has its head at one.
    """
    if not ink.any():
        return 0, 0
    columns = np.where(ink.any(axis=0))[0]
    spans = np.zeros(ink.shape[1], dtype=float)
    for column in columns:
        inked = np.where(ink[:, column])[0]
        spans[column] = inked[-1] - inked[0] + 1
    typical = float(np.median(spans[spans > 0]))
    tall = spans > _TICK_RISE * typical
    clusters, at_ends, inside = 0, set(), False
    zone = _END_ZONE * ink.shape[1]
    for column, value in enumerate(tall):
        if value and not inside:
            clusters += 1
            if column < zone:
                at_ends.add('start')
            elif column > ink.shape[1] - zone:
                at_ends.add('end')
        inside = bool(value)
    return min(clusters, _MAX_TICKS), len(at_ends)


def _ink_pieces(ink: np.ndarray) -> int:
    """Count the connected pieces of raw ink in the box: blocks of a bar, or letters."""
    if not ink.any():
        return 0
    count, _ = cv2.connectedComponents(ink.astype(np.uint8), connectivity=8)
    return int(count - 1)


def _flanked(
    candidate: Candidate, components: Sequence[Component], width: int, height: int
) -> bool:
    """
    Whether a drawing lies level with the mark on each side of it.

    A connector rule or an arrow sits between two views of one lithic;
    a scale bar does not.
    """
    box = candidate.box
    axis = 1 if candidate.upright else 0
    lo, hi = box[axis], box[axis + 2]
    reach = _FLANK_REACH * (hi - lo)
    cross = (box[1 - axis], box[3 - axis])
    centre = (cross[0] + cross[1]) / 2.0
    before = after = False
    for index, component in enumerate(components):
        if index in candidate.indices:
            continue
        if component.height < 0.03 * height and component.width < 0.05 * width:
            continue
        other = component.box
        if not (other[1 - axis] <= centre <= other[3 - axis]):
            continue
        if other[axis + 2] <= lo and lo - other[axis + 2] <= reach:
            before = True
        if other[axis] >= hi and other[axis] - hi <= reach:
            after = True
    return before and after


def _caption_marks(
    candidate: Candidate, components: Sequence[Component], width: int, height: int
) -> int:
    """Count the small marks set beside the candidate, as a caption would be."""
    x0, y0, x1, y1 = candidate.box
    short = min(box_width(candidate.box), box_height(candidate.box))
    reach = max(_CAPTION_REACH * short, _CAPTION_REACH_FLOOR * min(width, height))
    zone = [x0 - reach, y0 - reach, x1 + reach, y1 + reach]
    own = set(candidate.indices)
    count = 0
    for index, component in enumerate(components):
        if index in own:
            continue
        if component.height > _CAPTION_MAX_HEIGHT * height:
            continue
        if component.width > _CAPTION_MAX_WIDTH * width:
            continue
        bx0, by0, bx1, by1 = component.box
        if bx0 >= zone[0] and by0 >= zone[1] and bx1 <= zone[2] and by1 <= zone[3]:
            count += 1
    return count


def score_candidates(
    candidates: Sequence[Candidate],
    components: Sequence[Component],
    raw: np.ndarray,
    gray: np.ndarray,
    width: int,
    height: int,
    model: Model,
    threshold: float,
) -> None:
    """
    Measure and score every candidate in place, marking those at or above the threshold.

    A connector is never scored: a plain stroke with no caption that a
    drawing's box spans (a tick at a lithic's edge) or that lies between
    two drawings (the rule tying two views together) gets a score of 0
    and ``rejected_by`` set.
    """
    for candidate in candidates:
        candidate.features = candidate_features(candidate, components, raw, gray, width, height)
        candidate.rejected_by = _connector_rule(candidate.features)
        candidate.score = 0.0 if candidate.rejected_by else score(candidate.features, model)
        candidate.accepted = candidate.score >= threshold
        logging.debug(
            "Scale candidate %s: score %.2f, %s%s", candidate.box, candidate.score,
            'accepted' if candidate.accepted else 'rejected',
            f" ({candidate.rejected_by})" if candidate.rejected_by else '',
        )


def _connector_rule(features: Dict[str, float]) -> str:
    """Name the connector rule a plain, uncaptioned mark falls to, or ``''``."""
    if features['ticks'] or features['caption_marks']:
        return ''
    if features['enclosed'] and features['runs'] <= 1:
        return 'tick at a drawing'
    if features['flanked'] and features['runs'] <= 2:
        return 'rule between views'
    return ''
