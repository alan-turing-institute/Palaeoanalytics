"""
Per-page corrections for page segmentation.

Automatic grouping will occasionally split one artefact or merge two,
because plate layouts vary and drawing conventions are not uniform. The
corrections here are recorded in a CSV rather than passed as one-off
command-line flags, so a corrected run stays reproducible: re-running
the folder reproduces the same output without re-deriving the fixes.

Corrections CSV format::

    page_id,expect,join,split
    plate_014.jpg,5,,
    plate_015.jpg,,3+4,
    plate_021.jpg,7,,2

- ``expect``  target artefact count; grouping distances are swept to reach it
- ``join``    merge boxes by debug-image index, e.g. ``3+4`` or ``3+4 7+8``
- ``split``   cut a box at its widest ink-free column, e.g. ``2`` or ``2 5``

Indices refer to the numbering drawn on the debug image of a default
run, so the workflow is: run with ``--debug``, read the numbers, record
the corrections, re-run.
"""

import csv
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import BBox, union

# Ink-free runs shorter than this many pixels are not a usable cut line.
_MIN_SPLIT_RUN = 3
# Grayscale value below which a pixel counts as ink when finding a cut.
_INK_LEVEL = 128


@dataclass
class PageOverride:
    """
    Corrections recorded for a single page.

    Attributes
    ----------
    expect : int, optional
        Target artefact count.
    joins : list of list of int
        Groups of 1-based box indices to merge.
    splits : list of int
        1-based box indices to cut.
    """

    expect: Optional[int] = None
    joins: List[List[int]] = field(default_factory=list)
    splits: List[int] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """Report whether this override would change nothing."""
        return (
            self.expect is None and not self.joins and not self.splits
        )

    def applied_kinds(self) -> str:
        """Summarise which corrections this override carries."""
        kinds = []
        if self.expect is not None:
            kinds.append('expect')
        if self.joins:
            kinds.append('join')
        if self.splits:
            kinds.append('split')
        return '+'.join(kinds)


def load_overrides(path: Optional[str]) -> Dict[str, PageOverride]:
    """
    Read a corrections CSV, keyed by page filename.

    Parameters
    ----------
    path : str, optional
        Path to the CSV. Returns an empty mapping when None.

    Returns
    -------
    dict
        Map of page_id to PageOverride.

    Raises
    ------
    FileNotFoundError
        When a path is given but does not exist. A missing corrections
        file is a typo, not a reason to silently process uncorrected.
    """
    if not path:
        return {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Overrides file not found: {path}")

    overrides: Dict[str, PageOverride] = {}
    with open(path, newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            page_id = (row.get('page_id') or '').strip()
            if not page_id:
                continue
            override = _parse_row(row, page_id)
            if not override.is_empty:
                overrides[page_id] = override

    logging.info(
        "Corrections read for %d page(s) from %s",
        len(overrides), os.path.basename(path),
    )
    return overrides


def _parse_row(row: Dict[str, str], page_id: str) -> PageOverride:
    """Parse one corrections row, warning on unusable values."""
    override = PageOverride()

    expect = (row.get('expect') or '').strip()
    if expect:
        try:
            override.expect = int(expect)
        except ValueError:
            logging.warning(
                "The expect value '%s' for %s is not a number; not used", expect, page_id
            )

    for spec in (row.get('join') or '').split():
        indices = _parse_indices(spec.split('+'), page_id, 'join')
        if len(indices) > 1:
            override.joins.append(indices)

    override.splits = _parse_indices(
        (row.get('split') or '').split(), page_id, 'split'
    )
    return override


def _parse_indices(
    values: Sequence[str], page_id: str, field_name: str
) -> List[int]:
    """Parse a list of 1-based indices, skipping unusable entries."""
    parsed = []
    for value in values:
        try:
            parsed.append(int(value.strip()))
        except ValueError:
            logging.warning(
                "The %s index '%s' for %s is not a number; not used",
                field_name, value, page_id,
            )
    return parsed


def warn_unmatched(
    overrides: Dict[str, PageOverride], page_ids: Sequence[str]
) -> None:
    """
    Warn about corrections naming pages that were not processed.

    An unmatched page_id is almost always a typo, and silently ignoring
    it would leave the user believing a correction had been applied.
    """
    known = set(page_ids)
    for page_id in overrides:
        if page_id not in known:
            logging.warning(
                "The corrections name '%s', which is not in the input folder",
                page_id,
            )


### CORRECTION APPLICATION ###

def apply_expect(
    boxes: Sequence[BBox],
    expect: int,
    regroup,
    grouping: Dict,
) -> List[BBox]:
    """
    Sweep grouping distances until the expected artefact count is met.

    Candidate settings are tried in order of increasing deviation from
    the configured defaults, so the correction applied is the smallest
    one that works.

    Parameters
    ----------
    boxes : sequence of BBox
        Boxes from the default run, returned unchanged if already correct.
    expect : int
        Target artefact count.
    regroup : callable
        Takes ``(gap, narrow)`` and returns a list of boxes.
    grouping : dict
        Grouping configuration, providing the default distances.

    Returns
    -------
    list of BBox
        Boxes matching the expected count, or the original boxes when
        no candidate reaches it.
    """
    if len(boxes) == expect:
        return list(boxes)

    base_gap = grouping.get('gap', 0.025)
    base_narrow = grouping.get('narrow', 0.07)
    for gap, narrow in _sweep_candidates(base_gap, base_narrow):
        candidate = regroup(gap, narrow)
        if len(candidate) == expect:
            logging.info(
                "%d artefacts found with gap=%.3f narrow=%.3f",
                expect, gap, narrow,
            )
            return candidate

    logging.warning(
        "%d artefacts not possible; %d kept", expect, len(boxes)
    )
    return list(boxes)


def _sweep_candidates(
    base_gap: float, base_narrow: float
) -> List[Tuple[float, float]]:
    """Build (gap, narrow) candidates ordered by distance from defaults."""
    grid = [
        (gap / 1000, narrow / 1000)
        for gap in range(10, 82, 3)
        for narrow in range(20, 122, 10)
    ]
    grid.sort(
        key=lambda pair: (
            abs(pair[0] - base_gap) / base_gap
            + abs(pair[1] - base_narrow) / base_narrow
        )
    )
    return grid


def apply_joins(
    boxes: Sequence[BBox], joins: Sequence[Sequence[int]]
) -> List[BBox]:
    """
    Merge boxes named by 1-based debug-image index.

    Parameters
    ----------
    boxes : sequence of BBox
        Boxes in reading order.
    joins : sequence of sequence of int
        Groups of indices to merge.

    Returns
    -------
    list of BBox
        Boxes with each group merged into one.
    """
    result = [list(box) for box in boxes]

    for group in joins:
        indices = sorted({i - 1 for i in group}, reverse=True)
        if not all(0 <= i < len(result) for i in indices):
            logging.warning(
                "Join %s not used: the index is outside 1-%d",
                '+'.join(str(i) for i in group), len(result),
            )
            continue
        merged = result[indices[0]]
        for index in indices[1:]:
            merged = union(merged, result[index])
        for index in indices:
            result.pop(index)
        result.append(merged)
    return result


def apply_splits(
    boxes: Sequence[BBox],
    splits: Sequence[int],
    gray: np.ndarray,
) -> List[BBox]:
    """
    Cut boxes at their widest ink-free column.

    Parameters
    ----------
    boxes : sequence of BBox
        Boxes in reading order.
    splits : sequence of int
        1-based indices to cut.
    gray : np.ndarray
        Grayscale page, used to locate ink-free columns.

    Returns
    -------
    list of BBox
        Boxes with each named box replaced by two halves.
    """
    result = [list(box) for box in boxes]

    for index in sorted(splits, reverse=True):
        position = index - 1
        if not 0 <= position < len(result):
            logging.warning(
                "Split %d not used: the index is outside 1-%d", index, len(result)
            )
            continue
        halves = _split_at_widest_gap(result[position], gray)
        if halves is None:
            logging.warning(
                "Box %d has no empty column for a split", index
            )
            continue
        result[position] = halves[0]
        result.append(halves[1])
    return result


def _split_at_widest_gap(
    box: BBox, gray: np.ndarray
) -> Optional[Tuple[BBox, BBox]]:
    """Find the widest ink-free column in a box and cut through it."""
    x0, y0, x1, y1 = box
    empty = (gray[y0:y1, x0:x1] < _INK_LEVEL).sum(axis=0) == 0

    best_run, best_start = 0, 0
    run, start = 0, 0
    for column, is_empty in enumerate(empty):
        if not is_empty:
            run = 0
            continue
        if run == 0:
            start = column
        run += 1
        if run > best_run:
            best_run, best_start = run, start

    if best_run <= _MIN_SPLIT_RUN:
        return None

    cut = x0 + best_start + best_run // 2
    return [x0, y0, cut, y1], [cut, y0, x1, y1]
