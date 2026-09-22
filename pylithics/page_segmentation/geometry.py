"""
Bounding-box geometry helpers for page segmentation.

A box is a 4-element list ``[x0, y0, x1, y1]`` in page pixel
coordinates, with ``x1``/``y1`` exclusive. Overlap functions return
positive values when boxes overlap and negative values when they are
separated, so callers can test proximity and overlap with the same
primitive.
"""

from typing import List, Sequence

BBox = List[int]


def vertical_overlap(a: Sequence[int], b: Sequence[int]) -> int:
    """
    Measure vertical overlap between two boxes in pixels.

    Parameters
    ----------
    a, b : sequence of int
        Boxes as ``[x0, y0, x1, y1]``.

    Returns
    -------
    int
        Overlapping height. Negative when the boxes do not overlap.
    """
    return min(a[3], b[3]) - max(a[1], b[1])


def horizontal_overlap(a: Sequence[int], b: Sequence[int]) -> int:
    """
    Measure horizontal overlap between two boxes in pixels.

    Parameters
    ----------
    a, b : sequence of int
        Boxes as ``[x0, y0, x1, y1]``.

    Returns
    -------
    int
        Overlapping width. Negative when the boxes do not overlap.
    """
    return min(a[2], b[2]) - max(a[0], b[0])


def horizontal_gap(a: Sequence[int], b: Sequence[int]) -> int:
    """
    Measure the horizontal separation between two boxes in pixels.

    Returns
    -------
    int
        Gap width. Negative when the boxes overlap horizontally.
    """
    return max(b[0] - a[2], a[0] - b[2])


def vertical_gap(a: Sequence[int], b: Sequence[int]) -> int:
    """
    Measure the vertical separation between two boxes in pixels.

    Returns
    -------
    int
        Gap height. Negative when the boxes overlap vertically.
    """
    return max(b[1] - a[3], a[1] - b[3])


def union(a: Sequence[int], b: Sequence[int]) -> BBox:
    """
    Compute the smallest box enclosing both inputs.

    Returns
    -------
    list of int
        Enclosing box as ``[x0, y0, x1, y1]``.
    """
    return [
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    ]


def box_distance(a: Sequence[int], b: Sequence[int]) -> float:
    """
    Measure the edge-to-edge distance between two boxes.

    Overlapping boxes are zero distance apart on the overlapping axis,
    so touching or nested boxes return ``0.0``.

    Returns
    -------
    float
        Euclidean distance in pixels between the nearest edges.
    """
    dx = max(0, horizontal_gap(a, b))
    dy = max(0, vertical_gap(a, b))
    return (dx * dx + dy * dy) ** 0.5


def boxes_intersect(a: Sequence[int], b: Sequence[int]) -> bool:
    """Report whether two boxes overlap on both axes."""
    return horizontal_overlap(a, b) > 0 and vertical_overlap(a, b) > 0


def box_width(box: Sequence[int]) -> int:
    """Return the width of a box in pixels."""
    return box[2] - box[0]


def box_height(box: Sequence[int]) -> int:
    """Return the height of a box in pixels."""
    return box[3] - box[1]


def clamp_box(
    box: Sequence[int], padding: int, width: int, height: int
) -> BBox:
    """
    Expand a box by padding and clamp it to the page bounds.

    Parameters
    ----------
    box : sequence of int
        Box as ``[x0, y0, x1, y1]``.
    padding : int
        Pixels of whitespace to add on every side.
    width, height : int
        Page dimensions in pixels.

    Returns
    -------
    list of int
        Padded box clipped to ``[0, 0, width, height]``.
    """
    return [
        max(0, box[0] - padding),
        max(0, box[1] - padding),
        min(width, box[2] + padding),
        min(height, box[3] + padding),
    ]
