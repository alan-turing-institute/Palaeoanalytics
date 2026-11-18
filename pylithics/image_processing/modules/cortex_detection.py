"""
Cortex detection for PyLithics.

Detects cortex on lithic artifact surfaces. Cortex is distinguished from scars by its
stippled texture, which represents the original weathered surface of the stone.

Cortex is detected at the child contour level, labelled ``cortex 1``, ``cortex 2`` and
so on, and measured as an area and as a percentage of the parent surface area.

Stippling is many small pieces of ink. A scar needs at least
``min_stipple_pieces`` of them (default 3) before the density, variance
and edge tests are asked: one or two pieces are an arrow, a numeral or a
dot, and a small scar that holds an arrow passed the density test alone.

Some illustrators mark a cortex scar with a letter, usually C, instead of
or beside the stippling. A scar is also cortex when one of its marks (the
pieces of ink wholly inside it) reads as one of ``letter.letters``. The
letter speaks for the whole scar, so its whole area is the cortex area.
The reader is RapidOCR, the ``ocr`` extra; without it the letter test is
skipped and the log says so once. ``cortex_source`` records which test
decided: ``stippling`` or ``letter``.
"""

DEFAULT_MIN_STIPPLE_PIECES = 3
STIPPLE_MAX_AREA = 50   # a piece of ink up to this many pixels counts as a stipple
DEFAULT_LETTER = {'enabled': True, 'letters': ['C'], 'min_confidence': 0.8}
# A mark is scaled to this height and read by the recognizer alone. RapidOCR's text
# detector finds no line in one lone letter; its recognizer reads a cut-out C at
# 0.93 at this height and less well when the glyph is larger (0.54 at 96 px).
_READ_HEIGHT = 32
_READ_PAD = 12          # paper around the mark, so the reader sees a whole glyph
_reader = None
_reader_missing_reported = False

import cv2
import numpy as np
import logging
from typing import List, Dict, Any
from ..config import get_cortex_detection_config


def detect_cortex_in_child_contours(
    metrics: List[Dict[str, Any]],
    inverted_image: np.ndarray,
    working: "np.ndarray | None" = None,
    factor: int = 1,
) -> List[Dict[str, Any]]:
    """
    Detect cortex in child contours and relabel them by surface type.

    ``inverted_image`` is the binary image for the texture test; ``working``
    and ``factor`` are the working copy where the letter test reads the
    marks, as the arrow detector does.

    Returns:
        Updated metrics with cortex areas relabelled and non-cortex children
        renumbered per-surface.
    """
    config = get_cortex_detection_config()
    if not config.get('enabled', True):
        logging.debug("Cortex detection is disabled in configuration")
        return metrics

    logging.debug("Starting cortex detection in child contours")

    try:
        parents = [m for m in metrics if m["parent"] == m["scar"]]
        children = [m for m in metrics if m["parent"] != m["scar"]]

        parent_areas = {p.get("scar", ""): p.get("area", 0) for p in parents}
        parent_surface = {
            p.get("scar", ""): p.get("surface_type", "Unknown") for p in parents
        }

        letters_on = (working, factor) if working is not None else (inverted_image, 1)
        cortex_children, non_cortex_children = _classify_cortex_children(
            children, inverted_image, config, letters_on
        )
        _label_cortex_children(cortex_children, parent_areas)
        renumbered = _renumber_non_cortex_children(
            non_cortex_children, parent_surface
        )

        logging.debug(
            f"Cortex detection completed: {len(cortex_children)} cortex areas "
            f"detected, {len(renumbered)} non-cortex children processed"
        )
        return parents + cortex_children + renumbered

    except Exception as e:
        logging.error(f"Error in the cortex detection: {e}")
        return metrics


def _classify_cortex_children(
    children: List[Dict[str, Any]],
    inverted_image: np.ndarray,
    config: Dict[str, Any],
    letters_on: "tuple | None" = None,
) -> "tuple[List[Dict[str, Any]], List[Dict[str, Any]]]":
    """Split children into (cortex, non-cortex) by stippling, then by a cortex letter."""
    letters_on = letters_on or (inverted_image, 1)
    cortex, non_cortex = [], []
    for child in children:
        if not child.get("contour"):
            logging.warning(
                f"Child contour {child.get('scar', 'unknown')} "
                f"has no contour data"
            )
            non_cortex.append(child)
            continue

        contour_array = np.array(child["contour"], dtype=np.int32)
        if _detect_cortex_texture(contour_array, inverted_image, config):
            child["cortex_source"] = "stippling"
        elif _letter_in_scar(contour_array, *letters_on, config):
            child["cortex_source"] = "letter"
        else:
            non_cortex.append(child)
            continue
        cortex.append(child)
        logging.debug(f"Detected cortex in child {child.get('scar', 'unknown')} "
                      f"by {child['cortex_source']}")
    return cortex, non_cortex


def _letter_in_scar(
    contour: np.ndarray, ink: np.ndarray, factor: int, config: Dict[str, Any],
) -> bool:
    """
    Report whether a mark inside the scar reads as a cortex letter.

    A mark counts when the reader returns one line, that line is one of
    the configured letters (case ignored), and its confidence is at least
    ``min_confidence``.
    """
    settings = {**DEFAULT_LETTER, **(config.get('letter') or {})}
    if not settings.get('enabled', True):
        return False
    letters = {str(letter).strip().upper() for letter in settings.get('letters', [])}
    reader = _load_reader() if letters else None
    if reader is None:
        return False
    from .arrow_integration import marks_in_scar
    marks = marks_in_scar(contour, ink, factor)
    for mark in marks.contours:
        text, score = _read_mark(reader, marks.ink, mark)
        if text.upper() in letters and score >= float(settings.get('min_confidence', 0.8)):
            logging.debug(f"Cortex letter {text!r} read at {score:.2f}")
            return True
    return False


def _read_mark(reader, ink: np.ndarray, mark: np.ndarray) -> "tuple[str, float]":
    """Read one mark: dark ink on paper, enlarged and padded. Return its text and score."""
    x, y, w, h = cv2.boundingRect(mark)
    glyph = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(glyph, [mark - (x, y)], 0, 255, cv2.FILLED)
    glyph = np.where((glyph > 0) & (ink[y:y + h, x:x + w] > 0), 0, 255).astype(np.uint8)
    scale = _READ_HEIGHT / max(h, 1)
    glyph = cv2.resize(glyph, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    glyph = cv2.copyMakeBorder(glyph, _READ_PAD, _READ_PAD, _READ_PAD, _READ_PAD,
                               cv2.BORDER_CONSTANT, value=255)
    result, _ = reader(cv2.cvtColor(glyph, cv2.COLOR_GRAY2RGB),
                       use_det=False, use_cls=False, use_rec=True)
    if not result or len(result) != 1:
        return "", 0.0
    text, score = result[0][-2], result[0][-1]
    return "".join(ch for ch in str(text) if ch.isalnum()), float(score)


def _load_reader():
    """Load RapidOCR once per process; report its absence once."""
    global _reader, _reader_missing_reported
    if _reader is not None:
        return _reader
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        if not _reader_missing_reported:
            logging.info("RapidOCR is not installed, so a letter in a scar is not read as "
                         "cortex. Install it with: pip install '.[ocr]'")
            _reader_missing_reported = True
        return None
    _reader = RapidOCR()
    return _reader


def _label_cortex_children(
    cortex_children: List[Dict[str, Any]],
    parent_areas: Dict[str, float],
) -> None:
    """Relabel cortex children as `cortex N` and add cortex-area metrics."""
    for i, child in enumerate(cortex_children, start=1):
        original = child.get("scar", "unknown")
        parent_area = parent_areas.get(child.get("parent", ""), 0)
        cortex_area = child.get("area", 0)
        percentage = (cortex_area / parent_area * 100) if parent_area > 0 else 0

        child["scar"] = f"cortex {i}"
        child["surface_feature"] = f"cortex {i}"
        child["cortex_area"] = cortex_area
        child["cortex_percentage"] = round(percentage, 2)
        child["is_cortex"] = True

        logging.debug(
            f"Relabeled {original} -> {child['scar']} "
            f"(area: {cortex_area}, {percentage:.1f}% of surface)"
        )


def _renumber_non_cortex_children(
    non_cortex_children: List[Dict[str, Any]],
    parent_surface: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Rename Dorsal children to `scar N`, Lateral to `edge N`, drop Platform."""
    by_surface = {"Dorsal": [], "Platform": [], "Lateral": [], "Ventral": []}
    for child in non_cortex_children:
        surface = parent_surface.get(child.get("parent", ""), "Unknown")
        if surface in by_surface:
            by_surface[surface].append(child)

    renamed = []
    for i, child in enumerate(by_surface["Dorsal"], start=1):
        child["scar"] = f"scar {i}"
        child["surface_feature"] = f"scar {i}"
        child["is_cortex"] = False
        renamed.append(child)

    for child in by_surface["Platform"]:
        logging.debug(
            f"Excluding platform child (area={child.get('area', 0)}) "
            f"as likely empty space boundary"
        )

    for i, child in enumerate(by_surface["Lateral"], start=1):
        child["scar"] = f"edge {i}"
        child["surface_feature"] = f"edge {i}"
        child["is_cortex"] = False
        renamed.append(child)

    return renamed


def _detect_cortex_texture(
    contour: np.ndarray,
    inverted_image: np.ndarray,
    config: Dict[str, Any]
) -> bool:
    """
    Decide whether a contour region shows cortex-like texture.

    Cortex surfaces are characterized by a stippled pattern with many small
    connected components, high local variance, and high edge density. Fewer
    small components than ``min_stipple_pieces`` is never cortex.
    """
    try:
        roi_cropped, mask_cropped = _crop_contour_roi(contour, inverted_image)
        if roi_cropped is None:
            return False

        roi_area = cv2.countNonZero(mask_cropped)
        if roi_area == 0:
            return False

        pieces, stippling = _stippling_density(roi_cropped, roi_area)
        variance = _texture_variance(roi_cropped, mask_cropped)
        edge_density = _edge_density(roi_cropped, roi_area)

        cortex_detected = (
            pieces >= int(config.get('min_stipple_pieces', DEFAULT_MIN_STIPPLE_PIECES))
            and stippling > config.get('stippling_density_threshold', 0.2)
            and variance > config.get('texture_variance_threshold', 100)
            and edge_density > config.get('edge_density_threshold', 0.05)
        )
        logging.debug(
            f"Cortex analysis: stipple_pieces={pieces}, stippling_density={stippling:.3f}, "
            f"variance={variance:.1f}, edge_density={edge_density:.3f} -> "
            f"cortex={cortex_detected}"
        )
        return cortex_detected

    except Exception as e:
        logging.error(f"Error in the cortex texture detection: {e}")
        return False


def _crop_contour_roi(
    contour: np.ndarray, inverted_image: np.ndarray,
):
    """Return (roi, mask) cropped to the contour's bounding box, or (None, None)."""
    mask = np.zeros(inverted_image.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [contour], 255)
    roi = cv2.bitwise_and(inverted_image, mask)

    x, y, w, h = cv2.boundingRect(contour)
    roi_cropped = roi[y:y + h, x:x + w]
    mask_cropped = mask[y:y + h, x:x + w]
    if roi_cropped.size == 0:
        return None, None
    return roi_cropped, mask_cropped


def _stippling_density(roi_cropped: np.ndarray, roi_area: int) -> "tuple[int, float]":
    """Return the number of small connected components, and that number per 1000 pixels."""
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        roi_cropped, connectivity=8,
    )
    small = sum(
        1 for i in range(1, num_labels)
        if 1 <= stats[i, cv2.CC_STAT_AREA] <= STIPPLE_MAX_AREA
    )
    return small, small / roi_area * 1000


def _texture_variance(
    roi_cropped: np.ndarray, mask_cropped: np.ndarray,
) -> float:
    """Return the local variance of the Gaussian-blurred ROI within the mask."""
    if roi_cropped.shape[0] <= 5 or roi_cropped.shape[1] <= 5:
        return 0.0
    blurred = cv2.GaussianBlur(roi_cropped, (5, 5), 0)
    return float(np.var(blurred[mask_cropped > 0])) if np.any(mask_cropped) else 0.0


def _edge_density(roi_cropped: np.ndarray, roi_area: int) -> float:
    """Return the proportion of Canny edge pixels relative to the ROI area."""
    edges = cv2.Canny(roi_cropped, 50, 150)
    return float(np.sum(edges > 0)) / roi_area if roi_area > 0 else 0.0


def calculate_total_cortex_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate aggregate cortex metrics across all surfaces.

    Args:
        metrics (List[Dict]): List of contour metrics with cortex detection

    Returns:
        Dict[str, float]: Aggregate cortex statistics
    """
    cortex_areas = [m.get("cortex_area", 0) for m in metrics if m.get("is_cortex", False)]

    total_cortex_area = sum(cortex_areas)
    cortex_count = len(cortex_areas)
    average_cortex_size = total_cortex_area / cortex_count if cortex_count > 0 else 0

    return {
        "total_cortex_area": total_cortex_area,
        "cortex_count": cortex_count,
        "average_cortex_size": round(average_cortex_size, 2)
    }