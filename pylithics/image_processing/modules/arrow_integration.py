"""
Arrow integration for PyLithics.

Finds the arrows on the scars of a drawing and attaches each to its
scar. An arrow sits inside a scar, so the only marks tried are the
pieces of ink inside a recognised scar: a scar with a metric row. A
surface, a scar, or a mark outside every recognised scar is never
tried. Marks inside cortex are left alone.

The marks are cut from the working copy of the drawing when there is
one: the upscaled binary the ink was found on, before its return to the
source grid. A small arrow keeps there the notches that the return
removes. Each scar's outline is scaled to the copy, the ink inside it is
split into pieces, and each piece that does not touch the outline is a
candidate. The detector works at that scale and the points it returns
are brought back to source pixels here.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..config import get_arrow_integration_config
from .arrow_detection import analyze_child_contour_for_arrow

MARGIN = 4   # source pixels kept around a scar in its crop


def integrate_arrows(
    sorted_contours: Dict[str, List],
    metrics: List[Dict],
    image: np.ndarray,
    image_dpi: Optional[float] = None,
    debug_dir: Optional[str] = None,
    working: Optional[np.ndarray] = None,
    factor: int = 1,
) -> List[Dict]:
    """
    Detect arrows and attach them to the scar metrics; return the metrics.

    Parameters
    ----------
    sorted_contours : dict
        {"parents":..., "children":..., "nested_children":...}
    metrics : list
        Metric dictionaries to update with arrow info.
    image : np.ndarray
        The binary image on the source grid, ink white.
    image_dpi : float, optional
        DPI of the image, for the notch method's scaled thresholds.
    debug_dir : str, optional
        Folder for the per-mark debug files.
    working : np.ndarray, optional
        The binary image at the working resolution, ``factor`` times the
        size of ``image``. Without it the marks are cut from ``image``.
    factor : int, optional
        Pixels of ``working`` per pixel of ``image``.
    """
    if working is None:
        working, factor = image, 1
    tolerance = get_arrow_integration_config().get('area_match_tolerance', 1.0)
    found = cortex = 0
    for contour, metric in recognised_scars(metrics, sorted_contours["children"], tolerance):
        if metric.get('is_cortex', False):
            cortex += 1
            continue
        result = _best_arrow(
            marks_in_scar(contour, working, factor), metric['scar'], image_dpi, debug_dir,
        )
        if result is not None:
            _apply_arrow_result(metric, result)
            found += 1
    scars = sum(1 for metric in metrics if _is_scar(metric))
    logging.debug(f"{found} arrow(s) on {scars} scar(s); {cortex} cortex scar(s) not tried")
    return metrics


def _is_scar(metric: Dict) -> bool:
    """Say whether a metric row is a scar (not a surface)."""
    return metric["parent"] != metric["scar"]


def recognised_scars(
    metrics: List[Dict], children: List[np.ndarray], tolerance: float
) -> List[Tuple[np.ndarray, Dict]]:
    """Return each scar metric with its contour, matched by area."""
    pairs = []
    for metric in metrics:
        if not _is_scar(metric):
            continue
        for contour in children:
            if abs(cv2.contourArea(contour) - metric["area"]) < tolerance:
                pairs.append((contour, metric))
                break
    return pairs


### MARKS ###

@dataclass(frozen=True)
class Marks:
    """The ink inside one scar, cut from the image it is found on."""

    ink: np.ndarray                 # the crop around the scar, ink white
    origin: Tuple[int, int]         # the crop's top left corner in the image
    factor: int                     # pixels of the image per source pixel
    contours: List[np.ndarray]      # one outer contour per piece, in crop pixels

    def to_source(self, point) -> Tuple[int, int]:
        """Return a crop point in source pixels."""
        return (
            int(round((point[0] + self.origin[0]) / self.factor)),
            int(round((point[1] + self.origin[1]) / self.factor)),
        )


def marks_in_scar(scar: np.ndarray, ink: np.ndarray, factor: int = 1) -> Marks:
    """
    Return the pieces of ink inside a scar that do not touch its outline.

    ``scar`` is the scar's contour on the source grid. ``ink`` is the
    binary image, ``factor`` times the source size. The outline is scaled
    to the image and filled, and the ink of the crop is split into
    connected pieces. A piece with any pixel outside the fill is the
    outline itself, or joined to it, so it is not a mark. A piece inside
    the fill is a mark, even when its tip touches the outline's edge.
    """
    outline = np.rint(scar.reshape(-1, 2) * factor + (factor - 1) / 2.0).astype(np.int32)
    x, y, w, h = cv2.boundingRect(outline)
    margin = MARGIN * factor
    left, top = max(x - margin, 0), max(y - margin, 0)
    right, bottom = min(x + w + margin, ink.shape[1]), min(y + h + margin, ink.shape[0])
    crop = ink[top:bottom, left:right]
    region = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.drawContours(region, [outline.reshape(-1, 1, 2) - (left, top)], 0, 255, cv2.FILLED)
    count, labels = cv2.connectedComponents((crop > 0).astype(np.uint8), connectivity=8)
    outside = set(np.unique(labels[region == 0]))
    contours = [
        _outer_contour(labels == piece) for piece in range(1, count) if piece not in outside
    ]
    return Marks(crop, (left, top), factor, contours)


def _outer_contour(piece: np.ndarray) -> np.ndarray:
    """Return the outer contour of one connected piece of ink."""
    found, _ = cv2.findContours(
        piece.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    return max(found, key=cv2.contourArea)


### DECISION ###

def _best_arrow(
    marks: Marks, scar_name: str, image_dpi: Optional[float], debug_dir: Optional[str],
) -> Optional[Dict]:
    """Try every mark of a scar; return the best arrow found in source pixels, or None."""
    results = []
    for number, contour in enumerate(marks.contours, 1):
        entry = {"scar": f"{scar_name} mark {number}", "debug_dir": debug_dir}
        result = analyze_child_contour_for_arrow(
            contour, entry, marks.ink, image_dpi, marks.factor,
        )
        if result:
            results.append(result)
    if not results:
        return None
    best = max(results, key=lambda r: (r.get('stage') == 'triangle', r.get('score') or 0.0))
    logging.debug(f"Arrow on {scar_name} by stage {best.get('stage')}, "
                  f"compass {best.get('compass_angle', '?')}")
    return {
        **best,
        'arrow_back': marks.to_source(best['arrow_back']),
        'arrow_tip': marks.to_source(best['arrow_tip']),
    }


def _apply_arrow_result(metric: Dict, result: Dict) -> None:
    """Apply arrow detection result to a metric entry."""
    metric.update({
        "has_arrow": True,
        "arrow_angle_rad": round(result["angle_rad"], 2),
        "arrow_angle_deg": round(result["angle_deg"], 0),
        "arrow_angle": round(result["compass_angle"], 0),
        "arrow_tip": result["arrow_tip"],
        "arrow_back": result["arrow_back"],
    })
    if result.get("score") is not None:
        metric["arrow_score"] = result["score"]
