"""
Working resolution: upscale a drawing for detection, never for output.

A line drawing scanned or exported small has strokes a pixel or two
wide. At that size closing fuses neighbouring shapes, outlines break
into pieces and a numeral is too small to read. This module measures
the stroke width and the hatch clearance of a grayscale image and, when
either is below its floor, upscales a *working copy* by the smallest
integer factor that lifts them above it. Detection runs on the copy;
``restore_to_grid`` brings a binary result back to the source pixel
grid, so every crop, tag and measurement stays in source pixels and
the factor is provenance, nothing more.

The rule, for every caller: detect at working resolution, measure in
source coordinates. Never downscale.

Ported from lithic_editor ``processing/resolution.py``. The skeleton
comes from ``cv2.ximgproc.thinning`` rather than scikit-image, so the
package needs the OpenCV contrib build and nothing else new. The
neural models (ESPCN, FSRCNN) are bundled under ``pylithics/models``;
see the NOTICE file there.
"""

import logging
import math
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt, maximum_filter

# Factors the bundled models provide. 1 means no upscaling.
SUPPORTED_FACTORS = (1, 2, 3, 4)
SUPPORTED_MODELS = ('espcn', 'fsrcnn')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

_DEFAULTS = {
    'enabled': True,
    'model': 'espcn',
    'max_factor': 4,
    'restore_ink_coverage': 0.35,
    'dot_extent_in_stroke_widths': 4.0,
    'min_stroke_width_px': 6.0,
    'min_hatch_gap_px': 12.0,
    'max_working_pixels': 40_000_000,
}

_models: Dict[Tuple[str, int], object] = {}


class UpscalingUnavailable(RuntimeError):
    """The contrib build of OpenCV or a bundled model file is missing."""


@dataclass(frozen=True)
class LineGeometry:
    """Stroke width and hatch clearance of a drawing, in pixels."""

    stroke_width: float
    hatch_gap: float
    ink_fraction: float

    def describe(self) -> str:
        """One line for the log: ``stroke 2.1 px, gap 6.0 px``."""
        width = 'unknown' if math.isnan(self.stroke_width) else f'{self.stroke_width:.1f} px'
        gap = 'unknown' if math.isnan(self.hatch_gap) else f'{self.hatch_gap:.1f} px'
        return f'stroke {width}, gap {gap}'


@dataclass(frozen=True)
class WorkingCopy:
    """A grayscale image at working resolution, with how it got there."""

    gray: np.ndarray
    factor: int
    geometry: LineGeometry
    source_shape: Tuple[int, int]


def settings_for(config: Dict, section: str) -> Dict:
    """
    Return the working-resolution settings for one caller.

    ``config['working_resolution']`` holds the shared keys and one
    sub-section per caller (``pages``, ``analysis``) whose keys replace
    the shared ones. Missing keys take the defaults.
    """
    shared = config.get('working_resolution', {}) or {}
    own = shared.get(section, {}) or {}
    merged = dict(_DEFAULTS)
    merged.update({k: v for k, v in shared.items() if k not in ('pages', 'analysis')})
    merged.update(own)
    return merged


### MEASUREMENT ###

def binarize(gray: np.ndarray) -> np.ndarray:
    """Ink mask of a grayscale drawing by Otsu's threshold (inclusive)."""
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray <= threshold


def measure_line_geometry(gray: np.ndarray, dot_extent: float = 4.0) -> LineGeometry:
    """
    Measure the typical stroke width and the typical clearance between strokes.

    Width is twice the median distance-to-edge along the skeleton.
    Clearance is measured on the background enclosed by the drawing: the
    ridge of the background's distance-to-ink field runs midway between
    neighbouring strokes, and its median is the typical clearance.
    Dot-like components (stipple, specks) are left out of that
    measurement. Either value is NaN when it cannot be measured.
    """
    ink = binarize(gray)
    if not ink.any():
        return LineGeometry(math.nan, math.nan, 0.0)
    skeleton = cv2.ximgproc.thinning(ink.astype(np.uint8) * 255) > 0
    if skeleton.any():
        half_widths = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)[skeleton]
        stroke_width = max(1.0, 2.0 * float(np.median(half_widths)) - 1.0)
    else:
        stroke_width = math.nan
    return LineGeometry(stroke_width, _hatch_gap(ink, stroke_width, dot_extent), float(ink.mean()))


def _hatch_gap(ink: np.ndarray, stroke_width: float, dot_extent: float) -> float:
    """Median clearance between stroke-like components, or NaN."""
    strokes = _stroke_components(ink, stroke_width, dot_extent)
    enclosed = binary_fill_holes(strokes) & ~strokes
    if not enclosed.any():
        return math.nan
    clearance = distance_transform_edt(~strokes)
    ridge = enclosed & (clearance > 0) & (clearance >= maximum_filter(clearance, size=3))
    if not ridge.any():
        return math.nan
    return max(1.0, 2.0 * float(np.median(clearance[ridge])) - 1.0)


def _stroke_components(ink: np.ndarray, stroke_width: float, dot_extent: float) -> np.ndarray:
    """Ink with components shorter than ``dot_extent`` stroke widths removed."""
    if math.isnan(stroke_width) or dot_extent <= 0:
        return ink
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink.astype(np.uint8), connectivity=8
    )
    extent = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    keep = extent >= dot_extent * stroke_width
    keep[0] = False
    return keep[labels]


### FACTOR ###

def choose_factor(geometry: LineGeometry, shape: Tuple[int, int], settings: Dict) -> int:
    """
    Return the smallest supported factor that lifts both values above their floors.

    A floor of 0 turns that test off. The factor is capped by
    ``max_factor`` and so that the working image holds at most
    ``max_working_pixels``. 1 when nothing is below a floor, nothing
    could be measured, or the caps allow no more.
    """
    needed = 1.0
    width_floor = float(settings.get('min_stroke_width_px', 0) or 0)
    gap_floor = float(settings.get('min_hatch_gap_px', 0) or 0)
    if width_floor and not math.isnan(geometry.stroke_width) and geometry.stroke_width > 0:
        needed = max(needed, width_floor / geometry.stroke_width)
    if gap_floor and not math.isnan(geometry.hatch_gap) and geometry.hatch_gap > 0:
        needed = max(needed, gap_floor / geometry.hatch_gap)
    factor = math.ceil(needed - 1e-9)
    factor = min(factor, int(settings.get('max_factor', 4)), max(SUPPORTED_FACTORS))
    if factor not in SUPPORTED_FACTORS:
        factor = min(f for f in SUPPORTED_FACTORS if f >= factor)
    pixels = shape[0] * shape[1]
    limit = int(settings.get('max_working_pixels', _DEFAULTS['max_working_pixels']))
    while factor > 1 and pixels * factor * factor > limit:
        factor -= 1
    return factor


### UPSCALING ###

def upscale(gray: np.ndarray, factor: int, model: str = 'espcn') -> np.ndarray:
    """
    Enlarge a grayscale image by an integer factor with a bundled model.

    Raises
    ------
    UpscalingUnavailable
        When OpenCV has no ``dnn_superres`` (the contrib build is not
        installed) or the model file is missing.
    """
    if factor == 1:
        return gray
    engine = _load_model(model, factor)
    colour = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)      # the models take 3 channels
    return cv2.cvtColor(engine.upsample(colour), cv2.COLOR_BGR2GRAY)


def _load_model(model: str, factor: int):
    """Load a model once per run and keep it."""
    key = (model, factor)
    if key in _models:
        return _models[key]
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown upscaling model '{model}'; use one of {SUPPORTED_MODELS}")
    if not hasattr(cv2, 'dnn_superres'):
        raise UpscalingUnavailable(
            "OpenCV has no dnn_superres module. Install the contrib build: "
            "pip uninstall opencv-python-headless && "
            "pip install opencv-contrib-python-headless"
        )
    path = os.path.join(MODELS_DIR, f'{model.upper()}_x{factor}.pb')
    if not os.path.isfile(path):
        raise UpscalingUnavailable(f'Model file missing: {path}')
    engine = cv2.dnn_superres.DnnSuperResImpl_create()
    engine.readModel(path)
    engine.setModel(model, factor)
    _models[key] = engine
    logging.debug('Loaded %s x%d from %s', model.upper(), factor, path)
    return engine


### RESTORE ###

def restore_to_grid(
    mask: np.ndarray, shape: Tuple[int, int], ink_coverage: float = 0.35
) -> np.ndarray:
    """
    Resample a binary mask (ink = nonzero) back onto the source pixel grid.

    Ink coverage is area-averaged so a stroke keeps its footprint, then
    thresholded at ``ink_coverage`` so the result stays binary. A
    threshold a little under half keeps a thin line continuous after a
    fourfold reduction. Returns uint8 with ink = 255.
    """
    if mask.shape[:2] == tuple(shape):
        return mask
    coverage = (mask > 0).astype(np.float32)
    coverage = cv2.resize(coverage, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return np.where(coverage >= ink_coverage, 255, 0).astype(np.uint8)


### ENTRY POINT ###

def working_copy(gray: np.ndarray, settings: Dict, image_name: str = '') -> WorkingCopy:
    """
    Measure a grayscale image and return it at its working resolution.

    With the feature off, or when no upscaling is needed, the image
    comes back unchanged at factor 1. When upscaling is needed but not
    possible (no contrib build, no model), the error is logged once per
    run and the image comes back at factor 1, so a run never stops for
    this and the manifest records what happened.
    """
    geometry = measure_line_geometry(gray, float(settings.get('dot_extent_in_stroke_widths', 4.0)))
    shape = gray.shape[:2]
    if not settings.get('enabled', True):
        return WorkingCopy(gray, 1, geometry, shape)
    factor = choose_factor(geometry, shape, settings)
    if factor == 1:
        logging.debug('%s: %s; no upscaling', image_name, geometry.describe())
        return WorkingCopy(gray, 1, geometry, shape)
    try:
        upscaled = upscale(gray, factor, settings.get('model', 'espcn'))
    except UpscalingUnavailable as exc:
        _report_unavailable(exc)
        return WorkingCopy(gray, 1, geometry, shape)
    logging.info('%s: %s; upscaled x%d for detection', image_name, geometry.describe(), factor)
    return WorkingCopy(upscaled, factor, geometry, shape)


_unavailable_reported = False


def _report_unavailable(exc: Exception) -> None:
    """Say once per run why upscaling did not happen."""
    global _unavailable_reported
    if not _unavailable_reported:
        logging.error('Upscaling is not possible, images are processed as they are. %s', exc)
        _unavailable_reported = True
    else:
        logging.debug('Upscaling skipped: %s', exc)
