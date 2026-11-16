"""
Arrow detection for PyLithics, in two stages.

An arrow drawn on a scar gives the direction of the flake removal. A
candidate contour is decided in two stages.

Stage 1 is the notch method: an arrow has two notches where its head
meets its shaft. The two deepest convexity defects of the contour give
the base of the head triangle, the half-space with the lower solidity
holds the head, its farthest point is the tip, and the triangle must be
tall enough. Its thresholds are calibrated for 300 DPI and scaled to
the image. What this stage accepts is an arrow, and its direction is the
direction from the triangle. Nothing behind it overrules it.

Stage 2 runs only on what stage 1 rejects: the small arrows whose notches
are a pixel deep, and the styles without a closed head. A candidate that
holds other ink, or that is too short to show a head, is refused first.
The rest is measured by its width profile on its real ink
(``arrow_profile``), in both directions, with the notch measurements of
stage 1 added as features, and a random forest scores each reading as
"an arrow pointing this way". The higher score gives the direction; it
must reach the threshold. Without the model, stage 1 alone decides, and
the run says so once.

Both stages work on the image they are given. When that is the working
copy of the drawing, upscaled by ``factor``, the thresholds of stage 1
are scaled by the factor and the lengths of stage 2 are measured in
source pixels, so the decision is the same one at a higher resolution.
The points returned are in the pixels of the image given.
"""

import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ...forest import Model, ModelUnavailable, load_forest, score_features
from ..config import get_arrow_detection_config
from . import arrow_profile as profile

DEFAULT_MODEL = 'arrow_classifier.json'
DEFAULT_THRESHOLD = 0.5
MIN_LENGTH = 6.0   # source pixels; a shorter mark cannot show a head

# Stage 1 thresholds, calibrated for 300 DPI.
REFERENCE_THRESHOLDS = {
    'min_area': 1,
    'min_defect_depth': 2,
    'solidity_bounds': (0.4, 1.0),
    'min_triangle_height': 8,
    'min_significant_defects': 2,
}

_models: Dict[str, Optional[Model]] = {}


def _load_once(settings: Dict[str, Any]) -> Optional[Model]:
    """Load the model named in the settings once per run; None when it cannot be used."""
    if not settings.get('enabled', True):
        return None
    name = settings.get('model') or DEFAULT_MODEL
    if name not in _models:
        try:
            _models[name] = load_forest(name)
        except ModelUnavailable as exc:
            logging.warning("%s. The notch method alone decides.", exc)
            _models[name] = None
    return _models[name]


class _DebugWriter:
    """No-op wrapper so debug code paths can call `.write` unconditionally."""

    def __init__(self, target):
        self._target = target

    def write(self, msg: str) -> None:
        if self._target is not None:
            self._target.write(msg)


@dataclass(frozen=True)
class Assessment:
    """Everything the detector found out about one contour."""

    forward: Dict[str, float]
    reverse: Dict[str, float]
    scores: Tuple[Optional[float], Optional[float]]    # model scores, None without a model
    rule: Tuple[bool, bool]                            # the hand rule, each orientation
    reversed: bool
    accepted: bool
    stage: str          # triangle, model, no model, or the veto: ink inside, too short
    geometry: Dict[str, Any]

    @property
    def score(self) -> Optional[float]:
        """Return the model score of the chosen orientation, None without a model."""
        return self.scores[1] if self.reversed else self.scores[0]

    @property
    def features(self) -> Dict[str, float]:
        """Return the features of the chosen orientation."""
        return self.reverse if self.reversed else self.forward

    def result(self) -> Dict[str, Any]:
        """Return the arrow as the pipeline records it: geometry, stage and score."""
        result = dict(self.geometry)
        result['stage'] = self.stage
        if self.stage == 'model':
            result['score'] = self.score
        return result


class ArrowDetector:
    """Decide whether a contour is an arrow, and which way it points."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, model: Optional[Model] = None):
        """
        Make a detector from the ``arrow_detection`` section of the configuration.

        ``model`` replaces the bundled forest, for tests; without it the
        forest named in ``classifier.model`` is loaded once per run.
        """
        self.config = config or get_arrow_detection_config()
        self.reference_dpi = self.config.get('reference_dpi', 300.0)
        self.ref_thresholds = dict(REFERENCE_THRESHOLDS)
        classifier = self.config.get('classifier', {}) or {}
        self.threshold = float(classifier.get('threshold', DEFAULT_THRESHOLD))
        self.model = model if model is not None else _load_once(classifier)
        self.debug_enabled = self.config.get('debug_enabled', False)

    def scale_parameters_for_dpi(
        self, image_dpi: Optional[float], factor: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Scale detection parameters based on image DPI.

        Parameters
        ----------
        image_dpi : float, optional
            DPI of the current image being processed
        factor : float, optional
            Pixels of the image worked on per source pixel; above 1 when
            the contour comes from an upscaled working copy.

        Returns
        -------
        dict
            Dictionary of scaled parameters
        """
        if image_dpi is None or image_dpi <= 0:
            logging.warning("The DPI value is not valid. The reference thresholds are used.")
            if factor == 1:
                return self.ref_thresholds.copy()
            image_dpi = self.reference_dpi

        # Scale factor relative to reference DPI
        linear_scale = image_dpi * factor / self.reference_dpi
        area_scale = linear_scale * linear_scale

        # Get scale factors from config
        area_safety = self.config.get('min_area_scale_factor', 0.7)
        depth_safety = self.config.get('min_defect_depth_scale_factor', 0.8)
        height_safety = self.config.get('min_triangle_height_scale_factor', 0.8)

        ref = self.ref_thresholds
        return {
            'min_area': ref['min_area'] * area_scale * area_safety,
            'min_defect_depth': ref['min_defect_depth'] * linear_scale * depth_safety,
            'solidity_bounds': ref['solidity_bounds'],
            'min_triangle_height': ref['min_triangle_height'] * linear_scale * height_safety,
            'min_significant_defects': ref['min_significant_defects'],
        }

    ### DECISION ###

    def analyze_contour_for_arrow(
        self,
        contour: np.ndarray,
        entry: Dict[str, Any],
        image: np.ndarray,
        image_dpi: Optional[float] = None,
        factor: float = 1.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Return the arrow in a contour (back, tip, angles, stage, score), or None.

        ``image`` is the binary image the contour came from, ink white;
        the debug files are drawn on it, and stage 2 measures its ink.
        ``factor`` is its pixels per source pixel. The points returned are
        in its pixels.
        """
        contour_id = entry.get('scar', 'unknown')
        debug_dir = _debug_prefix(entry, contour_id) if self.debug_enabled else None
        params = self.scale_parameters_for_dpi(image_dpi, factor)
        debug_log = (
            self._setup_debug_logging(debug_dir, contour_id, image_dpi, params)
            if debug_dir else None
        )
        try:
            assessment = self._assess(contour, image, params, debug_dir, debug_log, factor)
            if assessment is None:
                return None
            if debug_log:
                _write_decision(debug_log, assessment)
            return assessment.result() if assessment.accepted else None
        except Exception as e:
            if debug_log:
                debug_log.write(f"Error in arrow detection: {str(e)}\n")
            logging.error(f"Arrow detection error for {contour_id}: {e}")
            return None
        finally:
            if debug_log:
                debug_log.close()

    def assess(
        self, contour: np.ndarray, ink: Optional[np.ndarray] = None,
        image_dpi: Optional[float] = None, factor: float = 1.0,
    ) -> Optional[Assessment]:
        """
        Decide a contour both stages through; None for a degenerate contour.

        ``ink`` is the binary image the contour came from (ink white).
        Without it stage 2 measures the filled contour and cannot apply
        the ink-inside veto. ``factor`` is the pixels of ``ink`` per
        source pixel.
        """
        params = self.scale_parameters_for_dpi(image_dpi, factor)
        return self._assess(contour, ink, params, None, None, factor)

    def _assess(
        self, contour: np.ndarray, ink: Optional[np.ndarray], params: Dict[str, Any],
        debug_dir: Optional[str], debug_log, factor: float = 1.0,
    ) -> Optional[Assessment]:
        """Run stage 1, then stage 2 on a rejection; None for a degenerate contour."""
        measured = profile.measure_profile(contour, ink, factor)
        if measured is None:
            if debug_log:
                debug_log.write("Degenerate contour: nothing to measure\n")
            return None
        triangle = self._run_detection_pipeline(contour, ink, debug_dir, debug_log, params)
        structure = self._structure(contour, measured)
        forward = {**profile.profile_features(measured), **structure}
        reverse = {**profile.profile_features(measured, reverse=True), **structure}
        scores = self._scores(forward, reverse)
        rule = (profile.rule_says_arrow(forward), profile.rule_says_arrow(reverse))
        if triangle is not None:
            return Assessment(
                forward, reverse, scores, rule, _points_back(measured, triangle), True,
                'triangle', {key: triangle[key] for key in _GEOMETRY_KEYS},
            )
        assessment = self._run_classifier(contour, ink, measured, forward, reverse, scores, rule,
                                          _DebugWriter(debug_log))
        if debug_dir and ink is not None:
            cv2.imwrite(f"{debug_dir}.png", _draw(ink, contour, assessment))
        return assessment

    def _scores(
        self, forward: Dict[str, float], reverse: Dict[str, float]
    ) -> Tuple[Optional[float], Optional[float]]:
        """Return the model's score for each orientation, or None without a model."""
        if self.model is None:
            return (None, None)
        return (score_features(forward, self.model), score_features(reverse, self.model))

    def _run_classifier(
        self, contour: np.ndarray, ink: Optional[np.ndarray], measured: profile.Profile,
        forward: Dict[str, float], reverse: Dict[str, float],
        scores: Tuple[Optional[float], Optional[float]], rule: Tuple[bool, bool], log,
    ) -> Assessment:
        """Stage 2: the vetoes, then the forest, on a contour stage 1 rejected."""
        veto = self._veto(contour, ink, forward['length'])
        if self.model is None:
            direction = (
                profile.rule_direction_score(forward), profile.rule_direction_score(reverse)
            )
            reversed_, accepted, stage = direction[1] > direction[0], False, 'no model'
        else:
            reversed_ = scores[1] > scores[0]
            accepted = not veto and max(scores) >= self.threshold
            stage = veto or 'model'
        log.write(f"Stage 2: {'refused, ' + veto if veto else 'scored'}\n")
        return Assessment(
            forward, reverse, scores, rule, reversed_, accepted, stage,
            profile.arrow_geometry(measured, reversed_),
        )

    def _veto(self, contour: np.ndarray, ink: Optional[np.ndarray], length: float) -> str:
        """Name the reason a candidate is refused before scoring, or ``''``."""
        if length < MIN_LENGTH:
            return 'too short'
        if ink is not None and profile.ink_inside(contour, ink):
            return 'ink inside'
        return ''

    def _structure(self, contour: np.ndarray, measured: profile.Profile) -> Dict[str, float]:
        """
        Return the notch measurements of stage 1 as ratios, zero where it could not measure.

        Depths are over the typical ink width of the mark, the triangle
        height over its length. A notch is a defect deeper than half the
        width.
        """
        features = dict.fromkeys(profile.STRUCTURE_FEATURES, 0.0)
        defects = self._find_significant_defects(contour, 0.0) or []
        if not defects:
            return features
        present = measured.span[measured.span > 0]
        width = max(float(np.median(present)) if len(present) else 0.0, 0.5)
        depths = [defect[3] for defect in defects]
        features['notches'] = float(sum(1 for depth in depths if depth > 0.5 * width))
        features['notch_depth_1'] = depths[0] / width
        features['notch_depth_2'] = depths[1] / width if len(depths) > 1 else 0.0
        base = self._identify_triangle_base(defects[:2])
        halves = self._analyze_halfspaces(contour, base) if base else None
        if halves is None:
            return features
        _, tip_side, first, second = halves
        features['shaft_solidity'] = max(first, second)
        features['head_solidity'] = min(first, second)
        tip = self._find_triangle_tip(self._divide_contour_points(contour, base)[tip_side], base[2])
        if tip is not None:
            height = math.hypot(tip[0] - base[2][0], tip[1] - base[2][1])
            features['triangle_height'] = height / max(measured.length, 1.0)
        return features

    ### STAGE 1: THE NOTCH METHOD ###

    def _run_detection_pipeline(
        self,
        contour: np.ndarray,
        image: np.ndarray,
        debug_dir: Optional[str],
        debug_log,
        params: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Run the 5-step detection pipeline and return arrow data or None."""
        if not self._validate_basic_properties(contour, params, debug_log):
            return None

        significant_defects = self._find_significant_defects(
            contour, params['min_defect_depth'],
        )
        if not self._validate_defects(significant_defects, params, debug_log):
            return None

        triangle_data = self._analyze_triangle_structure(
            contour, significant_defects, params, debug_log,
        )
        if triangle_data is None:
            return None

        arrow_data = self._calculate_arrow_properties(triangle_data, debug_log)

        if debug_dir and image is not None:
            self._create_debug_visualizations(
                contour, triangle_data, arrow_data, image, debug_dir,
            )

        if debug_log:
            debug_log.write("Arrow detection succeeded\n")
        return arrow_data

    def _setup_debug_logging(self,
                           debug_dir: str,
                           contour_id: str,
                           image_dpi: Optional[float],
                           params: Dict[str, Any]):
        """Open the debug log ``<prefix>.txt`` for one contour."""
        os.makedirs(os.path.dirname(debug_dir), exist_ok=True)
        debug_log = open(f"{debug_dir}.txt", 'w')

        debug_log.write(f"Arrow detection analysis for contour {contour_id}\n")
        if image_dpi:
            debug_log.write(f"Image DPI: {image_dpi}, scaling applied\n")

        debug_log.write("Scaled parameters:\n")
        for key, value in params.items():
            debug_log.write(f"  {key}={value}\n")

        return debug_log

    def _validate_basic_properties(self,
                                 contour: np.ndarray,
                                 params: Dict[str, Any],
                                 debug_log) -> bool:
        """Validate basic contour properties."""
        # Area test
        area = cv2.contourArea(contour)
        min_area = params['min_area']

        if debug_log:
            debug_log.write(f"Area test: {area:.2f} >= {min_area:.2f}\n")

        if area < min_area:
            if debug_log:
                debug_log.write("Failed: Area too small\n")
            return False

        # Solidity test
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0

        solidity_bounds = params['solidity_bounds']

        if debug_log:
            debug_log.write(f"Solidity test: {solidity:.3f} in range {solidity_bounds}\n")

        if solidity < solidity_bounds[0] or solidity > solidity_bounds[1]:
            if debug_log:
                debug_log.write("Failed: Solidity outside acceptable range\n")
            return False

        if debug_log:
            debug_log.write("Basic property tests: PASSED\n")

        return True

    def _find_significant_defects(self,
                                contour: np.ndarray,
                                min_defect_depth: float) -> Optional[List[Tuple]]:
        """Find significant convexity defects in the contour."""
        try:
            hull_indices = cv2.convexHull(contour, returnPoints=False)
            if hull_indices is None or len(hull_indices) < 3:
                return None

            defects = cv2.convexityDefects(contour, hull_indices)
            if defects is None or defects.shape[0] < 2:
                return None

            significant_defects = []
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                depth = d / 256.0
                if depth > min_defect_depth:
                    start = tuple(contour[s][0])
                    end = tuple(contour[e][0])
                    far = tuple(contour[f][0])
                    significant_defects.append((start, end, far, depth))

            # Sort by depth (deepest first)
            significant_defects.sort(
                key=lambda x: x[3], reverse=True
            )
            return significant_defects

        except (cv2.error, ValueError, IndexError):
            return None

    def _validate_defects(self,
                        significant_defects: Optional[List[Tuple]],
                        params: Dict[str, Any],
                        debug_log) -> bool:
        """Validate the found defects."""
        min_defects = params['min_significant_defects']

        if not significant_defects or len(significant_defects) < min_defects:
            if debug_log:
                count = len(significant_defects) if significant_defects else 0
                debug_log.write(f"Failed: Not enough defects ({count} < {min_defects})\n")
            return False

        if debug_log:
            depths = [d[3] for d in significant_defects]
            debug_log.write(f"Defect test: PASSED (found {len(significant_defects)} defects)\n")
            debug_log.write(f"Defect depths: {depths}\n")

        return True

    def _analyze_triangle_structure(self,
                                  contour: np.ndarray,
                                  significant_defects: List[Tuple],
                                  params: Dict[str, Any],
                                  debug_log) -> Optional[Dict[str, Any]]:
        """Analyze the triangle structure from defects."""
        log = _DebugWriter(debug_log)
        significant_defects = significant_defects[:2]

        triangle_base_info = self._identify_triangle_base(significant_defects)
        if triangle_base_info is None:
            log.write("Failed: Could not identify triangle base\n")
            return None
        base_p1, base_p2, base_midpoint, base_length = triangle_base_info
        log.write(f"Triangle base length: {base_length:.2f} pixels\n")

        halfspace_results = self._analyze_halfspaces(contour, triangle_base_info)
        if halfspace_results is None:
            log.write("Failed: Half-space analysis failed\n")
            return None
        shaft_halfspace, tip_halfspace, solidity1, solidity2 = halfspace_results
        log.write(f"Half-space solidities: {solidity1:.3f}, {solidity2:.3f}\n")
        log.write(f"Shaft: {shaft_halfspace}, Tip: {tip_halfspace}\n")

        halfspace_points = self._divide_contour_points(contour, triangle_base_info)
        triangle_tip = self._find_triangle_tip(
            halfspace_points[tip_halfspace], base_midpoint,
        )
        if triangle_tip is None:
            log.write("Failed: Could not find triangle tip\n")
            return None

        triangle_height = np.sqrt(
            (triangle_tip[0] - base_midpoint[0]) ** 2
            + (triangle_tip[1] - base_midpoint[1]) ** 2
        )
        min_height = params['min_triangle_height']
        log.write(
            f"Triangle height test: {triangle_height:.2f} >= {min_height:.2f}\n"
        )
        if triangle_height < min_height:
            log.write("Failed: Triangle height too small\n")
            return None

        log.write("Triangle analysis: PASSED\n")
        return {
            'base_p1': base_p1,
            'base_p2': base_p2,
            'base_midpoint': base_midpoint,
            'triangle_tip': triangle_tip,
            'triangle_height': triangle_height,
            'significant_defects': significant_defects,
        }

    def _calculate_arrow_properties(self,
                                  triangle_data: Dict[str, Any],
                                  debug_log) -> Dict[str, Any]:
        """Calculate final arrow properties."""
        base_midpoint = triangle_data['base_midpoint']
        triangle_tip = triangle_data['triangle_tip']

        # Arrow direction: from triangle tip to base midpoint
        arrow_back = triangle_tip
        arrow_tip = base_midpoint

        # Calculate angles
        dx = arrow_tip[0] - arrow_back[0]
        dy = arrow_tip[1] - arrow_back[1]

        angle_rad = math.atan2(dy, dx)
        angle_deg = (math.degrees(angle_rad) + 360) % 360
        compass_angle = (270 + angle_deg) % 360

        if debug_log:
            debug_log.write(f"Arrow direction: from {arrow_back} to {arrow_tip}\n")
            debug_log.write(f"Angles - deg: {angle_deg:.2f}°, compass: {compass_angle:.2f}°\n")

        return {
            'arrow_back': arrow_back,
            'arrow_tip': arrow_tip,
            'angle_rad': round(angle_rad, 1),
            'angle_deg': round(angle_deg, 1),
            'compass_angle': round(compass_angle, 1),
            'triangle_data': triangle_data
        }

    def _identify_triangle_base(self, defects: List[Tuple]) -> Optional[Tuple]:
        """Identify the triangle base from defect pairs."""
        defect_points = [defect[2] for defect in defects]  # 'far' points

        if len(defect_points) < 2:
            return None

        # Get the pair with largest distance as base
        p1, p2 = defect_points[0], defect_points[1]
        distance = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        base_midpoint = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

        return (p1, p2, base_midpoint, distance)

    def _divide_contour_points(self,
                             contour: np.ndarray,
                             triangle_base_info: Tuple) -> Dict[int, List]:
        """Divide contour points into half-spaces."""
        base_p1, base_p2, base_midpoint, _ = triangle_base_info

        base_vector = np.array([base_p2[0] - base_p1[0], base_p2[1] - base_p1[1]])
        normal = np.array([-base_vector[1], base_vector[0]])

        halfspace_points = defaultdict(list)

        for point in contour:
            p = point[0]
            vec = np.array([p[0] - base_midpoint[0], p[1] - base_midpoint[1]])
            side = np.dot(vec, normal)

            if side > 0:
                halfspace_points[1].append(p)
            else:
                halfspace_points[2].append(p)

        return halfspace_points

    def _analyze_halfspaces(self,
                          contour: np.ndarray,
                          triangle_base_info: Tuple) -> Optional[Tuple]:
        """Analyze solidity of each half-space."""
        halfspace_points = self._divide_contour_points(contour, triangle_base_info)

        if len(halfspace_points[1]) < 5 or len(halfspace_points[2]) < 5:
            return None

        try:
            # Calculate solidity for each half-space
            solidities = {}
            for i in [1, 2]:
                points = np.array(halfspace_points[i], dtype=np.int32).reshape(-1, 1, 2)
                hull = cv2.convexHull(points)

                area = cv2.contourArea(points)
                hull_area = cv2.contourArea(hull)

                solidities[i] = area / hull_area if hull_area > 0 else 0

            # More solid half-space is likely the shaft
            if solidities[1] > solidities[2]:
                return (1, 2, solidities[1], solidities[2])  # shaft, tip, sol1, sol2
            else:
                return (2, 1, solidities[1], solidities[2])

        except (cv2.error, ValueError, ZeroDivisionError):
            return None

    def _find_triangle_tip(self,
                         tip_halfspace_points: List,
                         base_midpoint: Tuple) -> Optional[Tuple]:
        """Find the triangle tip (furthest point from base midpoint)."""
        if not tip_halfspace_points:
            return None

        max_dist = -1
        tip_point = None

        for p in tip_halfspace_points:
            dist = np.sqrt((p[0] - base_midpoint[0])**2 + (p[1] - base_midpoint[1])**2)
            if dist > max_dist:
                max_dist = dist
                tip_point = tuple(p)

        return tip_point

    def _create_debug_visualizations(
        self,
        contour: np.ndarray,
        triangle_data: Dict[str, Any],
        arrow_data: Dict[str, Any],
        image: np.ndarray,
        debug_dir: str
    ) -> None:
        """Create debug visualization images."""
        if len(image.shape) == 3:
            vis = image.copy()
        else:
            vis = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)

        # Draw contour
        cv2.drawContours(vis, [contour], 0, (0, 255, 0), 2)

        # Draw hull
        hull = cv2.convexHull(contour)
        cv2.drawContours(vis, [hull], 0, (0, 0, 255), 1)

        # Draw defects
        for defect in triangle_data['significant_defects']:
            start, end, far, _ = defect
            cv2.circle(vis, far, 5, (255, 0, 0), -1)

        # Draw triangle base
        base_p1, base_p2 = triangle_data['base_p1'], triangle_data['base_p2']
        cv2.line(vis, base_p1, base_p2, (255, 0, 255), 2)

        # Draw triangle tip
        triangle_tip = triangle_data['triangle_tip']
        cv2.circle(vis, triangle_tip, 5, (0, 255, 255), -1)

        # Draw arrow
        arrow_back, arrow_tip = arrow_data['arrow_back'], arrow_data['arrow_tip']
        cv2.arrowedLine(vis, arrow_back, arrow_tip, (0, 255, 0), 2)

        # Draw angle text
        compass_angle = arrow_data['compass_angle']
        text_pos = (arrow_tip[0] + 10, arrow_tip[1] - 10)
        cv2.putText(vis, f"{compass_angle:.1f}°", text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(vis, f"{compass_angle:.1f}°", text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        # Save visualization as <prefix>.png, beside the <prefix>.txt log
        os.makedirs(os.path.dirname(debug_dir), exist_ok=True)
        cv2.imwrite(f"{debug_dir}.png", vis)


### HELPERS ###

_GEOMETRY_KEYS = ('arrow_back', 'arrow_tip', 'angle_rad', 'angle_deg', 'compass_angle')


def _write_decision(debug_log, assessment: Assessment) -> None:
    """Write the decision line and the features of the chosen orientation."""
    debug_log.write(f"arrow: {'yes' if assessment.accepted else 'no'}; "
                    f"stage {assessment.stage}; scores {assessment.scores}; "
                    f"rule {assessment.rule}; compass {assessment.geometry['compass_angle']}\n")
    for name, value in assessment.features.items():
        debug_log.write(f"  {name}={value:.3f}\n")


def _points_back(measured: profile.Profile, triangle: Dict[str, Any]) -> bool:
    """Say whether the triangle's direction runs from the profile's last end to its first."""
    (first_x, first_y), (last_x, last_y) = measured.ends
    back, tip = triangle['arrow_back'], triangle['arrow_tip']
    along = (last_x - first_x) * (tip[0] - back[0]) + (last_y - first_y) * (tip[1] - back[1])
    return along < 0


def _draw(image: np.ndarray, contour: np.ndarray, assessment: Assessment):
    """Return the image with the contour and the stage 2 arrow drawn on it."""
    canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    colour = (0, 200, 0) if assessment.accepted else (0, 140, 255)
    result = assessment.geometry
    cv2.drawContours(canvas, [contour], 0, (200, 200, 200), 1)
    cv2.arrowedLine(canvas, result['arrow_back'], result['arrow_tip'], colour, 2, tipLength=0.3)
    text = f"{result['compass_angle']:.0f}"
    if assessment.score is not None:
        text += f" ({assessment.score:.2f})"
    position = (result['arrow_tip'][0] + 6, result['arrow_tip'][1] - 6)
    cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return canvas


def _debug_prefix(entry: Dict[str, Any], contour_id: str) -> Optional[str]:
    """
    Build ``<debug_dir>/<contour>`` for this contour's debug files.

    The image's folder is ``results/arrow_debug/<image>/``. Each
    contour gets ``<contour>.png`` and ``<contour>.txt`` in it, so a
    page of scars does not overwrite one file.
    """
    debug_dir = entry.get('debug_dir')
    if not debug_dir:
        return None
    return os.path.join(debug_dir, str(contour_id).replace(' ', '_'))


def analyze_child_contour_for_arrow(
    contour: np.ndarray,
    entry: Dict[str, Any],
    image: np.ndarray,
    image_dpi: Optional[float] = None,
    factor: float = 1.0,
) -> Optional[Dict[str, Any]]:
    """
    Detect an arrow within a contour with a detector made from the current configuration.

    Module-level convenience function used by arrow_integration. The
    detector reads the configuration each time, so a change such as
    ``--arrow_debug`` takes effect; the model itself is read once per run.
    """
    return ArrowDetector().analyze_contour_for_arrow(contour, entry, image, image_dpi, factor)
