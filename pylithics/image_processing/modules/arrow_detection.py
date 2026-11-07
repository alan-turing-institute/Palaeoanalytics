"""
Arrow detection for PyLithics.

An arrow drawn on a scar gives the direction of the flake removal. A
candidate blob is measured by its width profile along its long axis
(``arrow_profile``): a thin, even shaft and a head that widens and then
narrows to a point. The profile is read in both directions, and a
random forest trained on the project's labelled arrows scores each
reading as "an arrow pointing this way". The higher score gives the
direction; it must reach the threshold for the blob to be an arrow.
Without the model (off in the configuration, or the file missing) a
hand rule on the same features decides, and the run says so once.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from ...forest import Model, ModelUnavailable, load_forest, score_features
from ..config import get_arrow_detection_config
from . import arrow_profile as profile

DEFAULT_MODEL = 'arrow_classifier.json'
DEFAULT_THRESHOLD = 0.5

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
            logging.warning("%s. The arrow rule decides for every scar.", exc)
            _models[name] = None
    return _models[name]


@dataclass(frozen=True)
class Assessment:
    """Everything the detector found out about one contour."""

    forward: Dict[str, float]
    reverse: Dict[str, float]
    scores: Tuple[Optional[float], Optional[float]]    # model scores, None without a model
    rule: Tuple[bool, bool]                            # the hand rule, each orientation
    reversed: bool
    accepted: bool
    geometry: Dict[str, Any]

    @property
    def score(self) -> Optional[float]:
        """Return the model score of the chosen orientation, None without a model."""
        return self.scores[1] if self.reversed else self.scores[0]

    @property
    def features(self) -> Dict[str, float]:
        """Return the features of the chosen orientation."""
        return self.reverse if self.reversed else self.forward


class ArrowDetector:
    """Decide whether a contour is an arrow, and which way it points."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, model: Optional[Model] = None):
        """
        Make a detector from the ``arrow_detection`` section of the configuration.

        ``model`` replaces the bundled forest, for tests; without it the
        forest named in ``classifier.model`` is loaded once per run.
        """
        self.config = config or get_arrow_detection_config()
        classifier = self.config.get('classifier', {}) or {}
        self.threshold = float(classifier.get('threshold', DEFAULT_THRESHOLD))
        self.model = model if model is not None else _load_once(classifier)
        self.debug_enabled = self.config.get('debug_enabled', False)

    def analyze_contour_for_arrow(
        self,
        contour: np.ndarray,
        entry: Dict[str, Any],
        image: np.ndarray,
        image_dpi: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Return the arrow in a contour (back, tip, angles, score), or None.

        ``image_dpi`` is accepted for the callers that pass it; the profile
        is measured in pixels and needs no DPI.
        """
        del image_dpi
        assessment = self.assess(contour)
        if assessment is None:
            return None
        if self.debug_enabled:
            self._write_debug(entry, contour, image, assessment)
        if not assessment.accepted:
            return None
        result = dict(assessment.geometry)
        result['score'] = assessment.score
        return result

    def assess(self, contour: np.ndarray) -> Optional[Assessment]:
        """Measure a contour both ways and decide; None for a degenerate contour."""
        measured = profile.measure_profile(contour)
        if measured is None:
            return None
        forward = profile.profile_features(measured)
        reverse = profile.profile_features(measured, reverse=True)
        rule = (profile.rule_says_arrow(forward), profile.rule_says_arrow(reverse))
        if self.model is not None:
            scores = (score_features(forward, self.model), score_features(reverse, self.model))
            reversed_ = scores[1] > scores[0]
            accepted = max(scores) >= self.threshold
        else:
            scores = (None, None)
            direction = (
                profile.rule_direction_score(forward), profile.rule_direction_score(reverse)
            )
            reversed_ = direction[1] > direction[0]
            accepted = rule[1] if reversed_ else rule[0]
        return Assessment(
            forward, reverse, scores, rule, reversed_, accepted,
            profile.arrow_geometry(measured, reversed_),
        )

    def _write_debug(
        self, entry: Dict[str, Any], contour: np.ndarray, image: np.ndarray,
        assessment: Assessment,
    ) -> None:
        """Write ``<prefix>.txt`` with the features and ``<prefix>.png`` with the arrow."""
        prefix = _debug_prefix(entry, entry.get('scar', 'unknown'))
        if not prefix:
            return
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        geometry = assessment.geometry
        with open(f"{prefix}.txt", 'w', encoding='utf-8') as handle:
            handle.write(f"arrow: {'yes' if assessment.accepted else 'no'}; "
                         f"scores {assessment.scores}; rule {assessment.rule}; "
                         f"compass {geometry['compass_angle']}\n")
            for name, value in assessment.features.items():
                handle.write(f"  {name}={value:.3f}\n")
        if image is not None:
            cv2.imwrite(f"{prefix}.png", _draw(image, contour, assessment))


def _draw(image: np.ndarray, contour: np.ndarray, assessment: Assessment):
    """Return the image with the contour and the arrow drawn on it."""
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
) -> Optional[Dict[str, Any]]:
    """
    Detect an arrow within a contour with a detector made from the current configuration.

    Module-level convenience function used by arrow_integration. The
    detector reads the configuration each time, so a change such as
    ``--arrow_debug`` takes effect; the model itself is read once per run.
    """
    return ArrowDetector().analyze_contour_for_arrow(contour, entry, image, image_dpi)
