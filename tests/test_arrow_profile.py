"""Tests for the width profile behind arrow detection."""

import math

import numpy as np
import pytest

from pylithics.image_processing.modules import arrow_profile as ap


# A filled arrow pointing up: a triangle head on a rectangular shaft.
_ARROW = np.array(
    [[50, 10], [70, 30], [60, 30], [60, 60], [40, 60], [40, 30], [30, 30]], dtype=float
)
# The same arrow with a dot at the tail, drawn as a polygon.
_DOT = np.array([[50 + 8 * math.cos(a), 68 + 8 * math.sin(a)]
                 for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)])


def _contour(points, degrees=0.0, scale=1.0):
    """Rotate and scale a polygon about its centre; return an OpenCV contour."""
    centre = points.mean(axis=0)
    angle = math.radians(degrees)
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    moved = (points - centre) @ rotation.T * scale + centre + 100
    return np.round(moved).astype(np.int32).reshape(-1, 1, 2)


def _direction(contour):
    """Return the compass angle of the orientation the rule prefers."""
    profile = ap.measure_profile(contour)
    forward, reverse = ap.profile_features(profile), ap.profile_features(profile, True)
    flipped = ap.rule_direction_score(reverse) > ap.rule_direction_score(forward)
    return ap.arrow_geometry(profile, flipped)['compass_angle']


@pytest.mark.unit
class TestMeasure:
    def test_the_arrow_is_measured_in_source_pixels(self):
        profile = ap.measure_profile(_contour(_ARROW))
        assert profile.length == pytest.approx(50, abs=1.5)
        assert len(profile.span) == ap.BINS
        assert profile.span.max() == pytest.approx(40, abs=2)
        assert profile.area == pytest.approx(1000, rel=0.1)

    def test_a_degenerate_contour_gives_none(self):
        assert ap.measure_profile(np.zeros((2, 1, 2), dtype=np.int32)) is None
        speck = np.array([[5, 5], [5, 5], [5, 5]], dtype=np.int32).reshape(-1, 1, 2)
        assert ap.measure_profile(speck) is None

    def test_a_hairline_is_still_measured(self):
        line = np.array([[0, 0], [10, 0], [20, 0]], dtype=np.int32).reshape(-1, 1, 2)
        profile = ap.measure_profile(line)
        assert profile is not None and profile.length == pytest.approx(20, abs=1)


@pytest.mark.unit
class TestFeatures:
    def test_the_head_end_shows_a_wide_tapering_head(self):
        profile = ap.measure_profile(_contour(_ARROW))
        tip_last = ap.profile_features(profile, reverse=True)
        assert tip_last['head_ratio'] > 1.5
        assert tip_last['tip_taper'] < 0.3
        assert tip_last['head_pos'] > 0.5
        assert tip_last['span_0'] == pytest.approx(1.0, abs=0.15)
        assert tip_last['span_9'] < tip_last['span_7']

    def test_reading_from_the_other_end_swaps_tail_and_head(self):
        profile = ap.measure_profile(_contour(_ARROW))
        forward, reverse = ap.profile_features(profile), ap.profile_features(profile, True)
        head_last = reverse if reverse['tip_taper'] < forward['tip_taper'] else forward
        head_first = forward if head_last is reverse else reverse
        assert head_last['tail_end'] == pytest.approx(1.0, abs=0.15)
        assert head_first['tail_end'] < 0.5
        assert forward['asymmetry'] == pytest.approx(1.0 / reverse['asymmetry'], rel=0.05)
        assert set(forward) == set(ap.FEATURES)

    def test_a_dot_at_the_tail_does_not_win_the_direction(self):
        profile = ap.measure_profile(_contour(np.vstack([_ARROW, _DOT])))
        forward, reverse = ap.profile_features(profile), ap.profile_features(profile, True)
        head_end = reverse if reverse['tip_taper'] < forward['tip_taper'] else forward
        dot_end = forward if head_end is reverse else reverse
        assert ap.rule_direction_score(head_end) > ap.rule_direction_score(dot_end)


@pytest.mark.unit
class TestDirection:
    @pytest.mark.parametrize('degrees, compass', [(0, 180), (90, 270), (180, 0), (270, 90)])
    def test_the_compass_angle_follows_the_rotation(self, degrees, compass):
        assert _direction(_contour(_ARROW, degrees)) == pytest.approx(compass, abs=3)

    def test_a_small_arrow_still_points_the_right_way(self):
        assert _direction(_contour(_ARROW, 0, 0.3)) == pytest.approx(180, abs=5)

    def test_geometry_gives_integer_ends_and_the_csv_frame(self):
        profile = ap.measure_profile(_contour(_ARROW))
        geometry = ap.arrow_geometry(profile, reverse=True)
        assert all(isinstance(v, int) for v in geometry['arrow_tip'] + geometry['arrow_back'])
        assert geometry['arrow_tip'][1] < geometry['arrow_back'][1]
        assert geometry['compass_angle'] == pytest.approx(180, abs=3)
        assert geometry['angle_deg'] == pytest.approx(270, abs=3)


@pytest.mark.unit
class TestRule:
    def test_a_plain_bar_is_not_an_arrow(self):
        bar = np.array([[0, 0], [60, 0], [60, 8], [0, 8]], dtype=float)
        profile = ap.measure_profile(_contour(bar))
        assert not ap.rule_says_arrow(ap.profile_features(profile))
        assert not ap.rule_says_arrow(ap.profile_features(profile, True))

    def test_the_rule_wants_a_head_and_a_shaft(self):
        features = {
            'head_ratio': 2.5, 'aspect': 4.0, 'tip_taper': 0.2, 'head_pos': 0.8,
            'asymmetry': 2.0,
        }
        assert ap.rule_says_arrow(features)
        assert not ap.rule_says_arrow({**features, 'head_ratio': 1.0})
        assert not ap.rule_says_arrow({**features, 'tip_taper': 0.95})
