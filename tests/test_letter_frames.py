"""Tests for taking out a thin frame drawn around a letter."""

import cv2
import numpy as np
import pytest

from pylithics.image_processing.modules.letter_frames import remove_letter_frames


def _face(frame_line=1, letter=True, frame=True):
    """A face with a thick outline and a thick ridge; a rectangle hangs from the ridge, holding a C.

    As on page 31 of the test project, the ridge runs along the frame's top side.
    """
    ink = np.zeros((300, 400), np.uint8)
    cv2.rectangle(ink, (20, 20), (380, 280), 255, 8)                 # the face outline
    cv2.line(ink, (20, 120), (380, 120), 255, 8)                      # a ridge across it
    if frame:
        cv2.rectangle(ink, (150, 120), (230, 210), 255, frame_line)   # top side on the ridge
    if letter:
        cv2.ellipse(ink, (190, 190), (10, 12), 0, 40, 320, 255, 3)   # a C inside the frame
    return ink


@pytest.mark.unit
class TestLetterFrames:

    def test_a_thin_frame_around_a_letter_is_taken_out(self):
        ink = _face()
        cleaned, _, frames = remove_letter_frames(ink, None, 1, {})
        assert len(frames) == 1
        assert cleaned[210, 160:225].max() == 0            # the frame's bottom side is gone
        assert cleaned[140:205, 150].max() == 0            # its left and right sides too
        assert cleaned[140:205, 230].max() == 0
        assert cleaned[178:203, 178:186].max() == 255      # the letter stays
        assert cleaned[120, 30:370].min() == 255           # the ridge is whole

    def test_a_thick_rectangle_is_a_scar_and_stays(self):
        ink = _face(frame_line=8)
        cleaned, _, frames = remove_letter_frames(ink, None, 1, {})
        assert frames == [] and np.array_equal(cleaned, ink)

    def test_an_empty_thin_rectangle_stays(self):
        ink = _face(letter=False)
        _, _, frames = remove_letter_frames(ink, None, 1, {})
        assert frames == []

    def test_switched_off_nothing_changes(self):
        ink = _face()
        cleaned, _, frames = remove_letter_frames(ink, None, 1, {'enabled': False})
        assert frames == [] and cleaned is ink

    def test_the_working_copy_loses_the_same_frame(self):
        ink = _face()
        working = cv2.resize(ink, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        _, cleaned_working, frames = remove_letter_frames(ink, working, 2, {})
        assert len(frames) == 1
        assert cleaned_working[420, 320:450].max() == 0                  # the bottom side, doubled
        assert cleaned_working[356:406, 356:372].max() == 255            # the letter stays
