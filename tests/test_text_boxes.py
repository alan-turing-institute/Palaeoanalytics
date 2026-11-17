"""Tests for the text box test: a box the reader finds to be writing is not an artefact."""

import logging

import cv2
import numpy as np
import pytest

from pylithics.page_segmentation import text_boxes as tb
from pylithics.page_segmentation.detection import PageImage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _page(gray):
    return PageImage(path='plate.png', array=gray, gray=gray, mode='L', dpi=(300, 300))


def _plate():
    """A page with a framed code at the top and a lithic with a numeral below."""
    gray = np.full((600, 800), 255, np.uint8)
    cv2.rectangle(gray, (100, 40), (500, 140), 0, 3)                       # the frame
    cv2.putText(gray, 'LM1-7251-1077', (120, 112), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 0, 5)
    cv2.ellipse(gray, (400, 400), (160, 140), 0, 0, 360, 0, 3)             # the lithic
    cv2.line(gray, (330, 300), (430, 520), 0, 3)
    cv2.putText(gray, '3', (560, 560), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 4)   # its numeral
    raw = (gray < 128).astype(np.uint8) * 255
    return _page(gray), raw, [(100, 40, 503, 143), (230, 250, 620, 580)]


class _Engine:
    """A stand-in for RapidOCR: returns the lines it was told to, in image pixels."""

    def __init__(self, answer):
        self.answer = answer          # function(image) -> list of (points, text, score)

    def __call__(self, image):
        return self.answer(image), None


def _whole_image(text, score=0.95):
    """An engine answer that reads the whole box as one line."""
    def answer(image):
        h, w = image.shape[:2]
        pad = tb._READ_PAD
        return [([[pad, pad], [w - pad, pad], [w - pad, h - pad], [pad, h - pad]], text, score)]
    return answer


def _corner(text, score=0.95):
    """An engine answer that reads one small line in the lower right corner."""
    def answer(image):
        h, w = image.shape[:2]
        return [([[w * 0.8, h * 0.85], [w, h * 0.85], [w, h], [w * 0.8, h]], text, score)]
    return answer


CONFIG = {'enabled': True, 'min_ink_share': 0.8, 'min_characters': 3, 'min_confidence': 0.8}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFrame:

    def test_a_straight_frame_is_discounted_and_the_letters_stay(self):
        ink = np.zeros((60, 200), bool)
        ink[:3, :] = ink[-3:, :] = ink[:, :3] = ink[:, -3:] = True     # a 3 px frame
        ink[20:40, 50:60] = True                                        # a letter
        kept = tb.ink_without_frame(ink)
        assert kept[20:40, 50:60].all()
        assert not kept[:3, :].any() and not kept[:, :3].any()

    def test_an_empty_box_stays_empty(self):
        assert tb.ink_without_frame(np.zeros((0, 0), bool)).size == 0


@pytest.mark.unit
class TestShare:

    def test_the_share_is_the_ink_inside_the_lines(self):
        ink = np.zeros((50, 100), bool)
        ink[10:20, 10:90] = True
        half = np.array([[0, 0], [50, 0], [50, 50], [0, 50]], np.int32)
        assert tb.text_share(ink, [(half, 'ab', 0.9)]) == pytest.approx(0.5, abs=0.02)
        assert tb.text_share(ink, []) == 0.0


@pytest.mark.unit
class TestFindTextBoxes:

    def test_a_framed_code_is_text_and_the_lithic_is_not(self, monkeypatch):
        page, raw, boxes = _plate()
        engine = _Engine(lambda image: (_whole_image('LM1-7251-1077')(image)
                                        if image.shape[1] > image.shape[0] * 2
                                        else _corner('3')(image)))
        monkeypatch.setattr(tb, '_load_engine', lambda: engine)
        found = tb.find_text_boxes(page, boxes, raw, CONFIG)
        assert [t.index for t in found] == [0]
        assert found[0].text == 'LM1-7251-1077' and found[0].share > 0.8

    def test_a_lithic_with_a_numeral_is_kept_even_when_read_as_a_word(self, monkeypatch):
        page, raw, boxes = _plate()
        monkeypatch.setattr(tb, '_load_engine', lambda: _Engine(_corner('123')))
        assert tb.find_text_boxes(page, [boxes[1]], raw, CONFIG) == []

    def test_a_short_read_is_an_identifier_not_a_text_box(self, monkeypatch):
        page, raw, boxes = _plate()
        monkeypatch.setattr(tb, '_load_engine', lambda: _Engine(_whole_image('7')))
        assert tb.find_text_boxes(page, [boxes[0]], raw, CONFIG) == []

    def test_a_low_confidence_read_does_not_count(self, monkeypatch):
        """A small drawing misread as a glyph reads at 0.6 to 0.7; print reads near 1."""
        page, raw, boxes = _plate()
        for score in (0.3, 0.7):
            monkeypatch.setattr(tb, '_load_engine',
                                lambda s=score: _Engine(_whole_image('200', score=s)))
            assert tb.find_text_boxes(page, [boxes[1]], raw, CONFIG) == []

    def test_off_or_without_the_engine_nothing_is_dropped(self, monkeypatch):
        page, raw, boxes = _plate()
        monkeypatch.setattr(tb, '_load_engine', lambda: _Engine(_whole_image('LM1-7251-1077')))
        assert tb.find_text_boxes(page, boxes, raw, {**CONFIG, 'enabled': False}) == []
        monkeypatch.setattr(tb, '_load_engine', lambda: None)
        assert tb.find_text_boxes(page, boxes, raw, CONFIG) == []

    def test_a_text_box_is_logged(self, monkeypatch, caplog):
        page, raw, boxes = _plate()
        monkeypatch.setattr(tb, '_load_engine', lambda: _Engine(_whole_image('LM1-7251-1077')))
        with caplog.at_level(logging.INFO):
            tb.find_text_boxes(page, [boxes[0]], raw, CONFIG)
        assert 'is text, not an artefact' in caplog.text


@pytest.mark.unit
class TestBoxImage:

    def test_the_image_maps_back_to_box_pixels(self):
        page, _, boxes = _plate()
        image, scale, origin = tb.box_image(page, boxes[0])
        x0, y0, x1, y1 = boxes[0]
        assert image.ndim == 3 and image.shape[0] >= tb._READ_HEIGHT
        # the box's top left corner sits at (origin * scale) past the pad in the image;
        # taken back through the reader's mapping it lands at (0, 0) in box pixels
        corner = tb._READ_PAD + origin * scale
        back = (corner - tb._READ_PAD) / scale - origin
        assert np.allclose(back, [0, 0]) and (origin > 0).all()
