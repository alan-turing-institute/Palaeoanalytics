"""Tests for hiding the page's text in an artefact crop."""

import numpy as np
import pytest

from pylithics.page_segmentation import hiding
from pylithics.page_segmentation.detection import Component
from pylithics.page_segmentation.grouping import Classified
from pylithics.page_segmentation.identifiers import Identifier, PageReads

PAPER = 230   # a grey scan's paper


def _component(box, fill=True):
    """A component whose mask is its whole box (or a hollow frame)."""
    w, h = box[2] - box[0], box[3] - box[1]
    mask = np.ones((h, w), bool)
    if not fill:
        mask[1:-1, 1:-1] = False
    return Component(box=list(box), width=w, height=h, ink=int(mask.sum()), mask=mask)


def _page():
    """A grey page: a hollow lithic, its identifier, a caption, a legend key, a glyph, an arrow."""
    page = np.full((300, 400), PAPER, np.uint8)
    pieces = {
        'lithic': _component((100, 100, 200, 220), fill=False),
        'identifier': _component((205, 205, 215, 220)),       # the "3" beside the lithic
        'caption': _component((60, 240, 180, 255)),
        'legend': _component((230, 120, 250, 140)),
        'other glyph': _component((50, 50, 60, 65)),           # a neighbour's identifier
        'arrow': _component((140, 230, 160, 236)),             # a flaking arrow below the outline
    }
    for c in pieces.values():
        x0, y0, x1, y1 = c.box
        page[y0:y1, x0:x1][c.mask] = 0
    classified = Classified(
        drawings=[pieces['lithic']],
        labels=[pieces['identifier'].box, pieces['other glyph'].box, pieces['arrow'].box],
        text=[pieces['caption'].box, pieces['legend'].box],
    )
    reads = PageReads(seen=[pieces['identifier'].box, pieces['other glyph'].box])  # not the arrow
    identifier = Identifier(label='3', source='read', confidence=0.9,
                            candidates=[('3', 0.9, tuple(pieces['identifier'].box))])
    return page, pieces, classified, reads, identifier


def _crop(page, box, pad=40):
    x0, y0, x1, y1 = box
    origin = (x0 - pad, y0 - pad)
    return page[y0 - pad:y1 + pad, x0 - pad:x1 + pad], origin


def _ink_left(crop, origin, piece):
    """Whether any of the piece's ink is still dark in the crop."""
    x0, y0, x1, y1 = piece.box
    region = crop[y0 - origin[1]:y1 - origin[1], x0 - origin[0]:x1 - origin[0]]
    return bool((region[piece.mask] < 128).any())


@pytest.mark.unit
class TestHideInCrop:

    def test_text_is_hidden_and_the_lithic_its_identifier_and_the_arrow_stay(self):
        page, pieces, classified, reads, identifier = _page()
        hide = hiding.make_hider(list(pieces.values()), classified, reads, {})
        crop, origin = _crop(page, pieces['lithic'].box, pad=60)
        result, hidden = hide(crop, origin, tuple(pieces['lithic'].box), identifier)
        assert not _ink_left(result, origin, pieces['caption'])
        assert not _ink_left(result, origin, pieces['legend'])
        assert not _ink_left(result, origin, pieces['other glyph'])
        assert _ink_left(result, origin, pieces['lithic'])
        assert _ink_left(result, origin, pieces['identifier'])
        assert _ink_left(result, origin, pieces['arrow'])
        assert hidden.counts == {'caption': 2, 'label': 1}
        assert hidden.text == 'caption 2; label 1'

    def test_hidden_pixels_take_the_paper_value_not_white(self):
        page, pieces, classified, reads, identifier = _page()
        hide = hiding.make_hider(list(pieces.values()), classified, reads, {})
        crop, origin = _crop(page, pieces['lithic'].box, pad=60)
        result, _ = hide(crop, origin, tuple(pieces['lithic'].box), identifier)
        x0, y0, x1, y1 = pieces['caption'].box
        region = result[y0 - origin[1]:y1 - origin[1], x0 - origin[0]:x1 - origin[0]]
        assert (region == PAPER).all()
        assert crop[y0 - origin[1], x0 - origin[0]] == 0      # the page itself is unchanged

    def test_the_identifier_box_goes_to_the_metadata_in_crop_pixels(self):
        page, pieces, classified, reads, identifier = _page()
        hide = hiding.make_hider(list(pieces.values()), classified, reads, {})
        crop, origin = _crop(page, pieces['lithic'].box, pad=60)
        _, hidden = hide(crop, origin, tuple(pieces['lithic'].box), identifier)
        x0, y0, x1, y1 = pieces['identifier'].box
        expected = (x0 - origin[0], y0 - origin[1], x1 - origin[0], y1 - origin[1])
        assert hidden.ignore_boxes == [expected]
        assert hidden.ignore_text == ' '.join(str(v) for v in hidden.ignore_boxes[0])

    def test_an_identifier_attached_inside_the_artefact_box_is_still_recorded(self):
        """The grouping attaches the identifier to the box, so the box holds it."""
        page, pieces, classified, reads, identifier = _page()
        hide = hiding.make_hider(list(pieces.values()), classified, reads, {})
        attached = (100, 100, 215, 220)                        # the box now holds the "3"
        crop, origin = _crop(page, attached, pad=60)
        result, hidden = hide(crop, origin, attached, identifier)
        assert _ink_left(result, origin, pieces['identifier'])
        assert len(hidden.ignore_boxes) == 1
        settings = {'hide_identifiers': True}
        hide = hiding.make_hider(list(pieces.values()), classified, reads, settings)
        result, hidden = hide(crop, origin, attached, identifier)
        assert not _ink_left(result, origin, pieces['identifier']) and hidden.ignore_boxes == []

    def test_hide_identifiers_hides_it_and_records_no_box(self):
        page, pieces, classified, reads, identifier = _page()
        settings = {'hide_identifiers': True}
        hide = hiding.make_hider(list(pieces.values()), classified, reads, settings)
        crop, origin = _crop(page, pieces['lithic'].box, pad=60)
        result, hidden = hide(crop, origin, tuple(pieces['lithic'].box), identifier)
        assert not _ink_left(result, origin, pieces['identifier'])
        assert hidden.ignore_boxes == [] and hidden.counts['identifier'] == 1

    def test_keep_text_gives_no_hider(self):
        page, pieces, classified, reads, _ = _page()
        settings = {'hide_text': False}
        assert hiding.make_hider(list(pieces.values()), classified, reads, settings) is None

    def test_hiding_follows_the_piece_s_own_ink_not_its_box(self):
        """A caption whose box crosses the outline is painted by its mask, so the outline stays."""
        page, pieces, classified, reads, identifier = _page()
        glyph = _component((195, 150, 210, 165))               # its box crosses the outline
        glyph.mask[:, 3:6] = False                             # but its ink does not
        classified.text.append(glyph.box)
        page[150:165, 195:210][glyph.mask] = 0
        components = list(pieces.values()) + [glyph]
        hide = hiding.make_hider(components, classified, reads, {})
        crop, origin = _crop(page, pieces['lithic'].box, pad=60)
        result, _ = hide(crop, origin, tuple(pieces['lithic'].box), identifier)
        assert (result[150 - origin[1]:165 - origin[1], 199 - origin[0]] < 128).any()
        assert not _ink_left(result, origin, glyph)


@pytest.mark.unit
class TestPaperValue:

    def test_grey_paper_is_measured(self):
        crop = np.full((20, 20), 230, np.uint8)
        crop[5:8, 5:8] = 0
        assert hiding.paper_value(crop) == 230

    def test_colour_paper_is_measured_per_channel(self):
        crop = np.zeros((10, 10, 3), np.uint8)
        crop[:, :] = (250, 245, 240)
        crop[2, 2] = (0, 0, 0)
        assert list(hiding.paper_value(crop)) == [250, 245, 240]

    def test_no_light_pixels_means_white(self):
        assert hiding.paper_value(np.zeros((4, 4), np.uint8)) == 255
