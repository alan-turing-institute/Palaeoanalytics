"""
Tests for PDF input to ``pylithics-pages``.

The fixtures are PDF files built in code with pypdfium2, so a known
raster is placed on a page and the render can be compared with it pixel
for pixel. Tests that need pypdfium2 skip without it; the missing
dependency test runs in every environment.
"""

import csv
import logging
import os

import numpy as np
import pytest
from PIL import Image

from pylithics.page_segmentation import pdf_input
from pylithics.page_segmentation.cli import main
from pylithics.page_segmentation.pdf_input import (
    DERIVED,
    INSTALL_MESSAGE,
    PageReport,
    PdfSupportMissing,
    RasterInfo,
    SELECTED,
    choose_dpi,
    parse_page_spec,
    render_pdf,
)

SAMPLE_PLATE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'pylithics', 'data', 'pages', 'sample_plate.png',
)
A4 = (595.0, 842.0)


### FIXTURE BUILDING ###

def figure(height: int, width: int, seed: int = 0) -> np.ndarray:
    """Return a grey drawing with lines, a frame and a speckle of dots."""
    rng = np.random.default_rng(seed)
    array = np.full((height, width), 255, np.uint8)
    array[height // 8:-height // 8, width // 8] = 0
    array[height // 2, width // 8:-width // 8] = 0
    array[0, :] = array[-1, :] = array[:, 0] = array[:, -1] = 40
    sparse = array[::6, ::6]
    dots = rng.integers(0, 2, size=sparse.shape).astype(bool)
    sparse[dots] = 90
    return array


def build_pdf(path, placements, page_size=A4, rotation=0, line=None, pages=1):
    """
    Write a PDF with ``placements`` on its first page.

    Each placement is ``(array, dpi, x_pt, y_pt)``: the array is placed
    with its bottom left corner at ``(x_pt, y_pt)`` at the given
    resolution. ``line`` draws a vector stroke ``(x0, y0, x1, y1)`` on
    top. Pages after the first are empty.
    """
    pdfium = pytest.importorskip('pypdfium2')
    import pypdfium2.raw as pdfium_c

    document = pdfium.PdfDocument.new()
    page = document.new_page(*page_size)
    for array, dpi, x_pt, y_pt in placements:
        image = pdfium.PdfImage.new(document)
        pil = Image.fromarray(array).convert('RGB')
        image.set_bitmap(pdfium.PdfBitmap.from_pil(pil))
        w_pt = array.shape[1] / dpi * 72.0
        h_pt = array.shape[0] / dpi * 72.0
        image.set_matrix(pdfium.PdfMatrix(w_pt, 0, 0, h_pt, x_pt, y_pt))
        page.insert_obj(image)
    if line is not None:
        path_obj = pdfium_c.FPDFPageObj_CreateNewPath(line[0], line[1])
        pdfium_c.FPDFPath_LineTo(path_obj, line[2], line[3])
        pdfium_c.FPDFPageObj_SetStrokeColor(path_obj, 0, 0, 0, 255)
        pdfium_c.FPDFPageObj_SetStrokeWidth(path_obj, 2)
        pdfium_c.FPDFPath_SetDrawMode(path_obj, 0, True)
        pdfium_c.FPDFPage_InsertObject(page, path_obj)
    if rotation:
        page.set_rotation(rotation)
    page.gen_content()
    for _ in range(pages - 1):
        document.new_page(*page_size).gen_content()
    document.save(str(path))
    return str(path)


def find_copy(rendered: np.ndarray, expected: np.ndarray, near) -> bool:
    """Return True when ``expected`` appears pixel for pixel within 3 px of ``near``."""
    h, w = expected.shape
    x_near, y_near = near
    for y in range(max(0, y_near - 3), y_near + 4):
        for x in range(max(0, x_near - 3), x_near + 4):
            window = rendered[y:y + h, x:x + w]
            if window.shape == expected.shape and np.array_equal(window, expected):
                return True
    return False


def device_origin(x_pt: float, top_pt: float, dpi: float):
    """Return the pixel position of a point ``x_pt`` from the left and ``top_pt`` from the top."""
    return round(x_pt * dpi / 72.0), round(top_pt * dpi / 72.0)


def render_one(tmp_path, pdf_path, config=None, pages=None):
    """Render a PDF and return the records and the grey array of the first page."""
    records = render_pdf(pdf_path, str(tmp_path), config or {}, pages)
    array = np.asarray(Image.open(records[0].path).convert('L'))
    return records, array


### RESOLUTION ###

@pytest.mark.unit
class TestResolution:
    """What resolution a page renders at, from what it holds."""

    def test_single_raster_is_copied_pixel_for_pixel(self, tmp_path):
        drawing = figure(900, 1200)
        pdf = build_pdf(tmp_path / 'one.pdf', [(drawing, 300.0, 100.0, 300.0)])
        records, rendered = render_one(tmp_path, pdf)
        assert records[0].dpi == pytest.approx(300.0)
        assert records[0].dpi_source == DERIVED
        assert Image.open(records[0].path).info['dpi'][0] == pytest.approx(300.0, abs=0.01)
        top = A4[1] - 300.0 - 900 / 300.0 * 72.0
        assert find_copy(rendered, drawing, device_origin(100.0, top, 300.0))

    def test_fractional_resolution_is_kept(self, tmp_path):
        drawing = figure(900, 1200)
        pdf = build_pdf(tmp_path / 'frac.pdf', [(drawing, 312.5, 100.0, 300.0)])
        records, rendered = render_one(tmp_path, pdf)
        assert records[0].dpi == pytest.approx(312.5, abs=0.01)
        assert Image.open(records[0].path).info['dpi'][0] == pytest.approx(312.5, abs=0.01)
        top = A4[1] - 300.0 - 900 / 312.5 * 72.0
        assert find_copy(rendered, drawing, device_origin(100.0, top, 312.5))

    def test_sharpest_raster_sets_the_resolution(self, tmp_path):
        background = figure(300, 400, seed=1)
        mask = figure(1200, 1600, seed=2)
        pdf = build_pdf(tmp_path / 'layers.pdf', [
            (background, 150.0, 100.0, 300.0), (mask, 600.0, 100.0, 300.0),
        ])
        records, rendered = render_one(tmp_path, pdf)
        assert records[0].dpi == pytest.approx(600.0)
        assert records[0].dpi_source == DERIVED
        top = A4[1] - 300.0 - 1200 / 600.0 * 72.0
        assert find_copy(rendered, mask, device_origin(100.0, top, 600.0))

    def test_small_image_does_not_set_the_resolution(self, tmp_path):
        drawing = figure(900, 1200)
        logo = figure(100, 100, seed=3)
        pdf = build_pdf(tmp_path / 'logo.pdf', [
            (drawing, 300.0, 100.0, 300.0), (logo, 508.0, 500.0, 760.0),
        ])
        records, _ = render_one(tmp_path, pdf)
        assert records[0].dpi == pytest.approx(300.0)
        assert records[0].filters == 'FlateDecode'

    def test_cap_limits_a_derived_resolution(self, caplog):
        raster = RasterInfo(3000, 2000, 3000.0, 3000.0, 8, ['FlateDecode'], 0.2, (0, 0, 1, 1))
        report = PageReport(number=1, rasters=[raster], image_count=1)
        dpi, source = choose_dpi(report, {'max_dpi': 1200})
        assert (dpi, source) == (1200.0, DERIVED)
        with caplog.at_level(logging.WARNING):
            pdf_input._log_report('big.pdf', report, dpi, source)
        assert 'more than the cap' in caplog.text

    def test_vector_page_uses_the_selected_resolution(self, tmp_path):
        pdf = build_pdf(tmp_path / 'vector.pdf', [], line=(100.0, 400.0, 400.0, 400.0))
        records, rendered = render_one(tmp_path, pdf, {'rasterise_dpi': 150})
        assert records[0].dpi == 150.0
        assert records[0].dpi_source == SELECTED
        assert records[0].filters == ''
        assert rendered.shape == (int(np.ceil(842 * 150 / 72)), int(np.ceil(595 * 150 / 72)))
        assert (rendered < 128).any()

    def test_pdf_dpi_option_sets_the_selected_resolution(self, tmp_path):
        pdf = build_pdf(tmp_path / 'vector.pdf', [], line=(100.0, 400.0, 400.0, 400.0))
        records, _ = render_one(tmp_path, pdf, {'rasterise_dpi': 72})
        assert records[0].dpi == 72.0


### CONTENT ###

@pytest.mark.unit
class TestContent:
    """What the render keeps."""

    def test_vector_line_over_a_raster_is_kept(self, tmp_path):
        drawing = figure(900, 1200)
        pdf = build_pdf(tmp_path / 'line.pdf', [(drawing, 300.0, 100.0, 300.0)],
                        line=(100.0, 200.0, 400.0, 200.0))
        records, rendered = render_one(tmp_path, pdf)
        top = A4[1] - 300.0 - 900 / 300.0 * 72.0
        assert find_copy(rendered, drawing, device_origin(100.0, top, 300.0))
        _, line_row = device_origin(0.0, A4[1] - 200.0, records[0].dpi)
        assert (rendered[line_row - 3:line_row + 4, :] < 128).any()

    def test_rotated_page_renders_upright(self, tmp_path):
        drawing = figure(900, 1200)
        pdf = build_pdf(tmp_path / 'rot.pdf', [(drawing, 300.0, 100.0, 300.0)], rotation=90)
        records, rendered = render_one(tmp_path, pdf)
        assert rendered.shape[1] > rendered.shape[0]
        turned = np.rot90(drawing, -1)
        ys, xs = np.where(rendered == turned[0, 0])
        assert find_copy(rendered, turned, (int(xs.min()), int(ys.min())))

    def test_grey_render_is_saved_as_greyscale(self, tmp_path):
        pdf = build_pdf(tmp_path / 'grey.pdf', [(figure(900, 1200), 300.0, 100.0, 300.0)])
        records, _ = render_one(tmp_path, pdf)
        assert Image.open(records[0].path).mode == 'L'

    def test_colour_render_stays_rgb(self, tmp_path):
        colour = np.dstack([figure(900, 1200)] * 3)
        colour[100:300, 100:300] = (200, 30, 30)
        pdf = build_pdf(tmp_path / 'colour.pdf', [(colour, 300.0, 100.0, 300.0)])
        records, _ = render_one(tmp_path, pdf)
        assert Image.open(records[0].path).mode == 'RGB'

    def test_warnings_for_low_resolution_and_jpeg(self, caplog):
        raster = RasterInfo(800, 600, 150.0, 150.0, 8, ['DCTDecode'], 0.3, (0, 0, 1, 1))
        report = PageReport(number=2, rasters=[raster], image_count=1)
        with caplog.at_level(logging.WARNING):
            pdf_input._log_report('web.pdf', report, 150.0, DERIVED)
        assert 'less than 200' in caplog.text
        assert 'JPEG compression' in caplog.text


### PAGES ###

@pytest.mark.unit
class TestPageSelection:
    """Which pages render, and how they are named."""

    def test_page_spec_is_parsed(self):
        assert parse_page_spec('4,7,11-14') == {4, 7, 11, 12, 13, 14}
        assert parse_page_spec(None) is None
        assert parse_page_spec('  ') is None

    @pytest.mark.parametrize('spec', ['a', '0', '5-3', '1,,2'])
    def test_bad_page_spec_is_refused(self, spec):
        with pytest.raises(ValueError):
            parse_page_spec(spec)

    def test_only_the_named_pages_render(self, tmp_path, caplog):
        pdf = build_pdf(tmp_path / 'three.pdf', [(figure(900, 1200), 300.0, 100.0, 300.0)],
                        pages=3)
        with caplog.at_level(logging.WARNING):
            records = render_pdf(pdf, str(tmp_path), {'rasterise_dpi': 72}, {1, 3, 9})
        assert [r.source_page for r in records] == [1, 3]
        assert sorted(os.listdir(tmp_path / 'pages_rendered')) == [
            'three_p001.png', 'three_p003.png'
        ]
        assert 'Not in the file: 9' in caplog.text


### COMMAND ###

def read_manifest(output_dir):
    """Read the manifest written into an output directory."""
    with open(os.path.join(str(output_dir), 'pages_manifest.csv'), newline='',
              encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


@pytest.mark.integration
class TestCommand:
    """``pylithics-pages`` on a folder that holds a PDF."""

    def _project(self, tmp_path, with_png=False):
        pages = tmp_path / 'pages'
        pages.mkdir()
        plate = np.asarray(Image.open(SAMPLE_PLATE).convert('L'))
        w_pt, h_pt = plate.shape[1] / 300.0 * 72.0, plate.shape[0] / 300.0 * 72.0
        build_pdf(pages / 'plate.pdf', [(plate, 300.0, 20.0, 30.0)],
                  page_size=(w_pt + 40.0, h_pt + 60.0))
        if with_png:
            scan = np.full((1500, 1500), 255, np.uint8)
            for x0 in (200, 900):
                scan[300:900, x0:x0 + 3] = 0
                scan[300:900, x0 + 400:x0 + 403] = 0
                scan[300:303, x0:x0 + 403] = 0
                scan[897:900, x0:x0 + 403] = 0
            Image.fromarray(scan).save(pages / 'scan.png', dpi=(300, 300))
        return tmp_path

    def test_pdf_pages_are_cut_and_their_source_recorded(self, tmp_path):
        project = self._project(tmp_path)
        assert main(['--data_dir', str(project), '--no_read_labels']) == 0
        assert os.path.isfile(project / 'pages_rendered' / 'plate_p001.png')
        rows = read_manifest(project)
        assert {r['input_page_id'] for r in rows} == {'plate_p001.png'}
        assert sum(1 for r in rows if r['image_type'] == 'artefact') == 5
        assert sum(1 for r in rows if r['image_type'] == 'scale_bar') == 1
        first = rows[0]
        assert first['source_pdf'] == 'plate.pdf'
        assert first['source_page'] == '1'
        assert first['render_dpi'] == '300'
        assert first['dpi_source'] == DERIVED
        assert first['source_filters'] == 'FlateDecode'
        assert first['dpi'] == '300'
        assert os.path.isfile(project / 'images' / 'plate_p001_box_01.png')

    def test_pdf_and_png_in_one_folder_both_process(self, tmp_path):
        project = self._project(tmp_path, with_png=True)
        assert main(['--data_dir', str(project), '--no_read_labels']) == 0
        rows = read_manifest(project)
        assert {r['input_page_id'] for r in rows} == {'plate_p001.png', 'scan.png'}
        png_rows = [r for r in rows if r['input_page_id'] == 'scan.png']
        assert {r['source_pdf'] for r in png_rows} == {''}
        assert {r['dpi_source'] for r in png_rows} == {''}

    def test_without_the_package_the_run_stops(self, tmp_path, monkeypatch, caplog):
        pages = tmp_path / 'pages'
        pages.mkdir()
        (pages / 'paper.pdf').write_bytes(b'%PDF-1.4\n')

        def missing():
            raise PdfSupportMissing(INSTALL_MESSAGE)

        monkeypatch.setattr(pdf_input, '_pdfium', missing)
        with caplog.at_level(logging.ERROR):
            assert main(['--data_dir', str(tmp_path)]) == 1
        assert "pip install 'pylithics[pdf]'" in caplog.text
        assert not os.path.isdir(tmp_path / 'images')
        assert not os.path.isdir(tmp_path / 'pages_rendered')

    def test_a_bad_page_list_stops_the_run(self, tmp_path, caplog):
        project = self._project(tmp_path)
        with caplog.at_level(logging.ERROR):
            assert main(['--data_dir', str(project), '--pdf_pages', '2-1']) == 1
        assert 'page list' in caplog.text
