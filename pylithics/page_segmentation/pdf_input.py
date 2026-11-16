"""
PDF input for page segmentation.

Renders each page of a PDF file to a PNG file, so that the rest of the
``pylithics-pages`` workflow reads it as it reads a scanned page. The
render keeps every pixel the PDF holds and adds none:

- A page that holds a raster image renders at the resolution of the
  sharpest large raster on it. The render is shifted by the fraction of
  a pixel that puts that raster on the pixel grid, and image smoothing
  is off, so the raster is copied pixel for pixel. The DPI is *derived*.
- A page with no raster (a vector drawing, a text page) renders at the
  configured resolution. The DPI is *selected*.

The PNG carries the render DPI. One log line for each page says what
the PDF held, with a warning for a low resolution and for JPEG
compression, which no later step can undo.

pypdfium2 is an optional dependency (``pip install 'pylithics[pdf]'``).
It is imported when a PDF is met, so the common path pays nothing.
"""

import logging
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image

PDF_EXTENSION = '.pdf'
RENDERED_DIRNAME = 'pages_rendered'
POINTS_PER_INCH = 72.0

DERIVED = 'derived'
SELECTED = 'selected'

# A render whose channels differ by no more than this is a grey page.
GREY_TOLERANCE = 2
# Below this effective resolution the lines of a drawing are thin.
LOW_DPI = 200.0
JPEG_FILTER = 'DCTDecode'
# pdfium stretches a raster to the whole pixels its box touches. The
# render is scaled down by this fraction of a pixel over the raster, and
# the box starts this far inside its first pixel, so the box touches
# exactly the raster's own number of pixels and the copy is 1:1.
FIT_DEFICIT = 0.4
FIT_MARGIN = 0.1

DEFAULTS: Dict[str, float] = {
    'rasterise_dpi': 300.0,
    'min_coverage': 0.05,
    'max_dpi': 1200.0,
}

INSTALL_MESSAGE = (
    "PyLithics cannot read PDF files without the pdf extra.\n"
    "       Install it: pip install 'pylithics[pdf]'\n"
    "       Or export the pages as PNG or TIFF at the scanned resolution."
)


class PdfSupportMissing(RuntimeError):
    """pypdfium2 is not installed."""


@dataclass
class RasterInfo:
    """One raster image placed on a PDF page."""

    width: int
    height: int
    dpi_x: float
    dpi_y: float
    bits: int
    filters: List[str]
    coverage: float
    bounds: Tuple[float, float, float, float]

    @property
    def dpi(self) -> float:
        """The larger of the two effective resolutions."""
        return max(self.dpi_x, self.dpi_y)


@dataclass
class PageReport:
    """The large rasters of one PDF page."""

    number: int
    rasters: List[RasterInfo]
    image_count: int


@dataclass
class RenderedPage:
    """One PDF page written as a PNG file."""

    path: str
    source_pdf: str
    source_page: int
    dpi: float
    dpi_source: str
    filters: str


### DEPENDENCY ###

def is_pdf(path: str) -> bool:
    """Return True when the file name ends in ``.pdf``."""
    return path.lower().endswith(PDF_EXTENSION)


def _pdfium():
    """Return the pypdfium2 modules, or raise :class:`PdfSupportMissing`."""
    try:
        import pypdfium2 as pdfium
        import pypdfium2.raw as pdfium_c
    except ImportError as exc:
        raise PdfSupportMissing(INSTALL_MESSAGE) from exc
    return pdfium, pdfium_c


def parse_page_spec(spec: Optional[str]) -> Optional[Set[int]]:
    """
    Return the 1-based page numbers that ``spec`` names, or None for all.

    ``spec`` is a list with commas of numbers and ranges: ``4,7,11-14``.
    """
    if spec is None or not spec.strip():
        return None
    pages: Set[int] = set()
    for part in spec.split(','):
        part = part.strip()
        bounds = part.split('-', 1) if '-' in part else [part, part]
        try:
            first, last = int(bounds[0]), int(bounds[1])
        except ValueError:
            raise ValueError(
                f"The page list '{spec}' is not correct. Give numbers and ranges "
                "with commas, for example 4,7,11-14."
            ) from None
        if first < 1 or last < first:
            raise ValueError(f"The page list '{spec}' is not correct: '{part}'.")
        pages.update(range(first, last + 1))
    return pages


### INSPECTION ###

def inspect_page(page, number: int, min_coverage: float) -> PageReport:
    """Return the rasters of ``page`` that cover at least ``min_coverage`` of it."""
    _, pdfium_c = _pdfium()
    width, height = _unrotated_size(page)
    page_area = width * height
    rasters, count = [], 0
    for obj in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE]):
        count += 1
        bounds = obj.get_bounds()
        coverage = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) / page_area
        if coverage < min_coverage:
            continue
        info = _raster_info(obj, bounds, coverage)
        if info is not None:
            rasters.append(info)
    return PageReport(number=number, rasters=rasters, image_count=count)


def _raster_info(obj, bounds, coverage: float) -> Optional[RasterInfo]:
    """Return the size, effective resolution and filters of one image object."""
    try:
        meta = obj.get_metadata()
        dpi_x, dpi_y, bits = meta.horizontal_dpi, meta.vertical_dpi, meta.bits_per_pixel
        width, height = meta.width, meta.height
    except Exception:  # noqa: BLE001 - pdfium cannot read every image's metadata
        width, height = obj.get_px_size()
        inches_w = (bounds[2] - bounds[0]) / POINTS_PER_INCH
        inches_h = (bounds[3] - bounds[1]) / POINTS_PER_INCH
        dpi_x = width / inches_w if inches_w else 0.0
        dpi_y = height / inches_h if inches_h else 0.0
        bits = 0
    if not width or not height or dpi_x <= 0 or dpi_y <= 0:
        return None
    return RasterInfo(width, height, float(dpi_x), float(dpi_y), int(bits),
                      list(obj.get_filters()), coverage, tuple(bounds))


def _unrotated_size(page) -> Tuple[float, float]:
    """Return the page size in points before ``/Rotate`` is applied."""
    width, height = page.get_size()
    if page.get_rotation() in (90, 270):
        return height, width
    return width, height


def choose_dpi(report: PageReport, config: Dict) -> Tuple[float, str]:
    """
    Return the render resolution of a page and how it was decided.

    The resolution of the sharpest large raster, with the cap
    ``max_dpi``, is ``derived``. Without a raster, ``rasterise_dpi`` is
    ``selected``.
    """
    if not report.rasters:
        return float(config.get('rasterise_dpi', DEFAULTS['rasterise_dpi'])), SELECTED
    cap = float(config.get('max_dpi', DEFAULTS['max_dpi']))
    return min(max(r.dpi for r in report.rasters), cap), DERIVED


### RENDERING ###

def render_page(page, dpi: float, align_to: Optional[RasterInfo]) -> Image.Image:
    """
    Render ``page`` at ``dpi`` and return it as a PIL image, ``L`` or ``RGB``.

    With ``align_to``, the render is shifted by the fraction of a pixel
    that puts that raster's top left corner on the pixel grid, scaled
    down by a fraction of one pixel over the raster (see ``FIT_DEFICIT``),
    and image smoothing is off, so the raster is copied pixel for pixel.
    The PNG still carries ``dpi``: the raster is at that resolution.
    """
    pdfium, pdfium_c = _pdfium()
    scale = dpi / POINTS_PER_INCH
    shift_x, shift_y = 0.0, 0.0
    flags = pdfium_c.FPDF_REVERSE_BYTE_ORDER | pdfium_c.FPDF_ANNOT
    if align_to is not None:
        scale *= 1.0 - FIT_DEFICIT / max(align_to.width, align_to.height)
        origin_x, origin_y = _device_origin(page, align_to.bounds)
        shift_x = FIT_MARGIN - ((origin_x * scale) % 1.0)
        shift_y = FIT_MARGIN - ((origin_y * scale) % 1.0)
        flags |= pdfium_c.FPDF_RENDER_NO_SMOOTHIMAGE
    width, height = page.get_size()
    size = (math.ceil(width * scale), math.ceil(height * scale))

    bitmap = pdfium.PdfBitmap.new_native(size[0], size[1], pdfium_c.FPDFBitmap_BGR,
                                         rev_byteorder=True)
    bitmap.fill_rect((255, 255, 255, 255), 0, 0, size[0], size[1])
    matrix = pdfium_c.FS_MATRIX(scale, 0, 0, scale, shift_x, shift_y)
    clip = pdfium_c.FS_RECTF(0, 0, size[0], size[1])
    pdfium_c.FPDF_RenderPageBitmapWithMatrix(bitmap, page, matrix, clip, flags)
    return _as_grey_when_grey(bitmap.to_pil())


def _device_origin(page, bounds) -> Tuple[float, float]:
    """
    Return the top left corner of ``bounds`` in points on the rendered page.

    ``bounds`` is (left, bottom, right, top) in page coordinates, y up,
    before ``/Rotate``. The render applies the rotation, so the four
    corners are mapped and the smallest x and y are the corner.
    """
    width, height = _unrotated_size(page)
    rotation = page.get_rotation()
    corners = [(bounds[0], bounds[1]), (bounds[2], bounds[1]),
               (bounds[0], bounds[3]), (bounds[2], bounds[3])]
    mapped = []
    for x, y in corners:
        if rotation == 90:
            mapped.append((y, x))
        elif rotation == 180:
            mapped.append((width - x, y))
        elif rotation == 270:
            mapped.append((height - y, width - x))
        else:
            mapped.append((x, height - y))
    return min(m[0] for m in mapped), min(m[1] for m in mapped)


def _as_grey_when_grey(image: Image.Image) -> Image.Image:
    """Return the image as ``L`` when its channels differ by at most the tolerance."""
    array = np.asarray(image)
    if array.ndim != 3:
        return image
    spread = array.max(axis=2).astype(np.int16) - array.min(axis=2)
    if int(spread.max()) > GREY_TOLERANCE:
        return image
    grey = (array.astype(np.uint16).sum(axis=2) // array.shape[2]).astype(np.uint8)
    return Image.fromarray(grey, mode='L')


### FILES ###

def render_pdf(
    path: str, output_dir: str, config: Dict, pages: Optional[Set[int]] = None,
) -> List[RenderedPage]:
    """
    Write each page of the PDF at ``path`` as a PNG and return the records.

    ``config`` is the ``pdf`` section of the page segmentation
    configuration. ``pages`` limits the 1-based page numbers; None
    renders all. The files go to ``<output_dir>/pages_rendered/`` as
    ``<stem>_p<NNN>.png`` and are written again on each run.

    Raises
    ------
    PdfSupportMissing
        When pypdfium2 is not installed.
    """
    pdfium, _ = _pdfium()
    folder = os.path.join(output_dir, RENDERED_DIRNAME)
    os.makedirs(folder, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    name = os.path.basename(path)
    min_coverage = float(config.get('min_coverage', DEFAULTS['min_coverage']))

    document = pdfium.PdfDocument(path)
    rendered = []
    try:
        wanted = _wanted_pages(name, len(document), pages)
        for number in wanted:
            page = document[number - 1]
            report = inspect_page(page, number, min_coverage)
            dpi, source = choose_dpi(report, config)
            _log_report(name, report, dpi, source)
            sharpest = max(report.rasters, key=lambda r: r.dpi) if report.rasters else None
            image = render_page(page, dpi, sharpest)
            out = os.path.join(folder, f"{stem}_p{number:03d}.png")
            image.save(out, dpi=(dpi, dpi))
            filters = sorted({f for r in report.rasters for f in r.filters})
            rendered.append(RenderedPage(out, name, number, dpi, source, ';'.join(filters)))
            page.close()
    finally:
        document.close()
    logging.info("%s: %d page(s) rendered to %s", name, len(rendered), folder)
    return rendered


def _wanted_pages(name: str, count: int, pages: Optional[Set[int]]) -> List[int]:
    """Return the page numbers to render, in order, with a warning for those not in the file."""
    if pages is None:
        return list(range(1, count + 1))
    missing = sorted(p for p in pages if p > count)
    if missing:
        logging.warning("%s has %d page(s). Not in the file: %s", name, count,
                        ', '.join(map(str, missing)))
    return sorted(p for p in pages if p <= count)


def _log_report(name: str, report: PageReport, dpi: float, source: str) -> None:
    """Write one log line for the page, and a warning for each quality problem."""
    where = f"{name} page {report.number}"
    if not report.rasters:
        logging.info("%s: no raster; rendered at %g dpi (%s)", where, dpi, source)
        return
    parts = [f"{r.width}x{r.height} px at {r.dpi:.1f} dpi, {_bits_text(r.bits)}"
             f"{', ' + ', '.join(r.filters) if r.filters else ''}" for r in report.rasters]
    logging.info("%s: %d raster(s): %s; rendered at %.1f dpi (%s)", where,
                 len(report.rasters), '; '.join(parts), dpi, source)
    sharpest = max(r.dpi for r in report.rasters)
    if sharpest < LOW_DPI:
        logging.warning("%s: the resolution is %.0f dpi, less than %.0f. The lines can be "
                        "too thin. The main run upscales the image for detection.",
                        where, sharpest, LOW_DPI)
    if sharpest > dpi:
        logging.warning("%s: the resolution %.0f dpi is more than the cap. Rendered at "
                        "%.0f dpi.", where, sharpest, dpi)
    if any(JPEG_FILTER in r.filters for r in report.rasters):
        logging.warning("%s: JPEG compression (%s). The ink edges can have artefacts.",
                        where, JPEG_FILTER)


def _bits_text(bits: int) -> str:
    """Return the bit depth as text, or ``unknown depth`` when pdfium gave none."""
    return f"{bits}-bit" if bits else "unknown depth"


def rendered_by_name(rendered: Sequence[RenderedPage]) -> Dict[str, RenderedPage]:
    """Return the records keyed by the PNG file name, for the manifest."""
    return {os.path.basename(r.path): r for r in rendered}
