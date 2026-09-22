"""
Page loading and ink-blob detection for page segmentation.

The binary mask built here exists only to locate ink on the page. It is
used to compute bounding boxes and is then discarded: nothing derived
from it reaches the saved crops, which are cut from the untouched source
array.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from pylithics.image_processing.importer import (
    calculate_dpi_scale_factor,
    get_image_dpi,
    normalise_dpi,
)

from .geometry import BBox

# Components below both thresholds are scanner speckle, not illustration.
_SPECKLE_MAX_INK = 6
_SPECKLE_MAX_EXTENT = 5

# Closing kernel size at the reference DPI, in pixels. Bridges the small
# breaks left by line-drawing stipple so a surface reads as one blob.
_BASE_CLOSE_KERNEL = 9


@dataclass
class PageImage:
    """
    A loaded source page.

    Attributes
    ----------
    path : str
        Absolute or relative path the page was read from.
    array : np.ndarray
        Source pixels, unmodified, in the source colour mode.
    gray : np.ndarray
        Grayscale view used for detection only.
    mode : str
        PIL colour mode of the source (``RGB``, ``RGBA``, ``L``).
    dpi : tuple of float, optional
        Source DPI as ``(x, y)``, or None when the file carried none.
    """

    path: str
    array: np.ndarray
    gray: np.ndarray
    mode: str
    dpi: Optional[Tuple[float, float]]

    @property
    def height(self) -> int:
        """Page height in pixels."""
        return self.gray.shape[0]

    @property
    def width(self) -> int:
        """Page width in pixels."""
        return self.gray.shape[1]

    @property
    def stem(self) -> str:
        """Page filename without directory or extension."""
        return os.path.splitext(os.path.basename(self.path))[0]


@dataclass
class Component:
    """
    A connected region of ink on the page.

    Attributes
    ----------
    box : list of int
        Bounding box as ``[x0, y0, x1, y1]``.
    width, height : int
        Box dimensions in pixels.
    ink : int
        Count of set pixels before morphological closing. Measured on
        the raw threshold so that closing cannot inflate the ink of a
        sparse outline into that of a solid mark.
    mask : np.ndarray, optional
        Boolean view of the unclosed ink within the bounding box. Used
        by shape tests that need the blob's internal structure rather
        than its proportions, such as scale bar detection.
    """

    box: BBox
    width: int
    height: int
    ink: int
    mask: Optional[np.ndarray] = field(
        default=None, repr=False, compare=False
    )

    @property
    def density(self) -> float:
        """Ink pixels as a fraction of bounding-box area."""
        area = self.width * self.height
        return self.ink / float(area) if area else 0.0


def load_page(path: str) -> Optional[PageImage]:
    """
    Load a page, preserving its colour mode and DPI.

    Parameters
    ----------
    path : str
        Path to the page image.

    Returns
    -------
    PageImage or None
        Loaded page, or None when the file cannot be read.
    """
    try:
        with Image.open(path) as img:
            img.load()
            mode = img.mode
            array = np.array(img)
            gray = np.array(img.convert("L"))
            dpi = _extract_dpi(img, path)
    except (OSError, ValueError) as exc:
        logging.error("Cannot read page %s: %s", path, exc)
        return None

    logging.debug(
        "Loaded page %s (%dx%d, mode=%s, dpi=%s)",
        os.path.basename(path), gray.shape[1], gray.shape[0], mode, dpi,
    )
    return PageImage(
        path=path, array=array, gray=gray, mode=mode, dpi=dpi
    )


def _extract_dpi(
    img: Image.Image, path: str
) -> Optional[Tuple[float, float]]:
    """
    Read DPI from an open image, returning None when absent.

    Unlike the main pipeline, a missing DPI is not defaulted to 300
    here. Writing a guessed DPI onto a crop would silently fabricate
    provenance, so an untagged page yields untagged crops.
    """
    dpi = img.info.get("dpi")
    if not dpi:
        logging.warning(
            "DPI missing for %s; the crops have no DPI tag",
            os.path.basename(path),
        )
        return None
    return (normalise_dpi(dpi[0]), normalise_dpi(dpi[1]))


def build_detection_mask(
    page: PageImage, config: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the binary masks used to locate ink.

    Ink is thresholded with Otsu and inverted so that marks are set
    pixels, then closed with a DPI-scaled kernel so a stippled or
    broken outline reads as one region.

    Parameters
    ----------
    page : PageImage
        The loaded page.
    config : dict
        Full configuration dictionary, used for DPI scaling.

    Returns
    -------
    tuple of np.ndarray
        ``(closed, raw)`` masks. ``raw`` measures true ink coverage;
        ``closed`` is what connected components are found on.
    """
    raw = cv2.threshold(
        page.gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    dpi_scale = _page_dpi_scale(page, config)
    size = max(3, int(_BASE_CLOSE_KERNEL * dpi_scale))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel)

    logging.debug(
        "Detection mask: closing %dx%d (dpi scale %.2f)",
        size, size, dpi_scale,
    )
    return closed, raw


def _page_dpi_scale(page: PageImage, config: Dict) -> float:
    """
    Derive the DPI scale factor for a page's detection kernels.

    Sizing from DPI rather than page width keeps kernel size tied to
    the resolution of the scan, so two 300 dpi plates of different
    physical sizes are treated identically.
    """
    if page.dpi is not None:
        image_dpi = page.dpi[0]
    else:
        image_dpi = get_image_dpi(page.path)
    return calculate_dpi_scale_factor(image_dpi, config)


def find_components(
    closed: np.ndarray, raw: np.ndarray
) -> List[Component]:
    """
    Find ink blobs on the page, discarding scanner speckle.

    Parameters
    ----------
    closed : np.ndarray
        Closed binary mask to label.
    raw : np.ndarray
        Unclosed mask, used to measure true ink coverage.

    Returns
    -------
    list of Component
        One entry per surviving blob, in no particular order.
    """
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        closed, connectivity=8
    )

    components = []
    for index in range(1, count):
        x, y, w, h = (int(v) for v in stats[index, :4])
        ink = int(cv2.countNonZero(raw[y:y + h, x:x + w]))
        if _is_speckle(ink, w, h):
            continue
        components.append(
            Component(
                box=[x, y, x + w, y + h], width=w, height=h, ink=ink,
                mask=raw[y:y + h, x:x + w] > 0,
            )
        )

    logging.debug(
        "Found %d ink components (%d raw labels)",
        len(components), count - 1,
    )
    return components


def _is_speckle(ink: int, width: int, height: int) -> bool:
    """Report whether a blob is too small to be part of a drawing."""
    return ink < _SPECKLE_MAX_INK and max(width, height) < _SPECKLE_MAX_EXTENT
