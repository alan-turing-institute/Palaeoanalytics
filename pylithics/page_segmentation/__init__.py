"""
Page segmentation for PyLithics.

Cuts a scanned plate of lithic illustrations into one image per
artefact, ready for the main PyLithics analysis run.

Archaeological plates commonly show several artefacts per page, and a
single artefact is usually drawn as a set of adjacent surface views
(platform, dorsal, ventral, lateral). A lithic drawn with four surfaces
is *one* artefact, not four, so this package groups adjacent views
rather than merely isolating ink.

This stage never alters saved pixels. Thresholding is used internally to
locate ink and is then discarded; crops are cut from the source image
untouched, at the source DPI and colour mode. All enhancement remains
the responsibility of the main pipeline, so measurements taken from a
crop stay comparable to measurements taken from a single-artefact scan.

Invoked via the ``pylithics-pages`` console script.
"""

from .detection import PageImage, Component, load_page, find_components
from .grouping import group_components, reading_order
from .export import export_page

__all__ = [
    "PageImage",
    "Component",
    "load_page",
    "find_components",
    "group_components",
    "reading_order",
    "export_page",
]
