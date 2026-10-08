#!/usr/bin/env python3
"""
Generate the sample plate fixture used by page segmentation tests.

Composites the five single-artefact drawings shipped in
``pylithics/data/images/`` onto one page, laid out the way an
archaeological plate is: artefacts in rows, each with its identifier
number set just under its ink, and a shared scale bar at the foot of
the page.

Each source drawing already shows two to four surface views, so the
composited page carries far more ink blobs than artefacts. Correct
segmentation yields five artefact crops plus one scale bar, which is the
property the tests assert. A splitter that returns one crop per blob
fails the fixture loudly.

Run from the repository root::

    python tests/fixtures/generate_sample_plate.py [OUTPUT.png]

Without an argument the plate is written to
``pylithics/data/pages/sample_plate.png``, the committed fixture.
"""

import os
import sys

from PIL import Image, ImageDraw, ImageFont, ImageOps

# Page geometry, in pixels at 300 dpi (approximately A4).
PAGE_WIDTH = 2480
PAGE_HEIGHT = 3508
PAGE_DPI = (300.0, 300.0)

# Source drawings are scaled down so a full plate fits one page.
SOURCE_SCALE = 0.42
SCALE_BAR_SCALE = 0.42

# Spacing must exceed the grouping distances so that neighbouring
# artefacts are never linked: gap (0.025 * width) and narrow
# (0.07 * width) both fall well below these values.
COLUMN_SPACING = 260
ROW_SPACING = 340
TOP_MARGIN = 220
# Each label sits this far below the ink of its own drawing, so that
# the reader finds it inside the identifier reach of the lithic it names.
LABEL_GAP = 10
# Source drawings are cut to their ink before placement, so the gap is
# measured from the lithic and not from the white margin of the file.
INK_THRESHOLD = 200
LABEL_FONT_SIZE = 76
CAPTION_FONT_SIZE = 64
SCALE_BAR_TOP_MARGIN = 430

ROWS = [
    ["awbari.png", "qesem_cave.png", "replica_1.png"],
    ["KL3_5313_1.png", "rub_al_khali.png"],
]
LABELS = ["1", "2", "3", "4", "5"]
SCALE_CAPTION = "5 cm"

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
]


def load_font(size: int) -> ImageFont.ImageFont:
    """Load a scalable font, falling back to PIL's bitmap default."""
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    print("warning: no TrueType font found; labels will be small")
    return ImageFont.load_default()


def load_drawing(images_dir: str, name: str) -> Image.Image:
    """Load a source drawing, flattened onto white, cut to its ink, scaled down."""
    source = Image.open(os.path.join(images_dir, name))
    flattened = Image.new("RGB", source.size, "white")
    if source.mode in ("RGBA", "LA"):
        flattened.paste(source, mask=source.split()[-1])
    else:
        flattened.paste(source.convert("RGB"))

    ink = ImageOps.invert(flattened.convert("L")).point(
        lambda value: 255 if value > 255 - INK_THRESHOLD else 0
    )
    bbox = ink.getbbox()
    if bbox:
        flattened = flattened.crop(bbox)

    size = (
        int(flattened.width * SOURCE_SCALE),
        int(flattened.height * SOURCE_SCALE),
    )
    return flattened.resize(size, Image.LANCZOS)


def paste_row(
    page: Image.Image,
    draw: ImageDraw.ImageDraw,
    drawings: list,
    labels: list,
    top: int,
    font: ImageFont.ImageFont,
) -> int:
    """
    Place one row of artefacts, centred, each with its letter just below it.

    Returns
    -------
    int
        The y coordinate just below the row's tallest drawing and its label.
    """
    total_width = sum(d.width for d in drawings)
    total_width += COLUMN_SPACING * (len(drawings) - 1)
    x = (PAGE_WIDTH - total_width) // 2
    row_height = max(d.height for d in drawings)

    for drawing, label in zip(drawings, labels):
        page.paste(drawing, (x, top))
        centre = x + drawing.width // 2
        draw.text(
            (centre, top + drawing.height + LABEL_GAP),
            label, fill="black", font=font, anchor="ma",
        )
        x += drawing.width + COLUMN_SPACING

    return top + row_height + LABEL_GAP + LABEL_FONT_SIZE


def paste_scale_bar(
    page: Image.Image,
    draw: ImageDraw.ImageDraw,
    scales_dir: str,
    top: int,
    font: ImageFont.ImageFont,
) -> None:
    """Place the shared page scale bar with its caption beneath it."""
    source = Image.open(os.path.join(scales_dir, "sc_001.png"))
    flattened = Image.new("RGB", source.size, "white")
    if source.mode in ("RGBA", "LA"):
        flattened.paste(source, mask=source.split()[-1])
    else:
        flattened.paste(source.convert("RGB"))

    size = (
        int(flattened.width * SCALE_BAR_SCALE),
        int(flattened.height * SCALE_BAR_SCALE),
    )
    bar = flattened.resize(size, Image.LANCZOS)

    x = (PAGE_WIDTH - bar.width) // 2
    page.paste(bar, (x, top))
    draw.text(
        (PAGE_WIDTH // 2, top + bar.height + 30),
        SCALE_CAPTION, fill="black", font=font, anchor="ma",
    )


def build_plate(data_dir: str) -> Image.Image:
    """Composite the sample plate."""
    images_dir = os.path.join(data_dir, "images")
    scales_dir = os.path.join(data_dir, "scales")

    page = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")
    draw = ImageDraw.Draw(page)
    label_font = load_font(LABEL_FONT_SIZE)
    caption_font = load_font(CAPTION_FONT_SIZE)

    label_index = 0
    top = TOP_MARGIN
    for row in ROWS:
        drawings = [load_drawing(images_dir, name) for name in row]
        labels = LABELS[label_index:label_index + len(row)]
        label_index += len(row)
        top = paste_row(page, draw, drawings, labels, top, label_font)
        top += ROW_SPACING

    paste_scale_bar(
        page, draw, scales_dir,
        top - ROW_SPACING + SCALE_BAR_TOP_MARGIN, caption_font,
    )
    return page


def main(argv=None) -> int:
    """Build the fixture and write it to the path given, or into the package."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    data_dir = os.path.join(root, "pylithics", "data")

    args = sys.argv[1:] if argv is None else argv
    if args:
        destination = args[0]
    else:
        destination = os.path.join(data_dir, "pages", "sample_plate.png")
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)

    plate = build_plate(data_dir)
    plate.save(destination, dpi=PAGE_DPI)
    print(f"Wrote {destination} ({plate.width}x{plate.height}, 300 dpi)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
