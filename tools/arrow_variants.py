r"""
Make labelled arrow rows from a folder of arrow images.

Each image holds one arrow, dark on white, drawn pointing the way
``--points`` says (``right`` unless told otherwise). The tool makes
variants of each arrow at the sizes arrows have on a drawing, at every
``--step`` degrees of rotation, and at three line weights (thin, as
drawn, thick). A variant that breaks into pieces (a dot tail at 10 pixels) is
skipped. Each variant is measured by the package's arrow detector
and written as one row of the same table ``arrow_candidates.py`` writes,
so ``train_arrow_classifier.py`` reads it unchanged.

The label needs no hand work. The true direction of a variant follows
from the rotation, so a variant is ``yes`` when the detector's chosen
direction agrees with it and ``flipped`` when it does not. Images under
``--others`` go through the same variants and are labelled ``no``; a
non-arrow image may hold several marks, and each piece is a row. The
``image`` column holds the source file, so grouped held-out scoring
keeps all variants of one source together.

Outputs under ``--out``: ``arrow_variants.csv`` and ``sheet_NN.png``
with a sample of the variants and the detector's direction drawn.

Usage::

    python tools/arrow_variants.py --arrows ~/arrows --out pylithics/data/results/arrow_variants
    python tools/train_arrow_classifier.py \\
        --table pylithics/data/results/arrow_variants/arrow_variants.csv \\
        --out pylithics/models/arrow_classifier.json
"""

import argparse
import csv
import math
import os
import sys
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arrow_candidates import (  # noqa: E402
    COLUMNS, list_images, row_fields, thumbnail, write_sheets,
)
from pylithics.image_processing.modules.arrow_detection import ArrowDetector  # noqa: E402

LENGTHS = (10, 16, 24, 40)      # long side of the variant, source pixels
STEP = 15                       # degrees between rotations
WEIGHTS = ('thin', 'drawn', 'thick')
INK = 128                       # grey level below which a pixel is ink
MARGIN = 6                      # white border around a variant, pixels
SAMPLES = 4                     # variants of each source on the sheets
WHOLE = 0.8                     # an arrow variant must keep this share of its ink in one piece
MIN_AREA = 4.0                  # a non-arrow piece smaller than this is not a candidate
EXTRA_COLUMNS = ['variant', 'truth']

# Unit vector of the drawn direction, in image coordinates (y down).
POINTS = {'right': (1.0, 0.0), 'left': (-1.0, 0.0), 'up': (0.0, -1.0), 'down': (0.0, 1.0)}


### VARIANTS ###

def read_ink(path: str) -> Optional[np.ndarray]:
    """Return the arrow as a white-on-black mask cropped to its ink, or None."""
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    mask = (image < INK).astype(np.uint8) * 255
    points = cv2.findNonZero(mask)
    if points is None:
        return None
    x, y, w, h = cv2.boundingRect(points)
    return mask[y:y + h, x:x + w]


def scaled(mask: np.ndarray, length: int) -> np.ndarray:
    """Return the mask with its long side at ``length`` pixels."""
    factor = length / max(mask.shape)
    size = (max(1, round(mask.shape[1] * factor)), max(1, round(mask.shape[0] * factor)))
    small = cv2.resize(mask, size, interpolation=cv2.INTER_AREA)
    return (small >= INK).astype(np.uint8) * 255


def weighted(mask: np.ndarray, weight: str) -> np.ndarray:
    """Return the mask one pixel thinner or thicker, or as drawn."""
    kernel = np.ones((3, 3), dtype=np.uint8)
    if weight == 'thin':
        thinner = cv2.erode(mask, kernel)
        return thinner if cv2.countNonZero(thinner) else mask
    if weight == 'thick':
        return cv2.dilate(mask, kernel)
    return mask


def rotated(mask: np.ndarray, degrees: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return the mask rotated about its centre with a white margin, and the 2x2 rotation."""
    height, width = mask.shape
    centre = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, degrees, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int(math.ceil(height * sin + width * cos)) + 2 * MARGIN
    new_h = int(math.ceil(height * cos + width * sin)) + 2 * MARGIN
    matrix[0, 2] += new_w / 2.0 - centre[0]
    matrix[1, 2] += new_h / 2.0 - centre[1]
    turned = cv2.warpAffine(mask, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return (turned >= INK).astype(np.uint8) * 255, matrix[:, :2]


def truth_angle(direction: Tuple[float, float], rotation: np.ndarray) -> float:
    """Return the compass angle of the drawn direction after the rotation (down 0, right 270)."""
    dx, dy = rotation @ np.array(direction)
    angle_deg = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
    return round((270.0 + angle_deg) % 360.0, 1)


def variants(
    mask: np.ndarray, direction: Tuple[float, float], lengths, step: int
) -> Iterator[Tuple[str, np.ndarray, float]]:
    """Yield (name, mask, true compass angle) for every size, rotation and weight."""
    for length in lengths:
        base = scaled(mask, length)
        for weight in WEIGHTS:
            shaped = weighted(base, weight)
            for degrees in range(0, 360, step):
                turned, rotation = rotated(shaped, degrees)
                yield f"L{length}_a{degrees:03d}_{weight}", turned, truth_angle(direction, rotation)


def whole_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Return the outer contour of the mask when the mask is one piece, else None.

    A small variant of an arrow with a ring or a dot at its tail breaks
    into pieces. One piece is not the arrow, so the variant is skipped.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    piece = np.zeros_like(mask)
    cv2.drawContours(piece, [biggest], 0, 255, cv2.FILLED)
    kept = cv2.countNonZero(cv2.bitwise_and(piece, mask)) / max(1, cv2.countNonZero(mask))
    return biggest if kept >= WHOLE else None


def every_piece(mask: np.ndarray) -> List[np.ndarray]:
    """Return every outer contour of the mask with at least ``MIN_AREA`` pixels."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [c for c in contours if cv2.contourArea(c) >= MIN_AREA]


### ROWS ###

def direction_label(angle: float, truth: Optional[float]) -> str:
    """Return ``yes`` when the angle is within a quarter turn of the truth, else ``flipped``."""
    if truth is None:
        return 'no'
    gap = abs((angle - truth + 180.0) % 360.0 - 180.0)
    return 'yes' if gap <= 90.0 else 'flipped'


def source_rows(
    path: str, direction: Optional[Tuple[float, float]], detector: ArrowDetector,
    args: argparse.Namespace,
) -> Tuple[List[Dict], List[np.ndarray]]:
    """Return the rows of one source image and the colour masks of its variants."""
    mask = read_ink(path)
    if mask is None:
        print(f"no ink in {path}", file=sys.stderr)
        return [], []
    rows, images = [], []
    for name, turned, truth in variants(mask, direction or POINTS['right'], args.lengths,
                                        args.step):
        pieces = [whole_contour(turned)] if direction else every_piece(turned)
        for contour in pieces:
            assessment = detector.assess(contour) if contour is not None else None
            if assessment is None:
                continue
            fields = row_fields(assessment)
            x, y, w, h = cv2.boundingRect(contour)
            fields.update({
                'image': os.path.basename(path), 'n': len(rows) + 1, 'depth': 0,
                'x0': x, 'y0': y, 'x1': x + w, 'y1': y + h, 'variant': name,
                'truth': truth if direction else '',
                'label': direction_label(fields['angle'], truth if direction else None),
                'label_source': 'known',
            })
            rows.append(fields)
            images.append(cv2.cvtColor(255 - turned, cv2.COLOR_GRAY2BGR))
    return rows, images


def sample_cells(rows: List[Dict], images: List[np.ndarray], count: int) -> List[np.ndarray]:
    """Return sheet cells for ``count`` evenly spaced variants of one source."""
    if not rows:
        return []
    picks = np.linspace(0, len(rows) - 1, min(count, len(rows))).astype(int)
    return [thumbnail(images[i], rows[i]) for i in picks]


def report(rows: List[Dict]) -> None:
    """Print how the present detector did on the variants, against the known truth."""
    arrows = [r for r in rows if r['label'] != 'no']
    others = [r for r in rows if r['label'] == 'no']
    found = sum(r['arrow'] for r in arrows)
    flipped = sum(1 for r in arrows if r['arrow'] and r['label'] == 'flipped')
    print(f"{len(rows)} row(s) from {len({r['image'] for r in rows})} source image(s)")
    print(f"arrows: {len(arrows)}; the present detector found {found}, "
          f"{flipped} of them pointing the wrong way")
    if others:
        print(f"non-arrows: {len(others)}; the present detector accepted "
              f"{sum(r['arrow'] for r in others)}")


### MAIN ###

def parse_args(argv=None) -> argparse.Namespace:
    """Return the command line options."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--arrows', required=True, help='folder of arrow images, one arrow each')
    parser.add_argument('--out', required=True, help='output folder')
    parser.add_argument('--others', help='folder of non-arrow images, labelled no')
    parser.add_argument('--points', choices=sorted(POINTS), default='right',
                        help='the way the arrows are drawn pointing (default right)')
    sizes = ' '.join(map(str, LENGTHS))
    parser.add_argument('--lengths', type=int, nargs='+', default=list(LENGTHS),
                        help=f'long sides of the variants in pixels (default {sizes})')
    parser.add_argument('--step', type=int, default=STEP,
                        help=f'degrees between rotations (default {STEP})')
    parser.add_argument('--model', help='model file to assess with (default: the configuration)')
    parser.add_argument('--samples', type=int, default=SAMPLES,
                        help=f'variants of each source shown on the sheets (default {SAMPLES})')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Make the variants, write the table and the sheets, report."""
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    detector = ArrowDetector(
        {'classifier': {'enabled': True, 'model': args.model}} if args.model else None
    )
    table, cells = [], []
    jobs = [(path, POINTS[args.points]) for path in list_images(args.arrows)]
    if args.others:
        jobs += [(path, None) for path in list_images(args.others)]
    for path, direction in jobs:
        rows, images = source_rows(path, direction, detector, args)
        for row in rows:
            row['k'] = len(table) + 1
            table.append(row)
        cells += sample_cells(rows, images, args.samples)
    table_path = os.path.join(args.out, 'arrow_variants.csv')
    with open(table_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS + EXTRA_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(table)
    sheets = write_sheets(cells, args.out)
    report(table)
    print(f"table -> {table_path}; {sheets} sheet(s) in {args.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
