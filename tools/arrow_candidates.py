r"""
List every arrow candidate on a folder of crops, for labelling.

Each crop is preprocessed as the analysis does it. The candidates are
the ones the analysis tries: the pieces of ink inside a closed contour,
cut from the working copy of the drawing (``arrow_integration``). The
analysis only tries the contours it recognised as scars; the tool has no
surface classification, so it tries every contour below the top level.
The package's detector decides each candidate in its two stages, the
notch method first and then, with the model given by ``--model``, the
classifier on the candidate's real ink. The table holds one row per
candidate: the decision, the ``stage`` that made it (``triangle``,
``model``, ``no model``, or the veto ``ink inside`` or ``too short``),
the scores, the chosen direction, the features in that orientation and,
as ``r_`` columns, in the other, and a ``label`` column. Boxes and
points are in source pixels.

Outputs under ``--out``: ``arrow_candidates.csv``; one overlay per crop in
``overlays/`` (green box and arrow = accepted, grey box = rejected); and
contact sheets ``sheet_NN.png`` of the candidates worth a look, numbered
by the ``k`` column. A cell shows the candidate's box in blue and, unless
``--plain``, the chosen direction in green (accepted) or orange.

Labels: ``yes`` (an arrow, the drawn direction is right), ``flipped`` (an
arrow, the drawn direction is reversed), ``no``. Known labels are read
from ``--labels`` and kept, matched by image and box, exactly or by the
box that overlaps most (at least half); the rest are prefilled from the
decision, with ``label_source`` saying ``known``, ``model`` or ``rule``.
A known label may carry the arrow's true compass ``angle``; then ``yes``
or ``flipped`` is decided against the direction drawn in this run, so a
label made against one model's direction stays right under the next.

Usage::

    python tools/arrow_candidates.py --images pylithics/data/images \\
        --out pylithics/data/results/arrow_candidates --model arrow_classifier.json
"""

import argparse
import csv
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pylithics.forest import load_forest  # noqa: E402
from pylithics.image_processing.config import get_config_manager  # noqa: E402
from pylithics.image_processing.importer import execute_preprocessing_pipeline  # noqa: E402
from pylithics.image_processing.modules import arrow_profile as profile  # noqa: E402
from pylithics.image_processing.modules.arrow_detection import (  # noqa: E402
    ArrowDetector, Assessment,
)
from pylithics.image_processing.modules.arrow_integration import marks_in_scar  # noqa: E402
from pylithics.image_processing.modules.contour_extraction import (  # noqa: E402
    extract_contours_with_hierarchy,
)

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
MAX_SIDE = 0.5          # a candidate's long side is at most this share of the image's
MIN_AREA = 4.0          # source pixels
MIN_OVERLAP = 0.5       # a known label applies to a box that overlaps its own by this share
SHEET_HEAD_RATIO = 1.3  # without a model, rejected rows below this stay off the sheets
SHEET_MIN_SCORE = 0.2   # with a model, rejected rows scoring under this stay off the sheets
CELL = 96               # thumbnail cell, pixels (``--cell``)
SHEET_COLUMNS = 10
DEFAULT_LABELS = os.path.join('pylithics', 'models', 'arrow_labels.csv')

COLUMNS = [
    'k', 'image', 'n', 'depth', 'x0', 'y0', 'x1', 'y1', 'arrow', 'stage', 'rule', 'reverse',
    'score_fwd', 'score_rev', 'angle', 'bx', 'by', 'tx', 'ty', *profile.FEATURES,
    *['r_' + name for name in profile.FEATURES], 'label', 'label_source',
]
Box = Tuple[int, int, int, int]


### CANDIDATES ###

def list_images(folder: str) -> List[str]:
    """Return the image files of a folder, sorted by name."""
    return sorted(
        os.path.join(folder, name) for name in os.listdir(folder)
        if name.lower().endswith(IMAGE_EXTENSIONS)
    )


def find_candidates(contours, hierarchy, ink, factor: int) -> List[Tuple]:
    """
    Return (marks, contour, depth, box) of every mark inside a contour below the top level.

    ``marks`` is the mark's crop (``arrow_integration.Marks``); the contour
    is in its pixels; the box is in source pixels. A mark is offered once,
    under the smallest contour that holds it.
    """
    limit = MAX_SIDE * max(ink.shape) / factor
    found, seen = [], set()
    order = sorted(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
    for i in order:
        depth = _depth(hierarchy, i)
        if depth == 0:
            continue
        marks = marks_in_scar(contours[i], ink, factor)
        for cnt in marks.contours:
            x, y, w, h = cv2.boundingRect(cnt)
            box = (*marks.to_source((x, y)), *marks.to_source((x + w, y + h)))
            if max(w, h) > limit * factor or w * h < MIN_AREA * factor ** 2 or box in seen:
                continue
            seen.add(box)
            found.append((marks, cnt, depth + 1, box))
    return found


def _depth(hierarchy, index: int) -> int:
    """Return how many ancestors a contour has in the hierarchy."""
    depth, parent = 0, hierarchy[index][3]
    while parent != -1:
        depth, parent = depth + 1, hierarchy[parent][3]
    return depth


def row_fields(assessment: Assessment, to_source=lambda point: point) -> Dict:
    """Return the table fields of one assessment; ``to_source`` maps its points to source pixels."""
    geometry = assessment.geometry
    rule = assessment.rule[1] if assessment.reversed else assessment.rule[0]
    back, tip = to_source(geometry['arrow_back']), to_source(geometry['arrow_tip'])
    row = {
        'arrow': int(assessment.accepted), 'stage': assessment.stage, 'rule': int(rule),
        'reverse': int(assessment.reversed),
        'score_fwd': _blank(assessment.scores[0]), 'score_rev': _blank(assessment.scores[1]),
        'angle': geometry['compass_angle'],
        'bx': back[0], 'by': back[1], 'tx': tip[0], 'ty': tip[1],
    }
    other = assessment.forward if assessment.reversed else assessment.reverse
    row.update({name: round(assessment.features[name], 4) for name in profile.FEATURES})
    row.update({'r_' + name: round(other[name], 4) for name in profile.FEATURES})
    return row


def _blank(value: Optional[float]) -> object:
    """Return a score rounded, or an empty cell for None."""
    return '' if value is None else round(value, 3)


def candidate_rows(
    path: str, config: Dict, detector: ArrowDetector
) -> Tuple[List[Dict], np.ndarray]:
    """Return the rows of one crop and the crop itself in colour."""
    image = cv2.imread(path)
    pre = execute_preprocessing_pipeline(path, config)
    if image is None or pre is None:
        return [], image
    contours, hierarchy = extract_contours_with_hierarchy(pre.image, os.path.basename(path), '')
    if not contours:
        return [], image
    rows = []
    dpi = _dpi(path)
    candidates = find_candidates(contours, hierarchy, pre.working, pre.upscale_factor)
    for marks, cnt, depth, box in candidates:
        assessment = detector.assess(cnt, marks.ink, dpi, marks.factor)
        if assessment is None:
            continue
        fields = row_fields(assessment, marks.to_source)
        fields.update({
            'image': os.path.basename(path), 'n': len(rows) + 1, 'depth': depth,
            'x0': box[0], 'y0': box[1], 'x1': box[2], 'y1': box[3],
        })
        rows.append(fields)
    return rows, image


def _dpi(path: str) -> Optional[float]:
    """Return the image's DPI tag, or None."""
    try:
        from PIL import Image
        with Image.open(path) as handle:
            dpi = handle.info.get('dpi')
        return float(dpi[0]) if dpi else None
    except (OSError, ValueError, TypeError):
        return None


### DRAWING ###

MIN_ARROW = 24   # a drawn arrow is at least this long on a sheet, so its head shows


def _draw_arrow(canvas, row: Dict, colour, scale: float, origin=(0, 0)) -> None:
    """Draw the row's arrow, scaled and shifted onto the canvas, never shorter than MIN_ARROW."""
    start = ((row['bx'] - origin[0]) * scale, (row['by'] - origin[1]) * scale)
    end = ((row['tx'] - origin[0]) * scale, (row['ty'] - origin[1]) * scale)
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = (dx * dx + dy * dy) ** 0.5
    if 0 < length < MIN_ARROW:
        grow = (MIN_ARROW - length) / 2.0
        start = (start[0] - dx / length * grow, start[1] - dy / length * grow)
        end = (end[0] + dx / length * grow, end[1] + dy / length * grow)
    cv2.arrowedLine(canvas, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])),
                    colour, 2, tipLength=0.35)


def draw_overlay(image: np.ndarray, rows: List[Dict], path: str) -> None:
    """Write the crop at twice its size with every candidate boxed and the arrows found."""
    canvas = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    for row in rows:
        colour = (0, 160, 0) if row['arrow'] else (150, 150, 150)
        cv2.rectangle(
            canvas, (2 * row['x0'], 2 * row['y0']), (2 * row['x1'], 2 * row['y1']), colour, 1,
        )
        cv2.putText(
            canvas, str(row['n']), (2 * row['x0'], max(2 * row['y0'] - 2, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1,
        )
        if row['arrow']:
            _draw_arrow(canvas, row, (0, 200, 0), 2.0)
    cv2.imwrite(path, canvas)


def thumbnail(
    image: np.ndarray, row: Dict, cell_size: int = CELL, plain: bool = False
) -> np.ndarray:
    """Return one sheet cell: the candidate, its box, its number and (unless plain) its arrow."""
    width, height = row['x1'] - row['x0'], row['y1'] - row['y0']
    pad = max(6, int(0.3 * max(width, height)))
    x0, y0 = max(row['x0'] - pad, 0), max(row['y0'] - pad, 0)
    x1, y1 = min(row['x1'] + pad, image.shape[1]), min(row['y1'] + pad, image.shape[0])
    crop = image[y0:y1, x0:x1]
    scale = (cell_size - 4) / max(crop.shape[0], crop.shape[1], 1)
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    cell = np.full((cell_size, cell_size, 3), 255, dtype=np.uint8)
    cell[2:2 + crop.shape[0], 2:2 + crop.shape[1]] = crop
    origin = (x0 - 2 / scale, y0 - 2 / scale)
    if not plain:
        _draw_arrow(cell, row, (0, 200, 0) if row['arrow'] else (0, 140, 255), scale, origin)
    cv2.rectangle(
        cell, (int((row['x0'] - origin[0]) * scale), int((row['y0'] - origin[1]) * scale)),
        (int((row['x1'] - origin[0]) * scale), int((row['y1'] - origin[1]) * scale)),
        (255, 200, 120), 1,
    )
    cv2.putText(
        cell, str(row['k']), (2, cell_size - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1,
    )
    cv2.rectangle(cell, (0, 0), (cell_size - 1, cell_size - 1), (200, 200, 200), 1)
    return cell


def write_sheets(cells: List[np.ndarray], out: str, cell_size: int = CELL) -> int:
    """Tile the cells onto numbered square sheets; return the number of sheets."""
    columns = max(1, SHEET_COLUMNS * CELL // cell_size)
    per_sheet = columns * columns
    count = 0
    for start in range(0, len(cells), per_sheet):
        batch = cells[start:start + per_sheet]
        sheet = np.full((columns * cell_size, columns * cell_size, 3), 255, dtype=np.uint8)
        for i, cell in enumerate(batch):
            r, c = divmod(i, columns)
            sheet[r * cell_size:(r + 1) * cell_size, c * cell_size:(c + 1) * cell_size] = cell
        count += 1
        cv2.imwrite(os.path.join(out, f'sheet_{count:02d}.png'), sheet)
    return count


### LABELS ###

Known = Tuple[Box, str, Optional[float]]   # box, label, true compass angle when known


def read_labels(path: str) -> Dict[str, List[Known]]:
    """Return the known labels of each image as (box, label, angle) rows, or an empty dict."""
    known: Dict[str, List[Known]] = {}
    if not path or not os.path.isfile(path):
        return known
    with open(path, newline='', encoding='utf-8') as handle:
        for r in csv.DictReader(handle):
            if r.get('label', '').strip():
                box = (int(r['x0']), int(r['y0']), int(r['x1']), int(r['y1']))
                angle = float(r['angle']) if r.get('angle', '').strip() else None
                known.setdefault(r['image'], []).append((box, r['label'].strip().lower(), angle))
    return known


def known_label(row: Dict, known: Dict[str, List[Known]]) -> Optional[str]:
    """
    Return the known label whose box overlaps the row's most, if at least half; else None.

    With a true angle on the label, an arrow is ``yes`` when the direction
    drawn in this row is within a right angle of it, else ``flipped``.
    """
    box = (row['x0'], row['y0'], row['x1'], row['y1'])
    best, best_overlap = None, MIN_OVERLAP
    for other, label, angle in known.get(row['image'], []):
        overlap = _overlap(box, other)
        if overlap >= best_overlap:
            best, best_overlap = (label, angle), overlap
    if best is None:
        return None
    label, angle = best
    if angle is None or label == 'no':
        return label
    apart = abs((float(row['angle']) - angle + 180.0) % 360.0 - 180.0)
    return 'yes' if apart <= 90.0 else 'flipped'


def _overlap(a: Box, b: Box) -> float:
    """Return the shared area of two boxes over the area of their union."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    shared = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - shared
    return shared / union if union > 0 else 0.0


def prefill(row: Dict, known: Dict[str, List[Known]], with_model: bool) -> None:
    """Set the label of a row from the known labels, else from the decision."""
    label = known_label(row, known)
    if label is not None:
        row['label'], row['label_source'] = label, 'known'
    else:
        row['label'] = 'yes' if row['arrow'] else 'no'
        row['label_source'] = 'model' if with_model else 'rule'


def _best_score(row: Dict) -> float:
    """Return the higher model score of a row, 0 without a model."""
    scores = [s for s in (row['score_fwd'], row['score_rev']) if s != '']
    return max(scores) if scores else 0.0


def _on_sheet(row: Dict, args: argparse.Namespace) -> bool:
    """Say whether a row goes on the sheets under the command line's choice."""
    if args.pick:
        return row['k'] in args.pick
    if args.sheet:
        return row['label'] in args.sheet.split(',')
    if args.all or row['arrow']:
        return True
    if args.model:
        return _best_score(row) >= args.min_score
    return bool(row['rule']) or row['head_ratio'] >= SHEET_HEAD_RATIO


def sheet_order(row: Dict) -> Tuple:
    """Sort key: accepted first, then by the best score, then by head ratio."""
    return (-row['arrow'], -_best_score(row), -row['head_ratio'])


### MAIN ###

def parse_args(argv=None) -> argparse.Namespace:
    """Return the parsed command line."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--images', default=os.path.join('pylithics', 'data', 'images'))
    parser.add_argument(
        '--out', default=os.path.join('pylithics', 'data', 'results', 'arrow_candidates'),
    )
    parser.add_argument('--labels', default=DEFAULT_LABELS, help='CSV of known labels to keep')
    parser.add_argument('--model', help='Model JSON for stage 2 (default: stage 1 alone)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Model score to accept')
    parser.add_argument('--limit', type=int, default=0, help='Only the first N images')
    parser.add_argument('--only', action='append', default=[], metavar='TEXT',
                        help='Only images whose name contains TEXT (repeatable)')
    parser.add_argument('--cell', type=int, default=CELL, help='Thumbnail cell size in pixels')
    parser.add_argument('--all', action='store_true', help='Put every candidate on the sheets')
    parser.add_argument('--sheet', metavar='LABELS',
                        help='Put only candidates with these labels on the sheets (comma list)')
    parser.add_argument('--pick', metavar='K', type=lambda s: {int(k) for k in s.split(',')},
                        help='Put only these k numbers on the sheets (comma list)')
    parser.add_argument('--min_score', type=float, default=SHEET_MIN_SCORE,
                        help='A rejected candidate scoring at least this goes on the sheets')
    parser.add_argument('--plain', action='store_true',
                        help='Draw no arrows on the sheets, only the candidate box')
    return parser.parse_args(argv)


def make_detector(args: argparse.Namespace) -> ArrowDetector:
    """Return a detector with the model given for stage 2, or stage 1 alone."""
    settings = {'classifier': {'enabled': bool(args.model), 'threshold': args.threshold}}
    model = load_forest(args.model) if args.model else None
    return ArrowDetector(settings, model=model)


def main(argv=None) -> int:
    """Run the tool."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')
    config = get_config_manager().config
    detector = make_detector(args)
    known = read_labels(args.labels)
    overlays = os.path.join(args.out, 'overlays')
    os.makedirs(overlays, exist_ok=True)
    images = list_images(args.images)
    if args.only:
        images = [p for p in images if any(text in os.path.basename(p) for text in args.only)]
    if args.limit:
        images = images[:args.limit]
    table, cells = [], []
    for position, path in enumerate(images, 1):
        rows, image = candidate_rows(path, config, detector)
        for row in rows:
            row['k'] = len(table) + 1
            prefill(row, known, bool(args.model))
            table.append(row)
        stem = os.path.splitext(os.path.basename(path))[0]
        draw_overlay(image, rows, os.path.join(overlays, stem + '.png'))
        cells.extend((row, image) for row in rows if _on_sheet(row, args))
        print(f'\r{position}/{len(images)} images, {len(table)} candidates', end='', flush=True)
    print()
    cells.sort(key=lambda pair: sheet_order(pair[0]))
    sheets = write_sheets(
        [thumbnail(image, row, args.cell, args.plain) for row, image in cells],
        args.out, args.cell,
    )
    table_path = os.path.join(args.out, 'arrow_candidates.csv')
    with open(table_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(table)
    kept = sum(r['label_source'] == 'known' for r in table)
    print(f'{len(table)} candidates: {sum(r["arrow"] for r in table)} accepted, '
          f'{sum(r["rule"] for r in table)} by the rule; {len(cells)} on {sheets} sheet(s); '
          f'known labels kept {kept}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
