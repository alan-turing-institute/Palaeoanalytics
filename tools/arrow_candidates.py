r"""
List every arrow candidate on a folder of crops, for labelling.

Each crop is preprocessed as the analysis does it. Every small contour at
any level of the hierarchy is a candidate: an arrow sits inside a closed
scar (nested), inside an open scar (at the scar's level) or in the
background (top level). The package's detector assesses each candidate
both ways along its axis, with the model given by ``--model`` (else the
hand rule), and the table holds one row per candidate: the decision, the
scores, the chosen direction, the profile features in that orientation
and, as ``r_`` columns, in the other, and a ``label`` column.

Outputs under ``--out``: ``arrow_candidates.csv``; one overlay per crop in
``overlays/`` (green box and arrow = accepted, grey box = rejected); and
contact sheets ``sheet_NN.png`` of the candidates worth a look, numbered
by the ``k`` column. A cell shows the candidate's box in blue and, unless
``--plain``, the chosen direction in green (accepted) or orange.

Labels: ``yes`` (an arrow, the drawn direction is right), ``flipped`` (an
arrow, the drawn direction is reversed), ``no``. Known labels are read
from ``--labels`` and kept; the rest are prefilled from the decision,
with ``label_source`` saying ``known``, ``model`` or ``rule``.

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
from pylithics.image_processing.modules.contour_extraction import (  # noqa: E402
    extract_contours_with_hierarchy,
)

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
MAX_SIDE = 0.2          # a candidate's long side is at most this share of the image's
MIN_AREA = 4.0          # source pixels
SHEET_HEAD_RATIO = 1.3  # without a model, rejected rows below this stay off the sheets
SHEET_MIN_SCORE = 0.2   # with a model, rejected rows scoring under this stay off the sheets
CELL = 96               # thumbnail cell, pixels (``--cell``)
SHEET_COLUMNS = 10
DEFAULT_LABELS = os.path.join('pylithics', 'models', 'arrow_labels.csv')

COLUMNS = [
    'k', 'image', 'n', 'depth', 'x0', 'y0', 'x1', 'y1', 'arrow', 'rule', 'reverse',
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


def find_candidates(contours, hierarchy, shape) -> List[Tuple[int, int, Box]]:
    """Return (index, depth, box) of every contour small enough to be an arrow."""
    limit = MAX_SIDE * max(shape)
    found = []
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if max(w, h) > limit or cv2.contourArea(cnt) < MIN_AREA:
            continue
        found.append((i, _depth(hierarchy, i), (x, y, x + w, y + h)))
    return found


def _depth(hierarchy, index: int) -> int:
    """Return how many ancestors a contour has in the hierarchy."""
    depth, parent = 0, hierarchy[index][3]
    while parent != -1:
        depth, parent = depth + 1, hierarchy[parent][3]
    return depth


def row_fields(assessment: Assessment) -> Dict:
    """Return the table fields of one assessment."""
    geometry = assessment.geometry
    rule = assessment.rule[1] if assessment.reversed else assessment.rule[0]
    row = {
        'arrow': int(assessment.accepted), 'rule': int(rule), 'reverse': int(assessment.reversed),
        'score_fwd': _blank(assessment.scores[0]), 'score_rev': _blank(assessment.scores[1]),
        'angle': geometry['compass_angle'],
        'bx': geometry['arrow_back'][0], 'by': geometry['arrow_back'][1],
        'tx': geometry['arrow_tip'][0], 'ty': geometry['arrow_tip'][1],
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
    for index, depth, box in find_candidates(contours, hierarchy, pre.image.shape):
        assessment = detector.assess(contours[index])
        if assessment is None:
            continue
        fields = row_fields(assessment)
        fields.update({
            'image': os.path.basename(path), 'n': len(rows) + 1, 'depth': depth,
            'x0': box[0], 'y0': box[1], 'x1': box[2], 'y1': box[3],
        })
        rows.append(fields)
    return rows, image


### DRAWING ###

def _draw_arrow(canvas, row: Dict, colour, scale: float, origin=(0, 0)) -> None:
    """Draw the row's arrow, scaled and shifted onto the canvas."""
    start = (int((row['bx'] - origin[0]) * scale), int((row['by'] - origin[1]) * scale))
    end = (int((row['tx'] - origin[0]) * scale), int((row['ty'] - origin[1]) * scale))
    cv2.arrowedLine(canvas, start, end, colour, 1, tipLength=0.3)


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

def read_labels(path: str) -> Dict[Tuple, str]:
    """Return known labels keyed by image and box, or an empty dict."""
    if not path or not os.path.isfile(path):
        return {}
    with open(path, newline='', encoding='utf-8') as handle:
        return {
            (r['image'], r['x0'], r['y0'], r['x1'], r['y1']): r['label'].strip().lower()
            for r in csv.DictReader(handle) if r.get('label', '').strip()
        }


def prefill(row: Dict, known: Dict[Tuple, str], with_model: bool) -> None:
    """Set the label of a row from the known labels, else from the decision."""
    key = (row['image'], str(row['x0']), str(row['y0']), str(row['x1']), str(row['y1']))
    if key in known:
        row['label'], row['label_source'] = known[key], 'known'
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
    parser.add_argument('--model', help='Model JSON to decide with (default: the hand rule)')
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
    """Return a detector that uses the model given, or the hand rule."""
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
