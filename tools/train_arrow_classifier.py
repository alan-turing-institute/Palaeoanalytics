r"""
Train the arrow classifier from a labelled candidate table.

Reads the table written by ``tools/arrow_candidates.py`` with its
``label`` column corrected to ``yes``, ``flipped`` or ``no`` (only rows
whose ``label_source`` is ``known`` count; the rest are prefilled). Each
candidate gives two training rows, one per orientation: a ``yes`` is
positive as drawn and negative reversed, a ``flipped`` the other way
round, a ``no`` negative both ways. The forest is scored with grouped
folds by crop, the misses, false arrows and flips are reported, and the
forest is written as JSON for ``pylithics/models/arrow_classifier.json``.

Usage::

    python tools/train_arrow_classifier.py \\
        --table pylithics/data/results/arrow_candidates/arrow_candidates.csv \\
        --out pylithics/models/arrow_classifier.json
"""

import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from forest_training import fit, held_out_scores, print_importances, write_model  # noqa: E402
from pylithics.image_processing.modules.arrow_profile import FEATURES  # noqa: E402

LABELS = ('yes', 'flipped', 'no')


def read_table(path: str) -> List[Dict]:
    """Return the labelled rows of the table."""
    with open(path, newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    return [
        r for r in rows
        if r['label'].strip().lower() in LABELS and r.get('label_source', 'known') == 'known'
    ]


def thin_negatives(rows: List[Dict], limit: int) -> List[Dict]:
    """Keep every arrow and at most ``limit`` non-arrows per crop, the best-scoring first."""
    kept, per_crop = [], {}
    for row in sorted(rows, key=lambda r: -float(r.get('head_ratio', 0))):
        if row['label'].strip().lower() != 'no':
            kept.append(row)
            continue
        count = per_crop.get(row['image'], 0)
        if count < limit:
            kept.append(row)
            per_crop[row['image']] = count + 1
    return kept


def training_rows(rows: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return two feature rows per candidate, their targets and their crop names."""
    x, y, groups = [], [], []
    for row in rows:
        label = row['label'].strip().lower()
        forward = [float(row[f]) for f in FEATURES]
        reverse = [float(row['r_' + f]) for f in FEATURES]
        x += [forward, reverse]
        y += [int(label == 'yes'), int(label == 'flipped')]
        groups += [row['image'], row['image']]
    return np.array(x, dtype=float), np.array(y, dtype=int), groups


def evaluate(rows: List[Dict], scores: np.ndarray, threshold: float) -> Dict[str, List[int]]:
    """Return the indices of the missed arrows, the false arrows and the flips."""
    result = {'missed': [], 'false': [], 'flipped': []}
    for i, row in enumerate(rows):
        forward, reverse = scores[2 * i], scores[2 * i + 1]
        label = row['label'].strip().lower()
        is_arrow = max(forward, reverse) >= threshold
        if label == 'no':
            if is_arrow:
                result['false'].append(i)
        elif not is_arrow:
            result['missed'].append(i)
        elif (forward >= reverse) != (label == 'yes'):
            result['flipped'].append(i)
    return result


def report(rows: List[Dict], scores: np.ndarray, threshold: float, show: int) -> None:
    """Print the held-out errors."""
    arrows = sum(1 for r in rows if r['label'].strip().lower() != 'no')
    outcome = evaluate(rows, scores, threshold)
    print(f"held out: {len(outcome['missed'])} missed and {len(outcome['flipped'])} flipped "
          f"of {arrows} arrow(s); {len(outcome['false'])} false arrow(s) of {len(rows) - arrows}")
    for kind, indices in outcome.items():
        for i in indices[:show]:
            row = rows[i]
            print(f"  {kind:<8} k={row['k']:>6} {row['image'][:50]:<50} n={row['n']} "
                  f"fwd {scores[2 * i]:.2f} rev {scores[2 * i + 1]:.2f}")


def write_scores(rows: List[Dict], scores: np.ndarray, path: str) -> None:
    """Write each row's key, label and held-out scores, for a look at the errors."""
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['k', 'image', 'x0', 'y0', 'x1', 'y1', 'label', 'fwd', 'rev', 'length',
                         'depth'])
        for i, row in enumerate(rows):
            writer.writerow([row['k'], row['image'], row['x0'], row['y0'], row['x1'], row['y1'],
                             row['label'], round(scores[2 * i], 3), round(scores[2 * i + 1], 3),
                             row['length'], row['depth']])


def main(argv=None) -> int:
    """Train, report and write the model."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--table', required=True, help='labelled arrow_candidates.csv')
    parser.add_argument('--out', required=True, help='where to write the model JSON')
    parser.add_argument('--trees', type=int, default=100)
    parser.add_argument('--depth', type=int, default=10)
    parser.add_argument('--leaf', type=int, default=2, help='smallest leaf, in rows')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--folds', type=int, default=10, help='grouped folds by crop')
    parser.add_argument('--show', type=int, default=15, help='errors listed per kind')
    parser.add_argument('--scores', help='write the held-out scores of every row to this CSV')
    parser.add_argument('--negatives', type=int, default=0,
                        help='keep at most this many no rows per crop (0 = all)')
    args = parser.parse_args(argv)

    rows = read_table(args.table)
    if args.negatives:
        rows = thin_negatives(rows, args.negatives)
    x, y, groups = training_rows(rows)
    labels = [r['label'].strip().lower() for r in rows]
    counts = {label: labels.count(label) for label in LABELS}
    print(f"{len(rows)} labelled candidate(s) on {len(set(groups))} crop(s): {counts}")
    settings = (args.trees, args.depth, args.leaf)
    scores = held_out_scores(x, y, groups, settings, args.folds)
    report(rows, scores, args.threshold, args.show)
    if args.scores:
        write_scores(rows, scores, args.scores)

    forest = fit(x, y, *settings)
    print_importances(FEATURES, forest)
    write_model(args.out, FEATURES, forest, args.threshold, {
        'rows': int(len(y)), 'crops': len(set(groups)), 'settings': settings,
    })
    print(f"model -> {args.out} ({os.path.getsize(args.out) // 1024} KB)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
