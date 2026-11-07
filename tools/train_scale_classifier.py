r"""
Train the scale bar classifier from a labelled candidate table.

Reads the table written by ``tools/scale_candidates.py``, with its
``label`` column corrected to ``yes`` or ``no``, fits a random forest,
reports the held-out errors page by page (leave one page out), and
writes the forest as JSON for ``pylithics/models/scale_bar_classifier.json``.

Usage::

    python tools/train_scale_classifier.py --table results/scale_candidates/scale_candidates.csv \\
        --out pylithics/models/scale_bar_classifier.json
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from forest_training import fit, held_out_scores, print_importances, write_model  # noqa: E402
from pylithics.page_segmentation.scale_classifier import FEATURES  # noqa: E402


def read_table(path):
    """Return the feature matrix, the labels and the page of each row."""
    rows = list(csv.DictReader(open(path, newline='', encoding='utf-8')))
    rows = [r for r in rows if r['label'].strip().lower() in ('yes', 'no') and not r.get('rule')]
    x = np.array([[float(r[f]) for f in FEATURES] for r in rows], dtype=float)
    y = np.array([r['label'].strip().lower() == 'yes' for r in rows], dtype=int)
    pages = [r['page'] for r in rows]
    return x, y, pages, rows


def main(argv=None) -> int:
    """Train, report and write the model."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--table', required=True, help='labelled scale_candidates.csv')
    parser.add_argument('--out', required=True, help='where to write the model JSON')
    parser.add_argument('--trees', type=int, default=100)
    parser.add_argument('--depth', type=int, default=8)
    parser.add_argument('--leaf', type=int, default=2, help='smallest leaf, in rows')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args(argv)

    x, y, pages, rows = read_table(args.table)
    print(f"{len(y)} labelled candidate(s): {int(y.sum())} yes, {int((1 - y).sum())} no, "
          f"on {len(set(pages))} page(s)")
    settings = (args.trees, args.depth, args.leaf)
    scores = held_out_scores(x, y, pages, settings)
    misses = [i for i in range(len(y)) if (scores[i] >= args.threshold) != bool(y[i])]
    print(f"leave-one-page-out: {len(misses)} wrong of {len(y)}")
    for index in misses:
        row = rows[index]
        print(f"  {row['label']:>3} scored {scores[index]:.2f}: {row['page'][:60]} "
              f"candidate {row['n']} at [{row['x0']}, {row['y0']}, {row['x1']}, {row['y1']}]")

    forest = fit(x, y, *settings)
    print_importances(FEATURES, forest)
    write_model(args.out, FEATURES, forest, args.threshold, {
        'rows': int(len(y)), 'pages': len(set(pages)), 'settings': settings,
    })
    print(f"model -> {args.out} ({os.path.getsize(args.out) // 1024} KB)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
