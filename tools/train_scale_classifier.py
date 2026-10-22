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
import json
import os
import sys
from datetime import date

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pylithics.page_segmentation.scale_classifier import FEATURES  # noqa: E402


def read_table(path):
    """Return the feature matrix, the labels and the page of each row."""
    rows = list(csv.DictReader(open(path, newline='', encoding='utf-8')))
    rows = [r for r in rows if r['label'].strip().lower() in ('yes', 'no') and not r.get('rule')]
    x = np.array([[float(r[f]) for f in FEATURES] for r in rows], dtype=float)
    y = np.array([r['label'].strip().lower() == 'yes' for r in rows], dtype=int)
    pages = [r['page'] for r in rows]
    return x, y, pages, rows


def fit(x, y, trees, depth, leaf):
    """Fit a random forest; return it."""
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=trees, max_depth=depth, min_samples_leaf=leaf,
        class_weight='balanced', random_state=0,
    )
    model.fit(x, y)
    return model


def leave_one_page_out(x, y, pages, settings, threshold):
    """Score every row with a model that never saw its page; return the misses."""
    pages = np.array(pages)
    misses = []
    for page in sorted(set(pages)):
        held = pages == page
        if held.all() or len(set(y[~held])) < 2:
            continue
        model = fit(x[~held], y[~held], *settings)
        scores = model.predict_proba(x[held])[:, 1]
        for index, score_value in zip(np.where(held)[0], scores):
            if (score_value >= threshold) != bool(y[index]):
                misses.append((index, float(score_value)))
    return misses


def tree_arrays(estimator):
    """Export one fitted tree as the parallel node arrays the package reads."""
    tree = estimator.tree_
    counts = tree.value[:, 0, :]
    share = counts[:, 1] / np.maximum(1e-9, counts.sum(axis=1))
    return {
        'left': tree.children_left.tolist(), 'right': tree.children_right.tolist(),
        'feature': np.where(tree.children_left == -1, 0, tree.feature).tolist(),
        'threshold': np.round(tree.threshold, 5).tolist(),
        'value': np.round(share, 4).tolist(),
    }


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
    misses = leave_one_page_out(x, y, pages, settings, args.threshold)
    print(f"leave-one-page-out: {len(misses)} wrong of {len(y)}")
    for index, score_value in misses:
        row = rows[index]
        print(f"  {row['label']:>3} scored {score_value:.2f}: {row['page'][:60]} "
              f"candidate {row['n']} at [{row['x0']}, {row['y0']}, {row['x1']}, {row['y1']}]")

    forest = fit(x, y, *settings)
    for name, weight in sorted(zip(FEATURES, forest.feature_importances_), key=lambda p: -p[1]):
        print(f"  {name:<16} {weight:.3f}")
    model = {
        'kind': 'forest', 'features': FEATURES,
        'trees': [tree_arrays(t) for t in forest.estimators_],
        'threshold': args.threshold, 'trained': date.today().isoformat(),
        'rows': int(len(y)), 'pages': len(set(pages)), 'settings': settings,
    }
    with open(args.out, 'w', encoding='utf-8') as handle:
        json.dump(model, handle, separators=(',', ':'))
    print(f"model -> {args.out} ({os.path.getsize(args.out) // 1024} KB)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
