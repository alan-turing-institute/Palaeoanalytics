"""
List every scale bar candidate on a folder of pages, for labelling.

For each page the candidate step of ``pylithics-pages`` runs on the raw
ink, and one row is written for each candidate: its page, box, features
and a ``label`` column (``yes`` or ``no``). The label comes from the
known labels file where the page and box match, else from the bundled
model, else from the shape rules; ``label_source`` says which. A
candidate the connector rule rejects is ``no`` with ``rule`` set, and
training leaves it out. Correct
the rows that are wrong, then train with ``tools/train_scale_classifier.py``.

An overlay for each page shows the candidates numbered, purple when the
label is ``yes`` and grey when it is ``no``, with the model's score, so
a row can be checked against the plate.

Usage::

    python tools/scale_candidates.py --pages pylithics/data/pages --out results/scale_candidates
"""

import argparse
import csv
import logging
import os
import sys
import unicodedata

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pylithics.image_processing.config import (  # noqa: E402
    get_config_manager, get_page_segmentation_config,
)
from pylithics.page_segmentation import grouping  # noqa: E402
from pylithics.page_segmentation.detection import (  # noqa: E402
    build_detection_mask, find_components, load_page,
)
from pylithics.page_segmentation.scale_classifier import (  # noqa: E402
    FEATURES, MODELS_DIR, ModelUnavailable, candidate_features, load_model, score,
    _connector_rule,
)

COLUMNS = (
    ['page', 'n', 'x0', 'y0', 'x1', 'y1', 'upright', 'by_rules']
    + FEATURES + ['score', 'rule', 'label', 'label_source']
)
PAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
DEFAULT_LABELS = os.path.join(MODELS_DIR, 'scale_bar_labels.csv')


def read_labels(path):
    """Return known labels keyed by (page, x0, y0, x1, y1)."""
    if not path or not os.path.isfile(path):
        return {}
    known = {}
    with open(path, newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            key = (unicodedata.normalize('NFC', row['page']),) + tuple(
                int(row[k]) for k in ('x0', 'y0', 'x1', 'y1')
            )
            known[key] = row['label'].strip().lower()
    return known


def candidates_for(path, config, full_config):
    """Return the page and its candidates with features measured."""
    page = load_page(path)
    masks = build_detection_mask(page, full_config)
    components = find_components(masks.closed, masks.raw)
    bar_config = config.get('scale_bars', {})
    rule_bars = grouping._scale_bar_indices(components, page.width, page.height, bar_config)
    found = grouping.find_scale_candidates(
        components, page.width, page.height, rule_bars, bar_config.get('classifier', {}),
        masks.raw,
    )
    for candidate in found:
        candidate.features = candidate_features(
            candidate, components, masks.raw, page.gray, page.width, page.height
        )
        candidate.rejected_by = _connector_rule(candidate.features)
    return page, found


def label_for(name, candidate, known, model, threshold):
    """Return ``(label, source)`` for one candidate."""
    key = (unicodedata.normalize('NFC', name),) + tuple(candidate.box)
    if key in known:
        return known[key], 'known'
    if model is not None:
        return ('yes' if candidate.score >= threshold else 'no'), 'model'
    return ('yes' if candidate.by_rules else 'no'), 'rules'


def table_rows(name, found, known, model, threshold):
    """Score the candidates of one page and return their table rows."""
    rows = []
    for n, candidate in enumerate(found, start=1):
        if model is not None and not candidate.rejected_by:
            candidate.score = score(candidate.features, model)
        label, source = label_for(name, candidate, known, model, threshold)
        if candidate.rejected_by:
            label, source = 'no', 'rule'
        row = {'page': name, 'n': n, 'x0': candidate.box[0], 'y0': candidate.box[1],
               'x1': candidate.box[2], 'y1': candidate.box[3],
               'upright': int(candidate.upright), 'by_rules': int(candidate.by_rules),
               'score': f"{candidate.score:.4f}", 'rule': candidate.rejected_by,
               'label': label, 'label_source': source}
        row.update({k: f"{v:.4f}" for k, v in candidate.features.items()})
        rows.append(row)
    return rows


def write_overlay(page, found, labels, path):
    """Draw the candidates on the page, numbered and scored, and save it."""
    canvas = cv2.cvtColor(page.gray, cv2.COLOR_GRAY2BGR)
    for n, (candidate, label) in enumerate(zip(found, labels), start=1):
        x0, y0, x1, y1 = candidate.box
        colour = (153, 60, 94) if label == 'yes' else (128, 128, 128)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, 2)
        cv2.putText(canvas, f"{n} {candidate.score:.2f}", (x0, max(12, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    cv2.imwrite(path, canvas)


def main(argv=None) -> int:
    """Write the candidate table and the overlays."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--pages', required=True, help='folder of plates')
    parser.add_argument('--out', required=True, help='folder for the table and overlays')
    parser.add_argument('--config_file', help='configuration file, default the built-in one')
    parser.add_argument('--labels', default=DEFAULT_LABELS,
                        help='known labels (page, box, label); default the bundled file')
    parser.add_argument('--model', help='model JSON to score with; default the bundled one')
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    full_config = get_config_manager(args.config_file).config
    config = get_page_segmentation_config(full_config)
    classifier = config.get('scale_bars', {}).get('classifier', {})
    threshold = float(classifier.get('threshold', 0.4))
    try:
        model = load_model(args.model or classifier.get('model'))
    except ModelUnavailable as exc:
        print(f"no model: {exc}; labels come from the rules")
        model = None
    known = read_labels(args.labels)
    print(f"{len(known)} known label(s) from {args.labels if known else 'nowhere'}")

    os.makedirs(os.path.join(args.out, 'overlays'), exist_ok=True)
    rows = []
    names = sorted(n for n in os.listdir(args.pages) if n.lower().endswith(PAGE_EXTENSIONS))
    for name in names:
        page, found = candidates_for(os.path.join(args.pages, name), config, full_config)
        page_rows = table_rows(name, found, known, model, threshold)
        rows.extend(page_rows)
        labels = [row['label'] for row in page_rows]
        write_overlay(page, found, labels, os.path.join(args.out, 'overlays', page.stem + '.png'))
        unknown = sum(1 for r in page_rows if r['label_source'] != 'known')
        print(f"{name}: {len(found)} candidate(s), {unknown} without a known label")

    table = os.path.join(args.out, 'scale_candidates.csv')
    with open(table, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} candidate(s) on {len(names)} page(s) -> {table}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
