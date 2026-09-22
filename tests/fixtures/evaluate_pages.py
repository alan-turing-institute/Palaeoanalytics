#!/usr/bin/env python3
"""
Score page segmentation against hand-verified artefact counts.

Grouping quality cannot be judged from unit tests alone: the rules encode
drawing conventions, and whether they fire correctly is a property of real
plates. This harness runs the grouping over a folder of pages and compares
the result against counts established by eye, so a change can be measured
instead of guessed at.

Ground truth below was verified by the project archaeologist against the
published figures. Note the conventions it encodes:

- A hatched section drawn beneath a flake is part of that artefact.
- Two views of the same lithic are one artefact, however far apart.
- A page scale bar is not an artefact, and connector rules between
  lithics are not scale bars.

Usage::

    python tests/fixtures/evaluate_pages.py [PAGE_DIR]

Pages absent from the folder are skipped, so the harness still runs on a
partial corpus. Exits non-zero if any page misses its expected counts.
"""

import glob
import logging
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from pylithics.image_processing.config import (  # noqa: E402
    get_config_manager,
    get_page_segmentation_config,
)
from pylithics.page_segmentation.detection import (  # noqa: E402
    build_detection_mask,
    find_components,
    load_page,
)
from pylithics.page_segmentation.grouping import group_components  # noqa: E402

DEFAULT_PAGE_DIR = os.path.join('pylithics', 'data', 'pages')

# filename prefix -> (label, expected artefacts, expected scale bars)
GROUND_TRUTH = {
    'sample_plate': ('sample_plate', 5, 1),
    '1977  Bulletin': ('Bulletin p229', 4, 1),
    'Flake modification in European Early': ('Flake fig6', 13, 1),
    'Mobility patterns and core technologies': ('Mobility fig3', 4, 1),
}


def match(filename):
    """Return the ground-truth entry for a page, or None if unknown."""
    base = os.path.basename(filename)
    for prefix, entry in GROUND_TRUTH.items():
        if base.startswith(prefix):
            return entry
    return None


def segment(path, config):
    """Run detection and grouping over one page."""
    page = load_page(path)
    if page is None:
        return None
    closed, raw = build_detection_mask(page, get_config_manager().config)
    components = find_components(closed, raw)
    boxes, bars = group_components(
        components, (page.width, page.height), config
    )
    return len(boxes), len(bars)


def main(argv):
    """Score every known page in the given directory."""
    logging.disable(logging.CRITICAL)
    page_dir = argv[1] if len(argv) > 1 else DEFAULT_PAGE_DIR
    config = get_page_segmentation_config(get_config_manager().config)

    print(f"{'page':<16}{'artefacts':>14}{'scale bars':>16}")
    score = total = 0
    for path in sorted(glob.glob(os.path.join(page_dir, '*'))):
        entry = match(path)
        if entry is None:
            continue
        label, want_boxes, want_bars = entry
        result = segment(path, config)
        if result is None:
            print(f'{label:<16}{"unreadable":>14}')
            continue
        boxes, bars = result
        ok_boxes, ok_bars = boxes == want_boxes, bars == want_bars
        score += ok_boxes + ok_bars
        total += 2
        print(
            f'{label:<16}{boxes:>7} /{want_boxes:<3} '
            f'{"OK" if ok_boxes else "X ":<3}'
            f'{bars:>7} /{want_bars:<3} {"OK" if ok_bars else "X "}'
        )

    if not total:
        print(f'No known pages found in {page_dir}')
        return 1
    print(f'\nscore {score}/{total}')
    return 0 if score == total else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
