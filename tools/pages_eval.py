r"""
Score a ``pylithics-pages`` run against a corrections file.

The corrections file is the ground truth: its ``expect`` column gives
the number of lithics on each page. This script counts the artefact
boxes each page produced and reports how many pages match, which
identifier rules fired, and every page where a rule fired without
bringing the count nearer to ``expect``. Those pages are the ones to
examine on the overlay.

Usage::

    python tools/pages_eval.py --manifest pylithics/data/pages_manifest.csv \\
        --corrections pylithics/data/corrections.csv [--before OLD_MANIFEST]

With ``--before``, each page also shows the earlier run's count, and
the totals say how many pages improved, worsened, and stayed.

Not part of the package. Development only.
"""

import argparse
import csv
import sys
import unicodedata
from collections import Counter
from typing import Dict, List, Optional

SPLIT, JOIN = 'identifier_split', 'identifier_join'
FLAGS = ('several_identifiers', 'no_identifier')


def load_manifest(path: str) -> Dict[str, List[dict]]:
    """Artefact rows of a manifest, grouped by page."""
    pages: Dict[str, List[dict]] = {}
    with open(path, newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            if row['image_type'] == 'artefact':
                name = unicodedata.normalize('NFC', row['input_page_id'])
                pages.setdefault(name, []).append(row)
    return pages


def load_expected(path: str) -> Dict[str, int]:
    """``page_id -> expect`` from a corrections file, skipping blanks."""
    expected = {}
    with open(path, newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            if (row.get('expect') or '').strip():
                expected[unicodedata.normalize('NFC', row['page_id'])] = int(row['expect'])
    return expected


def score_page(rows: List[dict]) -> dict:
    """Count the boxes, rules fired and flags of one page."""
    made = Counter()
    for row in rows:
        for rule in (SPLIT, JOIN):
            if rule in row.get('correction_applied', ''):
                made[rule] += 1
    flags = Counter(row.get('label_flag', '') for row in rows)
    return {'boxes': len(rows), 'split': made[SPLIT], 'joined': made[JOIN],
            'several': flags['several_identifiers'], 'none': flags['no_identifier']}


def short(name: str, width: int = 44) -> str:
    """Cut a page name to fit the table."""
    return name if len(name) <= width else name[:width - 1] + '…'


def report(
    manifest: Dict[str, List[dict]],
    expected: Dict[str, int],
    before: Optional[Dict[str, List[dict]]],
) -> int:
    """Print the per-page table and the totals. Returns 0."""
    print(f"{'page':<44} {'expect':>6} {'boxes':>5} {'diff':>5} "
          f"{'before':>6} {'split':>5} {'join':>4}  note")
    matched = improved = worsened = 0
    suspect = []
    for page, want in sorted(expected.items()):
        rows = manifest.get(page)
        if rows is None:
            print(f"{short(page):<44} {want:>6} {'-':>5}  not in manifest")
            continue
        now = score_page(rows)
        diff = now['boxes'] - want
        matched += diff == 0
        old = ''
        if before is not None and page in before:
            old_boxes = len(before[page])
            old = str(old_boxes)
            improved += abs(diff) < abs(old_boxes - want)
            worsened += abs(diff) > abs(old_boxes - want)
        note = ''
        if (now['split'] or now['joined']) and diff != 0 and (
                not old or abs(diff) >= abs(int(old) - want)):
            note = 'EXAMINE: rule fired, count not nearer'
            suspect.append(page)
        print(f"{short(page):<44} {want:>6} {now['boxes']:>5} {diff:>+5} "
              f"{old:>6} {now['split']:>5} {now['joined']:>4}  {note}")
    print(f"\nPages matching expect: {matched} of {len(expected)}")
    if before is not None:
        print(f"Against --before: {improved} nearer, {worsened} further, "
              f"{len(expected) - improved - worsened} unchanged")
    totals(manifest, before)
    if suspect:
        print(f"\nExamine on the overlay ({len(suspect)}):")
        for page in suspect:
            print(f"  {page}")
    return 0


def totals(manifest: Dict[str, List[dict]], before: Optional[Dict[str, List[dict]]]) -> None:
    """Print rule and flag counts over every page in the manifest."""
    now = Counter()
    for rows in manifest.values():
        now.update(score_page(rows))
    print(f"All pages: {now['split']} boxes split, {now['joined']} joined; "
          f"flags: {now['several']} several_identifiers, {now['none']} no_identifier")
    if before is not None:
        old = Counter()
        for rows in before.values():
            old.update(score_page(rows))
        print(f"Before:    flags: {old['several']} several_identifiers, "
              f"{old['none']} no_identifier")


def main(argv=None) -> int:
    """Score the manifest and print the report."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--manifest', required=True, help='pages_manifest.csv of the run')
    parser.add_argument('--corrections', required=True, help='corrections CSV with expect')
    parser.add_argument('--before', help='an earlier pages_manifest.csv to compare with')
    args = parser.parse_args(argv)
    before = load_manifest(args.before) if args.before else None
    return report(load_manifest(args.manifest), load_expected(args.corrections), before)


if __name__ == '__main__':
    sys.exit(main())
