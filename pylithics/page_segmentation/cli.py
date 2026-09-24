"""
Command-line interface for page segmentation.

Entry point for the ``pylithics-pages`` console script, which cuts a
folder of scanned plates into one image per artefact ready for the main
``pylithics`` analysis run.
"""

import argparse
from dataclasses import dataclass, field
import logging
import os
import sys
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple

from pylithics.image_processing.config import (
    get_config_manager,
    get_page_segmentation_config,
)

from . import overrides as overrides_module
from .detection import (
    PageImage,
    build_detection_mask,
    find_components,
    load_page,
)
from .export import (
    DEBUG_DIRNAME,
    ManifestRow,
    export_page,
    prepare_output_dir,
    remove_page_output,
    write_debug_overlay,
    write_metadata,
    write_manifest,
)
from .geometry import BBox
from .grouping import classify_components, group_components, reading_order
from .identifiers import read_identifiers
from .logging_setup import setup_logging

PAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
# Project layout, shared with the main analysis command: the plates
# live in pages/, and the crops go to images/ and scales/ beside it,
# where pylithics reads them.
PAGES_DIRNAME = 'pages'

DESCRIPTION = """
PyLithics page preparation
==========================

pylithics-pages cuts a published plate into one image for each artefact.
PyLithics analyses one artefact for each image. A published plate shows
many artefacts. This command makes the cut for you.

Give the command a folder of scanned pages. For each lithic, it writes
one image that contains all the surfaces of that lithic. It writes each
scale bar to its own image. It discards captions and legend text.

The command reads the identifier printed next to each lithic. It names
the crop with that identifier: <page>_figure_7.png. If it cannot read
one identifier for a crop, it names the crop by its box number on the
debug overlay: <page>_box_07.png. The manifest records the method and
the reason.

The command does not change the pixels. Each crop has the DPI and the
colour mode of the source page. No denoising, contrast change or
thresholding is applied. The pylithics analysis does those steps.

--data_dir is a project folder. The two commands share it:

    pylithics-pages  reads pages/               writes images/, scales/,
                                                pages_manifest.csv, meta_data.csv
    pylithics        reads images/, scales/,    writes results/
                     meta_data.csv

    images/          one crop for each artefact
    scales/          one crop for each scale bar
    pages_manifest.csv     one row for each crop: where it was cut, its identifier
    meta_data.csv    one row for each artefact: its scale bar, and a flag

meta_data.csv is the file that pylithics reads. The scale column (the
printed length of the bar, in millimetres) is empty. Fill it in. A row
with a flag has a problem for you to examine: no scale bar on the page,
several scale bars, or no identifier. pylithics analyses the row like
any other, in pixels when the scale is empty, and reports the flags at
the end. Remove the flag when the row is correct.

Every run cuts every plate in pages/. A plate cut before is cut again,
and its crops and rows are replaced. The scale values that you typed
stay. A new plate is added. Images and rows that you put in the project
yourself do not change.
"""

EPILOG = """
procedure:
  1. Put the scanned plates in <project>/pages/.
  2. Cut the plates. Use --debug to write the overlays:
       pylithics-pages --data_dir <project> --debug
  3. Examine <project>/pages_debug/. Each numbered box must contain one
     lithic and all of its surfaces. Each scale bar must have its own
     box. Each green identifier must match the number on the plate.
  4. Correct each page that is wrong (see below). Then start the command
     again:
       pylithics-pages --data_dir <project> --overrides corrections.csv
  5. Open <project>/meta_data.csv. Fill in the scale column. Correct each
     flagged row. Then remove its flag.
  6. Analyse the crops:
       pylithics --data_dir <project>

  To send the crops to a different folder, give --output_dir:
       pylithics-pages --data_dir ./plates --output_dir ./artefacts

correct a page:
  Write the corrections in a CSV file. Use one row for each page. Use
  the box numbers shown on the overlay:
    join=3+4     merge box 3 and box 4
    split=2      cut box 2 into two boxes
    expect=5     set the number of artefacts on the page to 5
  Pages that are not in the CSV file do not change.

  If all pages from one publication show the same error, change a
  distance:
    one artefact is cut into two boxes     increase --gap or --vertical_gap
    two artefacts are in one box           decrease --gap or --vertical_gap
    a profile view is not in its box       increase --narrow
    an identifier is far from its lithic   increase identifiers.reach in config.yaml
"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Run the page segmentation workflow.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code; 0 on success.
    """
    args = build_parser().parse_args(argv)
    _offer_update()
    pages_dir = _resolve_pages_dir(args.data_dir)
    if not args.output_dir:
        args.output_dir = args.data_dir

    console = setup_logging(_console_level(args), args.output_dir)
    logging.info("Reading pages from %s", pages_dir)
    logging.info("Writing output to %s", args.output_dir)

    try:
        config = _resolve_config(args)
        pages = _list_pages(pages_dir)
        corrections = overrides_module.load_overrides(args.overrides)
        overrides_module.warn_unmatched(
            corrections, [os.path.basename(p) for p in pages]
        )
        previous = prepare_output_dir(args.output_dir)
    except (FileNotFoundError, NotADirectoryError) as exc:
        logging.error("%s", exc)
        return 1

    run = _process_pages(pages, corrections, config, args, previous, console)
    if config.get('export', {}).get('manifest', True):
        kept = [r for r in previous if r['input_page_id'] not in run.recut]
        write_manifest(kept + run.rows, args.output_dir)
        write_metadata(run.rows, args.output_dir, run.removed)

    debug = args.debug or config.get('debug', {}).get('enabled', False)
    _log_summary(run.rows, len(pages), args.output_dir, debug)
    return 0


def _offer_update() -> None:
    """Once a day, tell the user about a newer release and offer it."""
    from pylithics.update_check import check_for_update
    enabled = get_config_manager().get_section('update_check').get('enabled', True)
    check_for_update(enabled)


@dataclass
class RunResult:
    """What one run cut, and what it replaced."""

    rows: List[ManifestRow] = field(default_factory=list)
    recut: set = field(default_factory=set)
    removed: set = field(default_factory=set)


def _process_pages(
    pages: Sequence[str],
    corrections: Dict[str, overrides_module.PageOverride],
    config: Dict,
    args: argparse.Namespace,
    previous: Sequence[Dict[str, str]],
    console=None,
) -> RunResult:
    """
    Segment every page, replacing what a previous run cut from it.

    Shows a live progress bar on a terminal. Segmenting a large plate
    takes a noticeable moment, and without it a long batch looks frozen.
    """
    padding = config.get('export', {}).get('padding', 20)
    by_page: Dict[str, List[Dict[str, str]]] = {}
    for row in previous:
        by_page.setdefault(row['input_page_id'], []).append(row)
    run = RunResult()

    with _page_progress(pages, console) as advance:
        for path in pages:
            name = os.path.basename(path)
            advance(name)
            override = corrections.get(name)
            try:
                page = load_page(path)
                if page is None:
                    continue
                run.removed |= remove_page_output(
                    args.output_dir, by_page.get(name, [])
                )
                run.recut.add(name)
                run.rows.extend(
                    _process_page(page, override, config, args, padding)
                )
            except FileExistsError as exc:
                logging.error("%s: %s", name, exc)
            except (OSError, ValueError) as exc:
                logging.error("Page not read, %s: %s", name, exc)
    return run


@contextmanager
def _page_progress(pages: Sequence[str], console):
    """
    Yield a callable that advances a progress bar, if one is shown.

    Falls back to a no-op when output is redirected, so piped or logged
    runs stay free of control codes.
    """
    if console is None or not sys.stdout.isatty() or not pages:
        yield lambda name: None
        return

    from rich.progress import (
        BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
        TextColumn, TimeElapsedColumn, TimeRemainingColumn,
    )

    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[cyan]Splitting pages[/]"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[dim]{task.fields[page]}[/]"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "splitting", total=len(pages), page="",
        )
        started = {'first': True}

        def advance(name: str) -> None:
            """Move to the next page, showing its name."""
            if started['first']:
                started['first'] = False
            else:
                progress.advance(task)
            progress.update(task, page=name)

        yield advance
        progress.advance(task)


def _process_page(
    page: PageImage,
    override: Optional[overrides_module.PageOverride],
    config: Dict,
    args: argparse.Namespace,
    padding: int,
) -> List[ManifestRow]:
    """Segment one loaded page and write its crops."""
    path = page.path
    closed, raw = build_detection_mask(page, get_config_manager().config)
    components = find_components(closed, raw)
    if not components:
        logging.warning(
            "%s: no ink found; no crops", os.path.basename(path)
        )
        return []

    boxes, bars = group_components(
        components, (page.width, page.height), config
    )
    boxes, applied = _apply_overrides(
        boxes, override, components, page, config
    )
    identifiers = _read_identifiers(
        page, components, boxes, bars, raw, closed, config
    )

    if args.debug or config.get('debug', {}).get('enabled', False):
        read = config.get('identifiers', {}).get('enabled', True)
        write_debug_overlay(
            page, boxes, bars, args.output_dir, identifiers if read else None
        )

    return export_page(
        page, boxes, bars, args.output_dir, padding, applied,
        _count_components(boxes, components), identifiers,
    )


def _read_identifiers(page, components, boxes, bars, raw, closed, config):
    """Read the plate's identifiers for the final boxes."""
    grouping = config.get('grouping', {})
    classified = classify_components(
        components, page.width, page.height,
        grouping.get('min_area', 0.0004), config.get('scale_bars', {}),
        config.get('text_rejection', {}),
    )
    return read_identifiers(
        page, components, classified, boxes, bars, raw, closed,
        config.get('identifiers', {}),
    )


def _count_components(
    boxes: Sequence[BBox], components: Sequence
) -> Dict[int, int]:
    """
    Count the ink blobs falling inside each artefact box.

    Recorded in the manifest so a suspiciously low count flags an
    artefact whose surface views were not all captured.
    """
    counts = {}
    for index, box in enumerate(boxes):
        counts[index] = sum(
            1 for component in components
            if _is_inside(component.box, box)
        )
    return counts


def _is_inside(inner: Sequence[int], outer: Sequence[int]) -> bool:
    """Report whether one box lies wholly within another."""
    return (
        inner[0] >= outer[0] and inner[1] >= outer[1]
        and inner[2] <= outer[2] and inner[3] <= outer[3]
    )


def _apply_overrides(
    boxes: Sequence[BBox],
    override: Optional[overrides_module.PageOverride],
    components: Sequence,
    page: PageImage,
    config: Dict,
) -> Tuple[List[BBox], str]:
    """Apply this page's corrections and report which ones ran."""
    if override is None or override.is_empty:
        return list(boxes), ''

    grouping = config.get('grouping', {})
    result = list(boxes)

    if override.expect is not None:
        result = overrides_module.apply_expect(
            result, override.expect,
            lambda gap, narrow: _regroup(
                components, page, config, gap, narrow
            ),
            grouping,
        )

    if override.joins:
        result = overrides_module.apply_joins(result, override.joins)
    if override.splits:
        result = overrides_module.apply_splits(
            result, override.splits, page.gray
        )
    if override.joins or override.splits:
        result = reading_order(result)

    return result, override.applied_kinds()


def _regroup(
    components: Sequence,
    page: PageImage,
    config: Dict,
    gap: float,
    narrow: float,
) -> List[BBox]:
    """Re-run grouping with swept distances, for the expect sweep."""
    trial = dict(config)
    trial['grouping'] = {
        **config.get('grouping', {}), 'gap': gap, 'narrow': narrow
    }
    boxes, _ = group_components(
        components, (page.width, page.height), trial
    )
    return boxes


### SETUP ###

def _resolve_config(args: argparse.Namespace) -> Dict:
    """Merge CLI overrides over the YAML configuration."""
    manager = get_config_manager(args.config_file)
    config = get_page_segmentation_config(manager.config)

    grouping = dict(config.get('grouping', {}))
    for name in ('gap', 'narrow', 'vertical_gap', 'bridge', 'min_area'):
        value = getattr(args, name, None)
        if value is not None:
            grouping[name] = value
            logging.debug("CLI override: grouping.%s = %s", name, value)

    export = dict(config.get('export', {}))
    if args.padding is not None:
        export['padding'] = args.padding

    bars = dict(config.get('scale_bars', {}))
    if args.disable_scale_bars:
        bars['enabled'] = False

    identifiers = dict(config.get('identifiers', {}))
    if args.read_labels is not None:
        identifiers['enabled'] = args.read_labels

    return {
        **config,
        'grouping': grouping,
        'export': export,
        'scale_bars': bars,
        'identifiers': identifiers,
    }


def _resolve_pages_dir(data_dir: str) -> str:
    """
    Locate the pages to split inside the project directory.

    ``--data_dir`` names a project root shared with the main analysis
    command: this reads ``<data_dir>/pages/`` and writes the crops to
    ``<data_dir>/images/`` and ``<data_dir>/scales/``, where
    ``pylithics`` reads them.

    A directory of pages passed directly is still accepted, so a folder
    of scans can be split without arranging it into a project first.
    """
    nested = os.path.join(data_dir, PAGES_DIRNAME)
    if os.path.isdir(nested):
        return nested
    return data_dir


def _list_pages(data_dir: str) -> List[str]:
    """
    List readable page images, reporting anything skipped.

    Files the tool cannot read are named rather than silently ignored.
    A folder of scans that yields no crops is otherwise indistinguishable
    from a folder in the wrong format, and the commonest case — a PDF of
    a paper — looks like a valid input to the person supplying it.

    Parameters
    ----------
    data_dir : str
        Directory holding the pages to split.

    Returns
    -------
    list of str
        Paths of readable pages, sorted.

    Raises
    ------
    NotADirectoryError
        When the input directory does not exist.
    """
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(f"Input directory not found: {data_dir}")

    pages, skipped = [], []
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if name.startswith('.') or not os.path.isfile(path):
            continue
        if name.lower().endswith(PAGE_EXTENSIONS):
            pages.append(path)
        else:
            skipped.append(name)

    _report_skipped(skipped)
    if not pages:
        logging.warning("No page images found in %s", data_dir)
    else:
        logging.info("Found %d page(s) in %s", len(pages), data_dir)
    return pages


def _report_skipped(skipped: Sequence[str]) -> None:
    """Warn about files that are not a readable page format."""
    if not skipped:
        return

    pdfs = [name for name in skipped if name.lower().endswith('.pdf')]
    others = [name for name in skipped if name not in pdfs]

    if pdfs:
        logging.warning(
            "%d PDF file(s) not read: PDFs are not supported. Export the "
            "figure pages as PNG or TIFF at their original resolution "
            "first, so that the DPI agrees with the scan.",
            len(pdfs),
        )
        for name in pdfs:
            logging.warning("  unsupported (PDF): %s", name)

    for name in others:
        extension = os.path.splitext(name)[1] or 'no extension'
        logging.warning(
            "  not supported (%s): %s — the formats read are %s",
            extension, name, ', '.join(PAGE_EXTENSIONS),
        )


def _console_level(args: argparse.Namespace) -> int:
    """
    Resolve the terminal logging level from the CLI flags.

    Affects the console only. The log file always records the full
    trace, so a run can be explained after the fact without being
    reproduced.
    """
    if args.verbose:
        return logging.DEBUG
    return getattr(logging, args.log_level, logging.INFO)


def _log_summary(
    rows: Sequence[ManifestRow], page_count: int, output_dir: str,
    debug: bool = False,
) -> None:
    """Report what the run produced, and where the overlays are."""
    artefacts = sum(1 for r in rows if r.image_type == 'artefact')
    bars = sum(1 for r in rows if r.image_type == 'scale_bar')
    logging.info(
        "Done: %d page(s) -> %d artefact(s), %d scale bar(s) in %s",
        page_count, artefacts, bars, output_dir,
    )
    if debug:
        logging.info(
            "Debug overlays: %s", os.path.join(output_dir, DEBUG_DIRNAME)
        )


### ARGUMENT PARSING ###

def build_parser() -> argparse.ArgumentParser:
    """Build the ``pylithics-pages`` argument parser."""
    parser = argparse.ArgumentParser(
        prog='pylithics-pages',
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_required_args(parser)
    _add_grouping_args(parser)
    _add_output_args(parser)
    _add_config_args(parser)
    return parser


def _add_required_args(parser: argparse.ArgumentParser) -> None:
    """Add required argument group."""
    group = parser.add_argument_group('NECESSARY ARGUMENTS')
    group.add_argument(
        '--data_dir', required=True, metavar='PATH',
        help='The project directory that contains pages/. You can also '
             'give a folder of pages directly.'
    )
    group.add_argument(
        '--output_dir', metavar='PATH',
        help='The folder for images/, scales/, pages_manifest.csv and '
             'meta_data.csv (default: the --data_dir project folder).'
    )


def _add_grouping_args(parser: argparse.ArgumentParser) -> None:
    """Add grouping tuning argument group."""
    group = parser.add_argument_group('GROUPING OPTIONS')
    group.add_argument(
        '--gap', type=float, metavar='FRAC',
        help='Two views closer than this fraction of the page width '
             'are one artefact (default: 0.025). Increase the value if '
             'one artefact is cut into two boxes. Decrease it if two '
             'artefacts are in one box.'
    )
    group.add_argument(
        '--narrow', type=float, metavar='FRAC',
        help='A narrow profile view joins the artefact next to it if '
             'the distance is less than this fraction of the page width '
             '(default: 0.07).'
    )
    group.add_argument(
        '--vertical_gap', type=float, metavar='FRAC',
        help='A short view joins the taller view above it if the '
             'distance is less than this fraction of the page height '
             '(default: 0.06).'
    )
    group.add_argument(
        '--bridge', type=float, metavar='FRAC',
        help='Views connected by a dash mark join if the dash is '
             'shorter than this fraction of the page (default: 0.06).'
    )
    group.add_argument(
        '--min_area', type=float, metavar='FRAC',
        help='Ink blobs smaller than this fraction of the page area are '
             'labels, not artefacts (default: 0.0004).'
    )
    group.add_argument(
        '--overrides', metavar='FILE',
        help='A CSV file of corrections, one row for each page. '
             'Columns: page_id, expect, join, split. Use the box numbers '
             'shown by --debug.'
    )


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    """Add output argument group."""
    group = parser.add_argument_group('OUTPUT OPTIONS')
    group.add_argument(
        '--padding', type=int, metavar='PX',
        help='The white margin around each crop, in pixels (default: 20).'
    )
    group.add_argument(
        '--debug', action='store_true',
        help='Write pages_debug/{page}.png for each page. The overlay '
             'shows each box, each scale bar and each identifier read.'
    )
    group.add_argument(
        '--disable_scale_bars', action='store_true',
        help='Do not write scale bar crops.'
    )
    group.add_argument(
        '--read_labels', dest='read_labels', action='store_true',
        default=None,
        help='Read the identifier printed next to each lithic and name '
             'the crop {page}_figure_{label}.png. This is the default. '
             'RapidOCR is necessary: pip install "PyLithics[ocr]".'
    )
    group.add_argument(
        '--no_read_labels', dest='read_labels', action='store_false',
        help='Do not read identifiers. Name the crops by box number only.'
    )


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    """Add configuration argument group."""
    group = parser.add_argument_group('CONFIGURATION OPTIONS')
    group.add_argument(
        '--config_file', metavar='FILE',
        help='A YAML configuration file to use in place of the default.'
    )
    group.add_argument(
        '--log_level', default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        metavar='LEVEL',
        help='The logging level (default: INFO).'
    )
    group.add_argument(
        '--verbose', '-v', action='store_true',
        help='Show the full trace for each page.'
    )


if __name__ == '__main__':
    sys.exit(main())
