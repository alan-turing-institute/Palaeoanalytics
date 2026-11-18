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
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple

from pylithics.image_processing.config import (
    get_config_manager,
    get_page_segmentation_config,
)

from pylithics.image_processing.utils import format_duration, progress_columns

from . import overrides as overrides_module
from .detection import (
    PageImage,
    build_detection_mask,
    find_components,
    load_page,
)
from .export import (
    skipped_row,
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
from .pdf_input import (
    PdfSupportMissing,
    RenderedPage,
    is_pdf,
    parse_page_spec,
    render_pdf,
    rendered_by_name,
)
from .grouping import (
    ScaleContext, bare_scale_bars, classify_components, group_components, reading_order,
)
from .identifiers import PageReads, assign_reads, read_page, report_identifiers
from .scale_classifier import ModelUnavailable, load_model
from .text_boxes import find_text_boxes
from .hiding import make_hider
from .regroup import JOIN, SPLIT, refine_boxes
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

The command does not change the pixels of the lithic. Each crop has the
DPI and the colour mode of the source page. The command does no
denoising, no contrast change and no thresholding. The pylithics
analysis does those steps. The one change is to the text of the page.
In the crop file, the command hides captions, legend keys, scale bar
ink and other identifiers under the paper colour. --keep_text keeps
them. pylithics analyses the crop file as it is, and does not hide
text. After you change these options, cut the pages again.

--data_dir is a project folder. The two commands share it:

    pylithics-pages  reads pages/               writes images/, scales/,
                                                pages_manifest.csv, meta_data.csv
    pylithics        reads images/, scales/,    writes results/
                     meta_data.csv

    images/          one crop for each artefact
    scales/          one crop for each scale bar
    pages_manifest.csv     one row for each crop: its position on the page, its identifier
    meta_data.csv    one row for each artefact: its scale bar, and a flag

meta_data.csv is the file that pylithics reads. The scale column (the
printed length of the bar, in millimetres) is empty. Type the value in
each row. A row with a flag has a problem for you to examine: no scale
bar on the page, several scale bars, a scale bar with a low score, or
no identifier. pylithics analyses a flagged row as it analyses the
other rows. If the scale is empty, it measures in pixels. At the end,
it reports the flags. Remove the flag when the row is correct.

Each run cuts all the plates in pages/. If an earlier run cut a plate,
this run cuts it again and replaces its crops and rows. The scale
values that you typed stay. The run adds each new plate. Images and
rows that you put in the project yourself do not change. If an earlier
run left a crop file with no manifest row, this run replaces the file
and reports the number. Two pages with the same name and different
extensions give the same crop names. The command does not write the
crops of the second page. Rename one of the pages.
"""

EPILOG = """
procedure:
  1. Put the scanned plates in <project>/pages/.
  2. Cut the plates. Use --debug to write the overlays:
       pylithics-pages --data_dir <project> --debug
  3. Examine <project>/pages_debug/. Each numbered box must contain one
     lithic and all of its surfaces. Each scale bar must have its own
     purple box, with its score beside it. A grey box is a mark that
     the command rejected. Each green identifier must match the number
     on the plate.
  4. Correct each page that is wrong (see "correct a page" below). Write
     a row in <project>/corrections.csv. The first run writes this file
     with no rows. Then start the command again:
       pylithics-pages --data_dir <project>
  5. Open <project>/meta_data.csv. Type the scale values in the scale
     column. Correct each flagged row. Then remove its flag.
  6. Analyse the crops:
       pylithics --data_dir <project>
     pylithics reads the crop files as pylithics-pages wrote them. After
     a change to the pages, the corrections or the options, run
     pylithics-pages again before pylithics.

  To write the crops to a different folder, give --output_dir:
       pylithics-pages --data_dir ./plates --output_dir ./artefacts

correct a page:
  1. Run the command one time with --debug. The run writes an empty
     corrections.csv into the project folder.
  2. Open pages_debug/<page>.png. Each box has a number.
  3. Add one row for the page to corrections.csv. Columns:
       page_id   the page file name with its extension (for a PDF, the
                 rendered PNG, for example report_p004.png)
       join      boxes to join: 1+2. Several pairs: 1+2 5+6
       split     one box to cut at its widest empty column: 3
       expect    the number of artefacts on the page. The command
                 changes its distances to get this number. 0 means
                 that the page has no artefacts. The command cuts
                 nothing and removes the earlier crops.
     Leave the columns you do not need empty. Use join or expect for
     a page, not both. Example, to join boxes 1 and 2 on page 4:
       page_id,expect,join,split
       report_p004.png,,1+2,
  4. Run the command again with no other options. The command cuts
     the pages in the file with their corrections. The other pages do
     not change.
  Use the same distance options for the debug run and the corrected
  run. If you do not, the box numbers change. Put a page name that has
  a comma in quotation marks. A spreadsheet writes them for you.

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
    int : exit code. 0 on success, 1 when a page was not cut or the project
    folder does not exist, 130 when stopped.
    """
    args = build_parser().parse_args(argv)
    logging.getLogger().addHandler(logging.NullHandler())   # no stdlib fallback handler
    get_config_manager(args.config_file)     # first use creates the process-wide manager
    _offer_update()
    if not _project_folder_exists(args):
        return 1
    pages_dir = _resolve_paths(args)

    console = setup_logging(_console_level(args), args.output_dir)
    started = time.monotonic()
    logging.info("Reading pages from %s", pages_dir)
    logging.info("Writing output to %s", args.output_dir)

    try:
        config, pages, corrections, previous, rendered, forgotten = _prepare(args, pages_dir)
    except (FileNotFoundError, NotADirectoryError, PdfSupportMissing, ValueError) as exc:
        logging.error("%s", exc)
        return 1

    try:
        run = _process_pages(pages, corrections, config, args, previous, console, rendered)
    except KeyboardInterrupt:
        logging.error(
            "The command stopped. The manifest names each crop that it wrote. "
            "Start the command again to cut the remaining pages."
        )
        return 130
    run.removed |= forgotten
    _write_tables(run, previous, config, args.output_dir)
    overrides_module.write_empty_corrections(args.data_dir)

    debug = args.debug or config.get('debug', {}).get('enabled', False)
    _log_summary(run, previous, len(pages), args.output_dir, debug,
                 time.monotonic() - started)
    if run.failed:
        logging.error("%d page(s) not cut: %s. See %s for the reason.", len(run.failed),
                      ', '.join(run.failed), os.path.join(args.output_dir, 'pylithics-pages.log'))
    return 1 if run.failed else 0


def _resolve_paths(args: argparse.Namespace) -> str:
    """Return the pages folder; default the output folder and the corrections file."""
    if not args.output_dir:
        args.output_dir = args.data_dir
    args.overrides = overrides_module.project_corrections(args.data_dir, args.overrides)
    return _resolve_pages_dir(args.data_dir)


def _project_folder_exists(args: argparse.Namespace) -> bool:
    """Report a ``--data_dir`` that is not there, before anything is written."""
    if os.path.isdir(args.data_dir):
        return True
    setup_logging(_console_level(args))
    logging.error("The folder %s does not exist. Give --data_dir the project folder "
                  "that contains pages/, or a folder of page images.", args.data_dir)
    return False


def _write_tables(run: 'RunResult', previous: Sequence[Dict[str, str]], config: Dict,
                  output_dir: str) -> None:
    """Write the manifest and the metadata when the export asks for them."""
    if not config.get('export', {}).get('manifest', True):
        return
    kept = [r for r in previous if r['input_page_id'] not in run.recut]
    write_manifest(kept + run.rows, output_dir)
    write_metadata(run.rows, output_dir, run.removed, _uncertain_scales(run.rows, config))


def _prepare(args: argparse.Namespace, pages_dir: str) -> Tuple:
    """Resolve the configuration, list and render the pages, read corrections, old manifest."""
    config = _resolve_config(args)
    pages, pdfs = _list_pages(pages_dir)
    rendered = _render_pdfs(pdfs, args, config)
    pages += [r.path for r in rendered]
    corrections = overrides_module.load_overrides(args.overrides)
    overrides_module.warn_unmatched(
        corrections, [os.path.basename(p) for p in pages]
    )
    previous = prepare_output_dir(args.output_dir)
    previous, forgotten = _forget_missing_sources(previous, pages, pdfs, args.output_dir)
    return config, pages, corrections, previous, rendered_by_name(rendered), forgotten


def _forget_missing_sources(
    previous: Sequence[Dict[str, str]], pages: Sequence[str], pdfs: Sequence[str],
    output_dir: str,
) -> Tuple[List[Dict[str, str]], set]:
    """
    Drop the rows of pages whose source is no longer in the project; remove their crops.

    A page file taken out of ``pages/``, or a PDF taken out, leaves crops
    and rows behind otherwise. A page of a PDF that is still there is
    kept, whether or not this run renders it.
    """
    present = {os.path.basename(p) for p in pages} | {os.path.basename(p) for p in pdfs}
    kept, gone = [], []
    for row in previous:
        source = row.get('source_pdf') or row['input_page_id']
        (kept if source in present else gone).append(row)
    removed = remove_page_output(output_dir, gone)
    if gone:
        names = sorted({r['input_page_id'] for r in gone})
        shown = ', '.join(names[:5]) + (' ...' if len(names) > 5 else '')
        logging.info("%d page(s) are no longer in the project. The command removed their "
                     "%d crop(s): %s", len(names), len(removed), shown)
    return kept, removed


def _render_pdfs(pdfs: Sequence[str], args: argparse.Namespace, config: Dict) -> List[RenderedPage]:
    """
    Render every PDF in the pages folder to PNG pages and return the records.

    A PDF that pypdfium2 cannot open is reported and the batch goes on.
    Without pypdfium2 the run stops before any page is processed.
    """
    if not pdfs:
        return []
    wanted = parse_page_spec(args.pdf_pages)
    rendered: List[RenderedPage] = []
    for path in pdfs:
        try:
            rendered += render_pdf(path, args.output_dir, config.get('pdf', {}), wanted)
        except PdfSupportMissing:
            raise
        except Exception as exc:  # noqa: BLE001 - one file must not stop the batch
            logging.error("%s: not read. %s: %s", os.path.basename(path),
                          type(exc).__name__, exc)
            logging.debug("Trace for %s", path, exc_info=True)
    return rendered


def _offer_update() -> None:
    """Once a day, tell the user about a newer release and offer it."""
    from pylithics.update_check import check_for_update
    enabled = get_config_manager().get_section('update_check').get('enabled', True)
    check_for_update(enabled)


@dataclass
class RunResult:
    """What one run cut, what it replaced, and what it could not cut."""

    rows: List[ManifestRow] = field(default_factory=list)
    recut: set = field(default_factory=set)
    removed: set = field(default_factory=set)
    orphans: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)


def _process_pages(
    pages: Sequence[str],
    corrections: Dict[str, overrides_module.PageOverride],
    config: Dict,
    args: argparse.Namespace,
    previous: Sequence[Dict[str, str]],
    console=None,
    rendered: Optional[Dict[str, RenderedPage]] = None,
) -> RunResult:
    """
    Segment every page, replacing what a previous run cut from it.

    A page that fails, for any reason, is reported and the batch goes
    on. The manifest is rewritten after every page, so a stopped run
    still names every crop it wrote. A progress bar shows on a terminal.
    ``rendered`` maps a PNG name to the PDF page it was rendered from;
    those rows record their source.
    """
    padding = config.get('export', {}).get('padding', 10)
    by_page: Dict[str, List[Dict[str, str]]] = {}
    for row in previous:
        by_page.setdefault(row['input_page_id'], []).append(row)
    run = RunResult()

    with _page_progress(pages, console) as advance:
        for path in pages:
            name = os.path.basename(path)
            advance(name)
            override = corrections.get(name)
            if override is not None and override.skipped:
                _skip_page(run, name, by_page.get(name, []), args.output_dir)
            else:
                _cut_page(run, path, name, override, config, args, padding, by_page)
            _record_source(run.rows, (rendered or {}).get(name), name)
            _checkpoint(run, previous, config, args.output_dir)
    return run


def _skip_page(run: RunResult, name: str, previous_rows: Sequence[Dict[str, str]],
               output_dir: str) -> None:
    """Cut nothing from a page marked ``expect=0``; remove what an earlier run cut from it."""
    run.removed |= remove_page_output(output_dir, previous_rows)
    run.recut.add(name)
    run.rows.append(skipped_row(name))
    logging.info("%s: no artefacts (expect=0); nothing cut%s", name,
                 ', the earlier crops removed' if previous_rows else '')


def _cut_page(run: RunResult, path: str, name: str, override, config: Dict,
              args: argparse.Namespace, padding: int, by_page: Dict) -> None:
    """Cut one page, replacing what a previous run cut from it; a failure is reported."""
    try:
        page = load_page(path)
        if page is None:
            return
        run.removed |= remove_page_output(args.output_dir, by_page.get(name, []))
        run.recut.add(name)
        taken = {r.output_crop_id: r.input_page_id for r in run.rows}
        run.rows.extend(_process_page(page, override, config, args, padding, taken, run.orphans))
    except FileExistsError as exc:
        run.failed.append(name)
        logging.error("%s: not cut. %s", name, exc)
    except (OSError, ValueError) as exc:
        logging.error("%s: not read. %s", name, exc)
    except Exception as exc:  # noqa: BLE001 - one page must not stop the batch
        run.failed.append(name)
        logging.error("%s: not cut. %s: %s", name, type(exc).__name__, exc)
        logging.debug("Trace for %s", name, exc_info=True)


def _record_source(rows: Sequence[ManifestRow], source: Optional[RenderedPage],
                   name: Optional[str] = None) -> None:
    """Write the PDF page a rendered page came from on every row of that page."""
    if source is None:
        return
    for row in rows:
        if name is not None and row.input_page_id != name:
            continue
        row.source_pdf = source.source_pdf
        row.source_page = str(source.source_page)
        row.render_dpi = f"{source.dpi:g}"
        row.dpi_source = source.dpi_source
        row.source_filters = source.filters


def _checkpoint(
    run: RunResult, previous: Sequence[Dict[str, str]], config: Dict, output_dir: str
) -> None:
    """Write the manifest as it stands: rows of this run, plus pages not yet recut."""
    if not config.get('export', {}).get('manifest', True):
        return
    kept = [r for r in previous if r['input_page_id'] not in run.recut]
    write_manifest(kept + run.rows, output_dir, quiet=True)


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

    from rich.progress import Progress

    with Progress(*progress_columns("Splitting pages", "page"), console=console) as progress:
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
    taken: Optional[Dict[str, str]] = None,
    orphans: Optional[List[str]] = None,
) -> List[ManifestRow]:
    """Segment one loaded page and write its crops."""
    path = page.path
    masks = build_detection_mask(page, get_config_manager().config)
    closed, raw = masks.closed, masks.raw
    components = find_components(closed, raw)
    if not components:
        logging.warning(
            "%s: no ink found; no crops", os.path.basename(path)
        )
        return []

    context = _scale_context(page, raw, config)
    boxes, bars = group_components(
        components, (page.width, page.height), config, context
    )
    _report_candidates(page, context)
    classified = _classify(page, components, config, context)
    reads = _read_page(page, components, classified, boxes, bars, raw, closed, config,
                       masks.working)
    boxes, tags = refine_boxes(
        boxes, reads, components, (page.width, page.height), config
    )
    boxes, applied = _apply_overrides(
        boxes, override, components, page, config, reads
    )
    text = find_text_boxes(page, boxes, raw, _text_config(config), masks.working)
    boxes, tags = _without(boxes, tags, {t.index for t in text})
    identifiers = assign_reads(reads, boxes)
    report_identifiers(page, identifiers)
    corrections = _box_corrections(boxes, tags, applied)

    _write_overlay(args, config, page, boxes, bars, identifiers, corrections, masks, context)

    return export_page(
        page, boxes, bars, args.output_dir, padding, '',
        _count_components(boxes, components), identifiers, corrections,
        masks.working, taken, orphans, context.bar_scores if context else None,
        text_boxes=text, hide=make_hider(components, classified, reads, config.get('export', {})),
        bar_boxes=bare_scale_bars(classified, config.get('scale_bars', {}), page.width),
    )


def _write_overlay(args, config, page, boxes, bars, identifiers, corrections, masks,
                   context) -> None:
    """Write the debug overlay for a page when the run asks for it."""
    if not (args.debug or config.get('debug', {}).get('enabled', False)):
        return
    read = config.get('identifiers', {}).get('enabled', True)
    write_debug_overlay(
        page, boxes, bars, args.output_dir,
        identifiers if read else None, corrections, masks.working.factor,
        context.candidates if context else None,
    )


def _text_config(config: Dict) -> Dict:
    """Return the text box settings."""
    return dict(config.get('text_boxes', {}))


def _without(boxes: Sequence[BBox], tags: Sequence[str], drop: set) -> Tuple[List, List]:
    """Return the boxes and their tags without the given positions."""
    kept = [i for i in range(len(boxes)) if i not in drop]
    aligned = list(tags) if len(tags) == len(boxes) else None
    return [boxes[i] for i in kept], ([aligned[i] for i in kept] if aligned else list(tags))


_SCALE_MODEL: Dict[str, object] = {}


def _scale_context(page: PageImage, raw, config: Dict) -> Optional[ScaleContext]:
    """
    Build the classifier's context for a page, or None when the rules decide.

    The model is read once for each run. When it cannot be read, that is
    reported once and every page falls back to the rules.
    """
    settings = config.get('scale_bars', {}).get('classifier', {})
    if not settings.get('enabled', True):
        return None
    if 'model' not in _SCALE_MODEL:
        try:
            _SCALE_MODEL['model'] = load_model(settings.get('model'))
        except ModelUnavailable as exc:
            logging.warning("%s. The scale bar rules decide for every page.", exc)
            _SCALE_MODEL['model'] = None
    model = _SCALE_MODEL['model']
    if model is None:
        return None
    return ScaleContext(
        raw=raw, gray=page.gray, model=model,
        threshold=float(settings.get('threshold', 0.4)), settings=settings,
    )


def _report_candidates(page: PageImage, context: Optional[ScaleContext]) -> None:
    """Log what the classifier decided on a page."""
    if context is None or not context.candidates:
        return
    accepted = sum(1 for c in context.candidates if c.accepted)
    logging.info(
        "%s: %d scale bar candidate(s) scored, %d accepted",
        os.path.basename(page.path), len(context.candidates), accepted,
    )


def _uncertain_scales(rows: Sequence[ManifestRow], config: Dict) -> set:
    """Return the scale bar crops whose score lies within the band around the threshold."""
    settings = config.get('scale_bars', {}).get('classifier', {})
    threshold = float(settings.get('threshold', 0.4))
    band = float(settings.get('uncertain_band', 0.15))
    return {
        row.output_crop_id for row in rows
        if row.image_type == 'scale_bar' and row.scale_score
        and abs(float(row.scale_score) - threshold) <= band
    }


def _classify(page, components, config, context=None):
    """Classify the page's ink pieces once more, for the reader and the hiding step."""
    grouping = config.get('grouping', {})
    return classify_components(
        components, page.width, page.height,
        grouping.get('min_area', 0.0004), config.get('scale_bars', {}),
        config.get('text_rejection', {}), context,
    )


def _read_page(page, components, classified, boxes, bars, raw, closed, config,
               working) -> PageReads:
    """Read the plate's identifiers, with their positions."""
    return read_page(
        page, components, classified, boxes, bars, raw, closed,
        config.get('identifiers', {}), working,
    )


def _box_corrections(
    boxes: Sequence[BBox], tags: Sequence[str], applied: str
) -> List[str]:
    """
    Record the corrections for each final box, for the manifest.

    A box the identifier rules made keeps its tag while the user's
    corrections leave it as it was; a box the user changed carries the
    user's corrections instead. Both are named when both apply.
    """
    made = {tuple(box): tag for box, tag in zip(boxes, tags)}
    result = []
    for box in boxes:
        kinds = [k for k in (made.get(tuple(box), ''), applied) if k]
        result.append('+'.join(kinds))
    return result


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
    reads: Optional[PageReads] = None,
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
                components, page, config, gap, narrow, reads
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
    reads: Optional[PageReads] = None,
) -> List[BBox]:
    """
    Re-run grouping with swept distances, for the expect sweep.

    The identifier rules run on each candidate too, so the count the
    sweep reaches is the count a default run would give.
    """
    trial = dict(config)
    trial['grouping'] = {
        **config.get('grouping', {}), 'gap': gap, 'narrow': narrow
    }
    size = (page.width, page.height)
    boxes, _ = group_components(components, size, trial)
    if reads is not None:
        boxes, _ = refine_boxes(boxes, reads, components, size, trial)
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
    if getattr(args, 'keep_text', False):
        export['hide_text'] = False
    if getattr(args, 'hide_identifiers', False):
        export['hide_identifiers'] = True

    bars = dict(config.get('scale_bars', {}))
    if args.disable_scale_bars:
        bars['enabled'] = False

    identifiers = dict(config.get('identifiers', {}))
    if args.read_labels is not None:
        identifiers['enabled'] = args.read_labels

    text = dict(config.get('text_boxes', {}))
    if getattr(args, 'keep_text_boxes', False):
        text['enabled'] = False

    pdf = dict(config.get('pdf', {}))
    if getattr(args, 'pdf_dpi', None) is not None:
        pdf['rasterise_dpi'] = args.pdf_dpi

    return {
        **config,
        'grouping': grouping,
        'export': export,
        'scale_bars': bars,
        'identifiers': identifiers,
        'text_boxes': text,
        'pdf': pdf,
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


def _list_pages(data_dir: str) -> Tuple[List[str], List[str]]:
    """
    List readable page images and PDF files, reporting anything skipped.

    Files the tool cannot read are named rather than silently ignored.
    A folder with no page at all is reported with where the pages go;
    the run goes on and writes empty tables, as before.

    Parameters
    ----------
    data_dir : str
        Directory holding the pages to split.

    Returns
    -------
    tuple of (list of str, list of str)
        Paths of readable pages, sorted, and paths of PDF files, sorted.

    Raises
    ------
    NotADirectoryError
        When the input directory does not exist.
    """
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(f"The folder {data_dir} does not exist.")

    pages, pdfs, skipped = [], [], []
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if name.startswith('.') or not os.path.isfile(path):
            continue
        if name.lower().endswith(PAGE_EXTENSIONS):
            pages.append(path)
        elif is_pdf(name):
            pdfs.append(path)
        else:
            skipped.append(name)

    _report_skipped(skipped)
    if not pages and not pdfs:
        logging.warning(
            "No page images in %s. Put the plates in a pages/ folder inside the project "
            "folder. The command reads these formats: %s and PDF.", data_dir,
            ', '.join(PAGE_EXTENSIONS),
        )
    else:
        logging.info("Found %d page(s) and %d PDF file(s) in %s", len(pages), len(pdfs),
                     data_dir)
    return pages, pdfs


def _report_skipped(skipped: Sequence[str]) -> None:
    """Warn about files that are not a readable page format."""
    for name in skipped:
        extension = os.path.splitext(name)[1] or 'no extension'
        logging.warning(
            "  not supported (%s): %s. The command reads these formats: %s and PDF.",
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
    run: RunResult, previous: Sequence[Dict[str, str]], page_count: int,
    output_dir: str, debug: bool = False, seconds: Optional[float] = None,
) -> None:
    """Report what the run produced, what it replaced, and where the overlays are."""
    rows = run.rows
    artefacts = sum(1 for r in rows if r.image_type == 'artefact')
    bars = sum(1 for r in rows if r.image_type == 'scale_bar')
    texts = sum(1 for r in rows if r.image_type == 'text')
    skipped = sum(1 for r in rows if r.image_type == 'skipped')
    logging.info(
        "Done%s: %d page(s) -> %d artefact(s), %d scale bar(s), %d text box(es) not cut, "
        "%d page(s) skipped (expect=0), in %s",
        f" in {format_duration(seconds)}" if seconds is not None else "",
        page_count, artefacts, bars, texts, skipped, output_dir,
    )
    cut_before = {r['input_page_id'] for r in previous} & run.recut
    if cut_before:
        logging.info(
            "An earlier run cut %d page(s). This run replaced their crops.",
            len(cut_before)
        )
    if run.orphans:
        logging.warning(
            "%d crop file(s) left by an earlier run had no manifest row. "
            "This run replaced them.", len(run.orphans),
        )
    upscaled = {r.input_page_id for r in rows if int(r.upscale_factor or 1) > 1}
    if upscaled:
        logging.info(
            "The command upscaled %d page(s) for detection. The crops are at the "
            "source size. See upscale_factor in pages_manifest.csv.", len(upscaled),
        )
    split = sum(SPLIT in r.correction_applied for r in rows)
    joined = sum(JOIN in r.correction_applied for r in rows)
    if split or joined:
        logging.info(
            "Identifiers split %d box(es) and joined %d. "
            "See correction_applied in pages_manifest.csv.", split, joined,
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
    _add_pdf_args(parser)
    _add_config_args(parser)
    return parser


def _add_required_args(parser: argparse.ArgumentParser) -> None:
    """Add required argument group."""
    group = parser.add_argument_group('NECESSARY ARGUMENTS')
    group.add_argument(
        '--data_dir', required=True, metavar='PATH',
        help='The project folder that contains pages/. You can also '
             'give a folder of page images.'
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
             'shown by --debug. Default: corrections.csv in the project '
             'folder. If this file is missing, the command writes an empty one.'
    )


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    """Add output argument group."""
    group = parser.add_argument_group('OUTPUT OPTIONS')
    group.add_argument(
        '--padding', type=int, metavar='PX',
        help='The white margin around each crop, in pixels (default: 10).'
    )
    group.add_argument(
        '--debug', action='store_true',
        help='Write pages_debug/{page}.png for each page. The overlay '
             'shows each box, each scale bar with its score, each '
             'rejected candidate in grey, and each identifier read.'
    )
    group.add_argument(
        '--disable_scale_bars', action='store_true',
        help='Do not write scale bar crops.'
    )
    group.add_argument(
        '--keep_text_boxes', action='store_true',
        help='Keep a box that the command identifies as text, for example a framed '
             'specimen code. By default, the manifest records such a box as text, and '
             'the command writes no crop for it.'
    )
    group.add_argument(
        '--keep_text', action='store_true',
        help='Keep the text of the page in each artefact crop: captions, legend keys, '
             'scale bar ink and other identifiers. By default, the command hides them '
             'under the paper colour of the crop. The identifier of the crop stays.'
    )
    group.add_argument(
        '--hide_identifiers', action='store_true',
        help='Also hide the identifier of the crop. By default, the identifier stays. '
             'The command writes its box to meta_data.csv, and pylithics does not '
             'analyse that box.'
    )
    group.add_argument(
        '--read_labels', dest='read_labels', action='store_true',
        default=None,
        help='Read the identifier printed next to each lithic and name '
             'the crop {page}_figure_{label}.png. This is the default. '
             'RapidOCR is necessary. See the installation page.'
    )
    group.add_argument(
        '--no_read_labels', dest='read_labels', action='store_false',
        help='Do not read identifiers. Name the crops by box number only.'
    )


def _add_pdf_args(parser: argparse.ArgumentParser) -> None:
    """Add the PDF input argument group."""
    group = parser.add_argument_group('PDF OPTIONS')
    group.add_argument(
        '--pdf_dpi', type=float, metavar='DPI',
        help='The resolution for a PDF page that contains no raster image, in '
             'dots per inch (default: 300). The command renders a page that '
             'contains a raster image at the resolution of that image.'
    )
    group.add_argument(
        '--pdf_pages', metavar='LIST',
        help='The pages to render from each PDF file, for example 4,7,11-14 '
             '(default: all pages).'
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
