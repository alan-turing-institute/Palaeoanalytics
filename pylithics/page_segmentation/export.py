"""
Crop export and manifest writing for page segmentation.

Crops are cut from the untouched source array, never from the detection
mask, and written at the source DPI and colour mode. Saved pixels are
therefore bit-identical to the page within the crop box, which keeps a
measurement taken from a crop comparable to one taken from a
single-artefact scan.

The output is the project layout the main pipeline consumes, so
``pylithics-pages --data_dir X`` can be followed by
``pylithics --data_dir X`` with no file movement. A project that already
holds images and a ``meta_data.csv`` is added to, never replaced::

    X/
    ├── pages/           the plates (input)
    ├── images/          artefact crops
    ├── scales/          scale bar crops
    ├── pages_debug/     numbered overlays, one for each page (--debug only)
    ├── pages_manifest.csv     how each crop was cut
    └── meta_data.csv    links each crop to its scale, with flags
"""

import csv
import logging
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from .detection import PageImage
from .geometry import BBox, box_height, box_width, clamp_box
from .identifiers import Identifier

IMAGES_DIRNAME = 'images'
SCALES_DIRNAME = 'scales'
# Named after the flag that makes it, as pylithics names its own
# threshold_debug/, scale_debug/ and arrow_debug/. The file inside is
# named after the source page.
DEBUG_DIRNAME = 'pages_debug'
MANIFEST_FILENAME = 'pages_manifest.csv'
METADATA_FILENAME = 'meta_data.csv'

METADATA_COLUMNS = ['image_id', 'scale_id', 'scale', 'flag']
# Flags written to meta_data.csv. The analysis reports a flagged row
# and runs it like any other, so the user knows which crops to examine.
NO_SCALE_FLAG = 'no_scale'
SEVERAL_SCALES_FLAG = 'several_scales'
_FLAG_SEPARATOR = ';'

MANIFEST_COLUMNS = [
    'output_crop_id', 'image_type', 'input_page_id', 'crop_id_index',
    'x0', 'y0', 'x1', 'y1', 'width_px', 'height_px',
    'dpi', 'colour_mode', 'n_components', 'correction_applied',
    'label', 'label_source', 'label_confidence', 'label_flag',
    'label_candidates',
]

# Manifest names for the source colour mode. Pillow's own names are
# terse ("L" for greyscale); the manifest spells them out for a reader.
_COLOUR_MODE_NAMES = {'L': 'greyscale', '1': 'bilevel', 'RGB': 'RGB', 'RGBA': 'RGBA'}

# Debug overlay colours, RGB.
_BOX_COLOUR = (215, 48, 39)
_BAR_COLOUR = (94, 60, 153)
_TEXT_COLOUR = (0, 0, 0)
_LABEL_COLOUR = (0, 140, 0)
# Height of the summary band above a labels overlay, before scaling.
_HEADER_HEIGHT = 40


@dataclass
class ManifestRow:
    """One exported crop, as recorded in the manifest."""

    output_crop_id: str
    image_type: str
    input_page_id: str
    crop_id_index: str
    x0: int
    y0: int
    x1: int
    y1: int
    width_px: int
    height_px: int
    dpi: str
    colour_mode: str
    n_components: int
    correction_applied: str
    label: str = ''
    label_source: str = ''
    label_confidence: str = ''
    label_flag: str = ''
    label_candidates: str = ''


def prepare_output_dir(output_dir: str) -> List[Dict[str, str]]:
    """
    Create the output layout and return the previous run's manifest.

    Every run cuts every plate. A plate cut before is cut again and
    its previous crops are replaced; the rows returned here say which
    files those are. Images and metadata the user placed in the
    project are never touched.

    Parameters
    ----------
    output_dir : str
        Destination directory.

    Returns
    -------
    list of dict
        The manifest rows of the previous run, or ``[]``.
    """
    for name in (IMAGES_DIRNAME, SCALES_DIRNAME):
        os.makedirs(os.path.join(output_dir, name), exist_ok=True)

    manifest = os.path.join(output_dir, MANIFEST_FILENAME)
    if os.path.isfile(manifest):
        return _read_manifest(manifest)
    return []


def remove_page_output(
    output_dir: str, previous: Sequence[Dict[str, str]]
) -> set:
    """
    Remove the crops a previous run cut from one page.

    Done before the page is cut again, so a crop the new run does not
    produce cannot linger with no manifest row explaining it. Only the
    files the manifest names are removed.

    Returns
    -------
    set of str
        The crop filenames removed.
    """
    removed = set()
    for row in previous:
        folder = (
            IMAGES_DIRNAME if row['image_type'] == 'artefact'
            else SCALES_DIRNAME
        )
        _remove(os.path.join(output_dir, folder, row['output_crop_id']))
        removed.add(row['output_crop_id'])
    return removed


def _read_manifest(path: str) -> List[Dict[str, str]]:
    """Read a manifest written by an earlier run."""
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def _read_metadata(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """Read ``meta_data.csv`` as its column names and rows."""
    with open(path, newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_rows(path: str, fieldnames: List[str], rows: List[Dict]) -> None:
    """Write CSV rows, filling any column a row lacks with ''."""
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval='')
        writer.writeheader()
        writer.writerows(rows)


def _remove(path: str) -> None:
    """Remove a file if it is there."""
    if os.path.isfile(path):
        os.remove(path)


def export_page(
    page: PageImage,
    boxes: Sequence[BBox],
    bars: Sequence[BBox],
    output_dir: str,
    padding: int,
    correction_applied: str = '',
    component_counts: Optional[Dict[int, int]] = None,
    identifiers: Optional[Sequence[Identifier]] = None,
) -> List[ManifestRow]:
    """
    Write every crop for one page and return its manifest rows.

    Parameters
    ----------
    page : PageImage
        The loaded source page.
    boxes, bars : sequence of BBox
        Artefact boxes in reading order, and scale bar boxes.
    output_dir : str
        Destination root.
    padding : int
        Pixels of whitespace kept around each crop.
    correction_applied : str
        Which corrections were applied to this page, for the manifest.
    component_counts : dict, optional
        Map of box index to the number of ink blobs it groups.
    identifiers : sequence of Identifier, optional
        The identifier read for each box. A read one names its crop
        ``{page}_figure_{label}``; otherwise the index name stays.

    Returns
    -------
    list of ManifestRow : one row per crop written.
    """
    identifiers = identifiers or [Identifier() for _ in boxes]
    _refuse_collisions(page, bars, output_dir, identifiers)
    rows = _export_artefacts(
        page, boxes, output_dir, padding,
        component_counts or {}, correction_applied, identifiers,
    )
    rows.extend(_export_scale_bars(page, bars, output_dir, padding))

    logging.info(
        "%s: %d artefact(s), %d scale bar(s)",
        os.path.basename(page.path), len(boxes), len(bars),
    )
    return rows


def _refuse_collisions(
    page: PageImage,
    bars: Sequence[BBox],
    output_dir: str,
    identifiers: Sequence[Identifier],
) -> None:
    """
    Stop before writing a page whose crop would replace a user's image.

    Checked for the whole page first, so a refused page leaves no crop
    behind. The crops of a previous run never collide: they were
    removed before the page was cut again.

    Raises
    ------
    FileExistsError
        Naming the first crop that already exists.
    """
    names = [
        (IMAGES_DIRNAME, _artefact_name(page.stem, index, identifier))
        for index, identifier in enumerate(identifiers, start=1)
    ]
    names += [
        (SCALES_DIRNAME, _scale_bar_name(page.stem, index, len(bars)))
        for index in range(1, len(bars) + 1)
    ]
    for folder, name in names:
        if os.path.exists(os.path.join(output_dir, folder, name)):
            raise FileExistsError(
                f"{folder}/{name} is already in the project. The page is "
                f"not written. Rename the file, or the page."
            )


def _export_artefacts(
    page: PageImage,
    boxes: Sequence[BBox],
    output_dir: str,
    padding: int,
    counts: Dict[int, int],
    correction_applied: str,
    identifiers: Sequence[Identifier],
) -> List[ManifestRow]:
    """Write the artefact crops, named from the plate where read."""
    rows = []
    for index, (box, identifier) in enumerate(zip(boxes, identifiers), start=1):
        row = _write_crop(
            page, box, output_dir, IMAGES_DIRNAME,
            _artefact_name(page.stem, index, identifier),
            'artefact', str(index), padding,
            counts.get(index - 1, 1), correction_applied,
        )
        row.label = identifier.label
        row.label_source = identifier.source
        row.label_confidence = (
            f"{identifier.confidence:.2f}" if identifier.named else ''
        )
        row.label_flag = identifier.flag
        row.label_candidates = identifier.candidate_text
        rows.append(row)
    return rows


def _artefact_name(stem: str, index: int, identifier: Identifier) -> str:
    """
    Name a crop from its read identifier, or its box number.

    The two forms differ on purpose: ``_figure_7`` is the number
    printed on the plate, ``_box_07`` is the seventh red box on the
    overlay in reading order. The filename alone says which, and the
    two can never be mistaken for each other.
    """
    if identifier.named:
        return f"{stem}_figure_{identifier.label}.png"
    return f"{stem}_box_{index:02d}.png"


def _export_scale_bars(
    page: PageImage,
    bars: Sequence[BBox],
    output_dir: str,
    padding: int,
) -> List[ManifestRow]:
    """Write the scale bar crops."""
    return [
        _write_crop(
            page, bar, output_dir, SCALES_DIRNAME,
            _scale_bar_name(page.stem, index, len(bars)),
            'scale_bar', '', padding, 1, '',
        )
        for index, bar in enumerate(bars, start=1)
    ]


def _scale_bar_name(stem: str, index: int, total: int) -> str:
    """Name a scale bar crop, numbering only when a page has several."""
    if total <= 1:
        return f"{stem}_scale_bar.png"
    return f"{stem}_scale_bar_{index:02d}.png"


def _write_crop(
    page: PageImage,
    box: BBox,
    output_dir: str,
    subdir: str,
    crop_id: str,
    image_type: str,
    crop_id_index: str,
    padding: int,
    n_components: int,
    correction_applied: str,
) -> ManifestRow:
    """Cut one crop from the source array and save it."""
    x0, y0, x1, y1 = clamp_box(box, padding, page.width, page.height)
    crop = page.array[y0:y1, x0:x1]

    destination = os.path.join(output_dir, subdir, crop_id)
    image = Image.fromarray(crop, mode=page.mode)
    image.save(destination, **_dpi_kwargs(page))

    logging.debug("Wrote %s (%dx%d)", crop_id, x1 - x0, y1 - y0)
    return ManifestRow(
        output_crop_id=crop_id,
        image_type=image_type,
        input_page_id=os.path.basename(page.path),
        crop_id_index=crop_id_index,
        x0=x0, y0=y0, x1=x1, y1=y1,
        width_px=x1 - x0,
        height_px=y1 - y0,
        dpi=f"{page.dpi[0]:g}" if page.dpi else '',
        colour_mode=_COLOUR_MODE_NAMES.get(page.mode, page.mode),
        n_components=n_components,
        correction_applied=correction_applied,
    )


def _dpi_kwargs(page: PageImage) -> Dict:
    """
    Build the DPI keyword for saving, omitted when the source had none.

    Writing a guessed DPI would fabricate provenance and silently
    corrupt any downstream calibration, so an untagged page stays
    untagged.
    """
    return {'dpi': page.dpi} if page.dpi else {}


def write_manifest(rows: Sequence, output_dir: str) -> str:
    """
    Write the manifest recording every crop cut from the plates.

    Parameters
    ----------
    rows : sequence of ManifestRow or dict
        Rows accumulated across all pages. Rows of a previous run,
        read back as dicts, come first.
    output_dir : str
        Destination root.

    Returns
    -------
    str
        Path to the written manifest.
    """
    path = os.path.join(output_dir, MANIFEST_FILENAME)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=MANIFEST_COLUMNS, restval='',
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row if isinstance(row, dict) else asdict(row))

    logging.info("Wrote manifest with %d row(s) to %s", len(rows), path)
    return path


def write_metadata(
    rows: Sequence[ManifestRow],
    output_dir: str,
    removed: Optional[set] = None,
) -> str:
    """
    Write ``meta_data.csv``, the file the analysis command reads.

    One row for each artefact crop: ``scale_id`` is the scale bar crop
    from the same page when that page has exactly one; ``scale`` (the
    bar's printed length, in millimetres) is left blank for the user;
    ``flag`` names what the user must correct first (no scale bar,
    several, or an identifier problem). The analysis reports flagged rows.

    An existing file is merged, not replaced: see
    :func:`_merge_metadata_rows`. A ``flag`` column is added if missing.

    Parameters
    ----------
    rows : sequence of ManifestRow
        Rows accumulated across all pages.
    output_dir : str
        Destination root. Returns the path of the written file.
    removed : set of str, optional
        Crops of a previous run removed before this run.
    """
    scales_by_page = _scales_by_page(rows)
    path = os.path.join(output_dir, METADATA_FILENAME)
    fieldnames, existing = (
        _read_metadata(path) if os.path.isfile(path)
        else (list(METADATA_COLUMNS), [])
    )
    if 'flag' not in fieldnames:
        fieldnames.append('flag')
    new = {
        row.output_crop_id: _metadata_row(
            row, scales_by_page.get(row.input_page_id, [])
        )
        for row in rows if row.image_type == 'artefact'
    }
    written, flagged = len(new), sum(1 for e in new.values() if e['flag'])
    merged = _merge_metadata_rows(existing, new, removed or set())

    _write_rows(path, fieldnames, merged)
    logging.info(
        "%d row(s) written to %s, %d with a flag. Fill in the scale column. "
        "Then remove each flag.", written, path, flagged,
    )
    return path


def _scales_by_page(rows: Sequence[ManifestRow]) -> Dict[str, List[str]]:
    """Map each page to the scale bar crops cut from it."""
    scales: Dict[str, List[str]] = {}
    for row in rows:
        if row.image_type == 'scale_bar':
            scales.setdefault(row.input_page_id, []).append(row.output_crop_id)
    return scales


def _merge_metadata_rows(
    existing: List[Dict[str, str]],
    new: Dict[str, Dict[str, str]],
    removed: set,
) -> List[Dict[str, str]]:
    """
    Merge this run's rows into the user's file.

    A row for a crop cut again gets the new ``scale_id`` and ``flag``
    and keeps the ``scale`` the user typed. A row for a crop removed
    and not cut again is dropped. The user's own rows do not change.
    New rows go at the end.
    """
    merged = []
    for row in existing:
        image_id = row.get('image_id', '')
        if image_id in new:
            entry = new.pop(image_id)
            entry['scale'] = row.get('scale', '')
            merged.append({**row, **entry})
        elif image_id not in removed:
            merged.append(row)
    merged.extend(new.values())
    return merged


def _metadata_row(row: ManifestRow, scales: Sequence[str]) -> Dict[str, str]:
    """Build one meta_data.csv row for an artefact crop."""
    flags = []
    scale_id = ''
    if len(scales) == 1:
        scale_id = scales[0]
    elif not scales:
        flags.append(NO_SCALE_FLAG)
    else:
        flags.append(SEVERAL_SCALES_FLAG)
    if row.label_flag:
        flags.append(row.label_flag)
    return {
        'image_id': row.output_crop_id,
        'scale_id': scale_id,
        'scale': '',
        'flag': _FLAG_SEPARATOR.join(flags),
    }


def write_debug_overlay(
    page: PageImage,
    boxes: Sequence[BBox],
    bars: Sequence[BBox],
    output_dir: str,
    identifiers: Optional[Sequence[Identifier]] = None,
) -> str:
    """
    Draw the page with its artefact boxes, scale bars and readings.

    Red numbered boxes are the crops (the numbers are what the
    corrections CSV refers to), purple boxes are scale bars. When
    identifiers were read, each is drawn in green beside its glyph with
    the crop's name under its number; a crop that could not be named
    shows its flag and every reading inside it in red. A header gives
    the page's totals, and says so when nothing was found.

    Parameters
    ----------
    page : PageImage
        The loaded source page.
    boxes, bars : sequence of BBox
        Artefact boxes in reading order; scale bar boxes.
    output_dir : str
        Destination root.
    identifiers : sequence of Identifier, optional
        The identifier read for each box.

    Returns
    -------
    str
        Path to the written overlay.
    """
    debug_dir = os.path.join(output_dir, DEBUG_DIRNAME)
    os.makedirs(debug_dir, exist_ok=True)
    scale = max(1, page.width // 700)
    canvas = _debug_canvas(page)
    lift = 0
    if identifiers:
        canvas = _with_header(canvas, _labels_summary(identifiers), scale)
        lift = _HEADER_HEIGHT * scale
    _draw_debug_boxes(canvas, boxes, bars, scale, lift)
    if identifiers:
        _draw_readings(canvas, boxes, identifiers, scale, lift)

    path = os.path.join(debug_dir, f"{page.stem}.png")
    Image.fromarray(canvas).save(path, **_dpi_kwargs(page))
    logging.debug("Wrote debug overlay %s", path)
    return path


def _draw_debug_boxes(
    canvas: np.ndarray,
    boxes: Sequence[BBox],
    bars: Sequence[BBox],
    scale: int,
    lift: int = 0,
) -> None:
    """Draw numbered artefact boxes and scale bar boxes onto a canvas."""
    for index, box in enumerate(boxes, start=1):
        cv2.rectangle(
            canvas, (box[0], box[1] + lift), (box[2], box[3] + lift),
            _BOX_COLOUR, 2 * scale,
        )
        cv2.putText(
            canvas, str(index), (box[0] + 5, box[1] + lift + 25 * scale),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, _TEXT_COLOUR, 2 * scale,
        )

    for bar in bars:
        cv2.rectangle(
            canvas, (bar[0], bar[1] + lift), (bar[2], bar[3] + lift),
            _BAR_COLOUR, 2 * scale,
        )


def _draw_readings(
    canvas: np.ndarray,
    boxes: Sequence[BBox],
    identifiers: Sequence[Identifier],
    scale: int,
    lift: int,
) -> None:
    """Draw each reading beside its glyph and each crop's verdict."""
    for box, identifier in zip(boxes, identifiers):
        colour = _LABEL_COLOUR if identifier.named else _BOX_COLOUR
        for text, _, glyph in identifier.candidates:
            _draw_reading(canvas, glyph, text, colour, scale, lift)
        verdict = (f"figure_{identifier.label}" if identifier.named
                   else identifier.flag)
        cv2.putText(canvas, verdict, (box[0] + 5, box[3] + lift - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, colour, 2 * scale)


def _draw_reading(canvas, glyph, text, colour, scale, lift) -> None:
    """Box a glyph and write what was read from it, just above."""
    cv2.rectangle(canvas, (glyph[0] - 2, glyph[1] + lift - 2),
                  (glyph[2] + 2, glyph[3] + lift + 2), colour, scale)
    cv2.putText(canvas, text, (glyph[2] + 4, glyph[1] + lift - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, colour, scale)


def _labels_summary(identifiers: Sequence[Identifier]) -> str:
    """One line stating what identifier reading found on the page."""
    named = sum(i.named for i in identifiers)
    if not named and not any(i.candidates for i in identifiers):
        return f"no identifiers found on this page ({len(identifiers)} crops)"
    several = sum(i.flag == 'several_identifiers' for i in identifiers)
    duplicate = sum(i.flag == 'duplicate' for i in identifiers)
    none = sum(i.flag == 'no_identifier' for i in identifiers)
    return (f"{named} of {len(identifiers)} crops named from the plate; "
            f"{several} hold several identifiers, {duplicate} duplicate, "
            f"{none} without one")


def _with_header(canvas: np.ndarray, text: str, scale: int) -> np.ndarray:
    """Add a white band above the page carrying one line of text."""
    band = np.full((_HEADER_HEIGHT * scale, canvas.shape[1], 3), 255, np.uint8)
    cv2.putText(band, text, (10, int(_HEADER_HEIGHT * scale * 0.65)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, _TEXT_COLOUR, scale)
    return np.vstack([band, canvas])


def _debug_canvas(page: PageImage) -> np.ndarray:
    """Build an RGB canvas from the page for annotation."""
    if page.array.ndim == 2:
        return cv2.cvtColor(page.gray, cv2.COLOR_GRAY2RGB)
    if page.array.shape[2] == 4:
        return page.array[:, :, :3].copy()
    return page.array.copy()
