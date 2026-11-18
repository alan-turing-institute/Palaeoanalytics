"""
The dashboard as one report: CSV, HTML and PDF.

Streamlit cannot save a page as a file, so the report is built from the
same data and the same chart functions as the pages. The Distributions
charts are recorded by running that page's functions with a recorder in
place of Streamlit (see ``distributions.recording``), so the report shows
the figures the page shows. The HTML keeps them as interactive Plotly
charts; the PDF draws them again with Matplotlib from the figures' own
data, so no browser is needed.

``review.csv`` needs none of this and is written after every analysis
run by :func:`write_review_csv`. Plotly, Streamlit and Matplotlib are
imported only when the HTML or the PDF is made.
"""

import csv
import html
import io
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd

from pylithics.image_processing.modules.dashboard.data import CHECKS, REVIEW_COLUMNS

REVIEW_FILENAME = "review.csv"
_PAGE = (11.69, 8.27)           # A4 landscape, inches
_TABLE_ROWS = 18                # review rows on one PDF page; a reason can wrap
_TAGS = re.compile(r"<[^>]+>")

# Streamlit's light theme gives a Plotly chart these colours, in this order, for
# every trace that sets none of its own (read from Streamlit's theme, version 1.65).
# The report applies them too, so its charts have the colours of the dashboard.
STREAMLIT_COLORWAY = [
    "#0068c9", "#83c9ff", "#ff2b2b", "#ffabab", "#29b09d",
    "#7defa1", "#ff8700", "#ffd16a", "#6d3fc0", "#d5dae5",
]
_RGB = re.compile(r"rgba?\(([^)]*)\)")
# Streamlit's Plotly template writes these placeholders, #000001 to #000010, and its
# browser code swaps them for the colours above. A figure saved outside the browser
# keeps the placeholders, which print as black, so the report swaps them itself.
_PLACEHOLDERS = {f"#0000{n:02d}": colour for n, colour in enumerate(STREAMLIT_COLORWAY, 1)}

# The Overview tiles, in their order, as (key, label).
OVERVIEW_TILES = [
    ("successful", "Successful"), ("failed", "Failed"),
    ("unclassified_surfaces", "Unclassified surfaces"),
    ("zero_scar_lithics", "Zero-scar lithics"),
    ("low_confidence_scales", "Low-confidence scales"),
    ("pixel_only_lithics", "Pixel-only"), ("mixed_dpi", "Mixed DPI"),
    ("missing_dpi", "Missing DPI"), ("lithics", "Lithics"), ("surfaces", "Surfaces"),
    ("scars", "Scars"), ("scars_with_arrows", "Scars with arrows"),
    ("cortex_regions", "Cortex regions"),
]


def run_name(processed_dir) -> str:
    """
    Return the name of the run: the project folder of its results.

    ``<project>/results`` gives ``<project>``. A results folder with any
    other name, given to ``--explore``, gives its own name.
    """
    from pathlib import Path
    folder = Path(str(processed_dir)).resolve()
    return folder.parent.name if folder.name == "results" and folder.parent.name else folder.name


def export_filename(processed_dir, extension: str) -> str:
    """Return the file name of an export: ``<run>_report.<extension>``."""
    return f"{run_name(processed_dir)}_report.{extension}"


### CSV ###

def report_csv(items: pd.DataFrame) -> str:
    """Return the review rows as CSV text, with the header on a clean run too."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=REVIEW_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in items.to_dict("records"):
        writer.writerow({k: row.get(k, "") for k in REVIEW_COLUMNS})
    return buffer.getvalue()


def write_review_csv(items: pd.DataFrame, results_dir: str) -> str:
    """Write ``review.csv`` into the results folder and return its path."""
    path = os.path.join(results_dir, REVIEW_FILENAME)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(report_csv(items))
    return path


### THE CHARTS OF THE DISTRIBUTIONS PAGE ###

class _Null:
    """A stand-in for a Streamlit container: a context manager that ignores calls."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getattr__(self, name):
        return lambda *args, **kwargs: _Null()


class _Recorder(_Null):
    """Record what the Distributions page draws: headings, figures, captions, notes."""

    def __init__(self):
        self.entries: List[Tuple[str, Any]] = []

    def header(self, text, *a, **k):
        self.entries.append(("section", str(text)))

    def subheader(self, text, *a, **k):
        self.entries.append(("heading", str(text)))

    def plotly_chart(self, fig, *a, **k):
        self.entries.append(("figure", fig))

    def caption(self, text, *a, **k):
        self.entries.append(("caption", str(text)))

    def info(self, text, *a, **k):
        self.entries.append(("note", str(text)))

    def columns(self, spec, *a, **k):
        return [_Null() for _ in range(spec if isinstance(spec, int) else len(spec))]

    def tabs(self, names, *a, **k):
        return [_Null() for _ in names]


def chart_entries(df: pd.DataFrame) -> List[Tuple[str, Any]]:
    """Run every Distributions tab on the unfiltered data and return what it drew."""
    from pylithics.image_processing.modules.dashboard.views import distributions

    with distributions.recording(_Recorder()) as recorder:
        distributions.render_all(df)
    return recorder.entries


### HTML ###

_STYLE = """
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:2rem auto;
max-width:1100px;padding:0 1rem;color:#222;background:#fff}
h1{margin-bottom:0}h2{border-bottom:2px solid #ddd;padding-bottom:.3rem;margin-top:2.5rem}
table{border-collapse:collapse;margin:.5rem 0 1rem;font-size:.85rem}
th,td{border:1px solid #ccc;padding:.25rem .5rem;text-align:left;vertical-align:top}
th{background:#f3f3f3}.note{color:#555}.caption{color:#666;font-size:.85rem}
.clear{color:#2e7d32}.review{color:#c62828}
@media print{h2{page-break-before:always}}
"""


def report_html(bundle: Dict[str, Any], counts: Dict[str, int], items: pd.DataFrame) -> str:
    """
    Return the whole dashboard as one self-contained HTML file.

    Overview, Distributions and Review, in the order of the pages. The
    Plotly library is written into the file once, so it opens without a
    network and the charts stay interactive.
    """
    body = [f"<h1>PyLithics Explorer</h1><p class='note'>{_e(_run_line(bundle))}</p>",
            "<h2>Overview</h2>", _overview_table(counts), "<h2>Distributions</h2>"]
    first = True
    for kind, value in chart_entries(bundle["metrics"]):
        if kind == "figure":
            body.append(_themed(value).to_html(full_html=False, include_plotlyjs=first))
            first = False
        elif kind == "section":
            body.append(f"<h3>{_e(value)}</h3>")
        elif kind == "heading":
            body.append(f"<h4>{_e(value)}</h4>")
        else:
            body.append(f"<p class='{'caption' if kind == 'caption' else 'note'}'>"
                        f"{_e(_plain(value))}</p>")
    body += ["<h2>Review</h2>", _review_html(items)]
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>PyLithics report</title><style>{_STYLE}</style></head><body>"
            + "\n".join(body) + "</body></html>")


def _themed(fig):
    """
    Return a copy of a figure with the dashboard's look: white ground, Streamlit colours.

    Every Streamlit colour placeholder in the figure becomes its colour, as
    the dashboard shows it; the trace colours the page sets stay.
    """
    import plotly.graph_objects as go
    spec = _swap_placeholders(fig.to_plotly_json())
    spec["layout"].pop("template", None)
    themed = go.Figure(spec)
    themed.update_layout(template="plotly_white", colorway=STREAMLIT_COLORWAY)
    return themed


def _swap_placeholders(value):
    """Replace Streamlit's colour placeholders everywhere in a figure's JSON."""
    if isinstance(value, str):
        return _PLACEHOLDERS.get(value.lower(), value)
    if isinstance(value, dict):
        return {k: _swap_placeholders(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_swap_placeholders(v) for v in value]
    return value


def _overview_table(counts: Dict[str, int]) -> str:
    """Return the Overview tiles as a two-column table."""
    rows = "".join(f"<tr><td>{_e(name)}</td><td>{counts.get(key, 0)}</td></tr>"
                   for key, name in OVERVIEW_TILES)
    return f"<table><tr><th>Tile</th><th>Value</th></tr>{rows}</table>"


def _review_html(items: pd.DataFrame) -> str:
    """Return the Review sections: meaning, steps and items of each check, then the clear ones."""
    parts, clear = [], []
    for check in CHECKS.values():
        rows = items[items["check"] == check["title"]]
        if rows.empty:
            clear.append(check["title"])
            continue
        steps = "".join(f"<li>{_e(step)}</li>" for step in check["steps"])
        table = "".join(
            "<tr>" + "".join(f"<td>{_e(str(r[c]))}</td>" for c in ("image_id", "item",
                                                                   "value", "reason"))
            + "</tr>" for r in rows.to_dict("records"))
        parts.append(
            f"<h3 class='review'>{_e(check['title'])}: {len(rows)}</h3>"
            f"<p>{_e(check['meaning'])}</p><p><b>What to do</b></p><ol>{steps}</ol>"
            "<table><tr><th>Image</th><th>Item</th><th>Value</th><th>Reason</th></tr>"
            f"{table}</table>")
    if clear:
        parts.append(f"<p class='clear'>Clear: {_e(', '.join(clear))}.</p>")
    return "\n".join(parts)


### PDF ###

def report_pdf(bundle: Dict[str, Any], counts: Dict[str, int], items: pd.DataFrame) -> bytes:
    """
    Return the whole dashboard as a PDF: Overview, Distributions, Review.

    The charts are drawn with Matplotlib from the data of the page's own
    Plotly figures. Long review tables continue on the next page.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        pdf.savefig(_overview_page(bundle, counts))
        heading = ""
        for kind, value in chart_entries(bundle["metrics"]):
            if kind in ("section", "heading"):
                heading = value
            elif kind == "figure":
                pdf.savefig(_chart_page(_themed(value), heading))
        for page in _review_pages(items):
            pdf.savefig(page)
    return buffer.getvalue()


def _new_page():
    """Return a blank A4 landscape Matplotlib figure, made without pyplot."""
    from matplotlib.figure import Figure
    return Figure(figsize=_PAGE)


def _overview_page(bundle: Dict[str, Any], counts: Dict[str, int]):
    """Draw the title and the Overview tiles as a table."""
    page = _new_page()
    page.text(0.05, 0.92, "PyLithics Explorer", fontsize=22, weight="bold")
    page.text(0.05, 0.87, _run_line(bundle), fontsize=10, color="#555")
    page.text(0.05, 0.80, "Overview", fontsize=16)
    lines = [f"{name:<24}{counts.get(key, 0):>8}" for key, name in OVERVIEW_TILES]
    page.text(0.05, 0.76, "\n".join(lines), fontsize=11, family="monospace", va="top")
    return page


def _chart_page(fig, heading: str):
    """Draw one Plotly figure again with Matplotlib, from the figure's own data."""
    page = _new_page()
    ax = page.add_axes([0.08, 0.1, 0.84, 0.75])
    title = _plain(getattr(fig.layout.title, "text", None) or heading)
    page.text(0.08, 0.92, title, fontsize=14)
    categories: Dict[Any, int] = {}       # one position for each category label, all traces
    for index, trace in enumerate(fig.data):
        _draw_trace(ax, trace, categories, _trace_colour(trace, index))
    if categories and len(categories) <= 40:
        ticks = sorted(categories.items(), key=lambda kv: kv[1])
        horizontal = any(getattr(t, "orientation", None) == "h" for t in fig.data)
        set_ticks = ax.set_yticks if horizontal else ax.set_xticks
        set_ticks([i for _, i in ticks], [str(c)[:24] for c, _ in ticks], fontsize=7)
    for shape in fig.layout.shapes or []:
        _draw_shape(ax, shape)
    ax.set_xlabel(_plain(getattr(fig.layout.xaxis.title, "text", None) or ""))
    ax.set_ylabel(_plain(getattr(fig.layout.yaxis.title, "text", None) or ""))
    if any(getattr(t, "name", None) and getattr(t, "showlegend", True) is not False
           for t in fig.data):
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)
    return page


def _draw_trace(ax, trace, categories: Dict[Any, int], colour) -> None:
    """Draw one Plotly trace on a Matplotlib axis: scatter, line, histogram, bar, violin."""
    kind = trace.type
    x = _values(getattr(trace, "x", None), categories)
    y = _values(getattr(trace, "y", None), categories)
    name = getattr(trace, "name", None) or None
    if kind in ("scatter", "scattergl"):
        if x is None or y is None:
            return
        mode = trace.mode or "markers"
        if "lines" in mode:
            ax.plot(x, y, linewidth=1.2, label=name, color=_line_colour(trace, colour))
        if "markers" in mode:
            ax.scatter(x, y, s=14, label=None if "lines" in mode else name,
                       c=_marker_colours(trace, colour, len(x)))
    elif kind == "histogram":
        values = x if x is not None else y
        if values is not None and len(values):
            ax.hist(values, bins=trace.nbinsx or "auto", label=name, color=colour,
                    orientation="vertical" if x is not None else "horizontal")
    elif kind == "bar" and x is not None and y is not None:
        ax.bar(x, y, label=name, color=colour)
    elif kind in ("violin", "box"):
        _draw_distribution(ax, trace, x, y, categories, colour)


def _trace_colour(trace, index: int):
    """
    Return the colour Plotly gives a trace on the dashboard, for Matplotlib.

    The trace's own marker, line or fill colour when it sets one; else the
    Streamlit colour for its place among the figure's traces, as the
    dashboard's theme does.
    """
    for owner, attr in (("marker", "color"), ("line", "color"), (None, "fillcolor")):
        value = getattr(getattr(trace, owner, None), attr, None) if owner else getattr(
            trace, attr, None)
        if isinstance(value, str):
            return _mpl_colour(value)
    return _mpl_colour(STREAMLIT_COLORWAY[index % len(STREAMLIT_COLORWAY)])


def _line_colour(trace, colour):
    """Return the trace's own line colour, or its colour."""
    value = getattr(getattr(trace, "line", None), "color", None)
    return _mpl_colour(value) if isinstance(value, str) else colour


def _marker_colours(trace, colour, count: int):
    """Return one colour for each point when the trace colours its points one by one."""
    value = getattr(getattr(trace, "marker", None), "color", None)
    if value is not None and not isinstance(value, str) and len(value) == count and all(
            isinstance(v, str) for v in value):
        return [_mpl_colour(v) for v in value]
    return [colour]


def _mpl_colour(value: str):
    """Turn a Plotly colour (``#hex``, ``rgb(...)``, ``rgba(...)``, a name) into Matplotlib's."""
    match = _RGB.fullmatch(value.strip().replace(" ", ""))
    if not match:
        return value
    parts = [float(v) for v in match.group(1).split(",")]
    rgb = tuple(v / 255 for v in parts[:3])
    return rgb + (parts[3],) if len(parts) == 4 else rgb


def _draw_distribution(ax, trace, x, y, categories: Dict[Any, int], colour) -> None:
    """Draw a violin or box trace as a box at its category's position."""
    horizontal = getattr(trace, "orientation", None) == "h"
    values, where = (x, getattr(trace, "y", None)) if horizontal else (y, getattr(trace, "x", None))
    if values is None or not len(values):
        return
    values = values[~pd.isna(values)]
    label = list(where)[0] if where is not None and len(where) else trace.name
    position = categories.setdefault(label, len(categories))
    parts = ax.boxplot(values, positions=[position], vert=not horizontal, widths=0.5,
                       manage_ticks=False, patch_artist=True)
    for box in parts["boxes"]:
        box.set_facecolor(colour)
        box.set_alpha(0.6)


def _draw_shape(ax, shape) -> None:
    """Draw a straight reference line of a Plotly figure: a mean or a limit."""
    if shape.type != "line":
        return
    if shape.x0 == shape.x1 and shape.xref == "x":
        ax.axvline(shape.x0, color="#888", linestyle="--", linewidth=1)
    elif shape.y0 == shape.y1 and shape.yref == "y":
        ax.axhline(shape.y0, color="#888", linestyle="--", linewidth=1)


def _values(data, categories: Dict[Any, int]):
    """
    Return trace data as numbers, or None.

    Category labels (image ids, group names) become positions, one for
    each label across all the traces of a figure, so the stems and the
    dots of a lollipop chart, or a violin and its points, stay together.
    A None in the data, Plotly's gap between line pieces, becomes NaN.
    """
    if data is None:
        return None
    series = pd.Series(list(data), dtype=object)
    if not len(series):
        return None
    numbers = pd.to_numeric(series, errors="coerce")
    if numbers.notna().any() or series.isna().all():
        return numbers.to_numpy(dtype=float)
    return pd.Series([float("nan") if v is None else categories.setdefault(v, len(categories))
                      for v in series]).to_numpy(dtype=float)


def _review_pages(items: pd.DataFrame) -> list:
    """Draw the Review: for each check with items, its meaning, steps and table."""
    pages, clear = [], []
    for check in CHECKS.values():
        rows = items[items["check"] == check["title"]].to_dict("records")
        if not rows:
            clear.append(check["title"])
            continue
        for start in range(0, len(rows), _TABLE_ROWS):
            page = _new_page()
            head = check["title"] + (" (continued)" if start else f": {len(rows)}")
            page.text(0.05, 0.93, head, fontsize=15, color="#c62828")
            top = 0.88
            if not start:
                text = _wrap(check["meaning"], 130) + "\n\nWhat to do:\n" + "\n".join(
                    f"  {n}. {s}" for n, s in enumerate(check["steps"], 1))
                page.text(0.05, top, text, fontsize=9, va="top")
                top = 0.88 - 0.021 * (text.count("\n") + 3)
            table = [f"{'Image':<41}{'Value':<35}Reason", ""]
            for r in rows[start:start + _TABLE_ROWS]:
                reason = _wrap(str(r["reason"]), 84).split("\n") or [""]
                table.append(f"{r['image_id'][:40]:<41}{str(r['value'])[:34]:<35}{reason[0]}")
                table += [" " * 76 + more for more in reason[1:]]
            page.text(0.05, top, "\n".join(table), fontsize=7, family="monospace", va="top")
            pages.append(page)
    page = _new_page()
    page.text(0.05, 0.93, "Review: clear", fontsize=15, color="#2e7d32")
    page.text(0.05, 0.88, ", ".join(clear) or "None.", fontsize=10, va="top")
    return pages + [page]


### TEXT ###

def _run_line(bundle: Dict[str, Any]) -> str:
    """Name the run: its results folder, the time of the run and the lithic count."""
    summary = bundle.get("run_summary") or {}
    when = summary.get("timestamp") or datetime.now().isoformat(timespec="seconds")
    lithics = bundle["metrics"]["image_id"].nunique() if not bundle["metrics"].empty else 0
    return f"{bundle.get('processed_dir', '')} · run {when} · {lithics} lithics"


def _plain(text: str) -> str:
    """Remove HTML tags and Markdown emphasis from a chart title or caption."""
    return _TAGS.sub(" ", str(text)).replace("**", "").replace("`", "").strip()


def _e(text: str) -> str:
    """Escape text for HTML."""
    return html.escape(str(text))


def _wrap(text: str, width: int) -> str:
    """Wrap text to a width in characters, for the PDF."""
    import textwrap
    return "\n".join(textwrap.wrap(text, width))
