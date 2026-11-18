"""Helpers for reading the metadata CSV file."""

import csv
import logging


def read_metadata(meta_file_path):
    """
    Read metadata from a CSV file.

    Returns a list of dictionaries with keys: image_id, scale_id, scale.
    """
    metadata = []
    try:
        with open(meta_file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                metadata.append(row)
        return metadata
    except FileNotFoundError:
        logging.error(f"The metadata file is missing: {meta_file_path}")
        return []
    except Exception as e:
        logging.error(f"Error when the metadata file {meta_file_path} was read: {e}")
        return []


def format_duration(seconds: float) -> str:
    """
    Write a run time for a person to read: ``8 s``, ``1 min 12 s``, ``2 h 5 min``.

    Used in the closing line of both commands, which goes to the screen
    and to the log.
    """
    total = int(round(max(seconds, 0)))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes} min"
    if minutes:
        return f"{minutes} min {secs} s"
    return f"{secs} s"


def progress_columns(label: str, field: str) -> tuple:
    """
    Return the progress bar columns that both commands use.

    The elapsed time comes first after the label and the bar has a fixed
    width, so in a narrow terminal the file name is shortened, never the
    time. ``field`` is the task field that holds the current file name.
    """
    from rich.progress import (
        BarColumn, MofNCompleteColumn, SpinnerColumn, TextColumn,
        TimeElapsedColumn, TimeRemainingColumn,
    )
    from rich.table import Column
    return (
        SpinnerColumn(style="cyan"),
        TextColumn(f"[cyan]{label}[/]", table_column=Column(no_wrap=True, min_width=len(label))),
        TimeElapsedColumn(table_column=Column(no_wrap=True, min_width=7)),
        BarColumn(bar_width=20, table_column=Column(no_wrap=True, min_width=20)),
        MofNCompleteColumn(table_column=Column(no_wrap=True, min_width=7)),
        TimeRemainingColumn(table_column=Column(no_wrap=True, min_width=7)),
        TextColumn(f"[dim]{{task.fields[{field}]}}[/]",
                   table_column=Column(no_wrap=True, overflow="ellipsis", ratio=1,
                                       min_width=8)),
    )
