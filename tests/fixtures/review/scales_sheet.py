#!/usr/bin/env python3
"""
Contact sheet of every exported scale crop, for review by eye.

Usage::

    python tests/fixtures/review/scales_sheet.py [OUTPUT_DIR]

Writes ``<OUTPUT_DIR>/review/scales_contact_sheet.png``, numbering the
scales in manifest order so a reviewer can refer to them by number.
"""
import csv
import math
import os
import sys

from PIL import Image, ImageDraw, ImageFont

COLS, TILE_W, TILE_H = 3, 520, 300


def main(output_dir):
    """Build the sheet from the manifest in ``output_dir``."""
    with open(os.path.join(output_dir, 'pages_manifest.csv'), encoding='utf-8') as handle:
        rows = [r for r in csv.DictReader(handle) if r['image_type'] == 'scale_bar']
    sheet_dir = os.path.join(output_dir, 'review')
    os.makedirs(sheet_dir, exist_ok=True)
    font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 15)
    height = max(1, math.ceil(len(rows) / COLS)) * TILE_H
    canvas = Image.new('RGB', (COLS * TILE_W, height), 'white')
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(rows):
        x, y = (index % COLS) * TILE_W, (index // COLS) * TILE_H
        tile = Image.open(os.path.join(output_dir, 'scales', row['output_crop_id'])).convert('RGB')
        tile.thumbnail((TILE_W - 20, TILE_H - 40))
        canvas.paste(tile, (x + 10, y + 32))
        draw.rectangle((x, y, x + TILE_W - 1, y + TILE_H - 1), outline=(200, 200, 200))
        label = f"{index + 1}. {row['input_page_id'][:58]}"
        draw.text((x + 8, y + 8), label, fill=(0, 0, 160), font=font)
    path = os.path.join(sheet_dir, 'scales_contact_sheet.png')
    canvas.save(path)
    print(f'{len(rows)} scale(s) -> {path}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join('pylithics', 'data'))
