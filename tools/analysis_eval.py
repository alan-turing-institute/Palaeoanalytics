"""
Measure how the analysis changes with input resolution.

Takes single-artefact images at a good resolution, makes smaller
variants (one half, one quarter, one eighth, by area averaging), runs
the preprocessing and contour extraction on each with the working
resolution on and off, and compares each variant with its source: the
number of surfaces found, the number of scars, and the area of the
largest surface in source pixels. Closer to the source is better.

Usage::

    python tools/analysis_eval.py --images pylithics/data/images --out results/analysis_eval

Only images tagged 300 DPI or more are used as sources. Not part of the
package. Development only.
"""

import argparse
import csv
import os
import sys
import tempfile
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pylithics.image_processing.config import get_config_manager  # noqa: E402
from pylithics.image_processing.importer import execute_preprocessing_pipeline  # noqa: E402
from pylithics.image_processing.modules.contour_extraction import (  # noqa: E402
    extract_contours_with_hierarchy,
)

FRACTIONS = (1.0, 0.5, 0.25, 0.125)
MIN_SOURCE_DPI = 300


def sources(folder: str) -> List[str]:
    """Image files in ``folder`` tagged at least ``MIN_SOURCE_DPI``."""
    found = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        try:
            with Image.open(path) as image:
                dpi = image.info.get('dpi', (0, 0))[0]
        except OSError:
            continue
        if dpi and dpi >= MIN_SOURCE_DPI:
            found.append(path)
    return found


def make_variant(path: str, fraction: float, workdir: str) -> str:
    """Write ``path`` scaled by ``fraction`` (area averaging) and return its path."""
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if fraction < 1.0:
        size = (max(1, round(image.shape[1] * fraction)), max(1, round(image.shape[0] * fraction)))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    out = os.path.join(workdir, f'{os.path.splitext(os.path.basename(path))[0]}_{fraction:g}.png')
    cv2.imwrite(out, image)
    return out


def measure(path: str, config: Dict, fraction: float, workdir: str) -> Optional[Dict]:
    """Surfaces, scars and largest area (in source pixels) for one variant."""
    processed = execute_preprocessing_pipeline(path, config)
    if processed is None:
        return None
    contours, hierarchy = extract_contours_with_hierarchy(
        processed.image, os.path.basename(path), workdir
    )
    if not contours or hierarchy is None:
        return {'surfaces': 0, 'scars': 0, 'largest_area': 0.0, 'factor': processed.upscale_factor}
    parents = [i for i, h in enumerate(hierarchy) if h[3] == -1]
    children = [i for i, h in enumerate(hierarchy) if h[3] != -1]
    areas = [cv2.contourArea(contours[i]) for i in parents] or [0.0]
    scale = 1.0 / (fraction * fraction)               # back to source pixel area
    return {'surfaces': len(parents), 'scars': len(children),
            'largest_area': max(areas) * scale, 'factor': processed.upscale_factor}


def run(images: str, out: str) -> int:
    """Run every source at every fraction, on and off, and write the table."""
    base = get_config_manager().config
    configs = {'on': base, 'off': {**base, 'working_resolution': {'enabled': False}}}
    rows = []
    os.makedirs(out, exist_ok=True)
    with tempfile.TemporaryDirectory() as workdir:
        for path in sources(images):
            reference = None
            for fraction in FRACTIONS:
                variant = make_variant(path, fraction, workdir)
                for mode, config in configs.items():
                    result = measure(variant, config, fraction, workdir)
                    if result is None:
                        continue
                    if fraction == 1.0 and mode == 'off':
                        reference = result
                    rows.append(_row(path, fraction, mode, result, reference))
    path = os.path.join(out, 'analysis_eval.csv')
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summarise(rows)
    print(f'\nRows written to {path}')
    return 0


def _row(path: str, fraction: float, mode: str, result: Dict, reference: Optional[Dict]) -> Dict:
    """One table row: the result and its distance from the full-size run."""
    has_ref = bool(reference and reference['largest_area'])
    return {'image': os.path.basename(path), 'fraction': fraction, 'mode': mode, **result,
            'area_vs_source': result['largest_area'] / reference['largest_area'] if has_ref else '',
            'surfaces_vs_source': result['surfaces'] - reference['surfaces'] if reference else '',
            'scars_vs_source': result['scars'] - reference['scars'] if reference else ''}


def summarise(rows: List[Dict]) -> None:
    """Print, per fraction and mode, the mean absolute error against the source."""
    print(f"{'fraction':>8} {'mode':>4} {'n':>3} {'|area-1|':>9} {'|surf|':>6} "
          f"{'|scars|':>7} {'factor':>6}")
    for fraction in FRACTIONS:
        for mode in ('off', 'on'):
            cell = [r for r in rows if r['fraction'] == fraction and r['mode'] == mode
                    and r['area_vs_source'] != '']
            if not cell:
                continue
            area = np.mean([abs(float(r['area_vs_source']) - 1) for r in cell])
            surf = np.mean([abs(r['surfaces_vs_source']) for r in cell])
            scars = np.mean([abs(r['scars_vs_source']) for r in cell])
            factor = np.mean([r['factor'] for r in cell])
            print(f"{fraction:>8g} {mode:>4} {len(cell):>3} {area:>9.3f} {surf:>6.2f} "
                  f"{scars:>7.2f} {factor:>6.2f}")


def main(argv=None) -> int:
    """Run the analysis on each variant and print the table."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--images', required=True, help='folder of single-artefact images')
    parser.add_argument('--out', default='results/analysis_eval', help='where to write the table')
    args = parser.parse_args(argv)
    return run(args.images, args.out)


if __name__ == '__main__':
    sys.exit(main())
