# Working Resolution

A small scan, or a small export from a PDF, has lines one or two
pixels wide. PyLithics cannot find the outlines in such an image. The
lines break and adjacent lithics merge. The numbers next to the
lithics are too small to read.

PyLithics measures each image. If the lines are too thin, it makes a
larger working copy of the image and finds the ink on that copy. Then
it returns the result to the pixel size of the original image. This
is the *working resolution*.

## The rule

**PyLithics detects at the working resolution. It measures at the
original resolution.**

- PyLithics cuts the saved crops from the original image. They keep
  the pixel size and the DPI of the original image.
- Every measurement is in the pixels of the original image. PyLithics
  calculates every millimetre value from the scale bar at its own
  resolution.
- PyLithics does not upscale the scale bar image.
- PyLithics never makes an image smaller.

The working resolution changes what PyLithics can find. It does not
change what PyLithics measures.

Arrow detection also reads the working copy. The return to the
original pixel grid removes the small notches of a small arrow. Thus
PyLithics cuts each scar from the working copy and looks for the
arrows there. The arrow positions in the CSV are in the pixels of the
original image.

## How it operates

1. PyLithics measures the width of the lines and the space between
   the hatch lines, in pixels.
2. If a value is below its limit, PyLithics selects the smallest
   factor of 2, 3 or 4 that makes the value large enough.
3. PyLithics makes the working copy with a neural network (ESPCN or
   FSRCNN). The package includes the two networks.
4. PyLithics finds the ink on the working copy. PyLithics multiplies
   the kernels of the threshold and the closing step by the factor.
5. PyLithics returns the black-and-white result to the pixel grid of
   the original image.
6. PyLithics keeps the working copy and looks for the arrows on it,
   scar by scar.

The two commands use different limits. A plate holds many small
drawings and fine hatching. Thus `pylithics-pages` uses the line
width only. The analysis of one lithic uses the line width and the
hatch space.

## Where the factor is recorded

| Output | Field | Value |
|---|---|---|
| `pages_manifest.csv` | `upscale_factor` | 1, 2, 3 or 4 for each crop of the page |
| `pages_manifest.csv` | `stroke_width_px` | The measured line width of the page |
| `meta_data.csv` | `flag` | `low_resolution` when the factor is more than 1 |
| `processed_metrics.csv` | `upscale_factor` | The factor for the image |
| The JSON file for each lithic | `calibration.upscale_factor` | The factor for the image |
| The log | | The measured values and the factor for each image |

A factor of 1 means that PyLithics used the image with no change.

The flag `low_resolution` in `meta_data.csv` tells you that the page
was thin at its original size. The crops from that page are also
thin. The analysis measures them again and upscales them again if
necessary. You do not have to do anything. If you can get a larger
scan of the page, use it.

## Configuration

The section `working_resolution` in `config.yaml`:

```yaml
working_resolution:
  enabled: true
  model: espcn               # espcn (fast) or fsrcnn
  max_factor: 4
  analysis:                  # the pylithics command
    min_stroke_width_px: 4.0
    min_hatch_gap_px: 12.0
    max_working_pixels: 40000000
  pages:                     # the pylithics-pages command
    min_stroke_width_px: 4.0
    min_hatch_gap_px: 0      # 0 turns this test off
    max_working_pixels: 40000000
```

- `enabled: false` turns the feature off for both commands. PyLithics
  then uses each image with no change and records the factor as 1.
- A limit of `0` turns that test off.
- `max_working_pixels` sets the maximum size of the working copy. If
  a factor makes the copy larger than this, PyLithics decreases the
  factor.

## Requirements

The neural networks need the contrib build of OpenCV. PyLithics
installs the package `opencv-contrib-python-headless`. If your
environment has the plain build, PyLithics tells you once for each
run. Then it uses each image with no change:

```
Upscaling is not possible, images are processed as they are.
```

To correct this, remove the plain build and install PyLithics again:

```bash
pip uninstall opencv-python-headless
pip install .
```
