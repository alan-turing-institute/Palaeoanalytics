# CLI Commands Reference

PyLithics installs two commands:

| Command | Purpose |
|---------|---------|
| `pylithics` | Measure single-artefact images: the analysis |
| `pylithics-pages` | Cut scanned plates into one image for each artefact |

This page lists each flag of both commands. `pylithics --help` and
`pylithics-pages --help` are the source. This page shows the same
flags with examples.

Most of this page is about `pylithics`. For `pylithics-pages`, go to
[Page Segmentation](#page-segmentation-pylithics-pages).

## The command

After installation, `pylithics` is a console command:

```bash
pylithics [options]
```

Type it with **no arguments** to see the welcome screen. The welcome
screen shows the logo and a panel with the most common command patterns.
Use the welcome screen to make sure that the installation is correct.
You can also use it as a quick reference in the terminal.

```bash
pylithics
```

With a flag (`--help` included), the command does not show the welcome
screen. It starts the normal command.

!!! danger "The project folder rule"
    `--data_dir` is a folder that contains these three items, with
    these exact names:

    ```
    <your folder>/
    ├── images/          the lithic images
    ├── scales/          the scale bar images
    └── meta_data.csv    which scale goes with which image
    ```

    The folder can have any name and can be anywhere. The three names
    inside it cannot change. PyLithics writes the analysis to a fourth
    item, `results/`, in the same folder.

## Necessary arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--data_dir` | The project folder that contains `images/`, `scales/` and `meta_data.csv`. The results go to `<data_dir>/results/`. | `pylithics/data` |
| `--meta_file` | The metadata CSV file (columns: `image_id`, `scale_id`, `scale`, `flag`). Default: `<data_dir>/meta_data.csv`. If some rows have no `scale`, the command asks once whether to measure those images in pixels. The command reports the flags. | `./lyon_2024.csv` |

```bash
pylithics --data_dir pylithics/data
```

## Configuration

| Option | Description | Default |
|--------|-------------|---------|
| `--config_file FILE` | A YAML configuration file to use in place of the default | the built-in defaults |
| `--threshold_method METHOD` | `auto`, `simple`, `otsu` or `adaptive`. `auto` finds the threshold from the image and keeps it between `threshold_value` and `threshold_ceiling` | `auto` |
| `--log_level LEVEL` | The logging level for the screen and the log file: `DEBUG`, `INFO`, `WARNING`, `ERROR`. The log file *always* has the full DEBUG trace. | `INFO` on the screen |
| `--verbose`, `-v` | Show the full trace for each step on the screen (the same as `--log_level DEBUG` for the screen only). | off |

```bash
pylithics --data_dir ./data --config_file ./my_config.yaml
```

The `working_resolution` section of the configuration sets when PyLithics upscales an image for detection. See [Working Resolution](../user-guide/working-resolution.md).

### Three levels

1. **Defaults** — in the code
2. **YAML file** — from `--config_file` (or `$PYLITHICS_CONFIG`)
3. **Command-line flags** — these replace the other two

### Example configuration file

```yaml
# config.yaml
thresholding:
  method: auto
  threshold_value: 127
  threshold_ceiling: 230
  max_value: 255
  debug_output: false

scale_calibration:
  enabled: true
  debug_output: false

arrow_detection:
  enabled: true
  reference_dpi: 300.0
  classifier:
    threshold: 0.5
    direction_threshold: 0.6
  debug_enabled: false

surface_classification:
  enabled: true
  tolerance: 0.1

scar_complexity:
  enabled: true
  distance_threshold: 10.0

cortex_detection:
  enabled: true
  min_stipple_pieces: 3
  letter:
    enabled: true          # --no_cortex_letter sets it to false
    letters: ["C"]         # the letters that mean cortex; case is ignored
    min_confidence: 0.8
  frames:
    enabled: true          # --keep_frames sets it to false
    max_line_ratio: 0.5    # a frame line is at most this share of the drawing line
  stippling_density_threshold: 0.2
  texture_variance_threshold: 100
  edge_density_threshold: 0.05

symmetry_analysis:
  enabled: true

voronoi_analysis:
  enabled: true
  padding_factor: 0.02
  images: [combined]       # any of: combined, voronoi, hull (--voronoi_images)

lateral_analysis:
  enabled: true

logging:
  level: INFO                    # file handler level (always DEBUG-capable)
  console_level: INFO            # console handler level; --verbose overrides to DEBUG
  log_to_file: true
  # log_file:                    # optional; default = <data_dir>/results/pylithics.log
```

## Analysis options

| Option | Description | Default |
|--------|-------------|---------|
| `--workers N` | The number of parallel worker processes for a batch. `auto` uses `cpu_count - 1`, with a maximum of 8 and a maximum of the batch size. `1` uses one process. Use it to find a problem with one image. | `auto` |
| `--threshold_debug` | Write the black-and-white image of each lithic to `results/threshold_debug/<image>.png` | off |
| `--closing BOOL` | Apply morphological closing | `True` |

### Parallel batches

By default, more than one worker process analyses a batch of more than
one image in parallel. Each worker analyses one image at a time.
Each worker writes its images to the results folder. It writes its
CSV rows to a `_partial/` folder. When all workers stop, the main
process merges the rows into `processed_metrics.csv`.

```bash
# Default: automatic worker count (cpu_count - 1, maximum 8, maximum the batch size)
pylithics --data_dir ./data

# One worker (to find a problem, to measure time, or on a computer with little RAM)
pylithics --data_dir ./data --workers 1

# A given worker count
pylithics --data_dir ./data --workers 4
```

The worker count is never more than the batch size. More workers than
images give no advantage. For a batch of 3 images or fewer, each worker
needs 2–3 s to start (for the imports). This start time can be more than
the decrease in the analysis time. Then `--workers 1` is frequently as fast.

The CSV file and the images are the same in sequential mode and in
parallel mode (with the usual differences in the last decimal places).

## DPI scaling

PyLithics reads the DPI of each image. By default it uses fixed kernel
sizes, which operate from 75 to 600 DPI. Set DPI-aware scaling on for
photographs with noise or scans of low quality.

| Option | Description | Default |
|--------|-------------|---------|
| `--enable_dpi_scaling` | Set DPI-aware kernel scaling on | off |
| `--dpi_scaling_mode MODE` | `conservative`, `standard` or `aggressive` | `standard` |
| `--dpi_reference DPI` | The reference DPI for the scaling | `300.0` |
| `--dpi_max_scale FACTOR` | The maximum scaling factor | `1.5` |

**Modes:**

- **conservative** — small changes; keeps fine line detail
- **standard** — moderate linear scaling, with limits
- **aggressive** — full proportional scaling; maximum noise removal

```bash
# Default
pylithics --data_dir ./data

# DPI scaling on
pylithics --data_dir ./data --enable_dpi_scaling

# Aggressive scaling
pylithics --data_dir ./data \
    --enable_dpi_scaling --dpi_scaling_mode aggressive

# Your own reference and limit
pylithics --data_dir ./data \
    --enable_dpi_scaling --dpi_reference 150 --dpi_max_scale 2.0
```

## Arrow detection

| Option | Description | Default |
|--------|-------------|---------|
| `--disable_arrow_detection` | Set arrow detection off | on |
| `--arrow_debug` | Write the arrow detection debug output to `results/arrow_debug/<image>/<scar>.png` and `.txt`: the direction found, the scores and the measurements | off |
| `--show-arrow-lines` | Draw a red line on each arrow in the labelled images | off |

```bash
pylithics --data_dir ./data --arrow_debug
pylithics --data_dir ./data --disable_arrow_detection
pylithics --data_dir ./data --show-arrow-lines
```

PyLithics decides a mark in two stages. Stage 1 is the notch method. The two notches where the head of an arrow meets its shaft give the head triangle. The triangle gives the direction. The stage 1 thresholds apply to 300 DPI. PyLithics scales them to the image with the `reference_dpi` and `*_scale_factor` keys of `arrow_detection`. What stage 1 accepts is an arrow. The model of stage 2 can turn the direction of that arrow. This occurs when the model scores the other direction at or above `classifier.direction_threshold` (default `0.6`). The score must also be higher than the direction of stage 1. Then the arrow takes the model's direction. Stage 2 runs only on the marks that stage 1 rejects. Stage 2 refuses a mark that holds other ink, or that is shorter than 6 pixels. For the other marks, a trained model (a random forest) measures the width of the mark's ink along its axis, in both directions. The model also uses the notch measurements of stage 1. It gives a score to each direction. The higher score gives the direction. The mark is an arrow when that score is at or above `classifier.threshold` (default `0.5`). With `classifier.enabled: false`, stage 1 alone decides. The model is `arrow_classifier.json` under `pylithics/models/`. PyLithics tries only the marks inside a recognised scar. It never tries a surface, a scar, or a mark outside every recognised scar. PyLithics can upscale the image for detection (see [Working Resolution](../user-guide/working-resolution.md)). Then it cuts the marks from the upscaled copy, so a small arrow keeps its notches. The arrow positions in the CSV are in the pixels of the original image. The CSV column `arrow_score` holds the score when stage 2 decided.

## Scale calibration

| Option | Description | Default |
|--------|-------------|---------|
| `--disable_scale_calibration` | Set scale bar detection off (pixel measurements) | on |
| `--scale_debug` | Write the scale image with the bar that PyLithics found to `results/scale_debug/<scale image>.png` | off |
| `--force_pixels` | Use pixel measurements only (no calibration). The command does not ask about images with no scale | off |

```bash
pylithics --data_dir ./data --scale_debug
pylithics --data_dir ./data --force_pixels
pylithics --data_dir ./data --disable_scale_calibration
```

## Cortex detection

| Option | Description | Default |
|--------|-------------|---------|
| `--disable_cortex_detection` | Set cortex detection off | on |
| `--cortex_sensitivity {low,medium,high}` | The detection sensitivity | `medium` |
| `--min_stipple_pieces N` | A scar is cortex only when it holds at least N small pieces of ink. One or two pieces are an arrow or a numeral | `3` |
| `--no_cortex_letter` | Do not read a letter in a scar as cortex. By default a scar that holds the letter C is cortex, because some illustrators mark cortex with a C. The letters are in `cortex_detection.letter.letters`. This option needs the `ocr` extra | on |
| `--keep_frames` | Keep a thin rectangle drawn around a letter. By default PyLithics removes it before it finds the scars, because the rectangle is not a scar. A rectangle with thick lines, or with no letter in it, stays | off |

```bash
pylithics --data_dir ./data --cortex_sensitivity high
pylithics --data_dir ./data --disable_cortex_detection
```

## Scar complexity

| Option | Description | Default |
|--------|-------------|---------|
| `--disable_scar_complexity` | Set scar adjacency analysis off | on |
| `--scar_complexity_distance_threshold PIXELS` | The adjacency distance in pixels | `10.0` |

```bash
pylithics --data_dir ./data \
    --scar_complexity_distance_threshold 15.0
```

## Surface analysis

| Option | Description | Default |
|--------|-------------|---------|
| `--disable_symmetry_analysis` | Set symmetry analysis off. The symmetry columns are `NA` | on |
| `--disable_voronoi_analysis` | Set Voronoi and convex hull analysis off. The Voronoi and convex hull columns are `NA`, and PyLithics writes no `_voronoi.png` | on |
| `--voronoi_images KINDS` | The Voronoi images to write: `combined` (the cells with the convex hull over them), `voronoi` (the cells only), `hull` (the convex hull only). To write more than one, use commas and no spaces. The measurements do not change | `combined` |

```bash
pylithics --data_dir ./data --disable_symmetry_analysis --disable_voronoi_analysis

# The cells and the convex hull as two images, with no overlay
pylithics --data_dir ./data --voronoi_images voronoi,hull
```

## Output

| Option | Description | Default |
|--------|-------------|---------|
| `--export_json` | Also write one JSON file for each lithic to `results/json/{image_stem}.json` | off |
| `--save_visualizations` | Write the labelled images and the Voronoi diagrams | on |
| `--hide_labels KINDS` | Do not draw these labels on the labelled image: `surfaces`, `scars`, `cortex`, `arrows`, or `all`. Give more than one kind in one list, with commas and no spaces. Give the option one time. A second option replaces the first. PyLithics always draws the outlines and the centroid dots | none hidden |
| `--explore [PATH]` | With no `PATH`: do the analysis, then open the dashboard at `http://localhost:8501` for `<data_dir>/results/`. With `PATH`: open the dashboard for that folder. PyLithics does no analysis. | off |

```bash
# Default: CSV only
pylithics --data_dir ./data

# CSV plus one JSON file for each lithic
pylithics --data_dir ./data --export_json

# Labelled images with only the arrow angles: one list, commas, no spaces
pylithics --data_dir ./data --hide_labels surfaces,scars,cortex

# No labels at all
pylithics --data_dir ./data --hide_labels all
```

The same setting is in `config.yaml`:

```yaml
data_export:
  csv: true
  json_per_lithic: true   # the same as --export_json
```

## Help

| Option | Description |
|--------|-------------|
| `-h`, `--help` | Show all options |
| `--help-config` | Show the documentation of the configuration file |
| `--help-examples` | Show examples of use |
| `--help-troubleshooting` | Show common problems and their procedures |
| `--docs` | Start the documentation server at <http://127.0.0.1:8000> |

```bash
pylithics --help
pylithics --help-config
pylithics --docs
```

## Common patterns

### A quick test with the sample data

```bash
pylithics --data_dir pylithics/data
```

### Find a problem with one image

```bash
pylithics --data_dir ./data \
    --log_level DEBUG \
    --arrow_debug \
    --scale_debug \
    --threshold_debug
```

### A faster analysis

```bash
# Set arrow detection off (the slowest optional step)
pylithics --data_dir ./data --disable_arrow_detection
```

### Replace a configuration value from the command line

A command-line flag always replaces the YAML value:

```bash
pylithics --data_dir ./data \
    --config_file ./site_config.yaml \
    --threshold_method adaptive
```

## Page Segmentation (`pylithics-pages`)

PyLithics analyses one artefact for each image. A published plate shows many artefacts. `pylithics-pages` cuts the plate into one image for each lithic, with all of its surfaces. It writes each scale bar to its own image. It reads the identifier printed next to each lithic and names the crop with it. See [Working from Published Plates](../user-guide/page-segmentation.md) for the procedure.

```bash
pylithics-pages --data_dir pylithics/data
```

### Necessary arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--data_dir` | The project directory that contains `pages/`. You can also give a folder of pages directly. | `pylithics/data` |

### Output Location

The two commands share one project folder:

| Command | Reads | Writes |
|---------|-------|--------|
| `pylithics-pages` | `<data_dir>/pages/` | `<data_dir>/images/`, `scales/`, `pages_manifest.csv`, `meta_data.csv`, `pages_debug/` with `--debug`, and `pages_rendered/` for PDF input |
| `pylithics` | `<data_dir>/images/`, `scales/`, `meta_data.csv` | `<data_dir>/results/` |

`meta_data.csv` has one row for each artefact crop. `scale_id` is the scale bar from the same page. `scale` is empty: you write the scale value in it. `flag` names a problem to correct first (`no_scale`, `several_scales`, `scale_uncertain`, `low_resolution`, or an identifier flag). See [meta_data.csv](../user-guide/page-segmentation.md#meta_datacsv).

### Crop Names

| Result | Filename |
|--------|----------|
| The command read one identifier in the crop | `<page>_figure_7.png` |
| The command did not read one identifier | `<page>_box_07.png` (the box number on the debug overlay, in reading order) |
| Scale bar | `<page>_scale_bar.png`, or `<page>_scale_bar_01.png` if the page has more than one |

The manifest columns `label`, `label_source`, `label_flag` and `label_candidates` record four items. In sequence, these are the identifier, the method, the reason for a missing name, and all identifiers read in the crop. The column `correction_applied` records the corrections that made each crop, from the identifier rules or the corrections CSV. See [pages_manifest.csv](../user-guide/page-segmentation.md#pages_manifestcsv) for all columns.

### Grouping Options

All distances are fractions of the page width or the page height. The same values apply to pages of different sizes.

| Flag | Description | Default |
|------|-------------|---------|
| `--gap FRAC` | Two views closer than this fraction of the page width are one artefact | `0.025` |
| `--narrow FRAC` | A narrow profile view joins the artefact next to it if the distance is less than this | `0.07` |
| `--vertical_gap FRAC` | A short view joins the taller view above it if the distance is less than this | `0.06` |
| `--bridge FRAC` | Two views that a dash mark connects join if the dash is shorter than this | `0.06` |
| `--min_area FRAC` | Ink blobs smaller than this fraction of the page area are labels, not artefacts | `0.0004` |
| `--overrides FILE` | A CSV file of corrections, one row for each page (`page_id`, `expect`, `join`, `split`). Use the box numbers that `--debug` shows | `corrections.csv` in the project folder. The first run writes it empty |

### Output Options

| Flag | Description | Default |
|------|-------------|---------|
| `--output_dir PATH` | The folder for `images/`, `scales/`, `pages_manifest.csv` and `meta_data.csv` | the `--data_dir` project folder |
| `--padding PX` | The white margin around each crop, in pixels | `10` |
| `--debug` | Write one overlay for each page to `pages_debug/<page>.png`. The overlay shows each box, each scale bar with its score, each rejected candidate in grey, and each identifier read | off |
| `--disable_scale_bars` | Do not write scale bar crops | off |
| `--read_labels` | Read the identifier next to each lithic and name the crop with it. RapidOCR is necessary; see [Installation](../installation.md#step-4-install-pylithics) | on |
| `--no_read_labels` | Do not read identifiers. Name the crops by box number only | off |
| `--keep_text_boxes` | Keep a box that the reader finds to be writing, such as a framed specimen code. By default, the manifest records it as `text` and the command does not cut it | off |
| `--keep_text` | Keep the page's text in each artefact crop. By default, the command hides captions, legend keys, scale bar ink and other identifiers. It paints over them with the crop's paper value. The crop's own identifier stays | off |
| `--hide_identifiers` | Hide the crop's own identifier too. By default, the identifier stays and the command writes its box to `meta_data.csv`. `pylithics` blanks that box before the analysis | off |

### PDF Options

The command renders each page of a PDF file in `pages/` to one PNG in `pages_rendered/`. Then it cuts the PNG like a scanned page. The optional package pypdfium2 is necessary: `pip install ".[pdf]"`. See [PDF input](../user-guide/page-segmentation.md#pdf-input).

| Flag | Description | Default |
|------|-------------|---------|
| `--pdf_dpi DPI` | The resolution for a PDF page that holds no raster image, in dots per inch. A page that holds a raster image renders at the resolution of that image | `300` |
| `--pdf_pages LIST` | The pages to render from each PDF file, for example `4,7,11-14` | all pages |

### Configuration Options

| Flag | Description | Default |
|------|-------------|---------|
| `--config_file FILE` | A YAML configuration file to use in place of the default | packaged `config.yaml` |
| `--log_level LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` |
| `--verbose`, `-v` | Show the full trace for each page | off |

The `pdf` section of `config.yaml` sets how the command renders a PDF page. `rasterise_dpi` is the resolution for a page with no raster image (default `300`). `min_coverage` is the share of the page that an image must cover to set the resolution (default `0.05`). `max_dpi` is the maximum derived resolution (default `1200`).

The `identifiers` section of `config.yaml` sets `reach` (default `1.5`). This is the distance in glyph heights at which an identifier outside a box belongs to the nearest box. Increase it for plates that print the identifier far from the lithic.

The `working_resolution` section sets when the command upscales a page for detection. See [Working Resolution](../user-guide/working-resolution.md).

The `scale_bars.classifier` section controls how the command decides a scale bar. `enabled` turns the classifier on or off (default `true`). When the classifier is off, the rules decide. `threshold` is the score at or above which a candidate is a scale bar (default `0.6`). `uncertain_band` is the distance from the threshold within which the page's artefacts get the flag `scale_uncertain` (default `0.15`). `min_aspect_ratio`, `max_short_side`, `min_long_side` and `max_long_side` set the shape of a candidate. The three side limits are fractions of the page's longer side. See [How a scale bar is decided](../user-guide/page-segmentation.md#how-a-scale-bar-is-decided).

With the `identifiers.regroup` section, the identifiers correct the grouping. The command cuts a box that holds several identifiers into one box for each identifier. A box that holds no identifier joins the labelled box next to it. `enabled` turns the rules on or off (default `true`). `join_gap` is the distance in which a box looks for its neighbour, as a fraction of page width (default `0.12`). See [How identifiers correct the grouping](../user-guide/page-segmentation.md#how-identifiers-correct-the-grouping).

### Re-runs

Every run cuts every plate. The command cuts again a plate that an earlier run cut, and replaces its crops and rows. The scale values that you typed stay. The command adds a new plate. It removes a crop that the new run does not make. It replaces a crop file that an earlier run left without a manifest row, and reports the number. The command never changes your own images and rows. Two pages with the same name and different extensions give the same crop names. The command does not write the second. See [Re-runs](../user-guide/page-segmentation.md#re-runs).

If the command cannot cut a plate, it reports it, cuts the other plates, writes the manifest, and exits with code `1`. The reason is in `pylithics-pages.log`. The command writes the manifest again after each plate. If you stop a run, the manifest names each crop that the run wrote. The next run replaces them.

### Common Patterns

```bash
# Cut the pages of a project and write the overlays
pylithics-pages --data_dir pylithics/data --debug

# Start again with corrections for each page
pylithics-pages --data_dir pylithics/data \
    --debug

# Cut a folder of scans that is not a project, and give the output folder
pylithics-pages --data_dir /mnt/archive/plates --output_dir ~/work/lyon

# Merge two artefacts that were cut into two boxes
pylithics-pages --data_dir pylithics/data --gap 0.05

# Name the crops by box number only
pylithics-pages --data_dir pylithics/data --no_read_labels

# Cut the figure pages of a downloaded paper
pylithics-pages --data_dir my_project --pdf_pages 4,7,11-14 --debug
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `PYLITHICS_CONFIG` | The default configuration file path when you do not give `--config_file` |
| `PYLITHICS_NO_UPDATE_CHECK` | Set it to any value to stop the daily check for a new release. The same as `update_check.enabled: false` in `config.yaml`. See [Update PyLithics](../installation.md#update-pylithics) |

```bash
export PYLITHICS_CONFIG=./default_config.yaml
pylithics --data_dir ./data
```

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success: the command processed all images, or at least one image |
| `1` | The input was not valid, the analysis stopped with an error, or an unexpected error occurred |
