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

Type it with **no arguments** to see the welcome screen: the logo and
a panel with the most common command patterns. Use it to make sure that
the installation is correct, and as a quick reference in the terminal.

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
| `--meta_file` | The metadata CSV file (columns: `image_id`, `scale_id`, `scale`, `flag`). Default: `<data_dir>/meta_data.csv`. If some rows have no `scale`, the command asks once whether to measure those images in pixels. Flags are reported. | `./lyon_2024.csv` |

```bash
pylithics --data_dir pylithics/data
```

## Configuration

| Option | Description | Default |
|--------|-------------|---------|
| `--config_file FILE` | A YAML configuration file to use in place of the default | the built-in defaults |
| `--threshold_method METHOD` | `simple`, `otsu`, `adaptive` or `default` | `default` |
| `--log_level LEVEL` | The logging level for the screen and the log file: `DEBUG`, `INFO`, `WARNING`, `ERROR`. The log file *always* has the full DEBUG trace. | `INFO` on the screen |
| `--verbose`, `-v` | Show the full trace for each step on the screen (the same as `--log_level DEBUG` for the screen only). | off |

```bash
pylithics --data_dir ./data --config_file ./my_config.yaml
```

### Three levels

1. **Defaults** — in the code
2. **YAML file** — from `--config_file` (or `$PYLITHICS_CONFIG`)
3. **Command-line flags** — these replace the other two

### Example configuration file

```yaml
# config.yaml
thresholding:
  method: otsu
  threshold_value: 127
  max_value: 255
  debug_output: false

scale_calibration:
  enabled: true
  debug_output: false

arrow_detection:
  enabled: true
  reference_dpi: 300.0
  min_area_scale_factor: 0.7
  min_defect_depth_scale_factor: 0.8
  min_triangle_height_scale_factor: 0.8
  debug_enabled: false

surface_classification:
  enabled: true
  tolerance: 0.1

scar_complexity:
  enabled: true
  distance_threshold: 10.0

cortex_detection:
  enabled: true
  stippling_density_threshold: 0.2
  texture_variance_threshold: 100
  edge_density_threshold: 0.05

symmetry_analysis:
  enabled: true

voronoi_analysis:
  enabled: true
  padding_factor: 0.02

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
| `--workers N` | The number of parallel worker processes for a batch. `auto` uses `cpu_count - 1`, with a maximum of 8 and a maximum of the batch size. `1` uses one process; use it to find a problem with one image. | `auto` |
| `--threshold_debug` | Write the black-and-white image of each lithic to `results/threshold_debug/<image>.png` | off |
| `--closing BOOL` | Apply morphological closing | `True` |

### Parallel batches

By default, a batch of more than one image is analysed in parallel by
more than one worker process. Each worker analyses one image at a time.
Each worker writes its images to the output directory. It writes its
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
images give no advantage. For a batch of 3 images or fewer, the start
time of each worker (2–3 s for the imports) can be more than the
decrease in the analysis time. Then `--workers 1` is often as fast.

The CSV output and the images are identical in sequential mode and in
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
| `--arrow_debug` | Write the arrow detection debug output to `results/arrow_debug/<image>/<scar>.png` and `.txt` | off |
| `--show-arrow-lines` | Draw a red line on each arrow in the labelled images | off |

```bash
pylithics --data_dir ./data --arrow_debug
pylithics --data_dir ./data --disable_arrow_detection
pylithics --data_dir ./data --show-arrow-lines
```

## Scale calibration

| Option | Description | Default |
|--------|-------------|---------|
| `--disable_scale_calibration` | Set scale bar detection off (pixel measurements) | on |
| `--scale_debug` | Write the scale image with the bar that was found to `results/scale_debug/<scale image>.png` | off |
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

## Output

| Option | Description | Default |
|--------|-------------|---------|
| `--export_json` | Also write one JSON file for each lithic to `results/json/{image_stem}.json` | off |
| `--save_visualizations` | Write the labelled images and the Voronoi diagrams | on |
| `--explore [PATH]` | With no `PATH`: do the analysis, then open the dashboard at `http://localhost:8501` for `<data_dir>/results/`. With `PATH`: open the dashboard for that folder. No analysis. | off |

```bash
# Default: CSV only
pylithics --data_dir ./data

# CSV plus one JSON file for each lithic
pylithics --data_dir ./data --export_json
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
| `pylithics-pages` | `<data_dir>/pages/` | `<data_dir>/images/`, `scales/`, `pages_manifest.csv`, `meta_data.csv`, and `pages_debug/` with `--debug` |
| `pylithics` | `<data_dir>/images/`, `scales/`, `meta_data.csv` | `<data_dir>/results/` |

`meta_data.csv` has one row for each artefact crop. `scale_id` is the scale bar from the same page. `scale` is empty for you to fill in. `flag` names a problem to correct first (`no_scale`, `several_scales`, or an identifier flag). See [meta_data.csv](../user-guide/page-segmentation.md#meta_datacsv).

### Crop Names

| Result | Filename |
|--------|----------|
| The command read one identifier in the crop | `<page>_figure_7.png` |
| The command did not read one identifier | `<page>_box_07.png` (the box number on the debug overlay, in reading order) |
| Scale bar | `<page>_scale_bar.png`, or `<page>_scale_bar_01.png` if the page has more than one |

The manifest columns `label`, `label_source`, `label_flag` and `label_candidates` record the identifier, the method, the reason for a missing name, and all identifiers read in the crop. The column `correction_applied` records the corrections used on the page. See [pages_manifest.csv](../user-guide/page-segmentation.md#pages_manifestcsv) for all columns.

### Grouping Options

All distances are fractions of the page width or the page height. The same values apply to pages of different sizes.

| Flag | Description | Default |
|------|-------------|---------|
| `--gap FRAC` | Two views closer than this fraction of the page width are one artefact | `0.025` |
| `--narrow FRAC` | A narrow profile view joins the artefact next to it if the distance is less than this | `0.07` |
| `--vertical_gap FRAC` | A short view joins the taller view above it if the distance is less than this | `0.06` |
| `--bridge FRAC` | Views connected by a dash mark join if the dash is shorter than this | `0.06` |
| `--min_area FRAC` | Ink blobs smaller than this fraction of the page area are labels, not artefacts | `0.0004` |
| `--overrides FILE` | A CSV file of corrections, one row for each page (`page_id`, `expect`, `join`, `split`) | none |

### Output Options

| Flag | Description | Default |
|------|-------------|---------|
| `--output_dir PATH` | The folder for `images/`, `scales/`, `pages_manifest.csv` and `meta_data.csv` | the `--data_dir` project folder |
| `--padding PX` | The white margin around each crop, in pixels | `20` |
| `--debug` | Write one overlay for each page to `pages_debug/<page>.png`. The overlay shows each box, each scale bar and each identifier read | off |
| `--disable_scale_bars` | Do not write scale bar crops | off |
| `--read_labels` | Read the identifier next to each lithic and name the crop with it. `pip install "PyLithics[ocr]"` is necessary | on |
| `--no_read_labels` | Do not read identifiers. Name the crops by box number only | off |

### Configuration Options

| Flag | Description | Default |
|------|-------------|---------|
| `--config_file FILE` | A YAML configuration file to use in place of the default | packaged `config.yaml` |
| `--log_level LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` |
| `--verbose`, `-v` | Show the full trace for each page | off |

The `identifiers` section of `config.yaml` sets `reach`, the distance in glyph heights at which an identifier outside a box belongs to the nearest box (default `1.5`). Increase it for plates that print the identifier far from the lithic.

### Re-runs

Every run cuts every plate. A plate cut before is cut again and its crops and rows are replaced; the scale values that you typed stay. A new plate is added. A crop that the new run does not make is removed. Your own images and rows are never changed, and a crop with the same filename as one of your images is not written. See [Re-runs](../user-guide/page-segmentation.md#re-runs).

### Common Patterns

```bash
# Cut the pages of a project and write the overlays
pylithics-pages --data_dir pylithics/data --debug

# Start again with corrections for each page
pylithics-pages --data_dir pylithics/data \
    --overrides ./corrections.csv --debug

# Cut a folder of scans that is not a project, and give the output folder
pylithics-pages --data_dir /mnt/archive/plates --output_dir ~/work/lyon

# Merge two artefacts that were cut into two boxes
pylithics-pages --data_dir pylithics/data --gap 0.05

# Name the crops by box number only
pylithics-pages --data_dir pylithics/data --no_read_labels
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `PYLITHICS_CONFIG` | Default config file path used when `--config_file` is omitted |
| `PYLITHICS_NO_UPDATE_CHECK` | Set to any value to switch off the daily check for a new release. The same as `update_check.enabled: false` in `config.yaml`. See [Update PyLithics](../installation.md#update-pylithics) |

```bash
export PYLITHICS_CONFIG=./default_config.yaml
pylithics --data_dir ./data
```

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success (all images processed, or partial success with at least one done) |
| `1` | The input was not valid, the analysis stopped with an error, or an unexpected error occurred |
