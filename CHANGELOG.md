# Changelog

All notable changes to PyLithics are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

New `pylithics-pages` command for preparing source material: cuts scanned plates
of lithic illustrations into one image per artefact, ready for the analysis run.
The two commands now share one flat project folder, and the analysis output
directory is renamed `results/`.

**Note for testing**: `pylithics-pages` is a new console script, so `git pull`
alone will not create it. Reinstall with `pip install . --upgrade`.

### Fixed

- **A page-wide rule is not a scale bar (`pylithics-pages`).** On a journal
  article the rule under the running header, 0.88 of the page wide, scored
  just above the threshold on every page. `scale_bars.classifier.max_long_side`
  falls from `0.9` to `0.5`: no labelled bar is wider than 0.44 of its page.
  The classifier threshold rises from `0.4` to `0.6`, and the model is
  retrained with 771 labelled candidates from the article's pages: held out
  on 153 pages, 2 false bars and 16 missed of 168, against 13 and 11 at 0.4.
  A caption joins a bar's crop only within three bar heights of it, and a bar
  fused with a drawing is exported alone.

- **A scar that holds an arrow is no longer cortex (`pylithics`).** The stippling
  test was a density of small ink pieces with no count, so in a scar under
  5,000 pixels one piece passed it, and an arrow is one piece. Cortex now needs
  at least `cortex_detection.min_stipple_pieces` small pieces (default 3; also
  `--min_stipple_pieces N`). On the research corpus 775 of 832 cortex marks held
  one or two pieces, and 252 arrows were lost behind them.
- **A missing project folder is reported, not created.** Both commands
  stop with "the folder ... does not exist" when `--data_dir` names a folder
  that is not there, before any log file or `results/` folder is made.
  When the project folder holds no page and no PDF, `pylithics-pages` says
  where the pages go.
- **Small crops no longer lose their lines in preprocessing (`pylithics`)** — a
  drawing cut from a PDF figure at 150 to 300 pixels wide came out of
  thresholding as a few dots: the fixed threshold of 127 dropped grey outlines,
  and the blur and closing kernels grew with the working-resolution factor to
  21 and 12 pixels, wider than the strokes. Now the threshold is found from the
  image (`thresholding.method: auto`, Otsu kept between `threshold_value` and
  `threshold_ceiling`; it rises above 127 only when the grey it adds is thin,
  so a shaded surface stays open), the blur and the closing are sized from the
  measured stroke width (half a stroke and one stroke, never larger than
  before), a DPI tag outside 50 to 1200 is treated as absent, and the contour
  size filter is at most `contour_filtering.min_area_share` of the image area
  (default one ten-thousandth), so a small arrow is kept and a speck is not.
  On a 270-pixel Gadeb flake with seven arrows, zero were found before and four
  after; large scans are unchanged.
- **`symmetry_analysis.enabled` and `voronoi_analysis.enabled` now work (`pylithics`)** —
  both flags were in `config.yaml` and the documentation but the code did not read
  them. Set to `false`, the stage is skipped: its CSV columns are `NA` and no
  `_voronoi.png` is written. New switches `--disable_symmetry_analysis` and
  `--disable_voronoi_analysis` do the same for one run.

### Changed
- **Dependencies** — `requirements.txt` lists only what the package needs to run.
  The test tools (`pytest`, `pytest-cov`, `psutil`, `pyflakes`, `pydocstyle`) are
  the `dev` extra and the documentation tools (`mkdocs`, `mkdocs-material`,
  `pymdown-extensions`) the `docs` extra: `pip install -e ".[dev,docs]"`.
  `setup.py` uses the SPDX licence `GPL-3.0-or-later` and ships the config,
  model and sample data folders without build warnings.
- **Travis CI removed** — the configuration, the README badge and the
  CONTRIBUTING text. The last build was in April 2025.
- **`.gitignore`** cut to the rules this project uses.

### Added

- **A page with no artefacts, and a page taken away (`pylithics-pages`).**
  `expect=0` in `corrections.csv` marks a page as holding no artefacts: nothing
  is cut, its earlier crops are removed, and the manifest keeps a `skipped`
  row. A page file or a PDF taken out of `pages/` has its earlier crops and
  rows removed, and the run names the pages.

- **The page's text is hidden in each crop (`pylithics-pages`).** A crop's margin
  took in captions, legend keys, scale bar ink and other identifiers, and
  `pylithics` measured them as surfaces. They are now painted over with the
  crop's paper value, by each piece's own ink, never a piece inside the
  artefact's box or a flaking arrow; the page is not changed. The manifest
  column `hidden` says what was hidden. The crop's own identifier stays and
  its box goes to `meta_data.csv` as `ignore_box`, which `pylithics` sets to
  paper before the analysis. `--keep_text` keeps every pixel;
  `--hide_identifiers` hides the identifier too.

- **The corrections file is part of the project (`pylithics-pages`).** The first
  run writes `corrections.csv` with its header into the project folder, and
  every run reads it from there, so a correction is a row to write and a
  rerun, with no `--overrides` to pass. `--overrides` still names a file
  elsewhere.

- **A box of text is not an artefact (`pylithics-pages`).** After the grouping,
  each artefact box is read with RapidOCR. A box whose lines read at
  `text_boxes.min_confidence` (0.8) or more cover at least
  `text_boxes.min_ink_share` (0.8) of its ink, a straight frame
  discounted, and spell at least `min_characters` (3), is writing: a framed
  specimen code, a site name. It is recorded in the manifest as `image_type`
  `text` with the text in `text_read`, and no crop is written.
  `--keep_text_boxes` keeps such boxes for one run.
- **PDF input (`pylithics-pages`)** — a PDF file in `pages/` is read. Each page
  is rendered to a PNG in `pages_rendered/` and cut like a scanned page. A page
  that holds a raster image renders at the resolution of the sharpest large
  image on it and copies that image pixel for pixel, so a scan inside a PDF
  keeps its pixels and a label or scale bar drawn over it stays. A page with no
  raster image renders at 300 dpi, or at `--pdf_dpi`. `--pdf_pages 4,7,11-14`
  limits the pages. The log says what each page holds and warns below 200 dpi
  and for JPEG compression. The manifest gains `source_pdf`, `source_page`,
  `render_dpi`, `dpi_source` and `source_filters`. The `pdf` section of
  `config.yaml` sets the limits. Needs the new optional extra:
  `pip install ".[pdf]"` (pypdfium2).
- **`--hide_labels` removes labels from the labelled image (`pylithics`)** — a
  list with commas of `surfaces`, `scars`, `cortex`, `arrows`, or `all`. The
  outlines and centroid dots always stay. The same setting is
  `visualization.labels` in `config.yaml`. Default: every label drawn.
- **Arrow detection in two stages (`pylithics`)** — the notch method stays
  and decides first: the two notches where the head meets the shaft give
  the head triangle and the direction, with its thresholds scaled from
  300 DPI. What it accepts is an arrow. A trained classifier now runs on
  the marks it rejects, the small arrows and the styles without a closed
  head. A mark that holds other ink, or is shorter than 6 pixels, is
  refused first. The rest is measured by the width of its ink along its
  axis (a thin, even shaft, a head that widens and then narrows to a
  point), on a four-times supersampled copy, in both directions, with the
  notch measurements added; a random forest trained on the project's
  labelled arrows scores each direction. The higher score gives the
  direction and must reach `arrow_detection.classifier.threshold` (default
  `0.5`). `classifier.enabled: false` leaves the notch method alone. The
  only marks tried are those inside a recognised scar; a surface, a scar
  or a mark outside every recognised scar is never tried. The marks are
  cut from the working copy of the drawing (the upscaled binary, when
  there is one), scar by scar, so a small arrow keeps the notches that
  the return to the source grid removes; the notch thresholds scale by
  the factor and the arrow positions come back in source pixels. The
  notch method's acceptance is final, but its direction is turned when the
  classifier scores the other direction at or above
  `classifier.direction_threshold` (default `0.6`) and higher: on the
  labelled corpus the notch method pointed 41 of 345 arrows the wrong way,
  and the classifier was right on 39 of them while disagreeing on 4 of
  the 304 it had right. The
  debug files are named by scar and mark (`scar_3_mark_1.png`) and show
  the scar at the working resolution; the `.txt` names the stage that
  decided, `triangle`, `triangle turned` or `model`. New CSV column
  `arrow_score`, filled when the classifier decided. The columns
  `triangle_base_length`, `triangle_height`, `shaft_solidity` and
  `tip_solidity` are gone from the CSV. Arrow detection runs before scar
  complexity and Voronoi. Model bundled as
  `pylithics/models/arrow_classifier.json`, with the hand labels it was
  trained from in `arrow_labels.csv` and the arrow templates in
  `training/arrows/` and `training/non_arrows/`; tools `tools/arrow_variants.py`,
  `tools/arrow_candidates.py` and `tools/train_arrow_classifier.py`.
  `arrow_integration` loses `min_solidity`, `max_solidity` and
  `reference_thresholds`. `pylithics.forest` reads a forest from JSON for
  both classifiers.
- **Scale bar classifier (`pylithics-pages`)** — a trained model (a random forest) decides
  which bar-shaped marks are scale bars, in place of the shape rules alone.
  Candidates include bars standing against a lithic, upright bars and
  rulers, which the rules took for annotations or text. The manifest records
  `scale_score` and `scale_method` for each scale bar; an artefact whose
  scale bar scored close to the threshold gets the flag `scale_uncertain` in
  `meta_data.csv`; the debug overlay draws rejected candidates in grey with
  their scores. Section `scale_bars.classifier` in `config.yaml`;
  `enabled: false` gives the rules of before. Model bundled as
  `pylithics/models/scale_bar_classifier.json`, its training labels as
  `pylithics/models/scale_bar_labels.csv`. Tools:
  `tools/scale_candidates.py` writes the labelling table, prefilled from
  the known labels and the model; `tools/train_scale_classifier.py`
  trains the model. A candidate's ink, not the boxes around it, decides
  whether it stands clear of the drawings, so a bar below a profile view
  is found; upright dashes form one bar; a thick bar fused with its
  caption is found from the raw ink; a plain stroke at a drawing's edge
  or between two views is a connector, never scored. A scale crop
  takes its caption (`0`, `5 cm`, and text that follows it) and not an
  arrow or an identifier at the bar's corner. Size limits are fractions
  of the page's longer side, so a strip cut from a page keeps its digits
  as labels. A large free-standing arrow on a drawing is no longer taken
  for a bar: the feature `head_end` compares the widest span at one end
  of a mark with the other end (a bar reads 1, with or without end ticks;
  an arrow about 2), and the labels gained 35 pages from a PDF with such
  arrows. Leave-one-page-out at the threshold: 11 false bars and 7 missed
  of 1,030 candidates, against 18 and 8 before.
- **Working resolution** — both commands measure the line width of each
  image and, when the lines are too thin for detection, find the ink on a
  working copy upscaled 2, 3 or 4 times with ESPCN or FSRCNN (bundled). The
  result is returned to the input pixel grid: crops, DPI tags and every
  measurement stay at the input size, and the factor is recorded
  (`upscale_factor` in `pages_manifest.csv`, `processed_metrics.csv` and the
  JSON; flag `low_resolution` in `meta_data.csv`). Never downscales. Section
  `working_resolution` in `config.yaml`. **Dependency change**: OpenCV is now
  `opencv-contrib-python-headless`; remove the plain build before reinstalling.
- **Identifier-guided grouping (`pylithics-pages`)** — the identifiers read
  from a plate now correct the grouping. A crop holding several identifiers is
  cut into one crop each where empty space between the drawings permits; a crop
  holding none joins the labelled crop beside it. A change is kept only when
  each crop then holds exactly one identifier. The manifest records
  `identifier_split` / `identifier_join` in `correction_applied`; the debug
  overlay marks the crops. Off switch: `identifiers.regroup.enabled`. On a
  dense plate with no empty run across the crop, each drawing goes to its
  identifier on the side the plate sets them.
- **A bad plate no longer stops the batch** — a plate that fails for any reason
  is reported with its cause, the other plates are cut, the manifest is written,
  and the exit code is 1. The manifest is also rewritten after every plate, so
  a stopped run leaves one that names every crop written, and the next run
  replaces those crops instead of refusing them as the user's own files.
- **Connector rules are not scale bars** — two short rules on one line, each
  between two views of one lithic, no longer export as a segmented scale bar.
- **Larger and bolder labels are read** — the glyph ceiling is never below
  64 px, so a figure cut small from a PDF keeps its letters as candidates, and
  a bold letter set close to its lithic passes the clear-space test.
- **Page segmentation (`pylithics-pages`)** — new console script that cuts a
  folder of scanned plates into one image per artefact. Adjacent surface views
  are grouped, so a lithic drawn with four surfaces yields one crop, not four.
  Crops are cut from the source pixels untouched, at the source DPI and colour
  mode: no denoising, contrast adjustment, or thresholding is applied to saved
  images, since the main pipeline already performs all of it and doing it twice
  would make measurements non-comparable.
- **Scale bar export** — page-level scale bars are exported to
  `scales/{page}_scale_bar.png` with their caption included, so the crop is
  self-describing. Segmented bars (the conventional alternating filled/open
  form) are rejoined from the blocks that survive thresholding.
- **Crop manifest** — `pages_manifest.csv` records every crop's source page, reading
  -order index, bounding box, DPI, colour mode, grouped-component count, and
  which corrections were applied.
- **Per-page corrections CSV** — `--overrides` accepts `expect`, `join`, and
  `split` per page, keyed by filename, so a manually corrected run is
  reproducible rather than living in shell history.
- **Debug overlays** — `--debug` writes numbered artefact boxes over the page;
  those numbers are what the corrections CSV refers to.
- **Sample plate fixture** — `pylithics/data/pages/sample_plate.png`, composited
  from the five shipped drawings, numbered 1 to 5 just under their ink, with a
  shared scale bar, for trying the workflow and for the test suite. Regenerate
  with `tests/fixtures/generate_sample_plate.py [OUTPUT.png]`.
- **`page_segmentation` configuration section** in `config.yaml`, with
  `get_page_segmentation_config()` following the established getter pattern.
- **`archaeological` pytest marker** for tests validating domain correctness.

- **One rule for debug output** — the folder is named after the flag, the file
  after the source image. `pylithics` writes `results/threshold_debug/`,
  `results/scale_debug/` and `results/arrow_debug/<image>/<scar>.png|.txt`;
  `pylithics-pages --debug` writes `pages_debug/<page>.png`. `--scale_debug` no longer writes to a stale
  `processed/` folder, `--arrow_debug` now writes (it never did), and
  `--show_thresholded_images` is renamed `--threshold_debug` and now writes the
  thresholded image (the old name still works).
- **Update check** — once a day, `pylithics` and `pylithics-pages` ask GitHub
  for the latest release. On a clone on `main` with an older version installed,
  the command offers to pull and reinstall, then points at the release notes.
  Silent offline, never asks in a script, off with `update_check.enabled: false`
  or `PYLITHICS_NO_UPDATE_CHECK`.
- **README splash in both GitHub themes** — `tests/fixtures/render_splash.py`
  writes `docs/assets/images/splash-dark.svg` and `splash-light.svg` from the
  real splash code; the README shows whichever matches the reader's theme.
- **Crop names say how they were named** — `{page}_figure_7.png` carries the
  identifier printed on the plate; `{page}_box_07.png` is the seventh box on
  the debug overlay, used when no identifier was read. The two forms cannot
  be mistaken for each other.
- **`meta_data.csv` written by `pylithics-pages`** — one row per artefact
  crop, paired with the scale bar from the same page. The millimetre `scale`
  value is left blank for the user. A new `flag` column marks rows that need
  attention (`no_scale`, `several_scales`, or an identifier flag).
- **The pixels question** — when some metadata rows have no `scale` value and
  a person is at the terminal, `pylithics` asks once whether to measure those
  images in pixels; `n` analyses only the images with a scale. Scripts are not
  asked and continue in pixels with a warning; `--force_pixels` and
  `--disable_scale_calibration` are never asked. Images left out are listed in
  `run_summary.json`.
- **Flags are reported, not blocking** — rows with a `flag` run like any
  other and are counted on screen at the end, listed in the log, and recorded
  in `run_summary.json`.
- **`--meta_file` defaults** to `<data_dir>/meta_data.csv`.
- **`--explore PATH`** opens the dashboard for a results folder without
  analysing. A bare `--explore` still analyses and then opens.

### Fixed
- **An orphaned crop locked its page out of every run.** A crop written by a
  run that stopped before the manifest was saved had no manifest row, so the
  next run took it for the user's own file and refused the page, every time.
  A file that carries a crop name of a page in `pages/` is now replaced, and
  the run reports how many such files it replaced. Two pages that differ only
  by extension are still refused, with a message that names the other page,
  and the run reports how many pages it cut again.
- **`pylithics --config_file` was ignored.** The update check created the
  process-wide configuration before the file named on the command line was
  read, so every run used the default configuration. The file is now read
  first. The same fault in `pylithics-pages` is fixed below.
- **A labelled crop could absorb a whole page.** The join rule let one
  labelled box take every unlabelled neighbour in turn, growing as it went;
  a page with one read label collapsed to one crop. A box now joins only a
  neighbour at least a third of its size.
- **`pylithics-pages` printed every log line twice**, including DEBUG lines
  whatever `--log_level` said: the first line logged at start-up made the
  standard library attach its own console handler. A placeholder handler
  now prevents it.
- **The OCR extra no longer reinstalls the plain OpenCV build.** RapidOCR
  declares `opencv-python` by name, so `pip install ".[ocr]"` put a second
  OpenCV beside the contrib build and left a mixed `cv2` that lost modules at
  random. The extra now carries RapidOCR's other dependencies, and RapidOCR is
  installed with `--no-deps`. See the installation page.
- **`pylithics-pages --config_file` was ignored** — the update check created the
  process-wide configuration before the file was read, so every run used the
  built-in defaults. The manager is now created from the file first.
- **Command-line overrides now reach parallel workers.** Worker processes were
  built from the configuration file alone, so on a multi-worker run every
  override (`--threshold_method`, `--disable_arrow_detection`, the debug
  flags, ...) was silently ignored. The main process now hands its merged
  configuration to each worker.
- **`--disable_arrow_detection` now disables arrow detection.** The pipeline
  never checked `arrow_detection.enabled`.
- **`--scale_debug`** wrote to a stale `processed/scale_debug/` folder;
  **`--arrow_debug`** wrote nothing; **`--show_thresholded_images`** did
  nothing. All three now write under `results/`.

### Changed
- **Python 3.10 or later is necessary** (was 3.8). The package is developed and
  tested on 3.12. The installation guide and the troubleshooting page name the
  contrib build of OpenCV, the only build with the upscaling module, and tell
  Apple Silicon users to check that Python is a native build: an Intel build
  runs through Rosetta and the upscaler is three times slower.
- **Crop padding is 10 px** (was 20). A scale crop still holds its caption,
  which is joined to the bar before the crop is cut.
- **One flat project folder for both commands.** `pylithics-pages` writes
  `images/`, `scales/`, `pages_manifest.csv` and `meta_data.csv` into the project
  folder itself (formerly `processed_pages/`), so the whole workflow is
  `pylithics-pages --data_dir X` then `pylithics --data_dir X`.
- **`processed_images/` is renamed `results/`.** The analysis output, the log
  and the optional `json/` folder now live in `<data_dir>/results/`.
- `pylithics-pages` adds to a project that already holds images and a
  `meta_data.csv`: crops go beside the images, rows go after the user's rows
  (a `flag` column is added if missing), and a page whose crop name is taken
  is refused whole. Every run cuts every plate: a plate cut before has its
  crops and rows replaced (typed scale values are kept), a new plate is added,
  and a crop the new run does not make is removed. There is no `--force`.
- Sample data: `setup.py` ships only `sample_plate.png`; the research corpus and
  all pipeline output are gitignored.

### Notes
- The sample project `pylithics/data` holds both hand-cropped images and a
  sample plate in `pages/`, so `pylithics-pages --data_dir pylithics/data`
  followed by `pylithics --data_dir pylithics/data` exercises the whole chain.

## [2.0.0] - 2026-06-21

Major rewrite. PyLithics 2.0 introduces an interactive Streamlit dashboard, a
rich-styled CLI, per-lithic JSON export, real-world scale calibration via
scale-bar detection, cortex texture analysis, and a fully modularised codebase
with substantially expanded test coverage.

### Added
- **Interactive results dashboard** — Streamlit app for exploring processed
  metrics with tabs for Size & shape, Symmetry, Scars, and Spatial analyses.
  Filters, ECDFs, lollipop charts, sized-circle dot plots, and
  Voronoi/centroid-hull visualisations. Per-lithic detail page with
  side-by-side labeled image and Voronoi diagram, raw metrics table, and the
  full per-lithic JSON document.
- **Per-lithic JSON export** — `--export_json` writes one JSON document per
  image alongside the CSV, with the full hierarchy of surfaces, scars, arrows,
  and Voronoi metrics.
- **Real-world calibration via scale bars** — comprehensive scale-bar
  calibration from CSV metadata; areas and linear dimensions are converted
  from pixels to mm. CSV `calibration_method` column records `scale_bar` or
  `pixels` per row.
- **Cortex detection and texture analysis** — identifies and quantifies cortex
  regions on dorsal surfaces, with configurable sensitivity via CLI
  (`--cortex_sensitivity`, `--disable_cortex_detection`) and `config.yaml`.
- **Scar adjacency analysis** — distance-based detection of scar
  border-sharing relationships, written out as per-scar complexity counts.
- **Voronoi tessellation** restricted to the dorsal surface, with per-cell
  area output and centroid convex hull metrics.
- **Lateral surface convexity analysis** and distance measurements.
- **Aspect ratio, perimeter, max-length, and max-width** metrics on every
  contour.
- **Extensive YAML configuration** — single `config.yaml` exposes 18
  documented sections covering every pipeline stage (DPI-aware preprocessing,
  thresholding, normalization, grayscale conversion, morphological closing,
  logging, contour filtering, arrow detection and integration, surface
  classification, symmetry analysis, lateral analysis, Voronoi analysis,
  visualization, cortex detection, scar complexity, data export, scale
  calibration). Every option has inline comments describing its effect,
  typical range, and recommended defaults.
- **Comprehensive `pylithics --help`** — every flag is documented with
  examples and tuning guidance for each pipeline stage.
- **End-to-end pipeline, batch-processing, and error-scenario test suites**.
- **MkDocs documentation site** with installation guide, user guide, CLI
  reference, and troubleshooting.

### Changed
- **CLI rebuilt** with rich-styled logging; focused INFO-by-default console
  output and a `--verbose` flag for deep trace.
- **Arrow detection rebuilt** as an object-oriented, DPI-aware pipeline with
  hierarchy-independent detection logic and improved cortex exclusion.
- **Surface classification** rewritten for archaeological accuracy with
  surface-based child-feature classification.
- **Width/height** measurements replaced with Y-axis-aligned **technical
  length and width**.
- **Configuration loader** rebuilt with validation, caching, and dependency
  inversion. CLI flags override YAML values; YAML values override built-in
  defaults.
- **Image-analysis pipeline** split into dedicated modules: contour
  processing, symmetry analysis, visualization, Voronoi analysis, arrow
  integration, contour metrics, surface classification.
- **All oversized functions decomposed**; type hints and PEP 8 docstrings
  added throughout.
- **Tests consolidated and tightened** with stronger numerical assertions and
  real geometric invariants in place of loose `isinstance` checks.
- **README** rewritten with v2 announcement and quick-start commands.

### Fixed
- Conversion factor not applied to area measurements when scale-bar
  calibration was active.
- Cortex being mistaken for arrows by the arrow-detection pipeline.
- Arrow detection returning `False` instead of `None` on triangle-height
  validation failure.
- Image format being appended twice in some output filenames.
- Index-out-of-bounds error in contour processing for edge cases.
- Division by zero on very small contours (minimum-area threshold added).
- `image_analysis` not loading configuration in some entry paths.
- Various morphological-closing config-missing edge cases handled gracefully.

### Removed
- Legacy debug directory creation from arrow integration.
- Unused `lru_cache` getters and dead code across multiple modules.
- Duplicate filter function and unused `load_config` from utilities.
- Old CNN-based arrow detection infrastructure (replaced by the
  hierarchy-independent geometric pipeline).

## [1.0.0] - 2022-01-24

Initial release. Published in the Journal of Open Source Software (JOSS) with
a Zenodo DOI. Provides the core image-processing pipeline for extracting
quantitative morphological data from 2D line drawings of prehistoric stone
artefacts.

[2.0.0]: https://github.com/alan-turing-institute/Palaeoanalytics/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/alan-turing-institute/Palaeoanalytics/releases/tag/v1.0.0
