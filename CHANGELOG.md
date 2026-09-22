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

### Added
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
  from the five shipped drawings with a shared scale bar, for trying the
  workflow and for the test suite. Regenerate with
  `tests/fixtures/generate_sample_plate.py`.
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
