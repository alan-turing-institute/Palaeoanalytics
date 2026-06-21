# Changelog

All notable changes to PyLithics are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
