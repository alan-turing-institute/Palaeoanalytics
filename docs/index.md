# Welcome to PyLithics

## Archaeological Lithic Analysis with Computer Vision

PyLithics is an open-source Python package for the automated quantitative analysis of two-dimensional line drawings of prehistoric stone artefacts. It processes scanned illustrations from archaeological publications, identifies dorsal, ventral, platform, and lateral surfaces along with their individual flake scars, and writes the resulting morphological, spatial, and structural measurements to analysis-ready CSV and per-lithic JSON files.

The processing pipeline combines image preprocessing (normalisation, grayscale conversion, configurable thresholding, morphological closing), hierarchical contour extraction, geometric surface classification, scar-centroid Voronoi tessellation, convex-hull spatial analysis, and DPI-aware arrow-direction detection from convexity defects. PyLithics does not use template matching or trained machine-learning models; every measurement is derived geometrically from the contour data, which keeps the pipeline transparent, deterministic, and reproducible.

## Key Features

### Surface and feature identification
- Automatically identifies dorsal, ventral, platform, and lateral surfaces
- Recognises individual flake scars within each surface
- Detects cortex regions using stippling-density, texture-variance, and edge-density thresholds
- Configurable detection parameters for different drawing conventions

### Comprehensive measurements
- Per-surface and per-scar size and shape metrics
- Technical length and width (Y-axis-aligned), plus max length and max width
- Geometric properties: area, perimeter, aspect ratio, bounding box
- Symmetry analysis (vertical and horizontal area-based)
- Scar complexity and adjacency relationships

### Spatial analysis
- Voronoi tessellation of dorsal scar centroids
- Convex-hull metrics (centroid-hull area, width, height)
- Lateral-edge convexity analysis
- Flaking-direction detection through geometric arrow recognition

### Real-world scale calibration
- Automatic scale-bar detection from accompanying scale images
- Areas and linear measurements converted from pixels to millimetres
- Falls back to pixel measurements with a clear `calibration_method` flag when a scale bar is not available

### Interactive results dashboard
- Streamlit-based browser dashboard for exploring batch results
- Tabs for Size & shape, Symmetry, Scars, and Spatial analyses with shared filters
- Per-lithic detail page with side-by-side labeled image and Voronoi diagram, raw metric tables, and the full per-lithic JSON document

### Configuration and customisation
- 18-section YAML configuration file with inline documentation for every option
- Command-line flags override YAML values; YAML values override built-in defaults
- Module-level toggles for arrow detection, cortex detection, scar complexity, and more
- Three thresholding methods: simple, Otsu, adaptive

### Research-ready output
- Consolidated `processed_metrics.csv` with one row per surface or scar
- Per-lithic JSON files with the full hierarchical structure
- Annotated visualisation images (labelled surfaces, scars, arrows, Voronoi diagram, convex hull)
- Run summary file recording every processed image and any failures
- Comprehensive logging for reproducibility


## Why PyLithics?

### For Researchers
- **Time-Saving**: Automate hours of manual measurement
- **Consistency**: Eliminate inter-observer variability
- **Scale**: Process entire assemblages efficiently
- **Reproducibility**: Ensure consistent, replicable results

### For Archaeological Science
- **Quantitative Analysis**: Move beyond qualitative descriptions
- **Pattern Recognition**: Identify subtle technological variations
- **Big Data**: Enable large-scale comparative studies
- **Open Science**: Free, open-source tool for the community

## Getting Started

Once PyLithics is installed, type `pylithics` on its own to see a welcome splash with the most common command patterns — quick start, run sample data and visualize, open an existing run in the browser, help, and the GitHub URL. Copy whichever command suits your situation.

Then dig in:

1. **[Installation Guide](installation.md)** - Set up PyLithics on your system
2. **[User Guide](user-guide/index.md)** - Learn how to use PyLithics effectively
3. **[Image Requirements](user-guide/image-requirements.md)** - Prepare your lithic illustrations
4. **[Basic Usage](user-guide/basic-usage.md)** - Run your first analysis

See the [CLI Commands Reference](reference/cli-commands.md) for complete configuration options.

## Support and Contributing

PyLithics is actively developed and maintained. We welcome contributions from the archaeological and computer science communities.

- **Issues**: Report bugs or request features on [GitHub](https://github.com/alan-turing-institute/Palaeoanalytics/issues)
- **Contributing**: See our [Contributing Guidelines](https://github.com/alan-turing-institute/Palaeoanalytics/blob/main/CONTRIBUTING.md)
- **Contact**: Reach out to the [team](about.md)

## Citation

If you use PyLithics in your research, please cite:

[![DOI](https://zenodo.org/badge/303727518.svg)](https://zenodo.org/badge/latestdoi/303727518)

## License

PyLithics is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0)