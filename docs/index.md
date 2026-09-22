# Welcome to PyLithics

## Lithic analysis with computer vision

PyLithics is an open-source Python package. It measures two-dimensional
line drawings of prehistoric stone artefacts. It reads scanned
illustrations from archaeological publications. It finds the dorsal,
ventral, platform and lateral surfaces, and the flake scars on each
surface. It writes the measurements to CSV files and to one JSON file
for each lithic.

The pipeline has these steps: image preprocessing (normalisation,
greyscale conversion, thresholding, morphological closing), contour
extraction, surface classification, Voronoi tessellation of the scar
centroids, convex-hull analysis, and arrow-direction detection.
PyLithics does not use template matching or trained machine-learning
models. It calculates each measurement from the contour geometry. The
same input gives the same result each time.

## Functions

### Surface and feature identification
- Finds the dorsal, ventral, platform and lateral surfaces
- Finds each flake scar on each surface
- Finds cortex areas from stipple density, texture variance and edge density
- You can set the detection parameters for different drawing conventions

### Measurements
- Size and shape metrics for each surface and each scar
- Technical length and width (along the Y axis), and maximum length and width
- Area, perimeter, aspect ratio and bounding box
- Symmetry (vertical and horizontal, from area)
- Scar complexity and adjacency

### Spatial analysis
- Voronoi tessellation of the dorsal scar centroids
- Convex-hull metrics (area, width and height of the centroid hull)
- Convexity of the lateral edges
- Flaking direction from the arrows in the drawing

### Scale calibration
- Finds the scale bar in a scale image
- Changes areas and lengths from pixels to millimetres
- Uses pixel measurements when there is no scale bar, and writes this in the column `calibration_method`

### Dashboard
- A Streamlit dashboard in the browser for the results of a batch
- Tabs for size and shape, symmetry, scars and spatial analysis, with shared filters
- A page for each lithic with the labelled image, the Voronoi diagram, the metric tables and the JSON file

### Configuration
- A YAML configuration file with 18 sections and a comment for each option
- Command-line flags replace the YAML values; YAML values replace the built-in defaults
- Switches for arrow detection, cortex detection, scar complexity and other modules
- Three threshold methods: simple, Otsu and adaptive

### Output
- `processed_metrics.csv`, with one row for each surface and each scar
- One JSON file for each lithic, with the full hierarchy
- Annotated images: labelled surfaces, scars, arrows, Voronoi diagram, convex hull
- A summary that lists each image and each error
- Full logs, so that you can repeat an analysis

## Why PyLithics?

### For researchers
- **Time**: measurements that take hours by hand take minutes
- **Consistency**: no differences between observers
- **Scale**: you can analyse a full assemblage
- **Reproducibility**: the same input gives the same result

### For archaeological science
- **Quantitative analysis**: numbers, not only descriptions
- **Pattern recognition**: small technological differences become visible
- **Large data sets**: comparative studies at scale
- **Open science**: a free, open-source tool for the community

## Start

After installation, type `pylithics` with no arguments. The command
shows the most common command patterns: quick start, sample data, a
previous analysis in the browser, help, and the GitHub URL. Copy the
command that you want.

Then read:

1. **[Installation Guide](installation.md)** — install PyLithics
2. **[User Guide](user-guide/index.md)** — learn to use PyLithics
3. **[Prepare Your Images](user-guide/image-requirements.md)** — the image formats, resolution and drawing style
4. **[Basic Usage](user-guide/basic-usage.md)** — do your first analysis

See the [CLI Commands Reference](reference/cli-commands.md) for all
options.

## Support and contributions

PyLithics is in active development. Contributions from the
archaeological and computer-science communities are welcome.

- **Issues**: report a bug or ask for a function on [GitHub](https://github.com/alan-turing-institute/Palaeoanalytics/issues)
- **Contributions**: see the [Contributing Guidelines](https://github.com/alan-turing-institute/Palaeoanalytics/blob/main/CONTRIBUTING.md)
- **Contact**: see the [team](about.md)

## Citation

If you use PyLithics in your research, cite:

[![DOI](https://zenodo.org/badge/303727518.svg)](https://zenodo.org/badge/latestdoi/303727518)

## License

PyLithics is released under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0).
