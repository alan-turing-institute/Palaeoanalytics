# Welcome to PyLithics

## Archaeological Lithic Analysis with Computer Vision

PyLithics is an open-source Python package that applies advanced computer vision techniques to extract quantitative morphological data from 2D line drawings of prehistoric stone artifacts.

## What is PyLithics?

PyLithics processes scanned 2D illustrations of stone tools (lithics) from archaeological publications, automatically identifying and measuring key technological and morphological features. The tool has been optimized for feature extraction using cutting-edge computer vision techniques including:

- Pixel intensity thresholding
- Edge detection and contour finding
- Custom template matching
- Advanced geometric analysis
- Machine learning-based feature recognition

## Key Features

### Accurate Surface Identification
- Automatically identifies dorsal, ventral, platform, and lateral surfaces
- Recognizes individual flake scars with high precision
- Detect cortex areas
- Configurable detection parameters for different drawing styles

### Comprehensive Measurements
- Complete size and shape metrics for whole flakes and individual scars
- Technical dimensions (length, width, thickness)
- Geometric properties (area, perimeter, aspect ratios)
- Symmetry analysis (vertical and horizontal)
- Scar complexity and adjacency relationships

### Advanced Analysis
- Flaking direction detection through arrow recognition
- Voronoi diagram generation for spatial analysis
- Convex hull calculations
- Lateral edge convexity analysis
- Configurable analysis modules that can be enabled/disabled

### Configuration and Customization
PyLithics offers extensive configuration options through both YAML configuration files and command-line arguments:

- Adjust thresholding methods (simple, Otsu, adaptive)
- Enable/disable specific analysis modules
- Fine-tune detection parameters
- Customize output formats

### Research-Ready Output
- Structured CSV data output with hierarchical organization
- Annotated visualization images for validation
- Comprehensive logging for reproducibility
- Compatible with standard statistical analysis software


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