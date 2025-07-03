# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyLithics is a Python package for archaeological stone tool analysis using computer vision and image processing. It analyzes 2D lithic illustrations to extract quantitative measurements of flake geometry, surface features, and technological attributes.

## Development Commands

### Installation and Setup
```bash
# Create virtual environment
python3 -m venv palaeo
source palaeo/bin/activate  # On Windows: .\palaeo\Scripts\activate

# Install package
pip install .

# Install with development dependencies
pip install -e ".[dev]"
```

### Testing
```bash
# Run all tests
pytest -s

# Run with coverage
pytest -s --cov=pylithics
```

### Running PyLithics
```bash
# Basic usage with test data
pylithics_run -c pylithics/config/config.yaml --input_dir data --output_dir output --metadata_filename meta_data.csv

# With arrow detection
pylithics_run -c pylithics/config/config.yaml --input_dir data --output_dir output --metadata_filename meta_data.csv --get_arrows

# Help
pylithics_run --help
```

## Architecture Overview

### Core Pipeline
1. **Image Preprocessing** (`importer.py`) - Grayscale conversion, normalization, thresholding
2. **Contour Extraction** (`contour_extraction.py`) - Hierarchical contour detection
3. **Metric Calculation** (`contour_metrics.py`) - Geometric measurements
4. **Surface Classification** (`surface_classification.py`) - Categorize as Dorsal/Ventral/Platform/Lateral
5. **Specialized Analysis** - Arrow detection, symmetry analysis, Voronoi diagrams
6. **Output Generation** (`visualization.py`) - CSV export and labeled images

### Key Components

#### Application Entry Points
- `PyLithicsApplication` class in `pylithics/app.py` - Main orchestrator
- Console commands: `pylithics` and `pylithics-run`

#### Configuration System
- `ConfigurationManager` in `pylithics/image_processing/config.py`
- YAML configuration files in `pylithics/config/`
- Default config: `pylithics/config/config.yaml`

#### Core Processing Modules
- `pylithics/image_processing/image_analysis.py` - Main processing coordinator
- `pylithics/image_processing/modules/` - Specialized analysis modules
- `pylithics/image_processing/measurement.py` - Unit conversion utilities

### Data Flow Architecture
```
Raw Image → Preprocessing → Binary Image → Contour Extraction → 
Metric Calculation → Surface Classification → Specialized Analysis → Output
```

## Development Guidelines

### Code Organization
- Main processing logic in `pylithics/image_processing/`
- Specialized modules in `pylithics/image_processing/modules/`
- Configuration management centralized in `config.py`
- Tests in `tests/` directory

### Configuration Management
- All processing parameters controlled via YAML configuration
- DPI-aware parameter scaling for arrow detection
- Multi-level configuration: defaults → file → environment → CLI
- Extensive validation and error handling

### Key Processing Features
- **Arrow Detection**: DPI-aware with configurable sensitivity parameters
- **Surface Classification**: Automated categorization based on geometric properties
- **Hierarchical Contours**: Parent-child relationships for surfaces and scars
- **Unit Conversion**: Pixel-to-millimeter scaling based on image metadata
- **Batch Processing**: Handles multiple images with metadata CSV files

### Testing Strategy
- Unit tests for individual modules
- Integration tests for complete pipeline
- Test data in `test_data/` and `tests/test_images/`
- TravisCI integration for automated testing

## File Structure Context

### Input Requirements
- Images in `input_dir/images/` (PNG format preferred)
- Scale images in `input_dir/scales/` (optional)
- Metadata CSV with columns: PA_ID, scale_ID, PA_scale

### Output Structure
- Processed images with contour overlays
- JSON files with hierarchical measurement data
- CSV export files with flattened metrics
- Debug visualizations (when enabled)

## Common Development Tasks

### Adding New Analysis Modules
1. Create module in `pylithics/image_processing/modules/`
2. Implement standardized interface with contour processing
3. Add configuration parameters to `config.yaml`
4. Update `image_analysis.py` to integrate module
5. Add tests in `tests/`

### Modifying Configuration
- Update `pylithics/config/config.yaml` for new parameters
- Modify `ConfigurationManager` validation if needed
- Document new parameters with comments

### Debugging Processing Issues
- Enable debug logging: set `logging.level: DEBUG` in config
- Enable arrow detection debug: set `arrow_detection.debug_enabled: true`
- Check log files in `pylithics/processed/logs/`

## Dependencies and Technology Stack

### Core Libraries
- **OpenCV**: Image processing and computer vision
- **NumPy/SciPy**: Numerical computations and spatial analysis
- **Pillow**: Image I/O and DPI metadata extraction
- **PyYAML**: Configuration file parsing
- **Pandas**: Data manipulation and CSV export
- **Matplotlib**: Visualization generation
- **Shapely**: Geometric operations

### Development Tools
- **pytest**: Testing framework
- **black**: Code formatting
- **flake8**: Linting
- **isort**: Import sorting