# CLI Commands Reference

## Command Overview

PyLithics is run using Python directly:

```bash
python pylithics/app.py [options]
```

## Required Arguments

Every PyLithics run requires these two arguments:

| Argument | Description | Example |
|----------|-------------|---------|
| `--data_dir` | Directory containing images/ and scales/ subdirectories | `pylithics/data` |
| `--meta_file` | Path to metadata CSV file | `pylithics/data/meta_data.csv` |

### Basic Usage

```bash
python pylithics/app.py --data_dir pylithics/data --meta_file pylithics/data/meta_data.csv
```

## Configuration Options

### Configuration Hierarchy

PyLithics uses a three-layer configuration system:

1. **Default settings** - Built into the code
2. **YAML configuration** - From config.yaml file
3. **CLI overrides** - Command-line arguments (highest priority)

### Using a Configuration File

| Option | Description | Default |
|--------|-------------|---------|
| `--config_file` | Path to YAML configuration file | None (uses defaults) |

```bash
python pylithics/app.py --data_dir pylithics/data --meta_file pylithics/data/meta_data.csv \
         --config_file ./my_config.yaml
```

### Example Configuration File

```yaml
# config.yaml
thresholding:
  method: otsu              # simple, otsu, or adaptive
  threshold_value: 127      # For simple method
  max_value: 255

scale_calibration:
  enabled: true             # Enable scale bar detection
  fallback_to_dpi: true     # Use DPI if scale bar fails
  fallback_to_pixels: true  # Use pixels if no calibration
  debug_output: false       # Save detection debug images

arrow_detection:
  enabled: true
  reference_dpi: 300.0
  min_area_scale_factor: 0.7
  max_area_scale_factor: 10.0
  min_aspect_ratio: 1.5
  debug_enabled: false

surface_classification:
  enabled: true
  classification_rules:
    dorsal_area_threshold: 0.6
    ventral_area_threshold: 0.4

scar_complexity:
  enabled: true
  distance_threshold: 10.0

cortex_detection:
  enabled: true
  min_area: 100
  detection_method: color_threshold

symmetry_analysis:
  enabled: true
  axis: both              # vertical, horizontal, or both

voronoi_analysis:
  enabled: true
  output_diagrams: true

lateral_analysis:
  enabled: true
  convexity_threshold: 0.8

logging:
  level: INFO            # DEBUG, INFO, WARNING, ERROR
  log_to_file: true
  log_file: pylithics.log
```

### Common Usage Patterns

#### Basic Analysis

```bash
# Simple processing with default settings
python pylithics/app.py --data_dir pylithics/data --meta_file pylithics/data/meta_data.csv
```

#### With Arrow Detection

```bash
# Enable arrow detection for flaking direction
python pylithics/app.py --data_dir pylithics/data --meta_file pylithics/data/meta_data.csv \
         --arrow_debug
```

#### Debug Mode

```bash
# Verbose output for troubleshooting
python pylithics/app.py --data_dir pylithics/data --meta_file pylithics/data/meta_data.csv \
         --log_level DEBUG
```

#### Fast Processing

```bash
# Disable time-consuming features for speed
python pylithics/app.py --data_dir pylithics/data --meta_file pylithics/data/meta_data.csv \
         --disable_arrow_detection \
         --disable_voronoi
```

#### Custom Thresholding

```bash
# Use adaptive thresholding for poor contrast
python pylithics/app.py --data_dir pylithics/data --meta_file pylithics/data/meta_data.csv \
         --threshold_method adaptive
```

### Thresholding Options

| Option | Values | Description | Default |
|--------|--------|-------------|---------|
| `--threshold_method` | `simple`, `otsu`, `adaptive` | Image binarization method | `simple` |
| `--threshold_value` | 0-255 | Threshold for simple method | 127 |
| `--max_value` | 0-255 | Maximum value after thresholding | 255 |

```bash
# Use Otsu automatic thresholding
python pylithics/app.py --data_dir pylithics/data --meta_file pylithics/data/meta_data.csv \
         --threshold_method otsu

# Use adaptive thresholding for poor contrast
python pylithics/app.py --data_dir pylithics/data --meta_file pylithics/data/meta_data.csv \
         --threshold_method adaptive

# Manual threshold adjustment
python pylithics/app.py --data_dir pylithics/data --meta_file pylithics/data/meta_data.csv \
         --threshold_method simple --threshold_value 100
```

## Feature Control

### Enabling/Disabling Analysis Modules

| Feature | Enable | Disable | Default |
|---------|--------|---------|---------|
| Scale Calibration | Default enabled | `--disable_scale_calibration` | Enabled |
| Arrow Detection | Default enabled | `--disable_arrow_detection` | Enabled |
| Voronoi Analysis | Default enabled | `--disable_voronoi` | Enabled |
| Symmetry Analysis | Default enabled | `--disable_symmetry` | Enabled |
| Scar Complexity | Default enabled | `--disable_scar_complexity` | Enabled |
| Cortex Detection | `--enable_cortex_detection` | Default disabled | Disabled |
| Lateral Analysis | Default enabled | `--disable_lateral` | Enabled |

### Scale Calibration Options

| Option | Description | Default |
|--------|-------------|---------|
| `--disable_scale_calibration` | Skip scale bar detection entirely | False |
| `--scale_debug` | Save scale detection debug images | False |
| `--force_scale_method` | Force specific calibration method: `scale_bar`, `dpi`, `pixels` | None |

```bash
# Enable scale debugging
pylithics --data_dir ./data --meta_file ./meta.csv --scale_debug

# Force DPI calibration only
pylithics --data_dir ./data --meta_file ./meta.csv --force_scale_method dpi

# Disable scale calibration (pixel measurements only)
pylithics --data_dir ./data --meta_file ./meta.csv --disable_scale_calibration
```

### Arrow Detection Options

| Option | Description | Default |
|--------|-------------|---------|
| `--disable_arrow_detection` | Skip arrow detection entirely | False |
| `--arrow_debug` | Save arrow detection debug images | False |
| `--show_arrow_lines` | Draw red arrow lines on output | False |
| `--arrow_reference_dpi` | Reference DPI for scaling | 300.0 |

```bash\n# Enable arrow debugging\npylithics --data_dir ./data --meta_file ./meta.csv --arrow_debug\n\n# Disable arrow detection for speed\npylithics --data_dir ./data --meta_file ./meta.csv --disable_arrow_detection\n\n# Show arrow direction lines\npylithics --data_dir ./data --meta_file ./meta.csv --show_arrow_lines
```

### Scar Complexity Options

| Option | Description | Default |
|--------|-------------|---------|
| `--scar_complexity_distance_threshold` | Adjacency detection distance (pixels) | 10.0 |
| `--disable_scar_complexity` | Skip scar adjacency analysis | False |

```bash\n# Adjust adjacency sensitivity\npylithics --data_dir ./data --meta_file ./meta.csv \\\n         --scar_complexity_distance_threshold 15.0
```

### Cortex Detection Options

| Option | Description | Default |
|--------|-------------|---------|
| `--enable_cortex_detection` | Enable cortex area detection | False |
| `--cortex_method` | Detection method: `color`, `texture`, `pattern` | `color` |
| `--cortex_threshold` | Detection sensitivity | 0.5 |

```bash\n# Enable cortex detection\npylithics --data_dir ./data --meta_file ./meta.csv --enable_cortex_detection
```

## Output Options

### Output Format

| Option | Values | Description | Default |
|--------|--------|-------------|---------|
| `--output_format` | `csv`, `json` | Data output format | `csv` |
| `--output_dir` | Directory path | Custom output location | `./processed` |

```bash\n# JSON output\npylithics --data_dir ./data --meta_file ./meta.csv --output_format json\n\n# Custom output directory\npylithics --data_dir ./data --meta_file ./meta.csv --output_dir ./results
```

### Visualization Options

| Option | Description | Default |
|--------|-------------|---------|
| `--save_visualizations` | Save all visualization outputs | True |
| `--no_images` | Skip image output generation | False |
| `--show_thresholded_images` | Display thresholding results | False |

```bash\n# Data only, no images\npylithics --data_dir ./data --meta_file ./meta.csv --no_images\n\n# Show intermediate processing steps\npylithics --data_dir ./data --meta_file ./meta.csv --show_thresholded_images
```

## Preprocessing Options

### Image Enhancement

| Option | Description | Default |
|--------|-------------|---------|
| `--closing` | Apply morphological closing | True |
| `--closing_kernel_size` | Closing operation kernel size | 3 |
| `--denoise` | Apply noise reduction | False |
| `--contrast_stretch` | Enhance image contrast | False |

```bash\n# Disable morphological closing\npylithics --data_dir ./data --meta_file ./meta.csv --no_closing\n\n# Enable denoising for noisy scans\npylithics --data_dir ./data --meta_file ./meta.csv --denoise
```

## Logging and Debug Options

### Logging Control

| Option | Values | Description | Default |
|--------|--------|-------------|---------|
| `--log_level` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | Logging verbosity | `INFO` |
| `--log_file` | File path | Custom log file location | `./processed/pylithics.log` |
| `--quiet` | - | Suppress console output | False |

```bash\n# Debug mode with verbose output\npylithics --data_dir ./data --meta_file ./meta.csv --log_level DEBUG\n\n# Quiet mode\npylithics --data_dir ./data --meta_file ./meta.csv --quiet\n\n# Custom log file\npylithics --data_dir ./data --meta_file ./meta.csv --log_file ./my_analysis.log
```

### Debug Options

| Option | Description | Output Location |
|--------|-------------|-----------------|
| `--arrow_debug` | Arrow detection debug images | `processed/arrow_debug/` |
| `--contour_debug` | Contour detection debug images | `processed/contour_debug/` |
| `--show_thresholded_images` | Display threshold results | Console |
| `--save_intermediate` | Save all intermediate processing steps | `processed/intermediate/` |

```bash\n# Full debug mode\npylithics --data_dir ./data --meta_file ./meta.csv \\\n         --log_level DEBUG \\\n         --arrow_debug \\\n         --contour_debug \\\n         --save_intermediate
```

## Performance Options

### Speed Optimization

```bash\n# Fastest processing (minimal features)\npylithics --data_dir ./data --meta_file ./meta.csv \\\n         --disable_arrow_detection \\\n         --disable_voronoi \\\n         --disable_symmetry \\\n         --disable_scar_complexity \\\n         --threshold_method simple \\\n         --no_images\n\n# Balanced speed/features\npylithics --data_dir ./data --meta_file ./meta.csv \\\n         --disable_arrow_detection \\\n         --threshold_method otsu
```

### Memory Management

| Option | Description |
|--------|-------------|
| `--batch_size` | Process images in batches |
| `--max_image_size` | Resize large images |
| `--cleanup_temp` | Remove temporary files |

```bash\n# Memory-efficient processing\npylithics --data_dir ./large_dataset --meta_file ./meta.csv \\\n         --batch_size 10 \\\n         --max_image_size 2048 \\\n         --cleanup_temp
```

## Configuration File Override

### CLI Override Pattern

Command-line arguments override configuration file settings using dot notation:

```bash\n# Override arrow detection settings\npylithics --data_dir ./data --meta_file ./meta.csv \\\n         --config_file ./config.yaml \\\n         --arrow_reference_dpi 600 \\\n         --disable_arrow_detection\n\n# Override thresholding settings\npylithics --data_dir ./data --meta_file ./meta.csv \\\n         --config_file ./config.yaml \\\n         --threshold_method adaptive \\\n         --threshold_value 120
```

### Complete Configuration Example

```yaml\n# config.yaml - Complete configuration template\nthresholding:\n  method: otsu\n  threshold_value: 127\n  max_value: 255\n  adaptive_block_size: 11\n  adaptive_constant: 2\n\nscale_calibration:\n  enabled: true\n  fallback_to_dpi: true\n  fallback_to_pixels: true\n  debug_output: false\n\narrow_detection:\n  enabled: true\n  reference_dpi: 300.0\n  min_area_scale_factor: 0.7\n  max_area_scale_factor: 10.0\n  min_aspect_ratio: 1.5\n  debug_enabled: false\n\nsurface_classification:\n  enabled: true\n  classification_rules:\n    dorsal_area_threshold: 0.6\n    ventral_area_threshold: 0.4\n    platform_area_threshold: 0.1\n\nscar_complexity:\n  enabled: true\n  distance_threshold: 10.0\n  min_scar_area: 50\n\ncortex_detection:\n  enabled: false\n  method: color_threshold\n  threshold: 0.5\n  min_area: 100\n\nsymmetry_analysis:\n  enabled: true\n  axis: both\n  tolerance: 0.1\n\nvoronoi_analysis:\n  enabled: true\n  min_scars: 3\n  output_diagrams: true\n  boundary_method: convex\n\nlateral_analysis:\n  enabled: true\n  convexity_threshold: 0.8\n  edge_sensitivity: 0.5\n\npreprocessing:\n  denoise: false\n  morphological_closing: true\n  closing_kernel_size: 3\n  contrast_stretch: false\n\nlogging:\n  level: INFO\n  log_to_file: true\n  log_file: pylithics.log\n\noutput:\n  format: csv\n  save_visualizations: true\n  save_intermediate: false\n```

## Help Commands

### Built-in Help

| Command | Description |
|---------|-------------|
| `--help`, `-h` | Show all available options |
| `--help-config` | Show configuration file options |
| `--help-examples` | Show usage examples |
| `--help-troubleshooting` | Show common issues and fixes |
| `--version` | Show PyLithics version |

```bash\n# Get help\npylithics --help\n\n# Configuration help\npylithics --help-config\n\n# Example commands\npylithics --help-examples\n\n# Troubleshooting guide\npylithics --help-troubleshooting
```

## Common Command Patterns

### Development and Testing

```bash\n# Quick test with sample data\npylithics --data_dir ./pylithics/data --meta_file ./pylithics/data/meta_data.csv\n\n# Test single image\npylithics --data_dir ./test_single --meta_file ./single_meta.csv --log_level DEBUG\n\n# Validation run with all debug output\npylithics --data_dir ./validation --meta_file ./val_meta.csv \\\n         --log_level DEBUG \\\n         --arrow_debug \\\n         --save_intermediate
```

### Production Analysis

```bash\n# Standard archaeological analysis\npylithics --data_dir ./assemblage --meta_file ./metadata.csv \\\n         --config_file ./site_config.yaml \\\n         --log_level INFO\n\n# Publication-quality analysis\npylithics --data_dir ./publication_data --meta_file ./pub_meta.csv \\\n         --arrow_debug \\\n         --save_visualizations \\\n         --output_format csv\n\n# Batch processing multiple sites\nfor site in site_*; do\n    pylithics --data_dir ./$site --meta_file ./$site/metadata.csv \\\n             --output_dir ./results/$site\ndone
```

### Performance-Optimized

```bash\n# Large dataset processing\npylithics --data_dir ./large_assemblage --meta_file ./large_meta.csv \\\n         --disable_arrow_detection \\\n         --disable_voronoi \\\n         --threshold_method simple \\\n         --no_images \\\n         --quiet\n\n# Memory-constrained environment\npylithics --data_dir ./data --meta_file ./meta.csv \\\n         --batch_size 5 \\\n         --max_image_size 1024 \\\n         --cleanup_temp
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Successful completion |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | File not found |
| 4 | Configuration error |
| 5 | Processing error |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PYLITHICS_CONFIG` | Default config file path | None |
| `PYLITHICS_DATA_DIR` | Default data directory | None |
| `PYLITHICS_LOG_LEVEL` | Default log level | INFO |

```bash\n# Set environment variables\nexport PYLITHICS_CONFIG=./default_config.yaml\nexport PYLITHICS_LOG_LEVEL=DEBUG\n\n# Then run with simplified command\npylithics --data_dir ./data --meta_file ./meta.csv
```