# CLI Commands Reference

This page lists every flag accepted by the `pylithics` CLI in the current release. The authoritative source is `pylithics --help`; this reference mirrors it with examples.

## Invocation

After installation, PyLithics is run as a console script:

```bash
pylithics [options]
```

## Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--data_dir` | Directory containing `images/` and `scales/` subdirectories | `pylithics/data` |
| `--meta_file` | CSV metadata file (columns: `image_id`, `scale_id`, `scale`) | `pylithics/data/meta_data.csv` |

```bash
pylithics --data_dir pylithics/data --meta_file pylithics/data/meta_data.csv
```

## Configuration

| Option | Description | Default |
|--------|-------------|---------|
| `--config_file FILE` | Path to a custom YAML configuration file | (use built-in defaults) |
| `--threshold_method METHOD` | One of `simple`, `otsu`, `adaptive`, `default` | `default` |
| `--log_level LEVEL` | Log verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` |

```bash
pylithics --data_dir ./data --meta_file ./meta.csv --config_file ./my_config.yaml
```

### Configuration Hierarchy

1. **Defaults** — baked into the code
2. **YAML file** — loaded from `--config_file` (or `$PYLITHICS_CONFIG`)
3. **CLI flags** — override everything else

### Example Configuration File

```yaml
# config.yaml
thresholding:
  method: otsu
  threshold_value: 127
  max_value: 255

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
  distance_threshold: 5.0

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
  level: INFO
  log_to_file: true
  log_file: pylithics/data/processed/pylithics.log
```

## Processing Options

| Option | Description | Default |
|--------|-------------|---------|
| `--show_thresholded_images` | Display thresholded images during analysis | off |
| `--closing BOOL` | Apply morphological closing | `True` |

## DPI Scaling

PyLithics auto-detects each image's DPI. By default it uses fixed kernel sizes that work across the 75–600 DPI range. Enable DPI-aware scaling for noisy photographs or degraded scans.

| Option | Description | Default |
|--------|-------------|---------|
| `--enable_dpi_scaling` | Enable DPI-aware kernel scaling | off |
| `--dpi_scaling_mode MODE` | `conservative`, `standard`, or `aggressive` | `standard` |
| `--dpi_reference DPI` | Reference DPI for scaling | `300.0` |
| `--dpi_max_scale FACTOR` | Maximum scaling factor | `1.5` |

**Modes:**

- **conservative** — minimal scaling, preserves fine line detail
- **standard** — moderate linear scaling with caps
- **aggressive** — full proportional scaling, maximum noise removal

```bash
# Default
pylithics --data_dir ./data --meta_file ./meta.csv

# Enable DPI scaling
pylithics --data_dir ./data --meta_file ./meta.csv --enable_dpi_scaling

# Aggressive scaling
pylithics --data_dir ./data --meta_file ./meta.csv \
    --enable_dpi_scaling --dpi_scaling_mode aggressive

# Custom reference & cap
pylithics --data_dir ./data --meta_file ./meta.csv \
    --enable_dpi_scaling --dpi_reference 150 --dpi_max_scale 2.0
```

## Arrow Detection

| Option | Description | Default |
|--------|-------------|---------|
| `--disable_arrow_detection` | Skip arrow detection | enabled |
| `--arrow_debug` | Save arrow-detection debug images to `processed/arrow_debug/` | off |
| `--show-arrow-lines` | Draw red arrow lines on labeled images | off |

```bash
pylithics --data_dir ./data --meta_file ./meta.csv --arrow_debug
pylithics --data_dir ./data --meta_file ./meta.csv --disable_arrow_detection
pylithics --data_dir ./data --meta_file ./meta.csv --show-arrow-lines
```

## Scale Calibration

| Option | Description | Default |
|--------|-------------|---------|
| `--disable_scale_calibration` | Skip scale-bar detection (use pixel measurements) | enabled |
| `--scale_debug` | Save scale-detection debug images | off |
| `--force_pixels` | Force pixel measurements only (skip calibration) | off |

```bash
pylithics --data_dir ./data --meta_file ./meta.csv --scale_debug
pylithics --data_dir ./data --meta_file ./meta.csv --force_pixels
pylithics --data_dir ./data --meta_file ./meta.csv --disable_scale_calibration
```

## Cortex Detection

| Option | Description | Default |
|--------|-------------|---------|
| `--disable_cortex_detection` | Skip cortex detection | enabled |
| `--cortex_sensitivity {low,medium,high}` | Detection sensitivity | `medium` |

```bash
pylithics --data_dir ./data --meta_file ./meta.csv --cortex_sensitivity high
pylithics --data_dir ./data --meta_file ./meta.csv --disable_cortex_detection
```

## Scar Complexity

| Option | Description | Default |
|--------|-------------|---------|
| `--disable_scar_complexity` | Skip scar adjacency analysis | enabled |
| `--scar_complexity_distance_threshold PIXELS` | Adjacency distance in pixels | `10.0` |

```bash
pylithics --data_dir ./data --meta_file ./meta.csv \
    --scar_complexity_distance_threshold 15.0
```

## Output

| Option | Description | Default |
|--------|-------------|---------|
| `--export_json` | Also write a per-lithic JSON file to `processed/json/{image_stem}.json` (in addition to the CSV) | off |
| `--save_visualizations` | Generate labeled images and Voronoi diagrams | on |

```bash
# Default — CSV only
pylithics --data_dir ./data --meta_file ./meta.csv

# CSV plus per-lithic JSON files
pylithics --data_dir ./data --meta_file ./meta.csv --export_json
```

The same behavior is available from `config.yaml`:

```yaml
data_export:
  csv: true
  json_per_lithic: true   # equivalent to --export_json
```

## Help

| Option | Description |
|--------|-------------|
| `-h`, `--help` | Show all available options |
| `--help-config` | Show built-in configuration documentation |
| `--help-examples` | Show usage examples |
| `--help-troubleshooting` | Show common problems and solutions |
| `--docs` | Launch the documentation server at <http://127.0.0.1:8000> |

```bash
pylithics --help
pylithics --help-config
pylithics --docs
```

## Common Patterns

### Quick test on the bundled sample data

```bash
pylithics --data_dir pylithics/data --meta_file pylithics/data/meta_data.csv
```

### Debug a problem image

```bash
pylithics --data_dir ./data --meta_file ./meta.csv \
    --log_level DEBUG \
    --arrow_debug \
    --scale_debug \
    --show_thresholded_images
```

### Faster runs

```bash
# Disable arrow detection (the most expensive optional step)
pylithics --data_dir ./data --meta_file ./meta.csv --disable_arrow_detection
```

### Override config from CLI

CLI flags always win over the YAML file:

```bash
pylithics --data_dir ./data --meta_file ./meta.csv \
    --config_file ./site_config.yaml \
    --threshold_method adaptive
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `PYLITHICS_CONFIG` | Default config file path used when `--config_file` is omitted |

```bash
export PYLITHICS_CONFIG=./default_config.yaml
pylithics --data_dir ./data --meta_file ./meta.csv
```

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success (all images processed, or partial success with at least one done) |
| `1` | Input validation failed, processing failed, or an unhandled error |
