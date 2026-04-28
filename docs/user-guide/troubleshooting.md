# Troubleshooting

This guide covers the most common problems you'll hit running PyLithics. If your issue isn't here, check `pylithics.log` in the output directory for the actual error, then [open a GitHub issue](https://github.com/alan-turing-institute/Palaeoanalytics/issues) with the log excerpt.

## Installation Issues

### Python Version Errors

**Problem**: "Python 3.8+ required" or compatibility errors

**Solutions**:

```bash
# Check your Python version
python --version

# Install correct Python version
# macOS (using Homebrew)
brew install python@3.11

# Ubuntu/Debian
sudo apt-get install python3.11

# Windows - download from python.org
```

### Package Installation Failures

**Problem**: `pip install .` fails with dependency errors

**Solutions**:

```bash
# Upgrade pip first
pip install --upgrade pip

# Install with verbose output to see details
pip install . -v

# Install OpenCV separately if it fails
pip install opencv-python-headless>=4.8.0
pip install .
```

### Virtual Environment Issues

**Problem**: Virtual environment not activating, or installed packages not found

**Solutions**:

```bash
# Verify the environment is active
which python  # Should show a path inside your venv

# Recreate if corrupted
deactivate
rm -rf palaeo/
python3 -m venv palaeo
source palaeo/bin/activate
pip install .
```

## Image Processing Issues

### No Contours Found

**Problem**: "No contours detected" or empty results

**Diagnosis**:

```bash
pylithics --data_dir ./data --meta_file ./meta.csv \
    --log_level DEBUG --show_thresholded_images
```

**Solutions**:

```bash
# Try Otsu thresholding (good for bimodal images)
pylithics --data_dir ./data --meta_file ./meta.csv --threshold_method otsu

# Adaptive thresholding for poor or uneven contrast
pylithics --data_dir ./data --meta_file ./meta.csv --threshold_method adaptive
```

### Poor Contour Detection

**Problem**: Incomplete or inaccurate contour boundaries

**Image quality checks**:

- High contrast (black lines on white background)
- Resolution at least 300 DPI
- No scanning artifacts or noise
- Closed/complete contour outlines

**Configuration adjustments** (in your `config.yaml`):

```yaml
thresholding:
  method: adaptive

morphological_closing:
  enabled: true
  kernel_size: 3
```

### Scale Calculation Errors

**Problem**: Unrealistic measurements (way too large or too small)

**Check your metadata.csv**:

```csv
image_id,scale_id,scale
artifact_001.png,scale_001.png,10
```

The `scale` column is in **millimeters**. Common mistakes:

- Scale value entered in centimetres instead of millimetres
- Wrong scale image associated with the artifact
- Missing or empty `scale_id`

You can also force pixel measurements to bypass calibration entirely:

```bash
pylithics --data_dir ./data --meta_file ./meta.csv --force_pixels
```

## Configuration Issues

### Config File Not Loading

**Problem**: Configuration changes not taking effect

**Solutions**:

```bash
# Verify the path PyLithics is using
pylithics --data_dir ./data --meta_file ./meta.csv \
    --config_file ./config.yaml --log_level DEBUG

# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Use absolute paths if relative paths confuse the shell
pylithics --data_dir "$(pwd)/data" --meta_file "$(pwd)/meta.csv" \
    --config_file "$(pwd)/config.yaml"
```

### CLI Overrides Not Working

**Problem**: Command-line arguments seem to be ignored

**Solutions**:

```bash
# List all real flags
pylithics --help

# Run with debug logging to confirm what was applied
pylithics --data_dir ./data --meta_file ./meta.csv --log_level DEBUG
```

CLI arguments override the YAML file, which overrides defaults. Always check spelling against `pylithics --help`.

## Feature-Specific Issues

### Arrow Detection Problems

**Problem**: Arrows not detected, or false positives

**Enable debug mode**:

```bash
pylithics --data_dir ./data --meta_file ./meta.csv \
    --arrow_debug --log_level DEBUG
```

**Check debug output** in `processed/arrow_debug/` for candidate detections.

**Tune via config.yaml**:

```yaml
arrow_detection:
  enabled: true
  reference_dpi: 300.0
  min_area_scale_factor: 0.5
  min_defect_depth_scale_factor: 0.7
```

### Cortex Detection Tuning

**Problem**: Too few or too many cortex regions detected

```bash
# Increase sensitivity
pylithics --data_dir ./data --meta_file ./meta.csv --cortex_sensitivity high

# Decrease sensitivity
pylithics --data_dir ./data --meta_file ./meta.csv --cortex_sensitivity low

# Disable entirely
pylithics --data_dir ./data --meta_file ./meta.csv --disable_cortex_detection
```

### Performance Problems

**Problem**: Processing is too slow

```bash
# Disable arrow detection — the most expensive optional stage
pylithics --data_dir ./data --meta_file ./meta.csv --disable_arrow_detection

# Use simple thresholding instead of adaptive
pylithics --data_dir ./data --meta_file ./meta.csv --threshold_method simple

# Or skip multiple optional analyses together
pylithics --data_dir ./data --meta_file ./meta.csv \
    --disable_arrow_detection \
    --disable_cortex_detection \
    --disable_scar_complexity
```

### Voronoi Analysis Issues

**Problem**: No Voronoi diagrams generated

Voronoi requires a Dorsal surface in the metrics. Check:

- Surface classification is producing a `Dorsal` row
- The Dorsal surface has at least one scar
- `voronoi_analysis.enabled` is `true` in config.yaml (default)

Voronoi cannot currently be toggled from the CLI; use the config file.

## Data Issues

### Missing Output Files

**Problem**: Expected output files not created

**Check write permissions**:

```bash
ls -la pylithics/data/processed/
```

**Check the log**:

```bash
tail -50 pylithics/data/processed/pylithics.log
grep ERROR pylithics/data/processed/pylithics.log
```

### Unrealistic Measurements

**Problem**: Measurements don't match expectations

1. Verify the `scale` column in metadata is in millimetres
2. Check the labeled image to confirm the correct contour was detected
3. Compare against known measurements in the source publication
4. Inspect the CSV `calibration_method` column — `pixels` means no real-world conversion was applied

```python
import pandas as pd

df = pd.read_csv('pylithics/data/processed/processed_metrics.csv')

print("Length range:", df['technical_length'].min(), "-", df['technical_length'].max())
print("Area range:", df['total_area'].min(), "-", df['total_area'].max())
print("Calibration methods:", df['calibration_method'].value_counts())
```

## Platform-Specific Issues

### macOS

**OpenCV install problems**:

```bash
# Try the headless build (no GUI requirements)
pip install opencv-python-headless

# Or install via conda
conda install opencv
```

### Windows

**PowerShell execution policy**:

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Path length limits**:

- Use shorter directory paths
- Move the project closer to the drive root
- Enable Windows long-path support

### Linux

**Missing system dependencies**:

```bash
# Ubuntu/Debian
sudo apt-get install python3-dev libopencv-dev

# CentOS/RHEL
sudo yum install python3-devel opencv-devel
```

## Error Message Guide

### "FileNotFoundError"

**Cause**: Missing image or scale files referenced in metadata
**Solution**: Verify every `image_id` and `scale_id` in the CSV resolves to an actual file

### "ValueError: could not convert string to float"

**Cause**: Invalid scale value in metadata
**Solution**: Make sure the `scale` column contains only numbers (PyLithics will skip rows it can't parse)

### "MemoryError"

**Cause**: Insufficient RAM for very large images
**Solution**: Reduce image size or process fewer images at a time

### "ImportError: No module named 'cv2'"

**Cause**: OpenCV not installed
**Solution**: `pip install opencv-python-headless`

### "yaml.scanner.ScannerError"

**Cause**: Invalid YAML syntax in config file
**Solution**: Check indentation (spaces, not tabs) and quoting

## Diagnostic Commands

### System check

```bash
python --version
pip --version
pylithics --help

# Verify dependencies
python -c "import cv2, numpy, pandas; print('Dependencies OK')"

# Test with the bundled sample data
pylithics --data_dir pylithics/data --meta_file pylithics/data/meta_data.csv
```

### Maximum debug output

```bash
pylithics --data_dir ./data --meta_file ./meta.csv \
    --log_level DEBUG \
    --show_thresholded_images \
    --arrow_debug \
    --scale_debug
```

## Getting Help

When reporting an issue, include:

1. **PyLithics version** (note the release tag in your install)
2. **Python version** (`python --version`)
3. **Operating system** and version
4. **The exact command you ran**
5. **The full error message**
6. **The contents of `pylithics.log`**
7. **A small sample dataset** that reproduces the problem, if possible

Open issues at <https://github.com/alan-turing-institute/Palaeoanalytics/issues>.
