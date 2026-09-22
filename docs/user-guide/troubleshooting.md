# Troubleshooting

This page gives the most common problems with PyLithics. If your
problem is not here, read `pylithics.log` in the output directory to
find the error. Then [open a GitHub
issue](https://github.com/alan-turing-institute/Palaeoanalytics/issues)
with the lines from the log.

## Installation

### Python version error

**Problem**: "Python 3.8+ required", or a compatibility error

**Procedure**:

```bash
# Show your Python version
python --version

# Install the correct Python version
# macOS (with Homebrew)
brew install python@3.11

# Ubuntu/Debian
sudo apt-get install python3.11

# Windows: get Python from python.org
```

### Package installation error

**Problem**: `pip install .` stops with a dependency error

**Procedure**:

```bash
# Update pip first
pip install --upgrade pip

# Install with full output to see the cause
pip install . -v

# If OpenCV is the cause, install it first
pip install opencv-python-headless>=4.8.0
pip install .
```

### Virtual environment

**Problem**: The virtual environment does not activate, or the installed packages are not found

**Procedure**:

```bash
# Make sure that the environment is active
which python  # The path must be in your virtual environment

# If the environment is damaged, make it again
deactivate
rm -rf palaeo/
python3 -m venv palaeo
source palaeo/bin/activate
pip install .
```

## Image analysis

### No contours found

**Problem**: "No contours detected", or empty results

**Find the cause**:

```bash
# --verbose shows the full trace for each step on the screen.
# The same trace is always in <data_dir>/results/pylithics.log
# (the log is next to the CSV for the --data_dir that you used).
# You can also start the command without --verbose and search the log
# file after.
pylithics --data_dir ./data \
    --verbose --threshold_debug
```

**Procedure**:

```bash
# Use Otsu thresholding (good for images with two clear tones)
pylithics --data_dir ./data --threshold_method otsu

# Use adaptive thresholding for low or uneven contrast
pylithics --data_dir ./data --threshold_method adaptive
```

### Contours are not correct

**Problem**: The contours are not complete, or not accurate

**Make sure that the image has**:

- High contrast (black lines on a white background)
- A resolution of 300 DPI or more
- No scan noise
- Closed outlines

**Configuration** (in your `config.yaml`):

```yaml
thresholding:
  method: adaptive

morphological_closing:
  enabled: true
  kernel_size: 3
```

### Scale calculation error

**Problem**: The measurements are much too large or much too small

**Examine your metadata CSV**:

```csv
image_id,scale_id,scale
artifact_001.png,scale_001.png,10
```

The `scale` column is in **millimetres**. Common errors:

- The scale value is in centimetres, not millimetres
- The wrong scale image is connected to the artefact
- `scale_id` is missing or empty

You can also use pixel measurements and no calibration:

```bash
pylithics --data_dir ./data --force_pixels
```

## Configuration

### The configuration file has no effect

**Problem**: Changes in the configuration file have no effect

**Procedure**:

```bash
# Show the path that PyLithics uses
pylithics --data_dir ./data \
    --config_file ./config.yaml --log_level DEBUG

# Examine the YAML syntax
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Use absolute paths if relative paths cause a problem in the shell
pylithics --data_dir "$(pwd)/data" \
    --config_file "$(pwd)/config.yaml"
```

### Command-line arguments have no effect

**Problem**: The command-line arguments have no effect

**Procedure**:

```bash
# Show all flags
pylithics --help

# Use debug logging to see which settings were applied
pylithics --data_dir ./data --log_level DEBUG
```

Command-line arguments replace the YAML values. YAML values replace the
defaults. Compare the spelling of each flag with `pylithics --help`.

## Analysis modules

### Arrow detection

**Problem**: Arrows are not found, or marks that are not arrows are found

**Write the debug output**:

```bash
pylithics --data_dir ./data \
    --arrow_debug --log_level DEBUG
```

**Examine the debug output** in `results/arrow_debug/<image>/`. Each scar has a `.png` with the arrow found and a `.txt` with the steps.

**Adjust the settings in `config.yaml`**:

```yaml
arrow_detection:
  enabled: true
  reference_dpi: 300.0
  min_area_scale_factor: 0.5
  min_defect_depth_scale_factor: 0.7
```

### Cortex detection

**Problem**: Too few or too many cortex areas are found

```bash
# Increase the sensitivity
pylithics --data_dir ./data --cortex_sensitivity high

# Decrease the sensitivity
pylithics --data_dir ./data --cortex_sensitivity low

# Set cortex detection off
pylithics --data_dir ./data --disable_cortex_detection
```

### The analysis is slow

**Problem**: The analysis takes too long

```bash
# Set arrow detection off. It is the slowest optional step.
pylithics --data_dir ./data --disable_arrow_detection

# Use simple thresholding, not adaptive
pylithics --data_dir ./data --threshold_method simple

# Set more than one optional analysis off
pylithics --data_dir ./data \
    --disable_arrow_detection \
    --disable_cortex_detection \
    --disable_scar_complexity
```

### Voronoi analysis

**Problem**: No Voronoi diagrams

A Voronoi diagram is only possible with a Dorsal surface. Make sure
that:

- The surface classification gives a `Dorsal` row
- The Dorsal surface has at least one scar
- `voronoi_analysis.enabled` is `true` in `config.yaml` (the default)

There is no command-line switch for Voronoi analysis. Use the
configuration file.

## Data

### Output files are missing

**Problem**: The output files are not there

**Make sure that you can write to the directory** (use your own `--data_dir`):

```bash
ls -la <data_dir>/results/
```

**Read the log**:

```bash
tail -50 <data_dir>/results/pylithics.log
grep ERROR <data_dir>/results/pylithics.log
```

### Measurements are not plausible

**Problem**: The measurements do not agree with what you expect

1. Make sure that the `scale` column in the metadata is in millimetres
2. Examine the labelled image to make sure that the correct contour was found
3. Compare the measurements with the known measurements in the publication
4. Examine the CSV column `calibration_method`. `pixels` means that no change to millimetres was applied

```python
import pandas as pd

df = pd.read_csv('<data_dir>/results/processed_metrics.csv')

print("Length range:", df['technical_length'].min(), "-", df['technical_length'].max())
print("Area range:", df['total_area'].min(), "-", df['total_area'].max())
print("Calibration methods:", df['calibration_method'].value_counts())
```

## Operating systems

### macOS

**OpenCV does not install**:

```bash
# Use the headless build (no GUI)
pip install opencv-python-headless

# Or install with conda
conda install opencv
```

### Windows

**PowerShell execution policy**:

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Path length limit**:

- Use shorter directory paths
- Move the project nearer to the root of the drive
- Set Windows long-path support on

### Linux

**Missing system dependencies**:

```bash
# Ubuntu/Debian
sudo apt-get install python3-dev libopencv-dev

# CentOS/RHEL
sudo yum install python3-devel opencv-devel
```

## Error messages

### "images have no scale value. Measure them in pixels?"

**Cause**: Some rows in `meta_data.csv` have no `scale` value. Those images can only be measured in pixels
**Procedure**: Answer `y` to analyse all images, in pixels where there is no scale. Answer `n` to analyse only the images with a scale. Then fill in the `scale` column and start the command again. See [Basic Usage](basic-usage.md#rows-with-no-scale-and-rows-with-a-flag)

### "images with no scale value not analysed"

**Cause**: You answered `n` to the question above
**Procedure**: Fill in the `scale` column of `meta_data.csv` for those images. `run_summary.json` lists them. Start the command again

### "row(s) in the metadata have a flag"

**Cause**: `pylithics-pages` put a flag on those rows in `meta_data.csv`. The images were analysed
**Procedure**: Examine each flagged row (the log names them). Fill in the scale, correct the row, remove the flag. See [meta_data.csv](page-segmentation.md#meta_datacsv)

### "FileNotFoundError"

**Cause**: An image or a scale file in the metadata is missing
**Procedure**: Make sure that each `image_id` and each `scale_id` in the CSV is the name of a file that is there

### "ValueError: could not convert string to float"

**Cause**: A scale value in the metadata is not a number
**Procedure**: Make sure that the `scale` column contains only numbers. PyLithics ignores a row that it cannot read

### "MemoryError"

**Cause**: Not enough RAM for very large images
**Procedure**: Make the images smaller, or analyse fewer images at one time

### "ImportError: No module named 'cv2'"

**Cause**: OpenCV is not installed
**Procedure**: `pip install opencv-python-headless`

### "yaml.scanner.ScannerError"

**Cause**: The YAML syntax in the configuration file is not correct
**Procedure**: Examine the indentation (spaces, not tabs) and the quotation marks

## Diagnostic commands

### System

```bash
python --version
pip --version
pylithics --help

# Make sure that the dependencies are installed
python -c "import cv2, numpy, pandas; print('Dependencies OK')"

# Analyse the sample data
pylithics --data_dir pylithics/data
```

### Maximum debug output

```bash
pylithics --data_dir ./data \
    --log_level DEBUG \
    --threshold_debug \
    --arrow_debug \
    --scale_debug
```

Each flag writes to its own folder in `results/`, and the run names
the folders at the end. See [Debug output](outputs.md#debug-output).

## Get help

When you report a problem, include:

1. **The PyLithics version** (the release tag of your installation)
2. **The Python version** (`python --version`)
3. **The operating system** and its version
4. **The exact command that you used**
5. **The full error message**
6. **The contents of `pylithics.log`**
7. **A small data set** that shows the problem, if possible

Open an issue at <https://github.com/alan-turing-institute/Palaeoanalytics/issues>.
