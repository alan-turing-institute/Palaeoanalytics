# Troubleshooting

This page gives the most common problems with PyLithics. If your
problem is not here, read `pylithics.log` in the results folder to
find the error. Then [open a GitHub
issue](https://github.com/alan-turing-institute/Palaeoanalytics/issues)
with the lines from the log.

## Installation

### Python version error

**Problem**: "Python 3.10+ required", or a compatibility error

**Procedure**:

```bash
# Show your Python version
python --version

# Install the correct Python version
# macOS (with Homebrew)
brew install python@3.12

# Ubuntu/Debian
sudo apt-get install python3.12

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

# If OpenCV is the cause, install the contrib build first.
# Only the contrib build has the upscaling module (dnn_superres).
pip install "opencv-contrib-python-headless>=4.8.0"
pip install .
```

### Virtual environment

**Problem**: The virtual environment does not activate, or Python does not find the installed packages

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
- The metadata connects the wrong scale image to the artefact
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

**Problem**: PyLithics does not find arrows, or it finds marks that are not arrows

**Write the debug files**:

```bash
pylithics --data_dir ./data \
    --arrow_debug --log_level DEBUG
```

**Examine the debug files** in `results/arrow_debug/<image>/`. Each mark that PyLithics tried has a `.png` and a `.txt`. The file name gives the scar and the number of the mark in that scar, for example `scar_3_mark_1.png`. The `.png` shows the scar at the working resolution with the direction found. The direction is green when the mark is an arrow and orange when it is not. The `.txt` shows the tests of stage 1, the stage that decided, the scores and the measurements. The stage that decided is `triangle`, `triangle turned` (when the model turned the direction), `model`, `ink inside`, `too short` or `no model`.

**Adjust the settings in `config.yaml`**:

```yaml
arrow_detection:
  enabled: true
  min_defect_depth_scale_factor: 0.8  # Decrease the value to let stage 1 accept shallower notches.
  classifier:
    threshold: 0.5  # Decrease the value to find more arrows with stage 2. Increase it to find fewer marks that are not arrows.
```

PyLithics tries only a mark inside a recognised scar. A mark that touches the outline of its scar is part of the outline, and PyLithics does not find it. PyLithics does not recognise a scar with a broken outline, so it does not find the arrows in it. The scars of a ventral or platform surface have no row in the results, so PyLithics does not find the arrows in them.

### Cortex detection

**Problem**: PyLithics finds too few or too many cortex areas

```bash
# Increase the sensitivity
pylithics --data_dir ./data --cortex_sensitivity high

# Decrease the sensitivity
pylithics --data_dir ./data --cortex_sensitivity low

# Set cortex detection off
pylithics --data_dir ./data --disable_cortex_detection
```

**Problem**: PyLithics marks a scar that holds an arrow or a numeral as cortex, and does not find its arrow

**Cause**: Cortex is stippling, many small pieces of ink. A scar is cortex only when it holds at least `min_stipple_pieces` of them (default 3). A larger value makes the test stricter
**Procedure**: `pylithics --data_dir ./data --min_stipple_pieces 4`, or set `cortex_detection.min_stipple_pieces` in `config.yaml`

**Problem**: A scar with the letter C is not cortex, or PyLithics incorrectly marks a scar with a letter as cortex

**Cause**: Some illustrators mark a cortex scar with the letter C. PyLithics reads each small mark inside a scar with RapidOCR. A scar is cortex when a mark reads as one of the letters in `cortex_detection.letter.letters` (default `C`) at a confidence of 0.8 or more. The column `cortex_source` says `letter` for these scars. Without RapidOCR, PyLithics does not read the letter, and the log tells you this one time
**Procedure**:

1. To read the letter, install the reader: `pip install '.[ocr]'`.
2. To add a letter that your illustrator used, set `letters: ["C", "K"]` under `cortex_detection.letter` in `config.yaml`.
3. To stop the letter test, run `pylithics --data_dir ./data --no_cortex_letter`.

A letter on a platform, lateral or ventral view, in no scar, makes the whole view cortex. A letter on the dorsal view outside every scar does not count.

**Problem**: A thin rectangle around a letter is a scar in the results

**Cause**: Some illustrators draw a thin rectangle around a label. The rectangle is a closed outline. PyLithics removes a thin rectangle that holds one letter before it finds the scars. A rectangle with lines more than half as thick as the drawing lines stays, because PyLithics reads it as a scar. The log gives each rectangle that PyLithics removed.
**Procedure**:

1. To keep all rectangles, run `pylithics --data_dir ./data --keep_frames`.
2. To remove rectangles with thicker lines, set a larger `max_line_ratio` under `cortex_detection.frames` in `config.yaml`.

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
    --disable_scar_complexity \
    --disable_symmetry_analysis \
    --disable_voronoi_analysis
```

### Voronoi analysis

**Problem**: No Voronoi diagrams

A Voronoi diagram is only possible with a Dorsal surface. Make sure
that:

- The surface classification gives a `Dorsal` row
- The Dorsal surface has at least one scar
- `voronoi_analysis.enabled` is `true` in `config.yaml` (the default)
- The run did not use `--disable_voronoi_analysis`

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
2. Examine the labelled image to make sure that PyLithics found the correct contour
3. Compare the measurements with the known measurements in the publication
4. Examine the CSV column `calibration_method`. `pixels` means that PyLithics did not change the measurements to millimetres

```python
import pandas as pd

df = pd.read_csv('<data_dir>/results/processed_metrics.csv')

print("Length range:", df['technical_length'].min(), "-", df['technical_length'].max())
print("Area range:", df['total_area'].min(), "-", df['total_area'].max())
print("Calibration methods:", df['calibration_method'].value_counts())
```

## Dashboard review checks

The dashboard's Review page and `results/review.csv` use these checks. Each row gives the reason in numbers.

### Failed

**Cause**: Images listed in meta_data.csv that gave no metrics.

**Procedure**:

1. Open pylithics.log in the results folder and find the image.
2. Correct the image, or its row in meta_data.csv.
3. Run the analysis again.

### Unclassified surfaces

**Cause**: PyLithics cannot name these outlines as a dorsal, ventral, platform or lateral surface from their size. A ventral surface must be within 10 percent of the dorsal surface in length, width and area (the tolerance setting).

**Procedure**:

1. Open the labelled image of the lithic. Find the outline with the label Unclassified.
2. If the outline is a view of the lithic, its size in the drawing does not agree with the dorsal view. You cannot correct this in PyLithics. Use the result as it is, or draw the views again to agree with the size rules in [Image Requirements: Views and their sizes](image-requirements.md#views-and-their-sizes).
3. If the outline is not part of the lithic, for example text, a number or a mark, remove it from the image. Then run the analysis again.

### Zero-scar lithics

**Cause**: Lithics whose dorsal face holds no scar.

**Procedure**:

1. Open the labelled image of the lithic.
2. If the dorsal face is plain in the drawing, the result is correct.
3. If scar lines are broken or missing, see Troubleshooting for the threshold settings.

### Low-confidence scales

**Cause**: Scale bars with a confidence below 0.8. Without a scale_box the confidence is the length of the bar divided by its thickness, divided by 10. A short, thick bar gets a low value even when it is measured correctly. A bar measured from a scale_box gets 0.95.

**Procedure**:

1. Open the scale image named in scale_id.
2. Make sure that the measured length is the bar alone, with no caption and no frame.
3. If it is, the value stands. To record the box of the bar, cut the pages again with pylithics-pages.

### Pixel-only

**Cause**: Lithics measured in pixels, not millimetres.

**Procedure**:

1. Open meta_data.csv and find the row of the image.
2. Type the printed length of the bar, in mm, in the scale column.
3. Make sure that the scale image named in scale_id is in scales/.
4. Run the analysis again.

### Mixed DPI

**Cause**: Images with a DPI that is different from most of the batch. You cannot compare pixel measurements from images with different DPI values.

**Procedure**:

1. Scan the listed images again at the DPI of the batch.
2. As an alternative, calibrate each image with a scale bar.

### Missing DPI

**Cause**: Images whose files have no DPI tag. PyLithics uses the default settings for them. Thus the sizes that use the DPI are less accurate.

**Procedure**:

1. Save the image again with its DPI.
2. As an alternative, calibrate the image with a scale bar.

## Operating systems

### macOS

**OpenCV does not install**:

```bash
# Use the headless contrib build (no GUI, with the upscaling module)
pip install opencv-contrib-python-headless
```

Do not install `opencv-python` or `opencv-python-headless` beside it.
Those builds have no `dnn_superres` module, and the upscaler stops.

**PyLithics is slow on an Apple Silicon Mac**:

An Intel build of Python runs on an Apple Silicon Mac through Rosetta,
and the upscaler is three times slower. Make sure that Python is
native:

```bash
# Must print arm64. If it prints x86_64, install a native Python
# (for example with Homebrew) and make the virtual environment again.
python -c "import platform; print(platform.machine())"
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

### "rows in meta_data.csv have no value in the scale column. Measure them in pixels?"

**Cause**: Some rows in `meta_data.csv` have no value in the `scale` column. That column holds the length of the scale bar in millimetres. You type this value. The `scale_id` column names the scale image. It is not sufficient alone. PyLithics can measure those images only in pixels
**Procedure**: Answer `y` to analyse all images, in pixels where there is no scale. Answer `n` to analyse only the images with a scale. Then type the values in the `scale` column and start the command again. See [Basic Usage](basic-usage.md#rows-with-no-scale-and-rows-with-a-flag)

### "rows with no value in the scale column are not analysed"

**Cause**: You answered `n` to the question above
**Procedure**: Type the values in the `scale` column of `meta_data.csv` for those images. `run_summary.json` lists them. Start the command again

### "PyLithics vN.N.N is available"

**Cause**: A newer release is on GitHub, and your clone is on the `main` branch
**Procedure**: Answer `y` to update, or `n` to continue. To stop the check, set `update_check.enabled: false` in `config.yaml` or set `PYLITHICS_NO_UPDATE_CHECK`. See [Update PyLithics](../installation.md#update-pylithics)

### "row(s) in the metadata have a flag"

**Cause**: `pylithics-pages` put a flag on those rows in `meta_data.csv`. PyLithics analysed the images
**Procedure**: Examine each row that has a flag (the log names them). Type the scale. Correct the row. Remove the flag. See [meta_data.csv](page-segmentation.md#meta_datacsv)

### "FileNotFoundError"

**Cause**: An image or a scale file in the metadata is missing
**Procedure**: Make sure that each `image_id` and each `scale_id` in the CSV is the name of a file that is there

### "the folder ... does not exist"

**Cause**: The folder that you gave to `--data_dir` is not there. Both commands stop before they write anything. A frequent cause is a space at the end of the folder name, which the Finder does not show
**Procedure**: Compare the name in the message with the name of the folder. Correct the name, or rename the folder. Then start the command again

### "No page images in ..."

**Cause**: `pylithics-pages` found no page image and no PDF file in the folder. The run completes with no crops. The pages must be in a `pages/` folder inside the project folder
**Procedure**: Put the plates in `<project>/pages/`. The command reads these formats: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp` and PDF. Then start the command again

### "ValueError: could not convert string to float"

**Cause**: A scale value in the metadata is not a number
**Procedure**: Make sure that the `scale` column contains only numbers. PyLithics ignores a row that it cannot read

### "MemoryError"

**Cause**: Not enough RAM for very large images
**Procedure**: Make the images smaller, or analyse fewer images at one time

### "ImportError: No module named 'cv2'"

**Cause**: The installation does not include OpenCV
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
