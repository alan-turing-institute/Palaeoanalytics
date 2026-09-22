# Metadata Setup

## Overview

The metadata CSV file connects each lithic image to its scale image.
PyLithics uses it for the scale calibration. The calibration finds and
measures the scale bar, and changes the measurements from pixels to
millimetres.

!!! danger "The project folder rule"
    `--data_dir` is a folder that contains these three items, with
    these exact names:

    ```
    <your folder>/
    ├── images/          the lithic images
    ├── scales/          the scale bar images
    └── meta_data.csv    which scale goes with which image
    ```

    The folder can have any name and can be anywhere. The three names
    inside it cannot change. PyLithics writes the analysis to a fourth
    item, `results/`, in the same folder.

## The CSV file

The metadata CSV file has these columns:

| Column | Description | Necessary | Example |
|--------|-------------|-----------|---------|
| `image_id` | The filename of the lithic image | Yes | `artifact_001.png` |
| `scale_id` | The filename of the scale image | No* | `scale_001.png` |
| `scale` | The length of the scale bar, in millimetres | No* | `50` |
| `flag` | Written by `pylithics-pages`. A note for you to examine. It does not stop the analysis. | No | `several_scales` |

*Necessary for scale bar calibration. Not necessary for pixel
measurements. A row with no `scale` value can only be measured in
pixels. `pylithics` asks once before it does that.

Name the file `meta_data.csv` and keep it in the project folder. Then
`pylithics --data_dir <project>` finds it. A file with another name or
place needs `--meta_file`.

`pylithics-pages` writes this file for you when you start from
published plates, with the `flag` column filled in. See
[meta_data.csv](page-segmentation.md#meta_datacsv).

## Calibration methods

PyLithics has two calibration methods:

### 1. Scale bar detection (recommended)
- **Method**: PyLithics finds and measures the scale bar in the scale image
- **Necessary**: the `scale_id` and `scale` columns
- **Scale bar styles**: horizontal and vertical bars, segmented bars, bars with tick marks
- **Accuracy**: the best accuracy for measurements in millimetres

### 2. Pixel measurements
- **Method**: the measurements are in pixels, with no calibration
- **Necessary**: nothing. This method always operates
- **Accuracy**: relative measurements only, with no real-world units

!!! info "Why there is no DPI calibration"
    DPI metadata is not reliable. A scanner does not always scan at
    the DPI it shows. The value can be an estimate. There is no way to
    make sure that it is correct. PyLithics uses a measured scale bar,
    or clear pixel measurements.

## Scales and images

!!! warning "Necessary for accuracy"
    Scan a scale image at the same DPI as its lithic image. If the DPI
    of the scale and the DPI of the image are different, the
    measurements are wrong.

### One scale for many images

One scale image can serve many artefacts, if the artefacts were all
drawn at the same scale:

```csv
image_id,scale_id,scale
flake_001.png,scale_50.png,50
flake_002.png,scale_50.png,50
flake_003.png,scale_50.png,50
```

### One scale for each image

Each artefact can have its own scale:

```csv
image_id,scale_id,scale
large_biface.png,scale_50.png,50
small_flake.png,scale_5.png,5
medium_core.png,scale_20.png,20
```

### Mixed calibration methods

You can mix the calibration methods in one data set:

```csv
image_id,scale_id,scale
artifact_001.png,scale_10.png,10    # Scale bar detection
artifact_002.png,,                  # Pixel measurements (empty scale columns)
artifact_003.png,scale_10.png,10    # Scale bar detection
artifact_004.png,,                  # Pixel measurements
```

### Scale bar styles

PyLithics finds these scale bar styles:

- **Simple horizontal and vertical lines**
- **Segmented scale bars** (black and white segments)
- **Scale bars with tick marks**
- **Scale bars with brackets or end marks**

!!! tip "Scale bars"
    - Make sure that the scale bar is clear, with high contrast
    - A black scale bar on a white background is best
    - PyLithics measures the longest dimension (horizontal or vertical)
    - Examine the result for a scale bar with an unusual design

## The directory structure

### The standard structure

```
my_project/                   # --data_dir
├── meta_data.csv             # Your metadata file
├── images/                   # The lithic illustrations
│   ├── artifact_001.png
│   ├── artifact_002.png
│   └── artifact_003.png
├── scales/                   # The scale bar images
│   ├── scale_001.png
│   └── scale_002.png
└── results/                  # pylithics writes the analysis here
```

### Filenames

Good filenames operate on all operating systems and prevent errors:

✅ **Good**:

- `lithic_001.png`
- `artifact_A1.png`
- `flake_site1_layer2.png`

**Why these are good:**

- They operate on Windows, Mac and Linux
- They go in a CSV file without quotation marks
- They sort correctly in a file browser
- They cause no problems on the command line

❌ **Not good**:

- Spaces: `artifact 001.png` → can cause errors in the CSV and on the command line
- Special characters: `artifact#1.png` → the shell can read them as comments or commands
- Very long names: `artifact_from_excavation_unit_4_level_3_find_number_127.png` → can be longer than the system limit

**Note**: These names do not stop PyLithics. But they can cause
unexpected results, or make quotation marks necessary in commands, or
cause confusion when you find a problem.

!!! tip "Recommended"
    Use underscores, not spaces. Keep the names short but clear. Use
    sequential numbers, so that the files sort correctly.

### CSV encoding

!!! warning "Excel CSV UTF-8"
    **Do not keep a CSV file as "CSV UTF-8" from Excel.** This format
    adds a Byte Order Mark (BOM), a character that you cannot see. The
    BOM prevents PyLithics from reading the column headers.

**The error**: `Column missing in the metadata: image_id` (when the column is there)

**Procedure**:

1. **Excel**: keep the file as "CSV (Comma delimited) (*.csv)", not "CSV UTF-8"
2. **Remove the BOM** from a UTF-8 CSV file:
   ```bash
   # Mac/Linux
   sed -i.bak '1s/^\xEF\xBB\xBF//' your_metadata.csv

   # Windows (PowerShell)
   (Get-Content your_metadata.csv -Raw) -replace '﻿', '' | Set-Content your_metadata.csv
   ```
3. **Text editors**: keep CSV files with VS Code, Sublime Text or a similar editor
4. **Examine the encoding**: make sure that the CSV file starts with `image_id`, not `﻿image_id`

## The scale value

### The `scale` column

The `scale` value is the real length of the scale bar, in millimetres.

Examples:

- A scale bar of 1 cm → `scale: 10`
- A scale bar of 5 cm → `scale: 50`
- A scale bar of 2 cm → `scale: 20`

### A scale bar with no value

If the value of the scale bar is not printed:

1. Find a known measurement in the publication
2. Measure that dimension on the artefact
3. Calculate the scale value from the ratio
4. Make sure with more than one measurement

### No scale image

PyLithics operates without scale information:

#### Pixel measurements
If there is no scale bar, PyLithics uses pixel measurements:

```csv
image_id,scale_id,scale
artifact_001.png,,     # Empty scale columns: pixel measurements
artifact_002.png,,     # Empty scale columns: pixel measurements
```

#### Pixel measurements for all images
To set all calibration off and use pixel measurements:

```bash
# No calibration
pylithics --data_dir ./artifacts --disable_scale_calibration
```

!!! warning "Pixel measurements"
    Without scale calibration, all measurements are in pixels. You
    cannot compare images from different sources. You cannot read the
    measurements in real-world units.

!!! note "The calibration method in the output"
    PyLithics writes the calibration method of each image in the
    output CSV (the column `calibration_method`: `scale_bar` or
    `pixels`). Use it to make sure that the measurements are correct.

    The pipeline has **three** calibration states. It writes them as
    **two** values in the CSV column:

    | Internal state              | CSV value    | Meaning                                                                                          |
    | --------------------------- | ------------ | ------------------------------------------------------------------------------------------------ |
    | `scale_bar`                 | `scale_bar`  | A scale image was given and the scale bar was found. The measurements are in millimetres.        |
    | `pixels_no_scale`           | `pixels`     | No scale image was given (or `--force_pixels` or `--disable_scale_calibration` was used).         |
    | `pixels_detection_failed`   | `pixels`     | A scale image was given but PyLithics did not find the scale bar. The measurements are in pixels. The log gives the reason as a `WARNING`. |

    If the CSV shows `pixels` for a lithic that has a scale, read the
    `WARNING` in the log to find which of the two pixel cases applies.

## Next steps

When your metadata file is ready:

1. [Basic usage](basic-usage.md) — start your first analysis
2. [Configuration](basic-usage.md#configuration) — set the analysis parameters
3. [Outputs](outputs.md) — read the results
