# Metadata Setup

## Overview

The metadata CSV file is essential for linking your lithic images to their scale references. PyLithics uses this information for its automatic scale calibration system, which detects and measures scale bars to convert pixel measurements to real-world units (millimeters).

## CSV File Structure

Your metadata CSV must contain these three columns:

| Column | Description | Required | Example |
|--------|-------------|----------|---------|
| `image_id` | Filename of the lithic image | Yes | `artifact_001.png` |
| `scale_id` | Filename of the scale image | No* | `scale_001.png` |
| `scale` | Scale measurement in millimeters | No* | `50` |

*Required for scale bar calibration. Optional if using pixel measurements only.

## Scale Calibration Methods

PyLithics uses a simple two-option calibration system:

### 1. Scale Bar Detection (Recommended)
- **How it works**: Computer vision automatically detects and measures scale bars in scale images
- **Requirements**: `scale_id` and `scale` columns must be provided
- **Supported formats**: Horizontal/vertical bars, segmented bars, bars with tick marks
- **Accuracy**: Highest precision for real-world measurements

### 2. Pixel Measurements (Fallback)
- **How it works**: Raw pixel measurements when no scale calibration is available
- **Requirements**: None - always works
- **Accuracy**: Relative measurements only, no real-world units

!!! info "Why No DPI Fallback?"
    DPI metadata is unreliable because scanners often don't scan at exact DPI settings, values can be estimated rather than measured, and there's no way to verify accuracy. PyLithics focuses on either precise scale bar measurement or clear pixel-based relative measurements.

## Understanding Scale Relationships

!!! warning "Critical for Accuracy"
    Scale images must be scanned at the same DPI as their corresponding lithic images. Mismatched DPI between scales and images will lead to incorrect measurements and compromise your analysis results.

### One Scale, Multiple Images

A single scale image can be used for multiple artifacts if they were all drawn at the same scale:

```csv
image_id,scale_id,scale
flake_001.png,scale_50.png,50
flake_002.png,scale_50.png,50
flake_003.png,scale_50.png,50
```

### Individual Scales

Each artifact can have its own scale if needed:

```csv
image_id,scale_id,scale
large_biface.png,scale_50.png,50
small_flake.png,scale_5.png,5
medium_core.png,scale_20.png,20
```

### Mixed Calibration Methods

You can mix calibration methods within a single dataset:

```csv
image_id,scale_id,scale
artifact_001.png,scale_10.png,10    # Scale bar detection
artifact_002.png,,                  # Pixel measurements (empty scale columns)
artifact_003.png,scale_10.png,10    # Scale bar detection
artifact_004.png,,                  # Pixel measurements
```

### Scale Bar Detection Examples

PyLithics can detect various scale bar styles:

- **Simple horizontal/vertical lines**
- **Segmented scale bars** (alternating black/white segments)
- **Scale bars with tick marks**
- **Scale bars with brackets or end markers**

!!! tip "Scale Bar Tips"
    - Ensure scale bars are clearly visible with good contrast
    - Black scale bars on white backgrounds work best
    - PyLithics measures the longest dimension (horizontal or vertical)
    - Complex scale bar designs may require manual verification

## Directory Organization

### Standard Structure

```
pylithics/
└── data/
    ├── meta_data.csv         # Your metadata file
    ├── images/               # Lithic illustrations for analysis
    │   ├── artifact_001.png
    │   ├── artifact_002.png
    │   └── artifact_003.png
    └── scales/               # Scale bar images
        ├── scale_001.png
        └── scale_002.png
```

### File Naming Conventions

Proper file naming ensures compatibility across different operating systems and prevents processing errors:

✅ **Good Naming**:

- `lithic_001.png`
- `artifact_A1.png`
- `flake_site1_layer2.png`

**Why these work well:**

- Compatible with all operating systems (Windows, Mac, Linux)
- Easy to reference in CSV files without escaping
- Sort properly in file browsers
- Prevent command-line issues

❌ **Avoid**:

- Spaces: `artifact 001.png` → Can cause parsing errors in CSV and command-line
- Special characters: `artifact#1.png` → May be interpreted as comments or commands
- Very long names: `artifact_from_excavation_unit_4_level_3_find_number_127.png` → Can exceed system path limits

**Note**: While these naming issues won't stop PyLithics from running, they may cause unexpected behavior, require extra quoting in commands, or create confusion when debugging. Following good naming conventions ensures smooth, predictable processing.

!!! tip "Best Practice"
    Use underscores instead of spaces, keep names descriptive but concise, and include sequential numbering for easy sorting.

### CSV File Encoding Issues

!!! warning "Excel CSV UTF-8 Problem"
    **Avoid saving CSV files as "CSV UTF-8" from Excel** - this adds an invisible Byte Order Mark (BOM) character that prevents PyLithics from recognizing column headers.

**Common Error**: `Missing required column in metadata: image_id` (even when the column exists)

**Solutions**:

1. **Excel Users**: Save as "CSV (Comma delimited) (*.csv)" instead of "CSV UTF-8"
2. **Remove BOM**: If you already have a UTF-8 CSV with BOM, remove it:
   ```bash
   # On Mac/Linux
   sed -i.bak '1s/^\xEF\xBB\xBF//' your_metadata.csv

   # On Windows (PowerShell)
   (Get-Content your_metadata.csv -Raw) -replace '\ufeff', '' | Set-Content your_metadata.csv
   ```
3. **Use Text Editors**: Save CSV files with VS Code, Sublime Text, or similar editors
4. **Verify Encoding**: Check that your CSV file starts directly with `image_id`, not `﻿image_id`

## Scale Value Determination

### Understanding the Scale Column

The `scale` value represents how many millimeters the scale bar represents in the real world.

Examples:

- A 1cm scale bar → `scale: 10`
- A 5cm scale bar → `scale: 50`
- A 2cm scale bar → `scale: 20`

### Measuring Unknown Scales

If the scale value is not labeled:

1. Identify any reference measurements in the publication
2. Measure a known dimension on the artifact
3. Use the ratio to calculate the scale value
4. Verify with multiple measurements

### Missing Scale Images

PyLithics can handle missing scale information:

#### Automatic Pixel Fallback
If scale bars aren't available, PyLithics automatically uses pixel measurements:

```csv
image_id,scale_id,scale
artifact_001.png,,     # Empty scale columns - will use pixels
artifact_002.png,,     # Empty scale columns - will use pixels
```

#### Force Pixel Measurements
To disable all calibration and use pixel measurements:

```bash
# Run without any calibration
pylithics --data_dir ./artifacts --meta_file ./metadata.csv --disable_scale_calibration
```

!!! warning "Pixel-Only Measurements"
    Without scale calibration, all measurements will be in pixels. This limits comparative analysis between different image sources and prevents real-world metric interpretation.

!!! note "Calibration Method Tracking"
    PyLithics automatically tracks which calibration method was used for each image in the output CSV (`calibration_method` column: either "scale_bar" or "pixels"), allowing you to validate measurement accuracy and identify potential issues.

## Next Steps

With your metadata file prepared:

1. [Learn basic usage](basic-usage.md) - Run your first analysis
2. [Configure settings](basic-usage.md#configuration-options) - Customize processing
3. [Understand outputs](outputs.md) - Interpret results