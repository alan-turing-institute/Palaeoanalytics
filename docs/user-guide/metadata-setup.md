# Metadata Setup

## Overview

For accurate measurement the metadata CSV file is essential for linking your lithic images to their corresponding scale references. This file tells PyLithics how to convert pixel measurements to real-world units (millimeters).

## CSV File Structure

Your metadata CSV must contain these three columns:

| Column | Description | Example |
|--------|-------------|------|
| `image_id` | Filename of the lithic image | `artifact_001.png` |
| `scale_id` | Filename of the scale image | `scale_001.png` |
| `scale` | Scale measurement in millimeters | `50` |

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

If no scale images are available, PyLithics can process in pixel units:

```bash
# Run without scale information
pylithics --data_dir ./artifacts --meta_file ./metadata_no_scales.csv
```

!!! warning "Pixel-Only Measurements"
    Without scales, all measurements will be in pixels. This limits comparative analysis between different image sources.

## Next Steps

With your metadata file prepared:

1. [Learn basic usage](basic-usage.md) - Run your first analysis
2. [Configure settings](basic-usage.md#configuration) - Customize processing
3. [Understand outputs](outputs.md) - Interpret results