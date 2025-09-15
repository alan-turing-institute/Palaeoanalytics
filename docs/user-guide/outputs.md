# PyLithics Outputs

## Overview

PyLithics generates multiple types of output files to help you analyze and validate your results. This page describes all the files created during processing and how to interpret them.

## Output Directory Structure

After running PyLithics, your output directory will contain:

```
processed/
├── measurements.csv           # Main data output
├── pylithics.log              # Processing log
├── labeled_images/            # Annotated visualizations
│   ├── artifact_001_labeled.png
│   └── artifact_002_labeled.png
├── voronoi_diagrams/          # Spatial analysis
│   ├── artifact_001_voronoi.png
│   └── artifact_002_voronoi.png
└── arrow_debug/               # Debug output (if enabled)
    ├── artifact_001_arrows.png
    └── artifact_002_arrows.png
```

## Primary Data Output

### measurements.csv

The main output file containing all quantitative measurements in CSV format.

#### Data Structure

The CSV is hierarchically organized:
- **Surfaces**: Dorsal, Ventral, Platform, Lateral
- **Features**: Individual scars, cortex areas, retouch zones
- **Relationships**: Parent-child contour associations

#### Key Columns

**Identification**
- `image_id`: Source image filename
- `surface_type`: Dorsal, Ventral, Platform, Lateral, or Unknown
- `surface_feature`: Surface, Scar, Cortex, Retouch, etc.
- `feature_id`: Unique identifier within image

**Scale Calibration Metadata**
- `calibration_method`: Method used ("scale_bar" or "pixels")
- `pixels_per_mm`: Conversion factor applied (null for pixel measurements)
- `scale_confidence`: Detection confidence score (0-1, scale_bar method only)

**Basic Measurements**
- `technical_length`: Platform-to-distal distance (mm)
- `technical_width`: Maximum perpendicular width (mm)
- `area`: Surface area (mm²)
- `perimeter`: Boundary perimeter (mm)
- `centroid_x`, `centroid_y`: Center coordinates

**Shape Properties**
- `aspect_ratio`: Length/width ratio
- `circularity`: Shape compactness (0-1)
- `convexity`: Boundary convexity
- `solidity`: Area density

**Advanced Metrics** (configurable)
- `symmetry_vertical`: Vertical balance (0-1)
- `symmetry_horizontal`: Horizontal balance (0-1)
- `scar_count`: Number of child scars
- `scar_complexity`: Adjacency relationships
- `has_arrow`: Arrow detection flag (true/false)
- `arrow_angle`: Flaking direction (degrees)
- `voronoi_cells`: Spatial tessellation count
- `convex_hull_area`: Convex hull area (mm²)
- `lateral_convexity`: Edge convexity measure

#### Example Data Row

```csv
image_id,surface_type,surface_feature,calibration_method,pixels_per_mm,scale_confidence,technical_length,technical_width,area,has_arrow,arrow_angle
artifact_001.png,Dorsal,Surface,scale_bar,25.2,0.95,45.2,32.1,1203.5,false,
artifact_001.png,Dorsal,Scar,scale_bar,25.2,0.95,12.3,8.7,89.4,true,145.6
artifact_002.png,Ventral,Surface,pixels,,,1138,812,30421,false,
artifact_003.png,Dorsal,Surface,pixels,,,1140,809,30378,false,
```

## Visualization Outputs

### Labeled Images

**Filename pattern**: `{image_id}_labeled.png`

**Content**:
- Original image with colored overlays
- Contour boundaries in different colors
- Surface classifications as labels
- Arrow indicators (if detected)
- Scale reference

**Color Coding**:
- **Purple** `RGB(94, 60, 153)`: Surface elements (dorsal/ventral/platform/lateral)
- **Orange** `RGB(253, 184, 99)`: Scar elements
- **Red** `RGB(215, 48, 39)`: Cortex elements
- **Mint Green** `RGB(128, 205, 193)`: Lateral edges
- **Light Purple** `RGB(178, 171, 210)`: Platform marks
- **Light Blue** `RGB(145, 191, 219)`: Arrows

### Voronoi Diagrams

**Filename pattern**: `{image_id}_voronoi.png`

**Content**:
- Spatial tessellation of scar centroids
- Voronoi cell boundaries
- Cell area colorization
- Statistical overlays

**Interpretation**:
- Dense patterns indicate intensive flaking
- Large cells suggest sparse scar distribution
- Regular patterns may indicate systematic reduction

## Debug and Diagnostic Outputs

### Scale Calibration Debug

**Location**: `processed/scale_debug/`
**Enabled by**: `--scale_debug` flag

**Files generated**:
- `debug_{scale_id}`: Scale bar detection visualization with bounding box
- Shows detected scale length in pixels and confidence score
- Helps troubleshoot scale detection issues

### Arrow Detection Debug

**Location**: `processed/arrow_debug/`
**Enabled by**: `--arrow_debug` flag

**Files generated**:
- `{image_id}_arrows.png`: Detected arrow overlays
- `{image_id}_contours.png`: All contour hierarchies
- `{image_id}_candidates.png`: Arrow candidates

### Processing Log

**Filename**: `pylithics.log`

**Content**:
- Processing timestamps
- Configuration settings used
- Error messages and warnings
- Performance metrics
- Feature detection statistics

**Example log entries**:
```
2024-01-15 10:30:15 [INFO] Starting PyLithics analysis
2024-01-15 10:30:15 [INFO] Configuration: arrow_detection=True, scale_calibration=True, voronoi=True
2024-01-15 10:30:16 [INFO] Processing artifact_001.png
2024-01-15 10:30:16 [INFO] Scale bar detected: 1260 pixels, confidence: 0.95, dimensions: 1260x34
2024-01-15 10:30:16 [INFO] Using scale bar calibration: 25.2 pixels/mm (1260 pixels = 50 mm)
2024-01-15 10:30:16 [INFO] - Found 15 contours
2024-01-15 10:30:16 [INFO] - Classified: Dorsal (1), Ventral (1) surfaces
2024-01-15 10:30:16 [INFO] - Detected 3 arrows
2024-01-15 10:30:16 [WARNING] - Low contrast in ventral surface
2024-01-15 10:30:16 [INFO] Converted measurements to millimeters using factor: 25.200
2024-01-15 10:30:17 [INFO] Processing artifact_002.png
2024-01-15 10:30:17 [INFO] No scale calibration available, measurements will be in pixels
2024-01-15 10:30:17 [INFO] Processing complete: 2.1 seconds
```

## Configuration-Dependent Outputs

### When Arrow Detection is Enabled

Additional columns in CSV:
- `has_arrow`: Boolean flag
- `arrow_angle`: Direction in degrees
- `arrow_length`: Arrow size (if measurable)

Additional visualizations:
- Arrow overlays on labeled images
- Arrow debug images (if `--arrow_debug`)

### When Voronoi Analysis is Enabled

Additional columns in CSV:
- `voronoi_cells`: Number of Voronoi cells
- `voronoi_area_mean`: Average cell area
- `voronoi_area_std`: Cell area standard deviation

Additional files:
- Voronoi diagram PNG files
- Spatial statistics summary

### When Scar Complexity is Enabled

Additional columns in CSV:
- `scar_complexity`: Adjacency count
- `adjacent_scars`: List of neighboring scar IDs
- `isolation_index`: Spatial isolation measure

## Output Validation

### Visual Validation

1. **Check labeled images**: Verify contour detection accuracy
2. **Review classifications**: Ensure surfaces are correctly identified
3. **Validate arrows**: Confirm arrow detection and directions
4. **Assess completeness**: Check for missed features

### Data Validation

1. **Measurement ranges**: Verify values are reasonable
2. **Unit consistency**: Ensure mm units throughout
3. **Missing data**: Check for incomplete records
4. **Outlier detection**: Identify unusual measurements

### Common Validation Checks

```python
import pandas as pd

# Load results
df = pd.read_csv('processed/measurements.csv')

# Check measurement ranges
print("Length range:", df['technical_length'].min(), "-", df['technical_length'].max())
print("Width range:", df['technical_width'].min(), "-", df['technical_width'].max())

# Check for missing critical measurements
missing_length = df['technical_length'].isna().sum()
print(f"Missing length measurements: {missing_length}")

# Validate aspect ratios
unrealistic_ratios = df[df['aspect_ratio'] > 10].shape[0]
print(f"Unrealistic aspect ratios: {unrealistic_ratios}")
```

## Working with Output Data

### Loading in R

```r
# Load and explore data
data <- read.csv("processed/measurements.csv")

# Surface summary
table(data$surface_type)

# Basic statistics
summary(data[c("technical_length", "technical_width", "area")])

# Plot length vs width
plot(data$technical_length, data$technical_width, 
     xlab="Length (mm)", ylab="Width (mm)",
     col=as.factor(data$surface_type))
```

### Loading in Python

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('processed/measurements.csv')

# Filter for surfaces only
surfaces = df[df['surface_feature'] == 'Surface']

# Basic plotting
plt.figure(figsize=(10, 6))
plt.scatter(surfaces['technical_length'], surfaces['technical_width'], 
           c=surfaces['surface_type'].astype('category').cat.codes)
plt.xlabel('Length (mm)')
plt.ylabel('Width (mm)')
plt.title('Lithic Dimensions by Surface Type')
plt.show()
```

## Customizing Outputs

### Output Format Options

```bash
# JSON output instead of CSV
pylithics --data_dir ./data --meta_file ./meta.csv --output_format json

# Custom output directory
pylithics --data_dir ./data --meta_file ./meta.csv --output_dir ./results

# Minimal output (data only)
pylithics --data_dir ./data --meta_file ./meta.csv --no_images
```

### Selective Visualization

```bash
# Save only specific visualizations
pylithics --data_dir ./data --meta_file ./meta.csv \
         --save_labeled_images \
         --no_voronoi
```

## File Management

### Organizing Results

```bash
# Create timestamped results
today=$(date +"%Y%m%d")
mkdir results_$today
pylithics --data_dir ./data --meta_file ./meta.csv \
         --output_dir ./results_$today
```

### Archiving Outputs

```bash
# Compress results for storage
tar -czf results_archive.tar.gz processed/

# Keep only CSV data
cp processed/measurements.csv ./analysis/
rm -rf processed/
```

## Next Steps

- [Voronoi Analysis](voronoi-analysis.md) - Understand spatial patterns
- [Troubleshooting](troubleshooting.md) - Resolve output issues
- [Glossary](glossary.md) - Reference for all metrics