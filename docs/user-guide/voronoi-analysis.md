# Voronoi Analysis

## Overview

Voronoi analysis in PyLithics provides spatial pattern analysis of scar distributions on lithic surfaces. This advanced feature generates tessellation diagrams that reveal technological patterns and reduction strategies.

## What is Voronoi Analysis?

### Mathematical Foundation

A Voronoi diagram divides a plane into regions based on distance to specific points (in our case, scar centroids). Each region contains all points closer to one scar than to any other scar.

### Archaeological Application

- **Flaking Intensity**: Dense patterns indicate intensive reduction
- **Spatial Organization**: Regular patterns suggest systematic flaking
- **Reduction Strategy**: Clustering reveals preferred flaking zones
- **Skill Assessment**: Regularity may indicate knapper expertise

## Enabling Voronoi Analysis

### Configuration

```yaml
# In config.yaml
voronoi_analysis:
  enabled: true
  output_diagrams: true
  min_scars: 3              # Minimum scars needed for analysis
  boundary_method: convex   # convex or rectangle
```

### Command Line

```bash
# Enable Voronoi analysis
pylithics --data_dir ./data --meta_file ./meta.csv

# Disable for faster processing
pylithics --data_dir ./data --meta_file ./meta.csv --disable_voronoi
```

## Generated Outputs

### Voronoi Diagram Images

**Location**: `processed/voronoi_diagrams/`
**Filename**: `{image_id}_voronoi.png`

**Visual Elements**:
- Scar centroid points
- Voronoi cell boundaries
- Color-coded cell areas
- Statistical overlays
- Scale reference

### CSV Data Columns

When Voronoi analysis is enabled, additional columns appear in the output CSV:

| Column | Description | Units |
|--------|-------------|-------|
| `voronoi_cells` | Number of Voronoi cells | count |
| `voronoi_area_mean` | Average cell area | mm² |
| `voronoi_area_std` | Standard deviation of cell areas | mm² |
| `voronoi_density` | Scars per unit area | scars/mm² |
| `spatial_distribution` | Regularity index (0-1) | ratio |

## Interpretation Guide

### Cell Size Patterns

**Large, uniform cells**:
- Systematic, controlled flaking
- Experienced knapper
- Planned reduction sequence

**Small, irregular cells**:
- Intensive flaking
- Opportunistic removal
- Possible reworking or resharpening

**Mixed cell sizes**:
- Multi-stage reduction
- Different flaking episodes
- Changing reduction strategies

### Spatial Organization

**Regular distribution**:
- Deliberate scar placement
- Efficient core utilization
- Systematic reduction strategy

**Clustered distribution**:
- Localized intensive flaking
- Platform preparation areas
- Reworking zones

**Random distribution**:
- Opportunistic flaking
- Less controlled reduction
- Possible expedient technology

## Convex Hull Analysis

### What is Convex Hull?

The convex hull is the smallest convex shape that contains all scar points. It provides:

- **Total flaking area**: Maximum extent of scar distribution
- **Utilization efficiency**: How much of available surface was used
- **Shape regularity**: Geometric properties of flaking zone

### Convex Hull Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| `convex_hull_area` | Area of convex hull | Total flaking zone |
| `hull_perimeter` | Perimeter of hull | Edge utilization |
| `hull_solidity` | Scar area / hull area | Flaking efficiency |
| `hull_aspect_ratio` | Length/width of hull | Shape preference |

## Configuration Options

### Analysis Parameters

```yaml
voronoi_analysis:
  enabled: true
  
  # Minimum requirements
  min_scars: 3              # Skip if fewer scars
  min_surface_area: 100     # Skip small surfaces (mm²)
  
  # Boundary definition
  boundary_method: convex   # convex, rectangle, or surface
  boundary_buffer: 5        # Buffer around scars (mm)
  
  # Output options
  output_diagrams: true     # Generate PNG files
  color_by_area: true       # Color cells by area
  show_centroids: true      # Mark scar centers
  
  # Statistical options
  calculate_density: true   # Spatial density metrics
  regularity_analysis: true # Distribution regularity
```

### Visual Customization

```yaml
visualization:
  voronoi:
    cell_alpha: 0.6         # Cell transparency
    boundary_color: black   # Cell border color
    centroid_size: 3        # Point size
    colormap: viridis       # Color scheme
```

## Analysis Examples

### High-Skill Reduction

**Characteristics**:
- Regular cell sizes
- Uniform distribution
- High hull solidity
- Low area standard deviation

**Interpretation**:
- Systematic flaking approach
- Efficient surface utilization
- Controlled reduction sequence
- Experienced knapper

### Opportunistic Flaking

**Characteristics**:
- Irregular cell sizes
- Clustered distribution
- Low hull solidity
- High area standard deviation

**Interpretation**:
- Expedient flaking strategy
- Focus on immediate needs
- Less systematic approach
- Possibly less experienced

### Multi-Stage Reduction

**Characteristics**:
- Mixed cell patterns
- Multiple clustering zones
- Moderate hull solidity
- Bimodal area distribution

**Interpretation**:
- Different reduction episodes
- Changing strategies
- Tool reuse or resharpening
- Complex reduction history

## Working with Voronoi Data

### Statistical Analysis in R

```r
# Load data
data <- read.csv("processed/measurements.csv")

# Filter for surfaces with Voronoi analysis
voronoi_data <- data[!is.na(data$voronoi_cells), ]

# Summary statistics
summary(voronoi_data$voronoi_area_mean)
summary(voronoi_data$voronoi_density)

# Compare between surface types
aggregate(voronoi_density ~ surface_type, voronoi_data, mean)

# Plot density vs. regularity
plot(voronoi_data$voronoi_density, voronoi_data$spatial_distribution,
     xlab="Scar Density", ylab="Spatial Regularity",
     main="Flaking Patterns")
```

### Python Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load and filter data
df = pd.read_csv('processed/measurements.csv')
voronoi_df = df[df['voronoi_cells'].notna()]

# Density distribution
plt.figure(figsize=(10, 6))
sns.histplot(voronoi_df['voronoi_density'], bins=20)
plt.xlabel('Scar Density (scars/mm²)')
plt.title('Distribution of Flaking Density')
plt.show()

# Regularity by surface type
sns.boxplot(data=voronoi_df, x='surface_type', y='spatial_distribution')
plt.ylabel('Spatial Regularity')
plt.title('Flaking Regularity by Surface Type')
plt.show()
```

## Troubleshooting Voronoi Analysis

### Common Issues

**No Voronoi diagrams generated**:
- Check that surfaces have ≥3 scars
- Verify `voronoi_analysis.enabled: true` in config
- Ensure output directory has write permissions

**Unrealistic cell areas**:
- Verify scale information in metadata
- Check for duplicate scar centroids
- Review contour detection accuracy

**Missing data columns**:
- Confirm Voronoi analysis is enabled
- Check for processing errors in log file
- Verify minimum requirements are met

### Performance Considerations

```bash
# For large datasets, disable Voronoi if not needed
pylithics --data_dir ./large_dataset --meta_file ./meta.csv \
         --disable_voronoi

# Or process subset first
pylithics --data_dir ./test_sample --meta_file ./test_meta.csv
```

## Archaeological Case Studies

### Levallois Technology

**Expected patterns**:
- Regular cell distribution
- High spatial organization
- Systematic centripetal flaking
- Efficient surface utilization

### Expedient Technology

**Expected patterns**:
- Irregular cell sizes
- Opportunistic distribution
- Lower spatial organization
- Variable surface utilization

### Blade Production

**Expected patterns**:
- Linear cell arrangements
- Parallel flaking zones
- Regular width patterns
- High aspect ratio hulls

## Research Applications

### Comparative Studies

- **Inter-site variation**: Compare flaking strategies
- **Temporal change**: Track technological evolution
- **Skill assessment**: Quantify knapping expertise
- **Cultural attribution**: Identify technological traditions

### Statistical Methods

- **Cluster analysis**: Group similar patterns
- **ANOVA**: Test between-group differences
- **Regression**: Model relationships
- **Multivariate analysis**: Integrate multiple metrics

## Next Steps

- [Troubleshooting](troubleshooting.md) - Resolve analysis issues
- [Glossary](glossary.md) - Reference for spatial metrics
- [CLI Commands](../reference/cli-commands.md) - Configuration options