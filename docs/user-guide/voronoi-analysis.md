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
  enabled: true              # default; set to false to skip
  padding_factor: 0.02       # padding around dorsal contour bounds (fraction)
  min_distance_threshold: 5.0  # minimum spacing between Voronoi points
```

Voronoi analysis is enabled by default and is not currently toggleable from the CLI; edit `config.yaml` to disable it.

### Command Line

```bash
# Default run (Voronoi enabled)
pylithics --data_dir ./data --meta_file ./meta.csv

# Use a custom config to disable Voronoi
pylithics --data_dir ./data --meta_file ./meta.csv --config_file ./no_voronoi.yaml
```

## Generated Outputs

### Voronoi Diagram Images

**Location**: `processed/`
**Filename**: `{image_stem}_voronoi.png`

**Visual elements**:

- Voronoi cell boundaries clipped to the dorsal contour
- Convex hull around all scar centroids
- Centroid points
- Axes in millimetres when scale calibration succeeded, pixels otherwise

### CSV Data Columns

When Voronoi analysis succeeds for an image, these columns are populated on the Dorsal parent row of `processed_metrics.csv`:

| Column | Units | Description |
|--------|-------|-------------|
| `voronoi_num_cells` | count | Number of Voronoi cells |
| `voronoi_cell_area` | mm² or px² | Area of the Voronoi cell containing this row's centroid |
| `convex_hull_width` | mm or px | Width of the convex hull around centroids |
| `convex_hull_height` | mm or px | Height of the convex hull |
| `convex_hull_area` | mm² or px² | Area of the convex hull |

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
  enabled: true              # Set false to skip Voronoi analysis
  padding_factor: 0.02       # Bounding-box padding as fraction of contour size
  min_distance_threshold: 5.0  # Minimum spacing between Voronoi points
```

These are the only Voronoi keys PyLithics reads. Visual elements (line colors, transparency) are not currently configurable.

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

### R

```r
data <- read.csv("pylithics/data/processed/processed_metrics.csv")

# Dorsal parents only — Voronoi columns live there
dorsal <- subset(data,
                 surface_type == "Dorsal" & surface_feature == "Dorsal")

# Summary
summary(dorsal$voronoi_num_cells)
summary(dorsal$convex_hull_area)

# Cells per dorsal surface area
dorsal$density <- dorsal$voronoi_num_cells / dorsal$total_area
hist(dorsal$density,
     xlab = "Cells per mm²", main = "Dorsal scar density")
```

### Python

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("pylithics/data/processed/processed_metrics.csv")

dorsal = df[(df["surface_type"] == "Dorsal") &
            (df["surface_feature"] == "Dorsal")]

# Cells vs. convex hull area
plt.scatter(dorsal["convex_hull_area"], dorsal["voronoi_num_cells"])
plt.xlabel("Convex hull area (mm²)")
plt.ylabel("Voronoi cell count")
plt.title("Scar count vs. dorsal coverage")
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

Voronoi cannot be disabled from the CLI. To skip it, set `voronoi_analysis.enabled: false` in your `config.yaml` and pass it via `--config_file`:

```bash
pylithics --data_dir ./large_dataset --meta_file ./meta.csv \
    --config_file ./no_voronoi.yaml
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