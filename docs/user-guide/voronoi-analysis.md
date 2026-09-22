# Voronoi Analysis

## Overview

The Voronoi analysis in PyLithics shows the spatial pattern of the
scars on a lithic surface. It makes a tessellation diagram that shows
the technological pattern and the reduction strategy.

## What is a Voronoi diagram?

### The mathematics

A Voronoi diagram divides a plane into regions. Each region belongs to
one point (here, one scar centroid). Each region contains all the
points of the plane that are nearer to that scar than to any other
scar.

### The archaeological use

- **Flaking intensity**: dense patterns show intensive reduction
- **Spatial organisation**: regular patterns show systematic flaking
- **Reduction strategy**: clusters show the preferred flaking zones
- **Skill**: regularity can show the experience of the knapper

## Set Voronoi analysis on or off

### Configuration

```yaml
# In config.yaml
voronoi_analysis:
  enabled: true              # default; set false to skip the analysis
  padding_factor: 0.02       # the margin around the dorsal contour (a fraction)
  min_distance_threshold: 5.0  # the minimum distance between Voronoi points
```

Voronoi analysis is on by default. There is no command-line switch.
Set `enabled: false` in `config.yaml` to set it off.

### Command line

```bash
# Default (Voronoi analysis on)
pylithics --data_dir ./data

# A configuration file that sets Voronoi analysis off
pylithics --data_dir ./data --config_file ./no_voronoi.yaml
```

## Output

### The Voronoi diagram

**Location**: `results/`
**Filename**: `{image_stem}_voronoi.png`

**The diagram shows**:

- The Voronoi cell boundaries, cut at the dorsal contour
- The convex hull around all scar centroids
- The centroid points
- The axes in millimetres when the scale calibration is correct, and in pixels when it is not

### CSV columns

When the Voronoi analysis is possible for an image, these columns are
in the Dorsal surface row of `processed_metrics.csv`:

| Column | Units | Description |
|--------|-------|-------------|
| `voronoi_num_cells` | count | The number of Voronoi cells |
| `voronoi_cell_area` | mm² or px² | The area of the Voronoi cell that contains this row's centroid |
| `convex_hull_width` | mm or px | The width of the convex hull around the centroids |
| `convex_hull_height` | mm or px | The height of the convex hull |
| `convex_hull_area` | mm² or px² | The area of the convex hull |

## How to read the diagram

### Cell size

**Large cells of the same size**:
- Systematic, controlled flaking
- An experienced knapper
- A planned reduction sequence

**Small cells of different sizes**:
- Intensive flaking
- Opportunistic removal
- Possibly rework or resharpening

**Mixed cell sizes**:
- Reduction in more than one stage
- Different flaking episodes
- A change of reduction strategy

### Spatial organisation

**Regular distribution**:
- Deliberate scar positions
- Efficient use of the core
- A systematic reduction strategy

**Clusters**:
- Intensive flaking in one area
- Platform preparation areas
- Rework zones

**Random distribution**:
- Opportunistic flaking
- Less controlled reduction
- Possibly expedient technology

## The convex hull

### What is the convex hull?

The convex hull is the smallest convex shape that contains all the
scar points. It gives:

- **The total flaking area**: the maximum extent of the scars
- **The use of the surface**: how much of the surface has scars
- **The shape of the flaking zone**: its geometric properties

### Convex hull metrics

| Metric | Description | Meaning |
|--------|-------------|---------|
| `convex_hull_area` | The area of the convex hull | The total flaking zone |
| `hull_perimeter` | The perimeter of the hull | The use of the edge |
| `hull_solidity` | The scar area divided by the hull area | The flaking efficiency |
| `hull_aspect_ratio` | The length of the hull divided by its width | The shape preference |

## Configuration

```yaml
voronoi_analysis:
  enabled: true              # Set false to skip the analysis
  padding_factor: 0.02       # The margin of the bounding box, as a fraction of the contour size
  min_distance_threshold: 5.0  # The minimum distance between Voronoi points
```

PyLithics reads only these Voronoi keys. You cannot set the colours or
the transparency of the diagram.

## Examples

### Skilled reduction

**Properties**:
- Cells of the same size
- Uniform distribution
- High hull solidity
- Low standard deviation of the cell area

**Meaning**:
- Systematic flaking
- Efficient use of the surface
- A controlled reduction sequence
- An experienced knapper

### Opportunistic flaking

**Properties**:
- Cells of different sizes
- Clusters
- Low hull solidity
- High standard deviation of the cell area

**Meaning**:
- An expedient flaking strategy
- Immediate use
- A less systematic method
- Possibly a less experienced knapper

### Reduction in more than one stage

**Properties**:
- Mixed cell patterns
- More than one cluster
- Moderate hull solidity
- A bimodal distribution of the cell area

**Meaning**:
- Different reduction episodes
- A change of strategy
- Reuse or resharpening of the tool
- A complex reduction history

## Use the Voronoi data

### R

```r
data <- read.csv("pylithics/data/results/processed_metrics.csv")

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

df = pd.read_csv("pylithics/data/results/processed_metrics.csv")

dorsal = df[(df["surface_type"] == "Dorsal") &
            (df["surface_feature"] == "Dorsal")]

# Cells vs. convex hull area
plt.scatter(dorsal["convex_hull_area"], dorsal["voronoi_num_cells"])
plt.xlabel("Convex hull area (mm²)")
plt.ylabel("Voronoi cell count")
plt.title("Scar count vs. dorsal coverage")
plt.show()
```

## Problems

### Common problems

**No Voronoi diagrams**:
- Make sure that the surfaces have 3 scars or more
- Make sure that `voronoi_analysis.enabled` is `true` in the configuration
- Make sure that you can write to the output directory

**Cell areas that are not plausible**:
- Make sure that the scale in the metadata is correct
- Look for scar centroids that are the same
- Examine the contours in the labelled image

**Columns are missing**:
- Make sure that the Voronoi analysis is on
- Read the log for errors
- Make sure that the surfaces have enough scars

### Speed

There is no command-line switch for the Voronoi analysis. To set it
off, set `voronoi_analysis.enabled: false` in your `config.yaml` and
give the file with `--config_file`:

```bash
pylithics --data_dir ./large_dataset \
    --config_file ./no_voronoi.yaml
```

## Archaeological examples

### Levallois technology

**Expected patterns**:
- A regular cell distribution
- High spatial organisation
- Systematic centripetal flaking
- Efficient use of the surface

### Expedient technology

**Expected patterns**:
- Cells of different sizes
- Opportunistic distribution
- Lower spatial organisation
- Variable use of the surface

### Blade production

**Expected patterns**:
- Cells in lines
- Parallel flaking zones
- Regular widths
- Hulls with a high aspect ratio

## Research uses

### Comparative studies

- **Between sites**: compare flaking strategies
- **Over time**: follow the change of a technology
- **Skill**: measure the experience of the knapper
- **Cultural attribution**: identify technological traditions

### Statistical methods

- **Cluster analysis**: group similar patterns
- **ANOVA**: test the differences between groups
- **Regression**: model the relationships
- **Multivariate analysis**: combine many metrics

## Next steps

- [Troubleshooting](troubleshooting.md) — common problems
- [Glossary](glossary.md) — the spatial metrics
- [CLI Commands](../reference/cli-commands.md) — configuration options
