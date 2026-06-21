# Glossary

This glossary defines every column PyLithics writes to `processed_metrics.csv` plus the archaeological and image-processing terms that appear in the documentation. Entries are grouped by where they appear in the CSV.

## Identification

**image_id**
: Filename of the source image.

**surface_type**
: Classification of a parent surface — one of `Dorsal`, `Ventral`, `Platform`, `Lateral`, or `Unclassified`.

**surface_feature**
: For parent rows, the surface name (e.g. `Dorsal`). For child rows, the feature label assigned during cortex-detection relabelling: `scar N`, `edge N`, or `cortex N`.

**total_dorsal_scars**
: Number of scars on the dorsal surface. Filled only on the Dorsal parent row; `NA` everywhere else.

## Position and Dimensions

All linear measurements are in millimetres when scale calibration succeeded; otherwise they are in pixels. Check `calibration_method` to confirm.

**centroid_x**, **centroid_y**
: Coordinates of the contour's geometric centroid.

**technical_width**
: Maximum width of a parent surface measured perpendicular to its `technical_length` axis.

**technical_length**
: Distance from the platform to the distal end, measured along the central axis perpendicular to the striking platform.

**max_width**
: Maximum dimension perpendicular to `max_length`.

**max_length**
: Longest point-to-point distance regardless of orientation.

**total_area**
: Area enclosed by the contour boundary, in mm² or px².

**perimeter**
: Length of the contour boundary.

**aspect_ratio**
: Ratio `technical_length / technical_width`. `None` (written as `NA`) when `technical_width` is zero.

**distance_to_max_width**
: Distance from the platform to the point on the contour where maximum width occurs.

## Voronoi & Convex Hull

These columns appear on the Dorsal parent row when the dorsal surface has scars.

**voronoi_num_cells**
: Number of Voronoi cells generated for the dorsal scar centroids.

**voronoi_cell_area**
: Area of the Voronoi cell containing this row's centroid.

**convex_hull_width**, **convex_hull_height**
: Dimensions of the convex hull around all scar centroids.

**convex_hull_area**
: Area of the convex hull.

## Symmetry

Calculated from the binary mask of the Dorsal parent contour, split at its centroid.

**top_area**, **bottom_area**, **left_area**, **right_area**
: Filled-pixel area in each quadrant relative to the centroid.

**vertical_symmetry**
: `1 − |top_area − bottom_area| / (top_area + bottom_area)`. Range 0–1, where 1.0 is perfect top/bottom balance.

**horizontal_symmetry**
: `1 − |left_area − right_area| / (left_area + right_area)`. Range 0–1.

## Lateral

**lateral_convexity**
: Ratio of the lateral surface contour area to its convex hull area. Range 0–1; 1.0 means the lateral edge is fully convex.

## Cortex

**is_cortex**
: `True` if a child contour was reclassified as cortex by the texture analysis. `False` otherwise.

**cortex_area**
: Area of the cortex region. Only meaningful when `is_cortex` is `True`.

**cortex_percentage**
: Cortex area as a percentage of the parent surface area.

## Arrows

**has_arrow**
: `True` if a directional arrow was detected for this scar.

**arrow_angle**
: Compass-style angle of the arrow in PyLithics's rotated frame. A downward-pointing arrow in image coordinates maps to `0°`; a rightward-pointing arrow maps to `270°`. Treat as a relative value when comparing scars within the same image.

## Scar Complexity

**scar_complexity**
: Number of other dorsal scars whose polygons lie within the configured adjacency distance (default 10 px).

## Scale Calibration Metadata

These columns appear when calibration metadata was passed through the pipeline.

**calibration_method**
: `scale_bar` when a scale image was detected and measured; `pixels` when calibration was unavailable or skipped.

**pixels_per_mm**
: Conversion factor used to translate pixel measurements into millimetres.

**scale_confidence**
: Confidence score (0–1) for scale-bar detection.

## Optional Arrow Geometry

These columns appear only when arrow detection produced detailed triangle geometry for at least one scar in the run.

**triangle_base_length**, **triangle_height**
: Geometry of the arrow-tip triangle inferred during detection.

**shaft_solidity**, **tip_solidity**
: Solidity ratios for the half-spaces split by the arrow's base.

## Surface Type Definitions

**Dorsal**
: Upper surface of the flake, showing scars from previous removals.

**Ventral**
: Lower surface formed during flake detachment; typically smooth and bears the bulb of percussion.

**Platform**
: Prepared striking surface on the core; appears as a small surface at the proximal end.

**Lateral**
: Side profile of the flake.

**Unclassified**
: A parent surface that could not be assigned to any of the four standard types.

## Feature Type Definitions

**scar N**
: A flake removal scar on the dorsal surface (numbered sequentially).

**edge N**
: A child contour on the lateral surface (numbered sequentially).

**cortex N**
: A child contour reclassified as cortex by texture analysis (numbered sequentially).

Platform child contours are excluded from the output as they typically represent empty-space boundaries rather than morphological features.

## Image Processing Terms

**DPI** (dots per inch)
: Image resolution. PyLithics uses fixed kernels by default across 75–600 DPI; `--enable_dpi_scaling` adds DPI-aware kernel sizing for noisy scans.

**Thresholding**
: Conversion of grayscale images to binary (black/white). PyLithics supports `simple` (fixed cutoff), `otsu` (auto bimodal split), `adaptive` (per-region), and `default` (`simple` with the default value).

**Contour**
: Boundary line around an object detected in the binarized image.

**Hierarchy**
: Parent–child relationships between contours. A surface is a parent contour; the scars and other features inside it are children.

**Morphological closing**
: Image processing step that fills small gaps in contours by dilating then eroding the binary image.

## Archaeological Terms

**Chaîne opératoire**
: The sequence of operations in tool production.

**Reduction sequence**
: The order in which flakes were removed during knapping.

**Debitage**
: Waste flakes produced during tool manufacture.

**Percussion**
: The striking technique used to remove flakes.

**Platform preparation**
: Shaping a striking surface on the core before flake removal.

**Ripple marks**
: Concentric lines on a flake surface showing how percussion force propagated. PyLithics works best on illustrations where ripples have been replaced by directional arrows (see [Image Requirements](image-requirements.md)).

## Units

| Quantity | Calibrated | Uncalibrated |
|----------|------------|--------------|
| Linear (length, width, distance) | mm | px |
| Area | mm² | px² |
| Angle | degrees | degrees |

## Typical Value Ranges for Stone Tools

Use these as sanity checks on your output:

| Quantity | Typical range | Most flakes |
|----------|---------------|-------------|
| `technical_length` | 10–200 mm | 20–80 mm |
| `technical_width` | 8–150 mm | 15–60 mm |
| `total_area` | 100–15,000 mm² | 300–3,000 mm² |
| `aspect_ratio` | 0.5–5.0 | 1.0–2.5 |
| Scars per dorsal surface | 0–50 | 2–15 |

Values well outside these ranges typically indicate scale calibration problems or contour-detection errors — review the `_labeled.png` for the affected image.

## Common Abbreviations

- **CV** — computer vision
- **DPI** — dots per inch
- **CSV** — comma-separated values
- **CLI** — command line interface
- **YAML** — YAML Ain't Markup Language
