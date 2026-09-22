# Glossary

This glossary defines each column that PyLithics writes to
`processed_metrics.csv`, and the archaeological and image-processing
terms in the documentation. The entries are in the order of the CSV.

## Identification

**image_id**
: The filename of the source image.

**surface_type**
: The classification of a parent surface: `Dorsal`, `Ventral`, `Platform`, `Lateral` or `Unclassified`.

**surface_feature**
: For a parent row, the surface name (for example `Dorsal`). For a child row, the feature label from the cortex detection: `scar N`, `edge N` or `cortex N`.

**scar_count**
: The number of scars on the dorsal surface. In the Dorsal parent row only. `NA` in all other rows.

## Position and dimensions

All lengths are in millimetres when the scale calibration is correct, and in pixels when it is not. The column `calibration_method` shows which.

**centroid_x**, **centroid_y**
: The coordinates of the geometric centroid of the contour.

**technical_width**
: The maximum width of a parent surface, perpendicular to its `technical_length` axis.

**technical_length**
: The distance from the platform to the distal end, along the central axis perpendicular to the striking platform.

**max_width**
: The maximum dimension perpendicular to `max_length`.

**max_length**
: The longest distance between two points of the contour, in any orientation.

**total_area**
: The area inside the contour, in mm² or px².

**perimeter**
: The length of the contour.

**aspect_ratio**
: `technical_length / technical_width`. `NA` when `technical_width` is zero.

**distance_to_max_width**
: The distance from the platform to the point of the contour where the width is maximum.

## Voronoi and convex hull

These columns are in the Dorsal parent row when the dorsal surface has scars.

**voronoi_num_cells**
: The number of Voronoi cells for the dorsal scar centroids.

**voronoi_cell_area**
: The area of the Voronoi cell that contains this row's centroid.

**convex_hull_width**, **convex_hull_height**
: The dimensions of the convex hull around all scar centroids.

**convex_hull_area**
: The area of the convex hull.

## Symmetry

Calculated from the binary mask of the Dorsal parent contour, divided at its centroid.

**top_area**, **bottom_area**, **left_area**, **right_area**
: The filled area on each side of the centroid.

**vertical_symmetry**
: `1 − |top_area − bottom_area| / (top_area + bottom_area)`. From 0 to 1. 1.0 is perfect top and bottom symmetry.

**horizontal_symmetry**
: `1 − |left_area − right_area| / (left_area + right_area)`. From 0 to 1.

## Lateral

**lateral_convexity**
: The area of the lateral surface contour divided by the area of its convex hull. From 0 to 1. 1.0 means that the lateral edge is fully convex.

## Cortex

**is_cortex**
: `True` if the texture analysis classified a child contour as cortex. `False` if not.

**cortex_area**
: The area of the cortex. Only meaningful when `is_cortex` is `True`.

**cortex_percentage**
: The cortex area as a percentage of the parent surface area.

## Arrows

**has_arrow**
: `True` if an arrow was found for this scar.

**arrow_angle**
: The angle of the arrow in the PyLithics frame, like a compass. An arrow that points down in the image is `0°`. An arrow that points right is `270°`. Use it to compare scars in the same image.

## Scar complexity

**scar_complexity**
: The number of other dorsal scars whose polygons are in the adjacency distance (default 10 px).

## Scale calibration

These columns are present when the pipeline had calibration metadata.

**calibration_method**
: `scale_bar` when a scale image was found and measured. `pixels` when there was no calibration.

**pixels_per_mm**
: The factor that changes pixel measurements to millimetres.

**scale_confidence**
: The confidence of the scale bar detection (0–1).

## Arrow geometry (optional)

These columns are present only when arrow detection found the triangle geometry for at least one scar.

**triangle_base_length**, **triangle_height**
: The geometry of the triangle of the arrow tip.

**shaft_solidity**, **tip_solidity**
: The solidity of the two halves of the arrow, divided at the base of the tip.

## Surface types

**Dorsal**
: The upper surface of the flake. It shows the scars of previous removals.

**Ventral**
: The lower surface, made when the flake was detached. It is usually smooth, with the bulb of percussion.

**Platform**
: The prepared striking surface of the core. It is a small surface at the proximal end.

**Lateral**
: The side view of the flake.

**Unclassified**
: A parent surface that does not match one of the four standard types.

## Feature types

**scar N**
: A flake removal scar on the dorsal surface (numbered in sequence).

**edge N**
: A child contour on the lateral surface (numbered in sequence).

**cortex N**
: A child contour that the texture analysis classified as cortex (numbered in sequence).

Platform child contours are not in the output. They are usually the boundaries of empty space, not morphological features.

## Image-processing terms

**DPI** (dots per inch)
: The image resolution. PyLithics uses fixed kernels by default from 75 to 600 DPI. `--enable_dpi_scaling` sets the kernel sizes from the DPI, for scans with noise.

**Thresholding**
: The change of a greyscale image to a black-and-white image. PyLithics has `simple` (a fixed value), `otsu` (an automatic division into two tones), `adaptive` (a value for each region) and `default` (`simple` with the default value).

**Contour**
: The boundary line around an object in the black-and-white image.

**Hierarchy**
: The parent–child relations between contours. A surface is a parent contour. The scars and the other features inside it are children.

**Morphological closing**
: An image-processing step that closes small gaps in contours. It dilates and then erodes the black-and-white image.

## Archaeological terms

**Chaîne opératoire**
: The sequence of operations in tool production.

**Reduction sequence**
: The order in which the flakes were removed during knapping.

**Debitage**
: The waste flakes from tool manufacture.

**Percussion**
: The strike that removes a flake.

**Platform preparation**
: The shaping of a striking surface on the core before a flake removal.

**Ripple marks**
: Curved lines, one inside the other, on a flake surface. They show how the force of the percussion moved. PyLithics gives the best results on illustrations where arrows replace the ripple marks (see [Prepare Your Images](image-requirements.md)).

## Units

| Quantity | Calibrated | Not calibrated |
|----------|------------|----------------|
| Length, width, distance | mm | px |
| Area | mm² | px² |
| Angle | degrees | degrees |

## Usual values for stone tools

Use these values to examine your output:

| Quantity | Usual range | Most flakes |
|----------|-------------|-------------|
| `technical_length` | 10–200 mm | 20–80 mm |
| `technical_width` | 8–150 mm | 15–60 mm |
| `total_area` | 100–15,000 mm² | 300–3,000 mm² |
| `aspect_ratio` | 0.5–5.0 | 1.0–2.5 |
| Scars on a dorsal surface | 0–50 | 2–15 |

A value far outside these ranges usually shows a scale calibration problem or a contour detection error. Examine the `_labeled.png` of that image.

## Abbreviations

- **CV** — computer vision
- **DPI** — dots per inch
- **CSV** — comma-separated values
- **CLI** — command-line interface
- **YAML** — YAML Ain't Markup Language
