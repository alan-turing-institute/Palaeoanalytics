# Glossary

## Metric Definitions

This comprehensive glossary defines all measurements and terms used in PyLithics output data.

## Basic Measurements

### Linear Dimensions

**technical_length** (mm)
: Distance from the platform to the distal end, measured along the central axis perpendicular to the striking platform

**technical_width** (mm)
: Maximum width measured perpendicular to the technical length axis

**max_length** (mm)
: Longest dimension of the artifact regardless of orientation

**max_width** (mm)
: Maximum width measured perpendicular to the maximum length

**thickness** (mm)
: Maximum distance between dorsal and ventral surfaces

**perimeter** (mm)
: Total length of the contour boundary

### Area Measurements

**area** (mm²)
: Surface area enclosed by the contour boundary

**convex_hull_area** (mm²)
: Area of the smallest convex shape that contains all points of the contour

**bounding_box_area** (mm²)
: Area of the smallest rectangle that contains the entire contour

## Shape Properties

### Ratios and Indices

**aspect_ratio**
: Ratio of technical length to technical width (length/width)

**circularity**
: Shape compactness measure: 4π × area / perimeter². Values near 1.0 indicate circular shapes

**convexity**
: Ratio of convex hull perimeter to actual perimeter. Values near 1.0 indicate convex shapes

**solidity**
: Ratio of contour area to convex hull area. Measures how "solid" the shape is

**rectangularity**
: Ratio of contour area to bounding box area. Measures how rectangular the shape is

### Geometric Properties

**centroid_x**, **centroid_y** (mm)
: Coordinates of the geometric center of the contour

**moment_hu_1** through **moment_hu_7**
: Hu moments - shape descriptors invariant to translation, rotation, and scaling

**orientation** (degrees)
: Angle of the major axis relative to horizontal (0-180°)

**eccentricity**
: Measure of shape elongation (0 = circle, 1 = line)

## Surface Classification

### Surface Types

**Dorsal**
: Upper surface of the flake showing previous removal scars

**Ventral**
: Lower surface formed during flake detachment, typically smooth

**Platform**
: Prepared surface where the flake was struck from the core

**Lateral**
: Side edges of the flake

**Unknown**
: Surface type could not be determined automatically

### Feature Types

**Surface**
: Main surface contour (parent contour)

**Scar**
: Individual flake removal visible on dorsal surface (child contour)

**Cortex**
: Original outer surface of the raw material

**Retouch**
: Intentional secondary modification of edges

**Platform_mark**
: Features on the striking platform

## Advanced Analysis Metrics

### Symmetry Analysis

**symmetry_vertical** (0-1)
: Measure of bilateral symmetry around vertical axis. 1.0 = perfect symmetry

**symmetry_horizontal** (0-1)
: Measure of symmetry around horizontal axis. 1.0 = perfect symmetry

**symmetry_axis_angle** (degrees)
: Angle of the best-fit symmetry axis

**symmetry_score** (0-1)
: Overall symmetry measure combining vertical and horizontal

### Scar Complexity

**scar_count**
: Number of individual scars detected on the surface

**scar_complexity**
: Number of adjacent scar relationships (scars sharing boundaries)

**scar_density** (scars/mm²)
: Number of scars per unit surface area

**isolation_index** (0-1)
: Measure of how isolated individual scars are from others

**adjacent_scars**
: List of scar IDs that share boundaries with this scar

### Arrow Detection

**has_arrow** (true/false)
: Whether a directional arrow was detected for this scar

**arrow_angle** (degrees)
: Direction of force application indicated by arrow (0-360°)

**arrow_length** (mm)
: Length of the detected arrow indicator

**arrow_confidence** (0-1)
: Confidence score for arrow detection

**force_direction** (degrees)
: Inferred direction of knapping force

### Voronoi Analysis

**voronoi_cells**
: Number of Voronoi cells in the tessellation

**voronoi_area_mean** (mm²)
: Average area of Voronoi cells

**voronoi_area_std** (mm²)
: Standard deviation of Voronoi cell areas

**voronoi_density** (cells/mm²)
: Spatial density of Voronoi cells

**spatial_distribution** (0-1)
: Regularity index of spatial distribution (1.0 = perfectly regular)

**nearest_neighbor_distance** (mm)
: Average distance to nearest neighboring scar

### Lateral Analysis

**lateral_convexity** (0-1)
: Measure of edge convexity (1.0 = perfectly convex)

**edge_angle** (degrees)
: Angle between dorsal and ventral surfaces at the edge

**edge_length** (mm)
: Length of the lateral edge

**use_wear_index** (0-1)
: Estimated use-wear based on edge characteristics

## Technical Terms

### Image Processing

**DPI** (dots per inch)
: Image resolution. PyLithics works best with 300+ DPI

**Thresholding**
: Process of converting grayscale images to binary (black/white)

**Contour**
: Boundary line around objects in the image

**Hierarchy**
: Parent-child relationships between contours (surfaces contain scars)

**Morphological Closing**
: Image processing operation to fill small gaps in contours

### Archaeological Terms

**Chaîne Opératoire**
: Sequence of operations in tool production

**Reduction Sequence**
: Order of flake removals during knapping

**Debitage**
: Waste flakes produced during tool manufacture

**Percussion**
: Striking technique used to remove flakes

**Platform Preparation**
: Creating suitable striking surface on core

**Ripple Marks**
: Concentric lines showing force propagation

## Data Organization

### File Structure

**image_id**
: Filename of the source image

**feature_id**
: Unique identifier for each detected feature within an image

**parent_id**
: ID of the parent contour (for hierarchical relationships)

**surface_feature**
: Type of feature (Surface, Scar, Cortex, etc.)

**processing_timestamp**
: When the analysis was performed

### Quality Metrics

**detection_confidence** (0-1)
: Confidence in the automated detection

**measurement_accuracy** (0-1)
: Estimated accuracy of measurements

**processing_notes**
: Automated notes about processing issues

**validation_required** (true/false)
: Flag indicating manual validation recommended

## Units and Scales

### Measurement Units

All linear measurements are in **millimeters (mm)**
All area measurements are in **square millimeters (mm²)**
All angular measurements are in **degrees (°)**

### Scale Conversion

**pixels_per_mm**
: Conversion factor from pixels to millimeters

**scale_reference**
: Scale bar used for conversion

**pixels_per_mm**
: Conversion factor from pixels to millimeters based on scale bar detection

**measurement_error** (mm)
: Estimated measurement uncertainty

## Statistical Summaries

### Assemblage-Level Metrics

**assemblage_size**
: Total number of artifacts analyzed

**mean_length**, **std_length**
: Mean and standard deviation of technical length

**mean_width**, **std_width**
: Mean and standard deviation of technical width

**size_distribution**
: Classification into size categories (small, medium, large)

**technology_index**
: Composite measure of technological sophistication

## Configuration Terms

### Processing Parameters

**threshold_method**
: Algorithm used for image binarization (simple, otsu, adaptive)

**threshold_value**
: Cutoff value for simple thresholding (0-255)

**min_contour_area**
: Minimum size for contour detection (pixels)

**edge_detection_sensitivity**
: Parameter controlling edge detection

### Analysis Modules

**feature_enabled**
: Whether specific analysis modules are active

**debug_mode**
: Whether diagnostic output is generated

**output_format**
: Format for data export (CSV, JSON)

**visualization_options**
: Settings for image output generation

## Common Abbreviations

**CV** - Computer Vision
**DPI** - Dots Per Inch
**CSV** - Comma-Separated Values
**JSON** - JavaScript Object Notation
**RGB** - Red, Green, Blue (color model)
**API** - Application Programming Interface
**CLI** - Command Line Interface
**YAML** - Yet Another Markup Language

## Value Ranges

### Typical Ranges for Stone Tools

**Technical Length**: 10-200 mm (most flakes 20-80 mm)
**Technical Width**: 8-150 mm (most flakes 15-60 mm)
**Area**: 100-15,000 mm² (most flakes 300-3,000 mm²)
**Aspect Ratio**: 0.5-5.0 (most flakes 1.0-2.5)
**Scar Count**: 0-50 (most surfaces 2-15 scars)

### Quality Indicators

**Good Detection**: Circularity 0.3-0.9, Convexity > 0.8
**Potential Issues**: Aspect ratio > 5.0, Area < 50 mm²
**Manual Review Needed**: Scar count > 30, Symmetry < 0.1

This glossary provides the foundation for understanding and interpreting PyLithics analysis results.