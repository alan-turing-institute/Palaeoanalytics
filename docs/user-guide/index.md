# User Guide

This guide gives the procedures for PyLithics, step by step.

## Contents

### [Prepare Your Images](image-requirements.md)
The image specifications and drawing conventions that give the best results:

- File formats and resolutions
- Drawing style
- Orientation
- Scale bar position
- Preparation of images

### [Working from Published Plates](page-segmentation.md)
`pylithics-pages` cuts a published plate into one image for each
artefact. It writes each scale bar to its own image. It reads the
identifier printed next to each lithic and names the crop with it. It
discards captions and legends.

### [Metadata Setup](metadata-setup.md)
How to prepare the metadata CSV file and the scale calibration:

- The columns and their format
- Scale bar detection
- Pixel measurements when there is no scale bar
- The link between an image and its scale
- The directory structure
- Missing scales and mixed calibration methods
- Example templates

### [Basic Usage](basic-usage.md)
How to do an analysis:

- The command line
- The necessary arguments
- Scale calibration examples and options
- The configuration file (`config.yaml`)
- The pipeline
- Analysis parameters

### [Outputs](outputs.md)
The files that PyLithics writes:

- The CSV data and its metrics
- The scale calibration columns
- The labelled images
- The Voronoi diagrams
- The log files and debug output
- The arrow detection results

### [Dashboard](dashboard.md)
The browser dashboard for your results:

- Summary numbers
- Distribution histograms with shared filters
- A page for each lithic, with the labelled image and the Voronoi diagram

### [Glossary](glossary.md)
The terms and metrics:

- All measurements
- Archaeological terms
- Technical definitions
- Units and calculations

## Sequence

1. **Start** → [Prepare Your Images](image-requirements.md) — make sure that your images are correct
2. **Next** → [Metadata Setup](metadata-setup.md) — prepare your CSV file
3. **Then** → [Basic Usage](basic-usage.md) — do your first analysis
4. **Last** → [Outputs](outputs.md) — read your results
