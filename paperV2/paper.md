---
title: 'PyLithics 2.0: Enhanced quantitative analysis of prehistoric stone tool illustrations'
tags:
  - Python
  - archaeology
  - lithic analysis
  - computer vision
  - prehistoric technology
  - human evolution
authors:
  - name: Jason J. Gellis
    orcid: 0000-0002-9929-789X
    corresponding: true
    affiliation: "1, 2, 3"
  - name: Camila Rangel Smith
    orcid: 0000-0002-0227-836X
    affiliation: 1
  - name: Robert A. Foley
    orcid: 0000-0003-0479-3039
    affiliation: "2, 3"
affiliations:
  - name: The Alan Turing Institute, United Kingdom
    index: 1
  - name: University of Cambridge, United Kingdom
    index: 2
  - name: Leverhulme Centre for Human Evolutionary Studies, United Kingdom
    index: 3
date: 17 June 2026
bibliography: paper.bib
---

# Summary

PyLithics 2.0 represents a major advancement in automated quantitative analysis of prehistoric stone tool (lithic) illustrations. Building on version 1.0 [@Gellis:2021], this release introduces sophisticated spatial analysis, morphological characterization, and expanded measurement capabilities. The software processes 2D line drawings of lithic artifacts using advanced computer vision techniques to extract comprehensive morphometric data. Key innovations include Voronoi-based spatial distribution analysis, bilateral symmetry quantification, scar complexity metrics based on adjacency relationships, lateral edge convexity analysis, automated cortex detection, structured per-lithic JSON export, and a Streamlit-based dashboard that surfaces assemblage-level distributions and per-lithic drilldowns without writing code. The modular architecture with YAML-based configuration enables researchers to customize analysis workflows for diverse illustration styles and research questions. PyLithics 2.0 addresses critical needs in archaeological lithic analysis by automating time-intensive measurements, ensuring inter-observer consistency, and enabling large-scale comparative studies essential for understanding prehistoric technological evolution and hominin behavior.

# Statement of Need

Lithic artifacts—stone tools created through controlled fracturing—provide the primary evidence for understanding prehistoric technology and cognitive evolution spanning over 3 million years [@Shea:2013]. While technological approaches to lithic analysis have become increasingly quantitative [@Clarkson:2013], significant methodological barriers persist. Manual measurement of complex morphological features, particularly flake scars on dorsal surfaces, is extremely time-consuming and subject to inter-observer variability [@Eren:2011]. Published lithic illustrations contain vast amounts of underutilized morphological information that could illuminate manufacturing techniques, skill levels, and cultural patterns [@Inizan:1999].

PyLithics 1.0 demonstrated the viability of automated feature extraction from lithic illustrations [@Gellis:2021], but with limited sophisticated analytical capabilities required for modern lithic studies. Version 2.0 addresses these limitations through: (1) spatial distribution analysis via Voronoi tessellation, enabling quantification of scar patterning indicative of reduction strategies [@Clarkson:2006]; (2) bilateral symmetry metrics critical for identifying intentional shaping versus opportunistic flaking [@Lycett:2007]; (3) scar complexity measures capturing reduction intensity and knapping skill [@Muller:2017]; (4) lateral convexity quantification relevant to functional interpretations [@Key:2016]; and (5) cortex detection for analyzing raw material economy and reduction stages [@Dibble:1995]. The modular, configuration-driven architecture accommodates diverse illustration conventions while maintaining analytical rigor. By dramatically reducing analysis time and eliminating inter-observer bias, PyLithics 2.0 enables the large-scale, high-dimensional datasets necessary for addressing fundamental questions about human technological evolution.

# Methods and Architecture

PyLithics 2.0 employs a modular pipeline architecture processing lithic illustrations through specialized analysis modules (\autoref{fig:pipeline}). The software accepts scanned images of lithic drawings following standard archaeological illustration conventions [@Martingell:1988], where artifacts are typically drawn at 1:1 scale with the striking platform oriented perpendicular to the page vertical axis.

The analysis workflow comprises:

1. **Image preprocessing and calibration**: Scale bar detection and DPI-based pixel-to-millimeter conversion using custom template matching and geometric validation
2. **Contour extraction**: Hierarchical contour detection (OpenCV RETR_TREE) preserving parent-child relationships between surfaces and scars
3. **Surface classification**: Rule-based classification of parent contours into Dorsal, Ventral, Platform, and Lateral surfaces based on area ratios and positional relationships
4. **Geometric metric calculation**: Comprehensive morphometric measurements including area, technical dimensions, aspect ratios, and shape descriptors
5. **Symmetry analysis**: Bilateral symmetry quantification using area-weighted centroids and reflection-based comparisons for dorsal surfaces
6. **Voronoi spatial analysis**: Tessellation of scar centroids with boundary-constrained Voronoi cell construction and convex hull calculations [@Okabe:2000]
7. **Scar complexity analysis**: Adjacency detection between neighboring scars using configurable distance thresholds to quantify reduction intensity
8. **Cortex detection**: Automated identification of cortical areas (original stone surface) using texture analysis and grayscale intensity profiling
9. **Arrow detection**: DPI-aware detection of flaking direction indicators using nested contour hierarchy analysis
10. **Lateral convexity analysis**: Curvature quantification of lateral edges using geometric approximation methods
11. **Data export**: Comprehensive CSV output, optional per-lithic JSON files (`--export_json`), and annotated visualization images

The modular design allows selective activation of analysis components through YAML configuration files or command-line arguments. All processing modules include graceful error handling with fallback mechanisms to maximize data recovery from imperfect illustrations.

An opt-in interactive dashboard (`--explore`) reads the same outputs in a browser-based Streamlit application, organising assemblage-level distributions across thematic tabs (size and shape, symmetry, scars, spatial, cortex) and providing per-lithic drill-downs against the labeled image and Voronoi diagram. It lets archaeologists inspect results without writing code, complementing downstream R/Python workflows.

PyLithics 2.0 depends on NumPy [@Harris:2020], SciPy [@Virtanen:2020], and Pandas [@McKinney:2010] for numerical operations, OpenCV [@opencv_library] for computer vision, Shapely [@Gillies:2007] for geometric operations, Matplotlib [@Hunter:2007] for static visualization, and Streamlit with Plotly for the interactive dashboard.

# Results and Output

PyLithics 2.0 generates four primary outputs:

**CSV data files** containing hierarchical measurements organized by surface type (Dorsal, Ventral, Platform, Lateral) and feature (individual scars, cortex areas). Measurements include: technical dimensions (length, width, thickness in mm), morphometric properties (area, perimeter, aspect ratio, circularity), symmetry scores (vertical and horizontal bilateral symmetry), spatial metrics (Voronoi cell areas, convex hull dimensions), complexity measures (scar adjacency counts), cortex coverage, and directional data (arrow angles indicating flaking direction). The tabular format enables direct integration with statistical software (R, Python, SPSS) for downstream analysis.

**Per-lithic JSON files** (`--export_json`) serialize the same measurements as structured records under `processed/json/`, with surfaces, scars, cortex blocks, symmetry, and arrows nested by relationship. JSON is well suited to programmatic pipelines and the interactive dashboard's drill-down view.

**Annotated visualization images and Voronoi diagrams** with color-coded contour overlays: purple for surface boundaries, orange for scars, red for cortex, light blue for arrows, and mint green for lateral edges (\autoref{fig:output}). Voronoi tessellations of dorsal scar centroids visualize spatial distribution and reduction strategies, with cells color-coded by area.

**Interactive dashboard** (`--explore`) provides a browser-based Streamlit interface that reads the CSV and JSON outputs in place. It organises distributions across thematic tabs, supports per-lithic drill-downs against the labeled image and Voronoi diagram, and gives non-programmer collaborators direct access to the data without changing the underlying outputs. Together these outputs ensure analytical transparency and let researchers verify automated measurements against archaeological expertise.

# Acknowledgements

PyLithics 2.0 was developed at University of Cambridge, and the Leverhulme Centre for Human Evolutionary Studies. Funding was provided by the British Academy.

# References
