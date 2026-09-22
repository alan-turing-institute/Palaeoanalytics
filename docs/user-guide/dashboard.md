# Dashboard

PyLithics has a dashboard for the results of a batch. It opens in your
browser. It reads the `processed_metrics.csv` that the analysis writes.
No other setup is necessary.

The analysis writes its output to `<data_dir>/results/`. To open the
dashboard for a previous analysis, give `--explore` the folder that
contains the `processed_metrics.csv` that you want to see. The folder
can have any name. You can keep more than one analysis, each in its
own folder.

## Open the dashboard

There is one flag, `--explore`, with two uses:

```bash
# First use: analyse and then open the dashboard.
# --data_dir is the project folder (with images/, scales/ and
# meta_data.csv). The analysis writes to ./assemblage/results/, and the
# dashboard opens for that folder.
pylithics --data_dir ./assemblage --explore

# Second use: open the dashboard for a previous analysis. No new
# analysis. Give the folder that contains processed_metrics.csv.
pylithics --explore ./assemblage/results
pylithics --explore ./tanzania_run_2025
```

The dashboard is at `http://localhost:8501` by default. Press `Ctrl+C`
in the terminal to stop it.

When the command starts, it shows one line and a progress bar.
PyLithics imports large libraries (OpenCV, pandas, Streamlit, Plotly)
before the browser opens. The progress bar shows this wait. When the
dashboard is ready, the URL shows and your browser opens.

If the folder that you give to `--explore` has no
`processed_metrics.csv`, the command stops with an error. Do an
analysis first.

## Pages

The dashboard has three pages. Select a page in the sidebar.

### 1. Overview

Summary tiles. Each tile has a small status: `✓ All clear` in green
when the count is zero, and `⚠ Review` in red when there is something
to examine:

- **Lithics processed** — the number of images
- **Calibrated (scale_bar)** — the number of images measured in millimetres
- **Arrow detection rate** — the fraction of dorsal scars with an arrow
- **Cortex prevalence** — the fraction of lithics with at least one cortex feature
- **Surface types** bar chart — the number of Dorsal, Ventral, Platform, Lateral and Unclassified surfaces
- **Calibration method** pie chart — `scale_bar` and `pixels`

### 2. Distributions

The numbers of the assemblage. A filter at the top of the page (surface
type and calibration method) applies to all charts on the page. The
page has four tabs:

**Size & shape.** A raincloud plot of `aspect_ratio` for each surface
type, a lollipop plot of `perimeter` for each lithic, a scatter plot of
length and width (with a colour for each surface type), and a scatter
plot of convex-hull area and total area for shape regularity.

**Symmetry.** Two charts, for dorsal surfaces only:
- *Signed asymmetry* — the horizontal and vertical offset of each dorsal centroid. Right of the y axis is a right offset. Above the x axis is a top offset. (0, 0) is perfect symmetry.
- *Paired ECDFs* — the cumulative distribution of the vertical and horizontal symmetry scores. Each dot is one lithic. Move the pointer over a dot to see the image ID.

**Scars.** Four sections:
- *Scarring relationships* — the number of scars and the scar coverage against the dorsal area, with a linear fit. A red circle marks a lithic more than 2 SD from the fit.
- *Scar complexity* — a histogram of the population, and a strip plot for each lithic (in the order of the median complexity).
- *Scar size & shape* — paired ECDFs of `total_area` (log x axis) and `aspect_ratio` for each scar.
- *Scar-size variability* — the coefficient of variation against the count. This separates one reduction strategy from mixed strategies.

**Spatial.** Three views of the scar positions on each dorsal surface:
- *Voronoi cells per dorsal surface* — a histogram of the cell counts.
- *Convex hull area vs dorsal area* — a scatter plot that shows how the hull of the scar centroids increases with the surface size.
- *Scar-Centroid Dispersion (Hull Area / Dorsal Area)* — a sorted lollipop plot of the ratio for each lithic.

Each chart that shows one lithic starts its tooltip with `Lithic:
<image_id>`. Each chart has an `About this plot` section with a
description of the chart and how to read it. The Plotly toolbar
(the camera icon) writes a chart as a PNG file.

### 3. Per-Lithic Detail

One lithic. Select an image in the list to see:

- The labelled image (`*_labeled.png`)
- The Voronoi diagram (`*_voronoi.png`)
- All CSV rows for the image, in a table that you can sort
- The JSON file for the lithic (if you used `--export_json`), as a tree

The labelled image and the Voronoi diagram are shown in a 700×700
square, so the two panels always have the same size on the screen.
The PNG files on the disk do not change.

## Operating systems

Streamlit and Plotly operate on Linux, macOS and Windows. The
dashboard opens the same way on all three. If port 8501 is in use,
Streamlit uses the next free port and shows it in the terminal.

## When the dashboard is not the correct tool

The dashboard is for a quick examination of one batch. For these
tasks, read `processed_metrics.csv` (or the JSON files from
`--export_json`) in R or Python:

- Compare two batches
- Make charts for a publication
- Do statistical tests on an assemblage

The dashboard is an addition to R and Python. It does not replace them.
