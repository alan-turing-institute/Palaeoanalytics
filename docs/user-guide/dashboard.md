# Dashboard

PyLithics ships with an interactive dashboard for exploring the results of a batch run. It opens in your default browser and reads from the same `processed/processed_metrics.csv` your normal analysis writes — no extra setup, no separate export step.

## Launching

There is one flag, `--explore`, with two flows:

```bash
# First-time use: analyze and immediately open the dashboard
pylithics --data_dir ./assemblage --meta_file ./meta.csv --explore

# Re-open the dashboard later (no re-analysis)
pylithics --data_dir ./assemblage --explore
```

The dashboard binds to `http://localhost:8501` by default. Press `Ctrl+C` in the terminal to stop it.

If you pass `--explore` without an existing `processed_metrics.csv` and without `--meta_file`, PyLithics exits with an error telling you to run analysis first.

## Pages

The dashboard has three pages, switchable from the sidebar.

### 1. Overview

Top-line headline numbers and two summary charts:

- **Lithics processed** — total distinct images
- **Calibrated (scale_bar)** — how many were measured in real-world units
- **Arrow detection rate** — fraction of dorsal scars with a detected arrow
- **Cortex prevalence** — fraction of lithics with at least one cortex feature
- **Surface types** bar chart — Dorsal / Ventral / Platform / Lateral / Unclassified counts
- **Calibration method** pie chart — `scale_bar` vs `pixels`

### 2. Distributions

Numerical exploration of the assemblage. A filter at the top of the page (surface type and calibration method) applies to every chart below it.

- Histograms of `technical_length`, `technical_width`, `total_area`, `aspect_ratio`
- Length × width scatter, colored by surface type
- Scars per dorsal surface — histogram
- `scar_complexity` histogram with the mean line overlaid
- Vertical vs horizontal symmetry scatter, with the 0.95–1.0 "near-perfect" zone shaded
- `voronoi_num_cells` histogram

Every chart has Plotly's built-in toolbar — hover over a chart to find the camera icon and download a PNG of just that chart.

### 3. Per-Lithic Detail

Drill-down view for a single lithic. Pick an image from the dropdown to see:

- The labeled image (`*_labeled.png`)
- The Voronoi diagram (`*_voronoi.png`)
- All CSV rows for that image as a sortable table
- The per-lithic JSON (if you ran with `--export_json`) as a collapsible tree

## Cross-platform notes

Streamlit and Plotly support Linux, macOS, and Windows. The dashboard launches the same way on all three. If port 8501 is already in use, Streamlit will pick the next free port and tell you in the terminal.

## When the dashboard is the wrong tool

The dashboard is designed for quick exploration of a single processed batch. If you want to:

- Compare two batches side-by-side
- Produce publication-figure-ready charts
- Run statistical tests across an assemblage

…load `processed_metrics.csv` (or the per-lithic JSON files from `--export_json`) into your usual R or Python workflow. The dashboard is a complement to those workflows, not a replacement.
