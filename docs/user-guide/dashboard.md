# Dashboard

PyLithics ships with an interactive dashboard for exploring the results of a batch run. It opens in your default browser and reads from the `processed_metrics.csv` your normal analysis writes — no extra setup, no separate export step.

When you analyze, PyLithics writes outputs to `<data_dir>/processed/`. When you later open the dashboard against an existing run, point `--data_dir` directly at whichever folder contains the `processed_metrics.csv` you want to explore — the folder name doesn't have to be `processed/`, and you can keep multiple runs in separate folders.

## Launching

There is one flag, `--explore`, with two flows:

```bash
# First-time use: analyze and immediately open the dashboard.
# --data_dir is the project root (contains images/ and scales/);
# analysis writes outputs to ./assemblage/processed/ and the
# dashboard opens against that folder automatically.
pylithics --data_dir ./assemblage --meta_file ./meta.csv --explore

# Re-open the dashboard later (no re-analysis).
# --data_dir is now the folder that actually contains processed_metrics.csv
# — usually <project_root>/processed/, but it can be any folder.
pylithics --data_dir ./assemblage/processed --explore
pylithics --data_dir ./other_run_folder --explore   # different assemblage, same flag
```

The dashboard binds to `http://localhost:8501` by default. Press `Ctrl+C` in the terminal to stop it.

When the command starts you'll see a one-line banner and a text progress bar — PyLithics imports a large stack (OpenCV, pandas, Streamlit, Plotly) before the browser opens, and the bar makes that wait visible rather than silent. Once the dashboard subprocess is listening on the port, the bar clears and the URL prints; your default browser opens automatically.

If you pass `--explore` without an existing `processed_metrics.csv` and without `--meta_file`, PyLithics exits with an error telling you to run analysis first.

## Pages

The dashboard has three pages, switchable from the sidebar.

### 1. Overview

Top-line headline tiles, each carrying a small status caption — `✓ All clear` in green when the count is zero and `⚠ Review` in red when there is something to look at:

- **Lithics processed** — total distinct images
- **Calibrated (scale_bar)** — how many were measured in real-world units
- **Arrow detection rate** — fraction of dorsal scars with a detected arrow
- **Cortex prevalence** — fraction of lithics with at least one cortex feature
- **Surface types** bar chart — Dorsal / Ventral / Platform / Lateral / Unclassified counts
- **Calibration method** pie chart — `scale_bar` vs `pixels`

### 2. Distributions

Numerical exploration of the assemblage. A filter at the top of the page (surface type and calibration method) applies to every chart below it. The page is organised into five thematic tabs:

**Size & shape.** Raincloud of `aspect_ratio` by surface, lollipop of `perimeter` per lithic, length × width scatter (colored by surface type), and a convex-hull-area vs. total-area scatter for shape regularity.

**Symmetry.** Two paired charts on dorsal surfaces only:
- *Signed asymmetry scatter* — horizontal vs. vertical bias of each dorsal centroid. Right of the y-axis = right-leaning, above x-axis = top-heavy. (0, 0) is perfect symmetry.
- *Paired ECDFs* — cumulative distribution of vertical and horizontal symmetry scores. Each dot is one lithic; hover for the image ID.

**Scars.** Four sections built around the relationships your statistical analyses validated as informative:
- *Scarring relationships* — scars-per-dorsal and coverage-% scatters against dorsal area, with a linear fit and red halos on lithics > 2 SD from the trend.
- *Scar complexity* — population histogram alongside a per-lithic strip plot (lithics ordered by median complexity).
- *Scar size & shape* — paired ECDFs of per-scar `total_area` (log-x) and `aspect_ratio`.
- *Scar-size variability* — coefficient-of-variation vs. count, separating monotonous from mixed-strategy reduction.

**Spatial.** Voronoi cell-count and convex-hull metrics for dorsal surfaces.

**Cortex.** Distribution of `cortex_percentage` and presence/absence by surface.

Every chart that identifies a specific lithic starts its tooltip with `Lithic: <image_id>` on the first line, and every analytical chart carries an `About this plot` expander with a plain-English description of what the visualisation shows and how to read it. Plotly's built-in toolbar (camera icon on hover) downloads any chart as a PNG.

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
