# Read PyLithics Output in Python, R and SPSS

This page shows how to read the two output formats that PyLithics
writes:

- **`processed_metrics.csv`** — a long-format table, with one row for each contour (a surface, a scar, a cortex area or an edge). PyLithics always writes it.
- **One JSON file for each lithic** — in `results/json/<image_id>.json`. PyLithics writes them only with `--export_json`.

Use the CSV for an analysis across an assemblage (each lithic and each
feature in one data frame). Use the JSON for the nested structure of
one lithic (for example, to go through the surfaces and the features
of one artefact, or for a web page for one lithic).

## CSV: one row for each contour

### Parent and child rows

Each row is a **parent** (a classified surface: Dorsal, Ventral,
Platform or Lateral) or a **child** (a feature on that surface: a scar,
an edge or a cortex area). Two columns show which:

- `surface_type` — the surface: `Dorsal`, `Ventral`, `Platform`, `Lateral` or `Unclassified`
- `surface_feature` — for a parent row, the same value as `surface_type`. For a child row, the feature: `scar 1`, `edge 1`, `cortex 1` and so on.

If `surface_feature == surface_type`, the row is a parent surface. If
not, the row is a child of the surface named in `surface_type`.

These columns are in each row: `image_id`, `surface_type`,
`surface_feature`, `centroid_x`, `centroid_y`, `technical_width`,
`technical_length`, `total_area`. Parent rows also have the surface
summaries (`scar_count`, `vertical_symmetry`, `horizontal_symmetry`,
`voronoi_num_cells`, `convex_hull_area`, and others). These are `NA` in
child rows. See [Outputs](outputs.md) and [Glossary](glossary.md) for
all columns.

### Python — pandas

```python
import pandas as pd

df = pd.read_csv("results/processed_metrics.csv")

# All surfaces (parent rows)
surfaces = df[df["surface_feature"] == df["surface_type"]]

# All scars on dorsal surfaces, for the full assemblage
dorsal_scars = df[
    (df["surface_type"] == "Dorsal")
    & df["surface_feature"].str.startswith("scar")
]

# The mean scar area for each lithic (dorsal only)
per_lithic_mean = (
    dorsal_scars
    .groupby("image_id")["total_area"]
    .mean()
    .rename("mean_dorsal_scar_area")
)
```

### R — tidyverse

```r
library(readr)
library(dplyr)

df <- read_csv("results/processed_metrics.csv")

# All surfaces (parent rows)
surfaces <- df %>% filter(surface_feature == surface_type)

# Scars on dorsal surfaces
dorsal_scars <- df %>%
  filter(surface_type == "Dorsal", startsWith(surface_feature, "scar"))

# The mean dorsal scar area for each lithic
per_lithic_mean <- dorsal_scars %>%
  group_by(image_id) %>%
  summarise(mean_dorsal_scar_area = mean(total_area, na.rm = TRUE))
```

### SPSS

SPSS reads the CSV with the import wizard. Missing values in the CSV
are the text `NA`. SPSS reads each column that contains `NA` as a
string variable, not a numeric variable. The procedure below imports
the file and then changes those columns to numeric.

**Import the CSV**:

1. Select **File > Import Data > CSV Data**.
2. Select `processed_metrics.csv` and click **Open**.
3. In the dialog, make sure that these options are set:
    - **First line contains variable names**: on
    - **Delimiter between values**: comma
    - **Decimal symbol**: period
    - **Text qualifier**: double quote
4. Click **OK**.

**Change the numeric columns**. Paste this syntax into a Syntax window
and start it. `ALTER TYPE` changes each `NA` to a system-missing
value. The `TO` ranges use the column order of the CSV.

```text
ALTER TYPE scar_count TO lateral_convexity (F12.4).
ALTER TYPE cortex_area TO scar_complexity (F12.4).
ALTER TYPE arrow_angle (F12.4).
ALTER TYPE pixels_per_mm TO scale_confidence (F12.4).
```

`is_cortex` and `has_arrow` stay as string variables with the values
`True` and `False`. If your CSV has the optional arrow geometry columns
(see [Glossary](glossary.md#arrow-geometry-optional)), add them to the
syntax.

**The same analysis as above**:

```text
* Parent rows: the surface name is in both columns.
COMPUTE is_surface = (surface_feature = surface_type).

* Scars on dorsal surfaces.
COMPUTE is_dorsal_scar =
    (surface_type = "Dorsal" AND CHAR.INDEX(surface_feature, "scar") = 1).
EXECUTE.

* The mean dorsal scar area for each lithic, in a new dataset.
DATASET NAME metrics.
DATASET COPY dorsal_scars.
DATASET ACTIVATE dorsal_scars.
SELECT IF is_dorsal_scar = 1.
AGGREGATE
    /OUTFILE=*
    /BREAK=image_id
    /mean_dorsal_scar_area = MEAN(total_area).
```

The aggregated dataset `dorsal_scars` has one row for each lithic. The
dataset `metrics` keeps all rows.

### Calibration

Measurements (`technical_width`, `total_area`, `voronoi_cell_area`, and
others) are in **millimetres** when the scale bar calibration is
correct, and in **pixels** when it is not. Examine `calibration_method`
before you mix values:

```python
mm_rows = df[df["calibration_method"] == "scale_bar"]
px_rows = df[df["calibration_method"] == "pixels"]
```

In SPSS: `SELECT IF calibration_method = "scale_bar".`

## JSON: one file for each lithic

Each JSON file in `results/json/<image_stem>.json` is one
object with this structure. SPSS does not read these files. Use the
CSV in SPSS.

```text
schema_version
image_id
calibration   { method, pixels_per_mm, scale_confidence }
surfaces      [ a list of surface objects, each with: ]
              surface_type, surface_feature,
              centroid_x, centroid_y, technical_width, technical_length, total_area,
              scar_count,                              (Dorsal only)
              voronoi { num_cells, cell_area, convex_hull_* },
              symmetry { top_area, bottom_area, vertical_symmetry, horizontal_symmetry },
              lateral_convexity,
              features [ a list of feature objects (scars, cortex, edges, arrows) ]
```

### Python — standard library

```python
import json
from pathlib import Path

records = [
    json.loads(p.read_text())
    for p in Path("results/json").glob("*.json")
]

# The total number of scars for all lithics
total_scars = sum(
    surface["scar_count"]
    for rec in records
    for surface in rec["surfaces"]
    if surface["scar_count"] is not None
)
```

### Python — pandas (one row for each scar)

```python
import json
from pathlib import Path
import pandas as pd

rows = []
for path in Path("results/json").glob("*.json"):
    rec = json.loads(path.read_text())
    for surface in rec["surfaces"]:
        for feat in surface.get("features", []):
            rows.append({
                "image_id": rec["image_id"],
                "surface_type": surface["surface_type"],
                "surface_feature": feat.get("surface_feature"),
                "total_area": feat.get("total_area"),
            })

scars_long = pd.DataFrame(rows)
```

### R — jsonlite

```r
library(jsonlite)
library(dplyr)
library(purrr)

paths <- list.files("results/json", pattern = "\\.json$", full.names = TRUE)
records <- lapply(paths, fromJSON, simplifyVector = FALSE)

# One row for each scar, for the full assemblage
scars_long <- map_dfr(records, function(rec) {
  map_dfr(rec$surfaces, function(surface) {
    if (length(surface$features) == 0) return(NULL)
    map_dfr(surface$features, function(feat) {
      tibble(
        image_id = rec$image_id,
        surface_type = surface$surface_type,
        surface_feature = feat$surface_feature %||% NA,
        total_area = feat$total_area %||% NA
      )
    })
  })
})
```

## Which format for which task

| Task | Use |
|---|---|
| Statistics across an assemblage (for example, the mean scar area for each site) | CSV |
| The full structure of one lithic | JSON |
| A database, or a statistics package such as SPSS | CSV |
| A web page for one lithic | JSON |
| The nested structure (surface → features) | JSON |
| One data frame for ggplot or plotnine | CSV |
