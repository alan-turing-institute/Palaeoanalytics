# Loading PyLithics Output in Python and R

This page shows how to load and work with the two output formats PyLithics writes:

- **`processed_metrics.csv`** — long-format table, one row per contour (surface OR scar OR cortex OR edge). Always written.
- **Per-lithic JSON** — one nested file per artefact in `processed/json/<image_id>.json`. Written only when `--export_json` is set.

Use the CSV for cross-assemblage analysis (every lithic and every feature in one frame). Use the JSON when you need the nested per-lithic structure (e.g. iterating surfaces → features for a single artefact, or feeding a per-lithic web view).

## CSV: one row per contour

### Row-type basics

Each row is either a **parent** (a classified surface — Dorsal, Ventral, Platform, Lateral) or a **child** (a feature on that surface — scar, edge, cortex). Two columns distinguish them:

- `scar` — the contour's own id (e.g. `parent 1`, `scar 1`, `cortex 1`)
- `parent` — the parent surface's id

If `scar == parent`, the row is a parent surface. Otherwise it's a child of the named parent.

Per-row columns that always exist include `image_id`, `surface_type`, `surface_feature`, `centroid_x`, `centroid_y`, `technical_width`, `technical_length`, `total_area`. Parent rows also carry the surface-level summaries (`scar_count`, `vertical_symmetry`, `horizontal_symmetry`, `voronoi_num_cells`, `convex_hull_area`, etc.) which are `NA` on child rows. See the [Outputs](outputs.md) and [Glossary](glossary.md) pages for the full column list.

### Python — pandas

```python
import pandas as pd

df = pd.read_csv("processed/processed_metrics.csv")

# All surfaces (parent rows)
surfaces = df[df["scar"] == df["parent"]]

# All scars on dorsal surfaces, across the whole assemblage
dorsal_parents = surfaces[surfaces["surface_type"] == "Dorsal"]
dorsal_scars = df.merge(
    dorsal_parents[["image_id", "scar"]].rename(columns={"scar": "parent"}),
    on=["image_id", "parent"],
)
dorsal_scars = dorsal_scars[dorsal_scars["scar"] != dorsal_scars["parent"]]

# Mean scar area per lithic (dorsal only)
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

df <- read_csv("processed/processed_metrics.csv")

# All surfaces (parent rows)
surfaces <- df %>% filter(scar == parent)

# Scars on dorsal surfaces
dorsal_parents <- surfaces %>% filter(surface_type == "Dorsal")
dorsal_scars <- df %>%
  inner_join(
    dorsal_parents %>% select(image_id, parent = scar),
    by = c("image_id", "parent")
  ) %>%
  filter(scar != parent)

# Mean dorsal scar area per lithic
per_lithic_mean <- dorsal_scars %>%
  group_by(image_id) %>%
  summarise(mean_dorsal_scar_area = mean(total_area, na.rm = TRUE))
```

### Calibration awareness

Linear measurements (`technical_width`, `total_area`, `voronoi_cell_area`, etc.) are in **millimetres** when scale-bar calibration succeeded and in **pixels** otherwise. Always check `calibration_method` before mixing values:

```python
mm_rows = df[df["calibration_method"] == "scale_bar"]
px_rows = df[df["calibration_method"].isin(
    ["pixels_no_scale", "pixels_detection_failed"]
)]
```

## JSON: one file per lithic

Each JSON file under `processed/json/<image_stem>.json` is a single object with this shape:

```text
schema_version
image_id
calibration   { method, pixels_per_mm, scale_confidence }
surfaces      [ list of surface objects, each with: ]
              surface_type, surface_feature,
              centroid_x, centroid_y, technical_width, technical_length, total_area,
              scar_count,                              (only on Dorsal)
              voronoi { num_cells, cell_area, convex_hull_* },
              symmetry { top_area, bottom_area, vertical_symmetry, horizontal_symmetry },
              lateral_convexity,
              features [ list of feature objects (scars, cortex, edges, arrows) ]
```

### Python — stdlib

```python
import json
from pathlib import Path

records = [
    json.loads(p.read_text())
    for p in Path("processed/json").glob("*.json")
]

# Total scar count across all lithics
total_scars = sum(
    surface["scar_count"]
    for rec in records
    for surface in rec["surfaces"]
    if surface["scar_count"] is not None
)
```

### Python — pandas (flatten to one row per scar)

```python
import json
from pathlib import Path
import pandas as pd

rows = []
for path in Path("processed/json").glob("*.json"):
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

paths <- list.files("processed/json", pattern = "\\.json$", full.names = TRUE)
records <- lapply(paths, fromJSON, simplifyVector = FALSE)

# Flatten to one row per scar across the whole assemblage
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

## When to use which

| Task | Use |
|---|---|
| Cross-assemblage statistics (e.g. mean scar area by site) | CSV |
| Compare a single lithic's full structure | JSON |
| Feed a relational database or stats package | CSV |
| Build a per-lithic web view | JSON |
| Need nested structure (surface → features) preserved | JSON |
| Want everything in one DataFrame for ggplot/plotnine | CSV |
