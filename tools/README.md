# Development tools

Scripts for the people who develop PyLithics. They are not installed with
the package and the user documentation does not describe them. Run each
one from the repository root with the project's Python, for example
`.venv/bin/python tools/arrow_variants.py --help`.

The training scripts (`train_arrow_classifier.py` and `train_scale_classifier.py`)
need scikit-learn: `pip install ".[train]"`. The package itself scores the bundled
models with numpy and does not need it.

## Training material

`training/` holds what the classifiers are trained from and nothing the package
reads: `training/arrows/`, one arrow for each image, all pointing down;
`training/non_arrows/`, marks that are not arrows. The hand labels that point into
real pages are `pylithics/models/arrow_labels.csv` and
`pylithics/models/scale_bar_labels.csv`. The generated arrow table is not kept.

## Arrow classifier

| Script | What it does |
|--------|--------------|
| `arrow_variants.py` | Makes training rows for the arrow classifier from a folder of arrow images (one arrow each, drawn pointing one way) and a folder of non-arrow marks. Each image gives variants at five sizes and its own, every 15 degrees and three line weights, each decided by the detector's two stages on its ink. The label comes from the rotation, so no hand labelling. |
| `train_arrow_classifier.py` | Trains the stage 2 forest from a row table (generated rows and labelled real rows together), reports what stage 1 and the vetoes did against the labels and the held-out misses, flips and false arrows on the rows left to the forest, and writes `pylithics/models/arrow_classifier.json`. |
| `arrow_candidates.py` | Runs the two-stage arrow detector over a folder of real drawings, on the marks inside each closed contour cut from the working copy as the analysis does it, and writes one row for each mark with the stage that decided it, an overlay for each drawing and numbered contact sheets. Boxes are in source pixels. Use it to see what the detector accepts and rejects on real plates, and to label marks by hand (`--labels` keeps known labels, matched by box overlap; the project's labels are `pylithics/models/arrow_labels.csv`, and a label with the arrow's true compass `angle` is read as `yes` or `flipped` against the direction drawn in the current run). |

## Scale bar classifier

| Script | What it does |
|--------|--------------|
| `scale_candidates.py` | Lists every scale bar candidate on a folder of pages with its features, for labelling, with overlays and contact sheets. |
| `train_scale_classifier.py` | Trains the scale bar random forest from a labelled candidate table and writes `pylithics/models/scale_bar_classifier.json`. |

## Shared

| Script | What it does |
|--------|--------------|
| `forest_training.py` | The fit, held-out scoring and JSON export that both trainers use. Not run on its own. |

## Checks

| Script | What it does |
|--------|--------------|
| `pages_eval.py` | Scores a `pylithics-pages` run against `pylithics/data/corrections.csv`: the `expect` column gives the number of lithics on each page and the `scales` column the number of scale bars. Reports the pages whose counts differ. |
| `analysis_eval.py` | Measures how the analysis changes with input resolution. Makes smaller copies of good images and compares the surfaces, scars and areas found on each copy with the source. |

A contact sheet is one image of many numbered thumbnails. It lets a person
check hundreds of detections by eye in minutes.
