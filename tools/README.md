# Development tools

Scripts for the people who develop PyLithics. They are not installed with
the package and the user documentation does not describe them. Run each
one from the repository root with the project's Python, for example
`.venv/bin/python tools/arrow_variants.py --help`.

## Arrow classifier

| Script | What it does |
|--------|--------------|
| `arrow_variants.py` | Makes the training rows for the arrow classifier from a folder of arrow images (one arrow each, drawn pointing one way) and a folder of non-arrow marks. Each image gives variants at four sizes, every 15 degrees and three line weights. The label comes from the rotation, so no hand labelling. |
| `train_arrow_classifier.py` | Trains the arrow random forest from a row table, reports the held-out misses, flips and false arrows, and writes `pylithics/models/arrow_classifier.json`. |
| `arrow_candidates.py` | Runs the arrow detector over a folder of real drawings and writes one row for each mark, an overlay for each drawing and numbered contact sheets. Use it to see what the detector accepts and rejects on real plates, and to label marks by hand. |

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
