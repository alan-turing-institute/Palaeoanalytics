# Working from Published Plates

`pylithics-pages` cuts a published plate into one image for each
artefact. PyLithics analyses one artefact for each image. A published
plate shows many artefacts. This command makes the cut for you.

Give the command a folder of scanned pages. For each lithic, it writes
one image that contains all the surfaces of that lithic. It writes each
scale bar to its own image. It discards captions and legend text. It
reads the identifier printed next to each lithic and names the crop
with it. The crops are ready for `pylithics`.

## Procedure

!!! danger "The project folder rule"
    `--data_dir` is a folder that contains these three items, with
    these exact names:

    ```
    <your folder>/
    ├── images/          the lithic images
    ├── scales/          the scale bar images
    └── meta_data.csv    which scale goes with which image
    ```

    The folder can have any name and can be anywhere. The three names
    inside it cannot change. `pylithics-pages` writes the first three for you from the plates
    in `pages/`. `pylithics` writes the analysis to `results/`.

**1. Put the scanned plates in `pages/` in your project folder.**
The two commands share the project folder. `--data_dir` is the same
folder for both:

| Command | Reads | Writes |
|---|---|---|
| `pylithics-pages` | `pages/` | `images/`, `scales/`, `pages_manifest.csv`, `meta_data.csv` |
| `pylithics` | `images/`, `scales/`, `meta_data.csv` | `results/` |

```
my_project/
├── pages/           your scanned plates
├── images/          one crop for each artefact      ← pylithics-pages writes
├── scales/          one crop for each scale bar     ← pylithics-pages writes
├── pages_manifest.csv     how each crop was cut           ← pylithics-pages writes
├── meta_data.csv    each crop, its scale, a flag    ← pylithics-pages writes
└── results/         the analysis                    ← pylithics writes
```

A project can already hold images and a `meta_data.csv`. The crops go
beside the images, and the new rows go after your rows. To try the
workflow, use the sample project: `pylithics/data/pages/` holds a
sample plate.

**2. Cut the plates. Use `--debug` to write the overlays:**

```bash
pylithics-pages --data_dir my_project --debug
```

**3. Examine the overlays** in `my_project/pages_debug/`. Each page has
one overlay. Each numbered box must contain one lithic and all of its
surfaces. Each scale bar must have its own purple box. Each green
identifier must match the number on the plate. Read the header of the
overlay: it gives the number of crops named and the number of crops
flagged.

**4. Correct each page that is wrong.** See [Correct a
page](#correct-a-page). Then start the command again:

```bash
pylithics-pages --data_dir my_project --overrides corrections.csv
```

**5. Open `meta_data.csv`. Fill in the `scale` column. Correct each
flagged row.** See [meta_data.csv](#meta_datacsv). For the sample
plate, the bar is 5 cm, so the value is `50`.

**6. Analyse the crops:**

```bash
pylithics --data_dir my_project
```

If some rows have no scale value, the command asks once whether to
measure those images in pixels. Answer `n` to analyse only the images
with a scale. A row with a flag runs like any other, and the flags are
counted on the screen at the end. The results go to
`my_project/results/`. See [Basic Usage](basic-usage.md).

You can also give a folder of scans that is not a project. Then give
the output folder:

```bash
pylithics-pages --data_dir /mnt/archive/plates --output_dir ~/work/lyon
```

The log shows the folders that the command reads and writes.

## One artefact, one crop

**A lithic drawn with four surfaces is one artefact. It is not four.**

An illustrator draws one flake as a set of views: platform, dorsal,
ventral, lateral. A section is below the flake. A profile is next to
it. A short rule can connect two views. `pylithics-pages` puts all of
these views in one crop. A page with five lithics gives five images.
The number of surfaces does not change this.

The command discards captions, legend text and running heads. It does
not make crops of them. It does not attach them to a lithic. It keeps
the identifier next to each lithic.

## Identifiers

A published plate prints a number or a letter next to each lithic. The
text of the publication refers to the lithic by that identifier.
`pylithics-pages` reads it and names the crop with it.

| Result | Filename | `label_source` |
|---|---|---|
| The command read one identifier in the crop | `<page>_figure_7.png` | `read` |
| The command did not read one identifier | `<page>_box_07.png` | `index` |

The filename shows the method. `_figure_7` is the number printed on
the plate: the publication calls this lithic "7". `_box_07` is the
number of the red box on the debug overlay: the seventh box on the
page in reading order, left to right and top to bottom. The two
numbers have nothing to do with each other. `_figure_7` and `_box_07`
on the same page are usually two different lithics.

!!! note "Read the two names"
    - `plate_014_figure_7.png` — the lithic that the plate calls 7.
    - `plate_014_box_07.png` — the seventh box on the page. Its
      identifier was not read. Find it on the overlay to see which
      lithic it is.

The command does not name a crop when it is not sure. The manifest
column `label_flag` gives the reason:

| `label_flag` | Meaning |
|---|---|
| `several_identifiers` | The crop contains more than one identifier. The crop contains more than one lithic. The column `label_candidates` lists the identifiers. |
| `duplicate` | Two crops on the page have the same identifier. |
| `no_identifier` | The command found no identifier in or near the crop. |
| `not_read` | Identifier reading was off, or RapidOCR is not installed. |

The optional RapidOCR package is necessary to read identifiers:

```bash
pip install "PyLithics[ocr]"
```

Without it, the command names all crops by box number and writes one
message. Use `--no_read_labels` to name crops by box number on purpose.

Some plates print the identifier far from the lithic. Then the command
does not find it. Increase `identifiers.reach` in `config.yaml`. The
value is a number of glyph heights. The default is 1.5.

## The pixels do not change

The command cuts each crop from the source image. The crop has the DPI
and the colour mode of the source. No denoising, contrast change or
thresholding is applied to the crop.

This is necessary. The `pylithics` analysis does those steps. If
PyLithics does the steps two times, a measurement from a crop is not
comparable with a measurement from a single-artefact scan. The command uses a threshold
to find the ink. It does not keep the result.

## Output

```
my_project/
├── images/          one crop for each artefact
├── scales/          one crop for each scale bar
├── pages_debug/     one overlay for each page, `<page>.png` (--debug only)
├── pages_manifest.csv     one row for each crop
└── meta_data.csv    one row for each artefact crop
```

Artefact crops are `<page>_figure_<label>.png` or `<page>_box_<NN>.png`.
See [Identifiers](#identifiers). Scale bar crops are
`<page>_scale_bar.png`. If a page has more than one scale bar, the
crops are numbered: `<page>_scale_bar_01.png`.

### pages_manifest.csv

The manifest has one row for each crop. It connects each crop to its
source page.

| Column | Meaning |
|---|---|
| `output_crop_id` | The filename of the crop in `images/` or `scales/`. |
| `image_type` | `artefact` or `scale_bar`. |
| `input_page_id` | The filename of the plate in `pages/`. |
| `crop_id_index` | The box number of the crop on the debug overlay, in reading order. The same number as in `_box_NN`. Empty for a scale bar. |
| `x0, y0, x1, y1` | The box of the crop on the source page, in pixels. |
| `width_px, height_px` | The size of the crop, in pixels. |
| `dpi` | The DPI of the source page. Empty if the page has no DPI. |
| `colour_mode` | `greyscale`, `RGB` or `RGBA`. |
| `n_components` | The number of ink blobs in the crop. |
| `correction_applied` | The corrections used on this page: `expect`, `join`, `split`, or a combination such as `join+split`. Empty if no correction was used. |
| `label` | The identifier read from the plate. |
| `label_source` | `read` or `index`. See [Identifiers](#identifiers). |
| `label_confidence` | The confidence of the reader in `label`, from 0 to 1. |
| `label_flag` | The reason a crop has no `label`. See [Identifiers](#identifiers). |
| `label_candidates` | All identifiers read in the crop, separated by `;`. |

Keep the manifest with the crops. It connects a measurement to its
page.

### meta_data.csv

This is the file that `pylithics` reads. It has one row for each
artefact crop, in the same format as a metadata file that you write by
hand (see [Metadata Setup](metadata-setup.md)), plus a `flag` column.

| Column | Content |
|---|---|
| `image_id` | The filename of the crop in `images/`. |
| `scale_id` | The scale bar crop from the same page, when the page has exactly one. |
| `scale` | Empty. Fill in the length of the bar, in millimetres. |
| `flag` | Empty when the row is ready. If not, what you must examine. It does not stop the analysis. |

The flags:

| `flag` | Meaning | What to do |
|---|---|---|
| `no_scale` | The page has no scale bar. | Give a `scale_id` and a `scale` from another page, or remove the flag to measure in pixels. |
| `several_scales` | The page has more than one scale bar. `scale_id` is empty. | Examine the page. Write the correct `scale_id`. |
| `no_identifier`, `several_identifiers`, `duplicate`, `not_read` | An identifier problem. See [Identifiers](#identifiers). | Examine the crop. Correct the filename, or accept it. |

More than one flag is separated by `;`. A flag does not stop the
analysis. `pylithics` runs the row like any other, in pixels when
`scale` is empty, and at the end gives one count for each flag on the
screen and each flagged row in the log. Remove the flag when you have
examined the row.

!!! note "Pixels until you fill in the scale"
    A crop with no `scale` value can only be measured in pixels.
    `pylithics` asks once before it does that, and `n` leaves those
    crops out. Fill in the scale and start `pylithics` again to get
    millimetres.

## Correct a page

Page layouts differ between illustrators and publishers. Some pages
come out wrong. Do not change the distance settings for one page. Write
the correction in a CSV file:

```csv
page_id,expect,join,split
plate_014.jpg,4,,
plate_015.jpg,,3+4,
plate_021.jpg,,,2
```

| Column | Effect |
|---|---|
| `expect` | The number of artefacts on the page. The command changes the distances until it finds this number. |
| `join` | Merge two boxes, for example `3+4`. Separate more than one pair with a space. |
| `split` | Cut one box into two boxes at its widest empty column. |

The box numbers are the numbers on the `--debug` overlay. The
procedure is: start the command with `--debug`, read the numbers,
write the correction, start the command again with `--overrides`.
Pages that are not in the CSV file are cut again with no correction.

!!! warning "Write the CSV file with a spreadsheet or a CSV writer"
    Page filenames often contain commas. Then `page_id` must be in
    quotation marks. This is easy to get wrong by hand.

!!! note "`expect` is not precise"
    `expect` changes the distances until the count matches. It does not
    know which grouping is correct. It can find the correct count for
    the wrong reason. Use it to confirm a page that you have examined
    on the overlay. Do not use it to correct a page that you have not
    examined.

### When all pages show the same error

Use the CSV file for one page. Use these settings when all pages from
one publication show the same error:

| Error | Setting |
|---|---|
| One artefact is cut into two boxes | Increase `--gap` or `--vertical_gap` |
| Two artefacts are in one box | Decrease `--gap` or `--vertical_gap` |
| A profile view is not in its box | Increase `--narrow` |
| Labels become crops | Increase `--min_area` |
| An identifier is far from its lithic | Increase `identifiers.reach` in `config.yaml` |

All distances are fractions of the page width or the page height. The
same settings apply to pages of different sizes at the same resolution.

## Re-runs

Every run cuts every plate in `pages/`. The rules:

| | |
|---|---|
| A plate cut before | Cut again. Its crops, its manifest rows and its rows in `meta_data.csv` are replaced. The scale values that you typed stay. |
| A new plate | Cut. Its crops and rows are added. |
| A crop that the new run does not make | Removed, with its row. |
| Your own images, scale images and rows | Never changed. |
| A crop with the same filename as one of your images | Not written. The page is skipped, and the file is named. |

So you can add plates, change a setting, or write a correction, and
start the command again. Nothing stops you, and nothing that you typed
is lost.

## Scale bars

The command finds scale bars and writes each one to its own crop. It
does not put a scale bar in an artefact crop. It finds these styles:

- **Block bars** — filled and open blocks in a row, or in two rows
  like a chessboard. The bar can be across the page or down the page.
- **Ruled scales** — a line with tick marks, usually with a numeral at
  each end.

If a caption such as `5 cm` is next to the bar, the crop includes it.

!!! note "You write the scale values by hand"
    `pylithics-pages` writes the scale bar image and connects it to
    each crop from the same page in `meta_data.csv`. It does not read
    the number printed on the bar. Write the value in millimetres in
    the `scale` column. See [meta_data.csv](#meta_datacsv).

## Analyse the crops

Fill in `meta_data.csv`. Then start the analysis on the project:

```bash
pylithics --data_dir my_project
```

See [Basic Usage](basic-usage.md) for the analysis step.
