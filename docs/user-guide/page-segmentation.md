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

    The folder can have any name and can be anywhere. You cannot
    change the three names inside it. `pylithics-pages` writes the
    first three for you from the plates in `pages/`. `pylithics`
    writes the analysis to `results/`.

**1. Put the scanned plates in `pages/` in your project folder.** You
can also put a PDF file there. See [PDF input](#pdf-input). The two
commands use the same project folder. `--data_dir` is the same folder
for the two commands:

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

A project can already hold images and a `meta_data.csv`. The command
puts the crops next to the images. It adds the new rows after your
rows. To try the workflow, use the sample project.
`pylithics/data/pages/` holds a sample plate.

**2. Cut the plates. Use `--debug` to write the overlays:**

```bash
pylithics-pages --data_dir my_project --debug
```

**3. Examine the overlays** in `my_project/pages_debug/`. Each page has
one overlay. Each numbered box must contain one lithic and all of its
surfaces. Each scale bar must have its own purple box, with its score
next to it. A grey box is a mark that the command rejected. Each
green identifier must match the number on the plate. Read the header
of the overlay. It gives the number of named crops and the number of
flagged crops.

**4. Correct each page that is wrong.** See [Correct a
page](#correct-a-page). Then start the command again:

```bash
pylithics-pages --data_dir my_project
```

**5. Open `meta_data.csv`. Fill in the `scale` column. Correct each
flagged row.** See [meta_data.csv](#meta_datacsv). For the sample
plate, the bar is 5 cm. Thus the value is `50`.

**6. Analyse the crops:**

```bash
pylithics --data_dir my_project
```

If some rows have no scale value, the command asks one time if it
must measure those images in pixels. Answer `n` to analyse only the
images with a scale. A row with a flag runs as all other rows do. At
the end, the command shows the count of the flags on the screen. The
results go to `my_project/results/`. See [Basic Usage](basic-usage.md).

You can also give a folder of scans that is not a project. Then give
the folder for the results:

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

## Text that is not an artefact

A plate has text that is not a label next to a lithic. Examples are a
specimen code in a frame, a site name and a caption. Two tests keep
this text off the artefact list.

The first test is on shape, before the grouping. A word is many
letter-sized pieces of ink on one baseline. The command discards it
and never exports it. A heading in large type does not join into one
word, thus each letter is a separate piece. A row of eight or more
such pieces on one baseline, each near the next, is a line of text.
The command discards it also. The identifiers of a row of lithics are
a lithic's width apart. Thus they do not make a line.

The second test is on reading, after the grouping. A framed code
passes the shape test, because the frame is one piece that holds most
of the ink. Thus `pylithics-pages` reads each artefact box with the
identifier reader (RapidOCR). A box is text when these two conditions
are true:

- The lines that the reader reads at a confidence of 0.8 or more cover
  a minimum of 80 percent of its ink. The command does not count a
  straight frame.
- These lines spell a minimum of three characters.

Print reads near 1. A small drawing that the reader reads as a glyph
reads at 0.6 to 0.7, and does not count. The reader reads the numeral
on a lithic that has a numeral. But the numeral is a small part of its
ink, thus the lithic stays. The command does not cut a text box. It
records the box in `pages_manifest.csv` with `image_type` `text`, and
the text in the column `text_read`. Thus you can see what the command
removed.

To keep such boxes for one run, give `--keep_text_boxes`. The settings
are in `page_segmentation.text_boxes` in `config.yaml`. Without
RapidOCR, the second test does not run.

## Identifiers

A published plate shows a number or a letter next to each lithic. The
text of the publication refers to the lithic by that identifier.
`pylithics-pages` reads it and names the crop with it.

| Result | Filename | `label_source` |
|---|---|---|
| The command read one identifier in the crop | `<page>_figure_7.png` | `read` |
| The command did not read one identifier | `<page>_box_07.png` | `index` |

The filename shows the method. `_figure_7` is the number printed on
the plate: the publication calls this lithic "7". `_box_07` is the
number of the red box on the debug overlay. It is the seventh box on
the page in reading order. The reading order is left to right and top
to bottom. The two numbers are not related. `_figure_7` and `_box_07`
on the same page are usually two different lithics.

!!! note "Read the two names"
    - `plate_014_figure_7.png` — the lithic that the plate calls 7.
    - `plate_014_box_07.png` — the seventh box on the page. The
      command did not read its identifier. Find it on the overlay to
      see which lithic it is.

The command does not name a crop when it is not sure. The manifest
column `label_flag` gives the reason:

| `label_flag` | Meaning |
|---|---|
| `several_identifiers` | The crop contains more than one identifier. The crop contains more than one lithic. The column `label_candidates` lists the identifiers. |
| `duplicate` | Two crops on the page have the same identifier. |
| `no_identifier` | The command found no identifier in or near the crop. |
| `not_read` | Identifier reading was off, or RapidOCR is not installed. |

The optional RapidOCR package is necessary to read identifiers. Run
these two commands from the PyLithics folder:

```bash
pip install ".[ocr]"
pip install --no-deps "rapidocr_onnxruntime>=1.3,<2"
```

The second command has `--no-deps` intentionally. RapidOCR asks for
the plain OpenCV package by name. You cannot install that package
together with the contrib package that PyLithics uses. The first
command installs the other packages that RapidOCR needs.

Without RapidOCR, the command names all crops by box number and writes
one message. Use `--no_read_labels` to name crops by box number
intentionally.

Some plates show the identifier far from the lithic. Then the command
does not find it. Increase `identifiers.reach` in `config.yaml`. The
value is a number of glyph heights. The default is 1.5.

### How identifiers correct the grouping

One lithic has one identifier. The command uses this rule to correct
two grouping errors:

- **A crop holds several identifiers.** The crop holds several lithics.
  The command cuts the crop at the widest empty space between the
  drawings. It cuts again until each piece holds one identifier.
  Each piece must hold one identifier and one drawing. On a dense
  plate, no empty space goes across the full crop. Then the command
  gives each drawing to its identifier. It uses the side of the lithic
  where the plate puts its identifiers, for example below or to the
  right. Two lithics with overlapping drawings stay together in one
  crop with the flag `several_identifiers`. The command still cuts the
  other lithics apart. If no cut gives one identifier and one drawing
  for each piece, the crop stays whole and keeps the flag.
- **A rule between two views is not a scale bar.** Two short rules on
  one line, each between two views of one lithic, are connectors. The
  command does not export them as a scale bar.
- **A crop holds no identifier.** The crop can be one view of the
  lithic next to it. The command joins the crop to its nearest
  neighbour that holds one identifier. All these conditions must be
  true:
    - The two crops are in line.
    - No other crop is between them.
    - The crop is not more than three times the size of the neighbour.
    - The joined crop holds one identifier.

    If the crop holds a glyph that the reader saw but did not read, the
    command does not join it. That glyph can be its own identifier.

The manifest column `correction_applied` shows `identifier_split` or
`identifier_join` for each crop that the rules made. On the debug
overlay, the crop shows `split` or `joined` after its name. Examine
these crops when a page looks wrong.

Set `identifiers.regroup.enabled: false` in `config.yaml` to turn the
rules off. `identifiers.regroup.join_gap` sets the distance in which a
crop looks for its neighbour. The value is a fraction of the page
width (default 0.12).

## PDF input

The command reads a PDF file in `pages/`. It renders each page of the
file to a PNG file in `pages_rendered/`, with the name
`<file>_p<NNN>.png`. Then it cuts those pages as it cuts a scanned
page. The PDF file stays in `pages/`. The command writes the PNG files
again on each run.

This function needs the optional package pypdfium2. Install it with
the `pdf` extra:

```bash
pip install ".[pdf]"
```

Without pypdfium2, a PDF file in `pages/` stops the command. The
message gives this install command.

### The resolution of a rendered page

A PDF page has no resolution of its own. The command reads the images
that the page holds and selects the resolution from them:

- **A page that holds a raster image** renders at the resolution of
  the sharpest large image on it. The render copies that image pixel
  for pixel. The render adds no pixel and loses no pixel. The manifest
  shows `derived`. Some scanned PDFs have a sharp 1-bit layer over a
  soft background. Such a PDF renders at the resolution of the sharp
  layer. Labels and scale bars drawn over the image stay in the
  render.
- **A page with no raster image** (a redrawn figure, a text page)
  renders at 300 dpi. `--pdf_dpi` sets a different value. The manifest
  shows `selected`.

A small image, for example a logo, does not set the resolution. The
`pdf` section of `config.yaml` sets these limits:

- `min_coverage` (default `0.05`): the part of the page that an image
  must cover.
- `max_dpi` (default `1200`).
- `rasterise_dpi` (default `300`).

The log gives one line for each page with the contents of the PDF. The
line gives the size of each image, its resolution, its bit depth and
its compression. Two warnings tell you what the render cannot correct:

| Warning | Meaning | What to do |
|---|---|---|
| `the resolution is N dpi, less than 200` | The publisher or the scanner decreased the size of the image. The lines are thin. | Nothing is necessary. The main run upscales the image for detection. Use a better scan if you have one. |
| `JPEG compression (DCTDecode)` | The PDF holds the image as JPEG. The ink edges can have small artefacts. | Nothing is necessary. Examine the crops. |

### Page selection

By default, each page of each PDF file renders. A long article has few
figure pages. Name the pages:

```bash
pylithics-pages --data_dir my_project --pdf_pages 4,7,11-14
```

The list applies to each PDF file in `pages/`. A text page gives no
crops. This is not a fault.

### Corrections for a PDF page

A rendered page is the same as the other pages. Its `page_id` in the
corrections CSV is the PNG file name, for example `report_p004.png`.
Its overlay is `pages_debug/report_p004.png`.

## The pixels do not change, with one exception

The command cuts each crop from the source image. The crop has the DPI
and the colour mode of the source. The command does not apply
denoising, contrast change or thresholding to the crop.

This is necessary. The `pylithics` analysis does those steps. If
PyLithics does the steps two times, you cannot compare a measurement
from a crop with a measurement from a single-artefact scan. The
command uses a threshold to find the ink. It does not keep the result.

The one exception is the text of the page. A crop is the box of the
artefact with a margin. The margin can include a caption, a legend
key, the end of a scale bar or the identifier of a neighbour.
`pylithics` measures each of these as a surface. Thus
`pylithics-pages` hides them in the crop file that it writes.
`pylithics` does not hide text. It analyses each crop file with no
change. The command paints over each piece with the paper value of the
crop. It paints the ink of the piece, with a margin of two pixels for
the grey edge of print. It never paints a flaking arrow. The command
paints a line of text as a band, thus the dot of an i and the specks
of print go with it. The command keeps the ink of the lithic at all
positions. Inside the box of the artefact, the command paints only
text, never a label glyph. The command does not change the page. The
manifest column `hidden` shows what the command hid in each crop.

The identifier of the crop stays, thus the crop still shows which
lithic it is. The box of the identifier goes into `meta_data.csv` in
the column `ignore_box`. `pylithics` sets that box to paper before it
looks for contours. If you move a crop out of its project, the crop
loses this box. Then `pylithics` measures the identifier.

- `--keep_text` keeps every pixel of the page in the crop.
- `--hide_identifiers` hides the identifier also, and writes no `ignore_box`.
- The settings are `hide_text` and `hide_identifiers` under
  `page_segmentation.export` in `config.yaml`.

If the lines of a page are thin, the command finds the ink on a
larger working copy of the page. The command still cuts the crops from
the original page at its own size. See [Working Resolution](working-resolution.md).

## Output

```
my_project/
├── images/          one crop for each artefact
├── scales/          one crop for each scale bar
├── pages_debug/     one overlay for each page, `<page>.png` (--debug only)
├── pages_rendered/  one PNG for each PDF page, `<file>_p<NNN>.png` (PDF input only)
├── pages_manifest.csv     one row for each crop
└── meta_data.csv    one row for each artefact crop
```

The names of artefact crops are `<page>_figure_<label>.png` or
`<page>_box_<NN>.png`. See [Identifiers](#identifiers). The names of
scale bar crops are `<page>_scale_bar.png`. If a page has more than
one scale bar, the command adds a number to each crop name:
`<page>_scale_bar_01.png`.

### pages_manifest.csv

The manifest has one row for each crop. It connects each crop to its
source page.

| Column | Meaning |
|---|---|
| `output_crop_id` | The filename of the crop in `images/` or `scales/`. |
| `image_type` | `artefact`, `scale_bar`, `text` for a box that the reader found to be text and that the command did not cut, or `skipped` for a page with the mark `expect=0`. |
| `input_page_id` | The filename of the plate in `pages/`, or of the rendered page in `pages_rendered/`. |
| `crop_id_index` | The box number of the crop on the debug overlay, in reading order. The same number as in `_box_NN`. Empty for a scale bar. |
| `x0, y0, x1, y1` | The box of the crop on the source page, in pixels. |
| `width_px, height_px` | The size of the crop, in pixels. |
| `dpi` | The DPI of the source page. Empty if the page has no DPI. |
| `colour_mode` | `greyscale`, `RGB` or `RGBA`. |
| `n_components` | The number of ink blobs in the crop. |
| `correction_applied` | The corrections that made this crop: `identifier_split` or `identifier_join` from the [identifier rules](#how-identifiers-correct-the-grouping); `expect`, `join`, `split` from the corrections CSV. A `+` joins a combination. Empty if the command used no correction. |
| `label` | The identifier that the command read from the plate. |
| `label_source` | `read` or `index`. See [Identifiers](#identifiers). |
| `label_confidence` | The confidence of the reader in `label`, from 0 to 1. |
| `label_flag` | The reason why a crop has no `label`. See [Identifiers](#identifiers). |
| `label_candidates` | All identifiers that the command read in the crop. A `;` separates them. |
| `stroke_width_px` | The measured line width of the page, in pixels. |
| `upscale_factor` | The factor of the working copy that the command used to detect the page: 1, 2, 3 or 4. See [Working Resolution](working-resolution.md). |
| `scale_score` | For a scale bar: the score of the classifier, from 0 to 1. See [Scale bars](#scale-bars). Empty for an artefact. |
| `scale_method` | For a scale bar: `classifier`, or `rules` when the classifier is off. |
| `scale_box` | For a scale bar: the box of the bar in the crop, as `x0 y0 x1 y1` in the pixels of the crop. The box does not include the caption. The command also writes it to `meta_data.csv`. |
| `source_pdf` | For a page rendered from a PDF file: the name of the file. Empty for a scanned page. |
| `source_page` | For a page rendered from a PDF file: the page number in the file, from 1. |
| `text_read` | For a `text` row: the text that the reader read in the box. |
| `hidden` | What the command hid in an artefact crop, for example `caption 1; scale bar 1`. Empty when the command hid nothing. See [The pixels do not change](#the-pixels-do-not-change-with-one-exception). |
| `render_dpi` | For a page rendered from a PDF file: the resolution of the render. See [PDF input](#pdf-input). |
| `dpi_source` | For a page rendered from a PDF file: `derived` from an image on the page, or `selected` by `--pdf_dpi`. |
| `source_filters` | For a page rendered from a PDF file: the compression of the images on the page. A `;` separates them. `DCTDecode` is JPEG. |

Keep the manifest with the crops. It connects a measurement to its
page.

### meta_data.csv

`pylithics` reads this file. It has one row for each artefact crop.
The format is the same as a metadata file that you write by hand, with
an added `flag` column. See [Metadata Setup](metadata-setup.md).

| Column | Content |
|---|---|
| `image_id` | The filename of the crop in `images/`. |
| `scale_id` | The scale bar crop from the same page, when the page has exactly one. |
| `scale` | Empty. Fill in the length of the bar, in millimetres. |
| `flag` | Empty when the row is ready. If not empty, the flag tells you what to examine. It does not stop the analysis. |
| `ignore_box` | The box of the identifier that stays in the crop, as `x0 y0 x1 y1` in the pixels of the crop. `pylithics` sets it to paper before the analysis. A `;` separates several boxes. Empty when there is no box. |
| `scale_box` | The box of the bar in the scale image that `scale_id` names, as `x0 y0 x1 y1` in the pixels of that image. The box does not include the caption. `pylithics` starts its search for the bar inside it. Empty when the row has no single scale. |

The flags:

| `flag` | Meaning | What to do |
|---|---|---|
| `no_scale` | The page has no scale bar. | Give a `scale_id` and a `scale` from a different page, or remove the flag to measure in pixels. |
| `several_scales` | The page has more than one scale bar. `scale_id` is empty. | Examine the page. Write the correct `scale_id`. |
| `no_identifier`, `several_identifiers`, `duplicate`, `not_read` | An identifier problem. See [Identifiers](#identifiers). | Examine the crop. Correct the filename, or accept it. |
| `low_resolution` | The lines of the page are thin. The command upscaled a working copy to find the crops. See [Working Resolution](working-resolution.md). | Nothing is necessary. Use a larger scan if you have one. |
| `scale_uncertain` | The classifier was not sure that the scale bar of this page is a scale bar. See [Scale bars](#scale-bars). | Open the scale crop. If it is not a scale bar, correct `scale_id`. |

A `;` separates two or more flags. A flag does not stop the analysis.
`pylithics` runs the row as all other rows. When `scale` is empty, it
measures in pixels. At the end, it shows on the screen one count for
each flag. It writes each flagged row in the log. Remove the flag when
you have examined the row.

!!! note "Pixels until you fill in the scale"
    For a crop with no `scale` value, `pylithics` can measure only in
    pixels. `pylithics` asks one time before it does that. The answer
    `n` leaves those crops out. Fill in the scale and start
    `pylithics` again to get millimetres.

## Correct a page

Illustrators and publishers use different page layouts. Some pages
come out wrong. Do not change the distance settings for one page.
Write the correction in `corrections.csv`.

1. Run the command one time with the overlays:
   ```bash
   pylithics-pages --data_dir my_project --debug
   ```
   The run writes an empty `corrections.csv` into the project folder,
   next to `pages/`.
2. Open `pages_debug/<page>.png` for the page that is wrong. Each box
   has a number.
3. Add one row for that page to `corrections.csv`. Leave the columns
   that you do not need empty.
4. Run the command again. Do not give other options:
   ```bash
   pylithics-pages --data_dir my_project
   ```
   The command cuts the pages in the file with their corrections. The
   other pages do not change. For each crop that a correction made,
   the manifest shows `join`, `split` or `expect` in
   `correction_applied`.

The columns:

| Column | Value |
|---|---|
| `page_id` | The page file name with its extension. For a PDF, the rendered PNG, for example `report_p004.png`. |
| `join` | The boxes to merge, with a plus: `1+2`. For several pairs: `1+2 5+6`. |
| `split` | One box to cut at its widest empty column: `3`. |
| `expect` | The number of artefacts on the page. The command changes its distances to get this number. For a page, use `join` or `expect`, not the two. `0` means that the page holds no artefacts. The command cuts nothing from the page and removes its crops from an earlier run. The manifest keeps a `skipped` row for the page. Use `0` for the title page, a map or a text page of a PDF. |

Example: skip pages 1 and 3, merge boxes 1 and 2 on page 4, and cut
box 3 on page 7.

```csv
page_id,expect,join,split
report_p001.png,0,,
report_p003.png,0,,
report_p004.png,,1+2,
report_p007.png,,,3
```

Use the same distance options for the debug run and the corrected
run. A different `--gap` can change the box numbers. To use a
corrections file in a different location, give its path with
`--overrides`.

!!! warning "Write the CSV file with a spreadsheet or a CSV writer"
    Page filenames often contain commas. Then `page_id` must be in
    quotation marks. When you write it by hand, it is easy to make an
    error.

!!! note "`expect` is not precise"
    `expect` changes the distances until the count is the same. It
    does not know which grouping is correct. It can find the correct
    count for the wrong reason. Use it to confirm a page that you
    examined on the overlay. Do not use it to correct a page that you
    did not examine.

### When all pages show the same error

Use the CSV file for one page. Use these settings when all pages from
one publication show the same error:

| Error | Setting |
|---|---|
| The command cuts one artefact into two boxes | Increase `--gap` or `--vertical_gap` |
| Two artefacts are in one box | Decrease `--gap` or `--vertical_gap` |
| A profile view is not in its box | Increase `--narrow` |
| Labels become crops | Increase `--min_area` |
| An identifier is far from its lithic | Increase `identifiers.reach` in `config.yaml` |

All distances are fractions of the page width or the page height.
Thus the same settings apply to pages of different sizes at the same
resolution.

## Re-runs

Each run cuts each plate in `pages/`. The rules:

| | |
|---|---|
| A plate that an earlier run cut | The command cuts it again. It replaces the crops, the manifest rows and the rows in `meta_data.csv` of the plate. The scale values that you typed stay. |
| A new plate | The command cuts it and adds its crops and rows. |
| A crop that the new run does not make | The command removes it and its row. |
| A page with the mark `expect=0` in `corrections.csv` | The command cuts nothing. It removes the crops and rows of the page from an earlier run. |
| A page file or a PDF that you removed from `pages/` | The command removes its crops and rows from an earlier run. The run tells you which pages. |
| Your own images, scale images and rows | The command never changes them. |
| A crop file from an earlier run that has no manifest row | The command replaces it. At the end, the run gives the number of these files. |
| Two pages with the same name and different extensions | They give the same crop names. The command does not write the second page. The message names the first page. Rename one page. The exit code is 1. |
| A plate that the command cannot cut | The command reports it, with the reason in `pylithics-pages.log`. It cuts the other plates. The exit code is 1. |
| A run that you stop | The manifest names each crop that the command wrote before the stop. Start the command again. The next run replaces these crops. |

Thus you can add plates, change a setting, or write a correction, and
start the command again. Nothing stops you. You do not lose the values
that you typed.

## Scale bars

The command finds scale bars and writes each one to its own crop. It
does not put a scale bar in an artefact crop. It finds these styles:

- **Block bars**: filled and open blocks in a row, or in two rows
  as on a chessboard. The bar can be across the page or down the page.
- **Ruled scales**: a line with tick marks, usually with a numeral at
  each end.

If a caption such as `5 cm` is next to the bar, the crop includes it.
The command does not measure the caption. The manifest and
`meta_data.csv` record the box of the bar only in `scale_box`.
`pylithics` measures the bar from this box.

### How a scale bar is decided

The command applies these rules in this sequence.

**1. The candidates**

- A candidate is a mark that is a minimum of 2.5 times longer than it
  is wide. It can be across the page or down the page.
- Its ink does not touch a drawing. A bar below the profile view of a
  lithic is a candidate. A line that touches a drawing is not a
  candidate.
- The solid blocks or the dashes of a segmented bar on one line are
  one candidate.
- When the ink of the caption touches a thick bar, the command cuts
  the bar from the raw ink. It exports the bar with its caption. The
  ink that touches the bar can be a drawing, many times the size of
  the bar. Then the command exports the bar alone.
- The size limits are in the configuration: `min_long_side`,
  `max_long_side` and `max_short_side`. Each limit is a fraction of
  the longer side of the page. `max_long_side` is `0.5`. On 130
  labelled pages, no scale bar is wider than 0.44 of its page. The
  rule below the running header of a journal page is 0.88.

**2. The candidates that the command rejects before the score**

- A line with no ticks and no caption at the edge of a drawing is a
  connector, not a scale bar.
- A line with no ticks and no caption between two views of one lithic
  is a rule between the views, not a scale bar.
- The score of these marks is `0`.

**3. The score**

A trained classifier (a random forest) measures each other candidate.
It gives the candidate a score from 0 to 1. It measures:

- The shape: length, width, aspect, density.
- The pattern of the ink: the number of blocks and their regularity,
  the tick marks, a continuous rule.
- The cross-section: a bar is solid, or solid blocks, with one width
  to its ends. The outline of a lithic surface is hollow, and becomes
  narrow at its ends.
- The tones of the pixels.
- The two ends of the candidate: a bar has the same width at its two
  ends, with or without end ticks. An arrow has its head at one end.
  A large arrow drawn on an artefact is not a bar.
- The small marks next to the candidate, for example the caption
  `5 cm`.
- If the box of a drawing contains the candidate, and the position of
  the candidate on the page.

A candidate with a score at or above the threshold (default `0.6`)
is a scale bar. The other candidates keep the role that the rules
give them.

**4. The caption**

- The crop of a scale bar contains the bar and its caption.
- A caption is text on the line of the bar. It can be numerals above
  or below the bar in its span, or a unit level with its end. It can
  also be text after such a caption (`cm` after `5`).
- A mark at a corner of the bar is not a caption. A thin line, for
  example an arrow, is not a caption.
- A bar-shaped mark on the line of a bar with a higher score is a
  caption of that bar, not a second bar. Examples: the `1` of a ruler,
  the `5 cm` next to a vertical bar.

**5. After the decision**

- The manifest records the score of each scale bar in `scale_score`.
- Sometimes the score is in `uncertain_band` (default `0.15`) of the
  threshold. Then the command puts the flag `scale_uncertain` on the
  artefacts of that page in `meta_data.csv`.
- With `--debug`, the overlay shows each accepted bar in purple with
  its score. It shows each rejected candidate in grey with its score.
- The command does not read the number on the bar. Write the value in
  millimetres in the `scale` column.

To use the rules only, set `scale_bars.classifier.enabled: false` in
the configuration. The package includes the model. If the command
cannot read the model, it tells you one time. Then the rules decide.

!!! note "You write the scale values by hand"
    `pylithics-pages` writes the scale bar image. It connects the
    image to each crop from the same page in `meta_data.csv`. It does
    not read the number printed on the bar. Write the value in
    millimetres in the `scale` column. See [meta_data.csv](#meta_datacsv).

## Analyse the crops

Fill in `meta_data.csv`. Then start the analysis on the project:

```bash
pylithics --data_dir my_project
```

`pylithics` reads the crop files as `pylithics-pages` wrote them. It
does not cut, hide or correct anything. After a change to the pages,
to `corrections.csv` or to a `pylithics-pages` option, run
`pylithics-pages` again. Then run `pylithics`.

See [Basic Usage](basic-usage.md) for the analysis step.
