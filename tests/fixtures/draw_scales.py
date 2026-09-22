"""
Draw the scale bar styles found in archaeological plates.

Scale bars vary far more than lithic drawings do, and the detector has
a separate route for each family. These generators reproduce the styles
seen in the source literature so the routes stay covered without
committing scanned figures to the repository.

Five families, matching what appears in published plates:

- ``chequered``  two rows of alternating filled and open cells
- ``ruled``      a plain rule carrying tick marks at each measure
- ``blocked``    one thick bar broken by open blocks, ticks at the ends
- ``zigzag``     an upright bar whose blocks step left and right
- ``broken``     a thin rule the scan has cut into segments, numerals above

Each returns a white RGB image with the scale drawn in black, sized and
proportioned as it would appear on a page.
"""

from PIL import Image, ImageDraw

# Proportions taken from measured examples: bars run several hundred
# pixels wide and a few tens tall, with numerals set above or below.
_BAR_LEFT = 30
_NUMERAL_DROP = 6


def draw_chequered(cells: int = 3, cell: int = 80) -> Image.Image:
    """
    Draw a two-row chequered bar with numerals above.

    The rows are offset by one cell, so filled and open squares
    alternate both along the bar and between the rows.
    """
    width = _BAR_LEFT * 2 + cells * cell
    image = Image.new('RGB', (width, 150), 'white')
    draw = ImageDraw.Draw(image)

    top, middle, bottom = 70, 90, 110
    for index in range(cells):
        left = _BAR_LEFT + index * cell
        right = left + cell
        upper = 'black' if index % 2 == 0 else 'white'
        draw.rectangle([left, top, right, middle], fill=upper)
        draw.rectangle(
            [left, middle, right, bottom],
            fill='white' if upper == 'black' else 'black',
        )

    draw.rectangle([_BAR_LEFT, top, _BAR_LEFT + cells * cell, bottom],
                   outline='black', width=2)
    for index in range(cells + 1):
        x = _BAR_LEFT + index * cell
        draw.text((x - 4, top - 30), str(index), fill='black')
    draw.text((width - 26, top - 30), 'cm', fill='black')
    return image


def draw_ruled(ticks: int = 9, spacing: int = 26) -> Image.Image:
    """
    Draw a plain rule carrying evenly spaced tick marks.

    The ticks sit on one side of the rule and the numerals clear of it,
    which is what lets the detector tell the two apart.
    """
    width = _BAR_LEFT * 2 + ticks * spacing
    image = Image.new('RGB', (width, 120), 'white')
    draw = ImageDraw.Draw(image)

    spine = 70
    right = _BAR_LEFT + ticks * spacing
    draw.rectangle([_BAR_LEFT, spine, right, spine + 2], fill='black')
    for index in range(ticks + 1):
        x = _BAR_LEFT + index * spacing
        draw.rectangle([x, spine - 10, x + 1, spine], fill='black')

    draw.text((_BAR_LEFT - 2, spine + _NUMERAL_DROP + 4), '0', fill='black')
    draw.text((right - 20, spine + _NUMERAL_DROP + 4),
              f'{ticks} cm', fill='black')
    return image


def draw_blocked(blocks: int = 3, block: int = 100) -> Image.Image:
    """
    Draw one thick bar broken by open blocks, with ticks at each end.

    Thresholding keeps only the filled parts, so this style reaches the
    detector either as one component with holes or as separate blocks,
    depending on whether the bar carries a drawn border.
    """
    width = _BAR_LEFT * 2 + blocks * block
    image = Image.new('RGB', (width, 140), 'white')
    draw = ImageDraw.Draw(image)

    top, bottom = 75, 105
    right = _BAR_LEFT + blocks * block
    draw.rectangle([_BAR_LEFT, top, right, bottom], fill='black')
    for index in range(blocks):
        if index % 2:
            left = _BAR_LEFT + index * block
            draw.rectangle(
                [left + 6, top + 6, left + block - 6, bottom - 6],
                fill='white',
            )

    for x in (_BAR_LEFT, right):
        draw.rectangle([x - 1, top - 14, x + 1, top], fill='black')
    draw.text((_BAR_LEFT - 4, top - 34), '0', fill='black')
    draw.text((right - 30, top - 34), f'{blocks} cm', fill='black')
    return image


def draw_zigzag(blocks: int = 5, block: int = 90) -> Image.Image:
    """
    Draw an upright bar whose blocks step left and right down its length.

    Plates of tall artefacts often carry the scale down the side rather
    than across the foot, and it carries no numerals. Successive blocks
    are offset by less than their own width, so the filled parts overlap
    through the centre and the whole scale reaches the detector as one
    component. Proportions follow six measured examples: eight to twelve
    times taller than wide, covering half to three quarters of their box.
    """
    bar, overlap = 34, 8
    step = bar - overlap
    width = _BAR_LEFT * 2 + bar + step
    height = _BAR_LEFT * 2 + blocks * block
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)

    for index in range(blocks):
        top = _BAR_LEFT + index * block
        left = _BAR_LEFT + (0 if index % 2 == 0 else step)
        draw.rectangle([left, top, left + bar, top + block], fill='black')
    return image


def draw_broken(segments: int = 4, segment: int = 100) -> Image.Image:
    """
    Draw a thin rule cut into segments, with a numeral at each cut.

    A cheap scan loses ink along a fine rule, so the line reaches the
    detector as a few long pieces rather than one unbroken run. The
    Revue de Comminges scales arrive this way. The cuts are narrower
    than the closing kernel, so the detector sees one blob whose raw
    ink is broken; short ticks at the cuts keep its box from being
    solid ink, as a real scan's is not.
    """
    gap = 6                       # under the closing kernel, so the pieces
    width = _BAR_LEFT * 2 + segments * segment    # arrive as one blob
    image = Image.new('RGB', (width, 110), 'white')
    draw = ImageDraw.Draw(image)

    rule = 70
    for index in range(segments):
        left = _BAR_LEFT + index * segment + (gap // 2 if index else 0)
        right = _BAR_LEFT + (index + 1) * segment - gap // 2
        draw.rectangle([left, rule, right, rule + 2], fill='black')
    for index in range(segments + 1):
        x = _BAR_LEFT + index * segment
        draw.rectangle([x, rule - 5, x, rule + 2], fill='black')
        draw.text((x - 4, rule - 30), str(index), fill='black')
    draw.text((width - 26, rule - 30), 'cm', fill='black')
    return image


STYLES = {
    'chequered': draw_chequered,
    'ruled': draw_ruled,
    'blocked': draw_blocked,
    'zigzag': draw_zigzag,
    'broken': draw_broken,
}


def page_with_scale(style: str, size=(2000, 1500)) -> Image.Image:
    """
    Place a drawn scale alone on a blank page of realistic size.

    Detection thresholds are page-relative, so a scale must be measured
    against a full page rather than a tight crop.
    """
    scale = STYLES[style]()
    page = Image.new('RGB', size, 'white')
    page.paste(scale, (200, size[1] - scale.height - 200))
    return page
