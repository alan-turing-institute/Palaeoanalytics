"""
Write the welcome splash as SVG images for the README.

GitHub shows one image in dark mode and another in light mode. A
screenshot of a dark terminal is a black box on a white page, so the
README carries two images, both rendered here from the same splash
code that ``pylithics`` prints. Run this after each change to the
splash, so the images stay in step with the command::

    python tests/fixtures/render_splash.py

Output: ``docs/assets/images/splash-dark.svg`` and
``docs/assets/images/splash-light.svg``.
"""

import os
import sys

from rich.console import Console
from rich.terminal_theme import SVG_EXPORT_THEME, TerminalTheme

from pylithics.cli_splash import print_splash

WIDTH = 100
ASSETS = os.path.join('docs', 'assets', 'images')

# The splash styles its text "cyan" and "white". On a light page those
# read as pale, so the light theme maps them to darker inks. The 16
# colours are the ANSI palette: black, red, green, yellow, blue,
# magenta, cyan, white, then the same eight in their bright form.
LIGHT_THEME = TerminalTheme(
    (255, 255, 255),
    (36, 41, 46),
    [
        (36, 41, 46), (203, 36, 49), (34, 134, 58), (176, 136, 0),
        (0, 92, 197), (111, 66, 193), (0, 117, 140), (60, 66, 72),
    ],
    [
        (96, 103, 112), (203, 36, 49), (34, 134, 58), (176, 136, 0),
        (0, 92, 197), (111, 66, 193), (0, 117, 140), (60, 66, 72),
    ],
)

THEMES = {
    'dark': SVG_EXPORT_THEME,
    'light': LIGHT_THEME,
}


def render(mode: str, theme: TerminalTheme, directory: str) -> str:
    """Write the splash as ``splash-<mode>.svg`` and return the path."""
    console = Console(
        record=True, width=WIDTH, force_terminal=True,
        color_system='truecolor', file=open(os.devnull, 'w'),
    )
    print_splash(console)
    path = os.path.join(directory, f'splash-{mode}.svg')
    console.save_svg(path, title='pylithics', theme=theme)
    return path


def main(directory: str = ASSETS) -> int:
    """Render both modes into ``directory``."""
    os.makedirs(directory, exist_ok=True)
    for mode, theme in THEMES.items():
        print(render(mode, theme, directory))
    return 0


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
