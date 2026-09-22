"""The README splash images are rendered from the real splash code."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fixtures"))

import render_splash  # noqa: E402


@pytest.mark.unit
def test_both_modes_are_written_as_svg(tmp_path):
    assert render_splash.main(str(tmp_path)) == 0
    for mode in ("dark", "light"):
        svg = (tmp_path / f"splash-{mode}.svg").read_text()
        assert svg.startswith("<svg")
        assert "Quick" in svg and "GitHub" in svg


@pytest.mark.unit
def test_light_mode_has_a_white_background(tmp_path):
    render_splash.main(str(tmp_path))
    light = (tmp_path / "splash-light.svg").read_text()
    dark = (tmp_path / "splash-dark.svg").read_text()
    assert "#ffffff" in light.lower()
    assert light != dark
