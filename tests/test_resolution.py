"""
Working resolution: measure, choose a factor, upscale, restore.

Drawings are synthetic so each test states one rule. The neural
upscaler is exercised for real once (the bundled models are small);
its absence is simulated by hiding the OpenCV module.
"""

import math

import numpy as np
import pytest

from pylithics.image_processing import resolution


def _drawing(stroke, gap, size=120):
    """A closed outline with horizontal hatching of given stroke width and clearance."""
    img = np.full((size, size), 255, dtype=np.uint8)
    w = stroke
    img[4:4 + w, 4:size - 4] = 0
    img[size - 4 - w:size - 4, 4:size - 4] = 0
    img[4:size - 4, 4:4 + w] = 0
    img[4:size - 4, size - 4 - w:size - 4] = 0
    row = 4 + w + gap
    while row + w < size - 4 - w - gap:
        img[row:row + w, 12:size - 30] = 0
        row += w + gap
    return img


PAGES = {'min_stroke_width_px': 4.0, 'min_hatch_gap_px': 0, 'max_factor': 4,
         'max_working_pixels': 40_000_000}
ANALYSIS = {'min_stroke_width_px': 6.0, 'min_hatch_gap_px': 12.0, 'max_factor': 4,
            'max_working_pixels': 40_000_000}


@pytest.mark.unit
class TestMeasure:
    def test_wide_strokes_and_clear_gaps(self):
        g = resolution.measure_line_geometry(_drawing(5, 12))
        assert g.stroke_width == pytest.approx(5, abs=1)
        assert g.hatch_gap == pytest.approx(12, abs=2)

    def test_thin_close_strokes(self):
        g = resolution.measure_line_geometry(_drawing(1, 2))
        assert g.stroke_width <= 2 and g.hatch_gap <= 3

    def test_blank_image_has_no_geometry(self):
        g = resolution.measure_line_geometry(np.full((40, 40), 255, dtype=np.uint8))
        assert math.isnan(g.stroke_width) and math.isnan(g.hatch_gap) and g.ink_fraction == 0

    def test_dots_do_not_shrink_the_gap(self):
        img = _drawing(5, 14)
        for y in range(20, 100, 4):
            for x in range(96, 112, 4):
                img[y:y + 2, x:x + 2] = 0
        assert resolution.measure_line_geometry(img).hatch_gap > 8


@pytest.mark.unit
class TestChooseFactor:
    def test_fine_drawing_needs_nothing(self):
        g = resolution.LineGeometry(10.0, 20.0, 0.1)
        assert resolution.choose_factor(g, (500, 500), ANALYSIS) == 1

    def test_thin_strokes_drive_the_factor(self):
        assert resolution.choose_factor(
            resolution.LineGeometry(1.8, 20.0, 0.1), (500, 500), ANALYSIS) == 4
        assert resolution.choose_factor(
            resolution.LineGeometry(4.7, 20.0, 0.1), (500, 500), ANALYSIS) == 2

    def test_close_hatching_drives_the_factor(self):
        assert resolution.choose_factor(
            resolution.LineGeometry(10.0, 5.0, 0.1), (500, 500), ANALYSIS) == 3

    def test_a_zero_floor_turns_that_test_off(self):
        """Pages: fine hatching on a plate must not force x4."""
        assert resolution.choose_factor(
            resolution.LineGeometry(10.0, 5.0, 0.1), (500, 500), PAGES) == 1

    def test_capped_by_max_factor(self):
        g = resolution.LineGeometry(0.5, 0.5, 0.1)
        assert resolution.choose_factor(g, (500, 500), ANALYSIS) == 4
        assert resolution.choose_factor(g, (500, 500), {**ANALYSIS, 'max_factor': 2}) == 2

    def test_capped_by_working_pixels(self):
        g = resolution.LineGeometry(1.0, 20.0, 0.1)                # wants x4
        big = {**PAGES, 'max_working_pixels': 40_000_000}
        tiny = {**PAGES, 'max_working_pixels': 1_000}
        assert resolution.choose_factor(g, (3000, 3000), big) == 2
        assert resolution.choose_factor(g, (3000, 3000), tiny) == 1

    def test_unmeasurable_geometry_means_no_upscaling(self):
        g = resolution.LineGeometry(math.nan, math.nan, 0.0)
        assert resolution.choose_factor(g, (500, 500), ANALYSIS) == 1

    def test_never_below_one(self):
        g = resolution.LineGeometry(100.0, 100.0, 0.1)
        assert resolution.choose_factor(g, (10, 10), ANALYSIS) == 1


@pytest.mark.unit
class TestUpscale:
    def test_factor_one_returns_the_image(self):
        img = _drawing(3, 6)
        assert resolution.upscale(img, 1) is img

    def test_espcn_doubles_the_size_and_keeps_the_drawing(self):
        pytest.importorskip('cv2.dnn_superres')
        img = _drawing(2, 4)
        big = resolution.upscale(img, 2, 'espcn')
        assert big.shape == (240, 240) and big.dtype == np.uint8
        assert resolution.measure_line_geometry(big).stroke_width > \
            resolution.measure_line_geometry(img).stroke_width

    def test_model_is_loaded_once(self, monkeypatch):
        pytest.importorskip('cv2.dnn_superres')
        resolution._models.clear()
        calls = []
        real = resolution._load_model

        def counting(model, factor):
            calls.append((model, factor))
            return real(model, factor)

        monkeypatch.setattr(resolution, '_load_model', counting)
        resolution.upscale(_drawing(2, 4), 2)
        resolution.upscale(_drawing(2, 4), 2)
        assert len(resolution._models) == 1 and calls == [('espcn', 2)] * 2

    def test_missing_contrib_build_is_an_error(self, monkeypatch):
        import cv2
        resolution._models.clear()
        monkeypatch.delattr(cv2, 'dnn_superres', raising=False)
        with pytest.raises(resolution.UpscalingUnavailable):
            resolution.upscale(_drawing(2, 4), 2)

    def test_missing_model_file_is_an_error(self, monkeypatch):
        pytest.importorskip('cv2.dnn_superres')
        resolution._models.clear()
        monkeypatch.setattr(resolution, 'MODELS_DIR', '/nonexistent')
        with pytest.raises(resolution.UpscalingUnavailable):
            resolution.upscale(_drawing(2, 4), 3)

    def test_unknown_model_name(self):
        resolution._models.clear()
        with pytest.raises(ValueError):
            resolution.upscale(_drawing(2, 4), 2, 'edsr')


@pytest.mark.unit
class TestBundledModels:
    """Every bundled model file loads and upscales by its factor."""

    @pytest.mark.parametrize('model', resolution.SUPPORTED_MODELS)
    @pytest.mark.parametrize('factor', (2, 3, 4))
    def test_model_loads_and_scales(self, model, factor):
        pytest.importorskip('cv2.dnn_superres')
        resolution._models.clear()
        big = resolution.upscale(_drawing(2, 4, size=60), factor, model)
        assert big.shape == (60 * factor, 60 * factor)


@pytest.mark.unit
class TestRestore:
    def test_same_shape_is_returned_unchanged(self):
        img = np.full((30, 30), 255, dtype=np.uint8)
        assert resolution.restore_to_grid(img, (30, 30)) is img

    def test_thin_line_survives_a_fourfold_reduction(self):
        big = np.zeros((120, 120), dtype=np.uint8)
        big[58:64, 8:112] = 255                           # a 6 px ink line at x4
        small = resolution.restore_to_grid(big, (30, 30))
        assert small.shape == (30, 30) and small.dtype == np.uint8
        assert (small > 0).any(axis=0)[2:28].all()


@pytest.mark.unit
class TestWorkingCopy:
    def test_off_means_factor_one_but_still_measured(self):
        wc = resolution.working_copy(_drawing(1, 2), {**PAGES, 'enabled': False})
        assert wc.factor == 1 and not math.isnan(wc.geometry.stroke_width)

    def test_thin_drawing_is_upscaled(self):
        pytest.importorskip('cv2.dnn_superres')
        wc = resolution.working_copy(_drawing(1, 6), PAGES)
        assert wc.factor == 4 and wc.gray.shape == (480, 480) and wc.source_shape == (120, 120)

    def test_unavailable_upscaler_gives_factor_one_and_one_error(self, monkeypatch, caplog):
        import cv2
        resolution._models.clear()
        resolution._unavailable_reported = False
        monkeypatch.delattr(cv2, 'dnn_superres', raising=False)
        with caplog.at_level('ERROR'):
            first = resolution.working_copy(_drawing(1, 6), PAGES)
            second = resolution.working_copy(_drawing(1, 6), PAGES)
        assert first.factor == 1 and second.factor == 1
        assert caplog.text.count('Upscaling is not possible') == 1

    def test_settings_merge_shared_and_own(self):
        cfg = {'working_resolution': {'model': 'fsrcnn', 'pages': {'min_stroke_width_px': 3.0}}}
        pages = resolution.settings_for(cfg, 'pages')
        assert pages['model'] == 'fsrcnn' and pages['min_stroke_width_px'] == 3.0
        assert 'analysis' not in pages and pages['max_factor'] == 4
