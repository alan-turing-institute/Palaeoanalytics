"""Integration tests for the PyLithics analysis pipeline orchestrator."""

import os
import tempfile
import time
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import pytest

from pylithics.image_processing.image_analysis import process_and_save_contours


def _make_inverted_binary_image(size=(200, 300), rect=(50, 50, 200, 100)):
    """Return an inverted binary image (white object on black background)."""
    image = np.zeros(size, dtype=np.uint8)
    x, y, w, h = rect
    cv2.rectangle(image, (x, y), (x + w, y + h), 255, thickness=-1)
    return image


def _make_image_with_scar():
    """Inverted image with a parent surface containing a scar (inner hole)."""
    image = np.zeros((200, 300), dtype=np.uint8)
    cv2.rectangle(image, (40, 40), (260, 160), 255, thickness=-1)  # parent
    cv2.rectangle(image, (100, 70), (200, 130), 0, thickness=-1)   # scar hole
    return image


@pytest.mark.integration
class TestProcessAndSaveContours:
    """End-to-end tests for the pipeline orchestrator."""

    def test_produces_csv_with_expected_schema(self):
        """A real pipeline run on a simple artifact writes a populated CSV."""
        inverted = _make_inverted_binary_image()

        with tempfile.TemporaryDirectory() as temp_dir:
            process_and_save_contours(
                inverted,
                conversion_factor=0.1,
                output_dir=temp_dir,
                image_id="simple_shape.png",
                image_dpi=300,
            )

            csv_path = os.path.join(temp_dir, "processed_metrics.csv")
            assert os.path.exists(csv_path), "pipeline did not write processed_metrics.csv"

            df = pd.read_csv(csv_path)
            assert len(df) >= 1, "CSV contains no rows"

            required_columns = {
                "image_id", "surface_type", "centroid_x", "centroid_y",
                "total_area",
            }
            missing = required_columns - set(df.columns)
            assert not missing, f"CSV missing columns: {missing}"

            # Dorsal surface must be classified on this single-shape image
            assert (df["surface_type"] == "Dorsal").any()

    def test_no_contours_logs_warning_and_returns(self):
        """A blank image produces no contours — pipeline exits cleanly."""
        blank = np.zeros((100, 100), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("pylithics.image_processing.image_analysis.logging") as mock_log:
                process_and_save_contours(
                    blank,
                    conversion_factor=0.1,
                    output_dir=temp_dir,
                    image_id="blank.png",
                )
                mock_log.warning.assert_called()

            assert not os.path.exists(
                os.path.join(temp_dir, "processed_metrics.csv")
            )

    def test_arrow_stage_failure_does_not_abort_pipeline(self):
        """If arrow integration raises, the pipeline still writes the CSV."""
        inverted = _make_image_with_scar()

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "pylithics.image_processing.image_analysis.integrate_arrows",
                side_effect=RuntimeError("simulated arrow failure"),
            ), patch(
                "pylithics.image_processing.image_analysis.logging"
            ) as mock_log:
                process_and_save_contours(
                    inverted,
                    conversion_factor=0.1,
                    output_dir=temp_dir,
                    image_id="arrow_fail.png",
                )

            # Exception was logged and pipeline continued to write the CSV
            mock_log.exception.assert_called()
            assert os.path.exists(
                os.path.join(temp_dir, "processed_metrics.csv")
            )

    def test_critical_error_is_caught_and_logged(self):
        """A failure in contour extraction is caught by the top-level try/except."""
        inverted = _make_inverted_binary_image()

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "pylithics.image_processing.image_analysis.extract_contours_with_hierarchy",
                side_effect=RuntimeError("boom"),
            ), patch(
                "pylithics.image_processing.image_analysis.logging"
            ) as mock_log:
                # Must not raise — the orchestrator logs and returns.
                process_and_save_contours(
                    inverted,
                    conversion_factor=0.1,
                    output_dir=temp_dir,
                    image_id="crash.png",
                )
                mock_log.exception.assert_called()


@pytest.mark.performance
class TestPipelinePerformance:
    """Performance bounds for the pipeline."""

    def test_single_image_completes_under_thirty_seconds(self):
        inverted = _make_inverted_binary_image()

        with tempfile.TemporaryDirectory() as temp_dir:
            start = time.time()
            process_and_save_contours(
                inverted,
                conversion_factor=0.1,
                output_dir=temp_dir,
                image_id="perf.png",
                image_dpi=300,
            )
            elapsed = time.time() - start

        assert elapsed < 30.0, f"pipeline took {elapsed:.1f}s (limit 30s)"

    def test_memory_increase_under_one_hundred_megabytes(self):
        psutil = pytest.importorskip("psutil")
        process = psutil.Process(os.getpid())
        inverted = _make_inverted_binary_image()

        initial = process.memory_info().rss
        with tempfile.TemporaryDirectory() as temp_dir:
            process_and_save_contours(
                inverted,
                conversion_factor=0.1,
                output_dir=temp_dir,
                image_id="mem.png",
                image_dpi=300,
            )
        increase = process.memory_info().rss - initial

        assert increase < 100 * 1024 * 1024, (
            f"pipeline allocated {increase / 1024 / 1024:.1f} MB (limit 100 MB)"
        )
