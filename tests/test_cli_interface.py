"""Tests for the CLI argument parser and main() entry point."""

import os
import tempfile
from unittest.mock import patch

import pytest

from pylithics.app import (
    PyLithicsApplication,
    create_argument_parser,
    main,
)


# ---------------------------------------------------------------------------
# Argument parser: our own flags (not argparse library behavior)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreateArgumentParser:
    """Cover the shape of OUR parser, not argparse's library semantics."""

    def test_data_dir_and_meta_file_are_parsed_verbatim(self):
        parser = create_argument_parser()
        args = parser.parse_args([
            "--data_dir", "/some/data",
            "--meta_file", "/some/meta.csv",
        ])
        assert args.data_dir == "/some/data"
        assert args.meta_file == "/some/meta.csv"

    def test_optional_flags_populate_expected_attributes(self):
        parser = create_argument_parser()
        args = parser.parse_args([
            "--data_dir", "/d",
            "--meta_file", "/m.csv",
            "--threshold_method", "otsu",
            "--log_level", "DEBUG",
            "--disable_arrow_detection",
            "--arrow_debug",
        ])
        assert args.threshold_method == "otsu"
        assert args.log_level == "DEBUG"
        assert args.disable_arrow_detection is True
        assert args.arrow_debug is True

    def test_invalid_threshold_method_exits(self):
        """Our parser must reject threshold methods we don't support."""
        parser = create_argument_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--data_dir", "/d",
                "--meta_file", "/m.csv",
                "--threshold_method", "not_a_method",
            ])


# ---------------------------------------------------------------------------
# main(): exit-code contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMainExitCodes:
    """main() translates run_batch_analysis outcomes into exit codes."""

    def _invoke_with(self, argv, validate_return, batch_result):
        """Drive main() with specific validate/batch outcomes."""
        with patch("sys.argv", argv), \
             patch.object(PyLithicsApplication, "validate_inputs",
                          return_value=validate_return), \
             patch.object(PyLithicsApplication, "run_batch_analysis",
                          return_value=batch_result) as mock_run:
            return main(), mock_run

    def test_full_success_exits_zero(self):
        code, _ = self._invoke_with(
            ["pylithics", "--data_dir", "/d", "--meta_file", "/m.csv"],
            validate_return=True,
            batch_result={
                "success": True, "processed_successfully": 2,
                "total_images": 2, "failed_images": [],
            },
        )
        assert code == 0

    def test_partial_success_still_exits_zero(self):
        code, _ = self._invoke_with(
            ["pylithics", "--data_dir", "/d", "--meta_file", "/m.csv"],
            validate_return=True,
            batch_result={
                "success": True, "processed_successfully": 2,
                "total_images": 3, "failed_images": ["bad.png"],
            },
        )
        assert code == 0

    def test_validation_failure_exits_one(self):
        code, _ = self._invoke_with(
            ["pylithics", "--data_dir", "/d", "--meta_file", "/m.csv"],
            validate_return=False,
            batch_result={"success": False, "error": "Input validation failed"},
        )
        assert code == 1

    def test_batch_failure_exits_one(self):
        code, _ = self._invoke_with(
            ["pylithics", "--data_dir", "/d", "--meta_file", "/m.csv"],
            validate_return=True,
            batch_result={"success": False, "error": "boom"},
        )
        assert code == 1

    def test_value_error_during_setup_exits_one(self):
        argv = ["pylithics", "--data_dir", "/d", "--meta_file", "/m.csv"]
        with patch("sys.argv", argv), \
             patch.object(PyLithicsApplication, "validate_inputs",
                          side_effect=ValueError("bad input")):
            assert main() == 1


# ---------------------------------------------------------------------------
# main(): flags are forwarded into the app
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMainFlagForwarding:
    """Verify that CLI flags reach run_batch_analysis / update_configuration."""

    def _run(self, argv_extras):
        argv = [
            "pylithics",
            "--data_dir", "/d",
            "--meta_file", "/m.csv",
            *argv_extras,
        ]
        with patch("sys.argv", argv), \
             patch.object(PyLithicsApplication, "validate_inputs",
                          return_value=True), \
             patch.object(PyLithicsApplication, "run_batch_analysis",
                          return_value={
                              "success": True,
                              "processed_successfully": 1,
                              "total_images": 1,
                              "failed_images": [],
                          }) as mock_run, \
             patch.object(PyLithicsApplication, "update_configuration") as mock_update:
            code = main()
        return code, mock_run, mock_update

    def test_threshold_method_override_triggers_update_configuration(self):
        code, _, mock_update = self._run(["--threshold_method", "otsu"])
        assert code == 0
        # update_configuration should have been called with thresholding.method
        forwarded = {}
        for call in mock_update.call_args_list:
            forwarded.update(call.kwargs)
        assert forwarded.get("thresholding.method") == "otsu"

    def test_disable_arrow_detection_forwards_to_update_configuration(self):
        code, _, mock_update = self._run(["--disable_arrow_detection"])
        assert code == 0
        forwarded = {}
        for call in mock_update.call_args_list:
            forwarded.update(call.kwargs)
        assert forwarded.get("arrow_detection.enabled") is False

    def test_run_batch_analysis_receives_cli_paths(self):
        code, mock_run, _ = self._run([])
        assert code == 0
        assert mock_run.called
        # First positional args should be (data_dir, meta_file)
        args = mock_run.call_args.args
        kwargs = mock_run.call_args.kwargs
        data_dir = args[0] if args else kwargs.get("data_dir")
        meta_file = args[1] if len(args) > 1 else kwargs.get("meta_file")
        assert data_dir == "/d"
        assert meta_file == "/m.csv"


# ---------------------------------------------------------------------------
# PyLithicsApplication.update_configuration: dot-notation key handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUpdateConfiguration:
    """update_configuration parses `section.key` into the config manager."""

    def test_dotted_keys_update_the_correct_section(self):
        app = PyLithicsApplication()
        app.update_configuration(**{
            "thresholding.method": "otsu",
            "arrow_detection.enabled": False,
        })
        assert app.config_manager.get_value("thresholding", "method") == "otsu"
        assert app.config_manager.get_value(
            "arrow_detection", "enabled"
        ) is False

    def test_non_dotted_keys_are_ignored_with_warning(self):
        app = PyLithicsApplication()
        with patch("pylithics.app.logging") as mock_log:
            app.update_configuration(**{"not_a_dotted_key": "whatever"})
        mock_log.warning.assert_called()


# ---------------------------------------------------------------------------
# PyLithicsApplication.process_single_image: missing/failing cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProcessSingleImage:

    def test_returns_false_when_image_file_missing(self, tmp_path):
        app = PyLithicsApplication()
        result = app.process_single_image(
            "nonexistent", None, str(tmp_path), str(tmp_path)
        )
        assert result is False

    def test_returns_false_when_preprocessing_returns_none(self, tmp_path):
        image_path = tmp_path / "fake.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\n")  # PNG header, nothing else

        app = PyLithicsApplication()
        with patch(
            "pylithics.app.execute_preprocessing_pipeline", return_value=None
        ):
            result = app.process_single_image(
                "fake.png", 15.0, str(tmp_path), str(tmp_path)
            )
        assert result is False
