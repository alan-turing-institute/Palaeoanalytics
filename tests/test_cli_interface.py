"""Tests for the CLI argument parser and main() entry point."""

import os
import tempfile
from unittest.mock import patch

import pytest

import logging

from pylithics.app import (
    PyLithicsApplication,
    _resolve_explore_dir,
    create_argument_parser,
    default_meta_file,
    flagged_rows,
    main,
)
from pylithics.image_processing.config import clear_config_cache


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

    def test_explore_alone_is_a_switch(self):
        args = create_argument_parser().parse_args(["--data_dir", "/d", "--explore"])
        assert args.explore is True

    def test_explore_takes_a_results_folder(self):
        args = create_argument_parser().parse_args(["--explore", "/p/results"])
        assert args.explore == "/p/results"

    def test_explore_before_another_flag_is_still_a_switch(self):
        args = create_argument_parser().parse_args(["--explore", "--data_dir", "/d"])
        assert args.explore is True
        assert args.data_dir == "/d"

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


    def test_missing_meta_file_and_no_default_exits_one(self, tmp_path, capsys):
        with patch("sys.argv", ["pylithics", "--data_dir", str(tmp_path)]):
            assert main() == 1
        assert "meta_data.csv" in capsys.readouterr().out

    def test_meta_data_csv_in_the_project_is_the_default(self, tmp_path):
        (tmp_path / "meta_data.csv").write_text("image_id,scale_id,scale\n")
        argv = ["pylithics", "--data_dir", str(tmp_path)]
        with patch("sys.argv", argv), \
             patch.object(PyLithicsApplication, "validate_inputs",
                          return_value=True), \
             patch.object(PyLithicsApplication, "run_batch_analysis",
                          return_value={
                              "success": True, "processed_successfully": 0,
                              "total_images": 0, "failed_images": [],
                          }) as mock_run:
            assert main() == 0
        assert mock_run.call_args[0][1] == str(tmp_path / "meta_data.csv")

    def test_explore_with_a_path_opens_that_folder_and_nothing_else(self):
        argv = ["pylithics", "--explore", "/p/results"]
        with patch("sys.argv", argv), \
             patch("pylithics.app._launch_explore", return_value=0) as launch, \
             patch.object(PyLithicsApplication, "run_batch_analysis") as run:
            assert main() == 0
        launch.assert_called_once_with("/p/results")
        run.assert_not_called()


# ---------------------------------------------------------------------------
# Debug flags map to the configuration keys the modules read
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDebugFlagOverrides:
    """Each debug flag sets the key that its module checks."""

    @staticmethod
    def _overrides_for(*flags):
        from pylithics.app import _apply_config_overrides
        args = create_argument_parser().parse_args(["--data_dir", "/d", *flags])
        recorded = {}

        class FakeApp:
            def update_configuration(self, **kwargs):
                recorded.update(kwargs)

        _apply_config_overrides(FakeApp(), args)
        return recorded

    def test_threshold_debug(self):
        assert self._overrides_for("--threshold_debug") == {
            "thresholding.debug_output": True
        }

    def test_old_show_thresholded_images_name_still_works(self):
        assert self._overrides_for("--show_thresholded_images") == {
            "thresholding.debug_output": True
        }

    def test_scale_debug(self):
        assert self._overrides_for("--scale_debug") == {
            "scale_calibration.debug_output": True
        }

    def test_arrow_debug(self):
        assert self._overrides_for("--arrow_debug") == {
            "arrow_detection.debug_enabled": True
        }

    def test_no_flags_no_overrides(self):
        assert self._overrides_for() == {}


# ---------------------------------------------------------------------------
# Metadata: the default file and the flag column
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMetadataDefaults:
    """meta_data.csv is found by convention, and flagged rows are held."""

    def test_default_meta_file_is_found(self, tmp_path):
        (tmp_path / "meta_data.csv").write_text("image_id,scale_id,scale\n")
        assert default_meta_file(str(tmp_path)) == str(tmp_path / "meta_data.csv")

    def test_default_meta_file_is_none_when_absent(self, tmp_path):
        assert default_meta_file(str(tmp_path)) is None

    def test_rows_without_a_flag_are_not_listed(self):
        rows = [
            {"image_id": "a.png", "scale_id": "s.png", "scale": "50", "flag": ""},
            {"image_id": "b.png", "scale_id": "", "scale": ""},
        ]
        assert flagged_rows(rows) == []

    def test_flagged_rows_are_listed_with_their_flag(self):
        rows = [
            {"image_id": "a.png", "scale_id": "", "scale": "",
             "flag": "several_scales"},
            {"image_id": "b.png", "scale_id": "", "scale": "", "flag": "  "},
        ]
        assert flagged_rows(rows) == [
            {"image_id": "a.png", "flag": "several_scales"}
        ]


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


# ---------------------------------------------------------------------------
# --explore data_dir resolution
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResolveExploreDir:
    """``--data_dir`` for --explore must accept either the parent folder or
    the processed folder directly."""

    def test_resolves_parent_to_processed_subfolder_when_csv_present(
        self, tmp_path
    ):
        processed = tmp_path / "results"
        processed.mkdir()
        (processed / "processed_metrics.csv").write_text("image_id\n")

        resolved = _resolve_explore_dir(str(tmp_path))

        assert resolved == str(processed)

    def test_falls_back_to_data_dir_when_no_processed_subfolder(
        self, tmp_path
    ):
        (tmp_path / "processed_metrics.csv").write_text("image_id\n")

        resolved = _resolve_explore_dir(str(tmp_path))

        assert resolved == str(tmp_path)

    def test_falls_back_to_data_dir_when_processed_subfolder_lacks_csv(
        self, tmp_path
    ):
        (tmp_path / "processed").mkdir()  # empty subfolder, no CSV inside

        resolved = _resolve_explore_dir(str(tmp_path))

        assert resolved == str(tmp_path)


# ---------------------------------------------------------------------------
# setup_logging() log-file resolution
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSetupLoggingLogFile:
    """The default log file should follow the user's ``--data_dir``, not the
    shell's current working directory."""

    def _config_path(self, tmp_path, logging_section):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "logging:\n"
            + "\n".join(f"  {k}: {v}" for k, v in logging_section.items())
            + "\n"
        )
        return str(config_path)

    def test_derives_log_path_from_data_dir_when_config_omits_log_file(
        self, tmp_path
    ):
        clear_config_cache()
        config_path = self._config_path(
            tmp_path, {"level": "INFO", "log_to_file": "true"},
        )
        data_dir = tmp_path / "my_dataset"

        app = PyLithicsApplication(config_file=config_path)
        app.setup_logging(data_dir=str(data_dir))

        expected = str(
            data_dir / "results" / "pylithics.log"
        )
        assert app.log_file_path == expected
        assert os.path.isdir(os.path.dirname(expected))

    def test_honours_explicit_log_file_when_config_sets_it(self, tmp_path):
        clear_config_cache()
        explicit_path = tmp_path / "custom.log"
        config_path = self._config_path(
            tmp_path,
            {
                "level": "INFO",
                "log_to_file": "true",
                "log_file": str(explicit_path),
            },
        )

        app = PyLithicsApplication(config_file=config_path)
        # data_dir should be ignored when an explicit log_file is configured.
        app.setup_logging(data_dir=str(tmp_path / "elsewhere"))

        assert app.log_file_path == str(explicit_path)

    def test_no_file_handler_when_no_data_dir_and_no_config_log_file(
        self, tmp_path
    ):
        clear_config_cache()
        config_path = self._config_path(
            tmp_path, {"level": "INFO", "log_to_file": "true"},
        )

        app = PyLithicsApplication(config_file=config_path)
        app.setup_logging()  # no data_dir, no config log_file

        assert app.log_file_path is None
        # Verify no FileHandler is attached to the root logger.
        assert not any(
            isinstance(h, logging.FileHandler)
            for h in logging.getLogger().handlers
        )
