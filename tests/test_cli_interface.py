"""
PyLithics CLI Interface Tests
=============================

Tests for the command-line interface including PyLithicsApplication class,
argument parsing and validation, batch processing workflows, and configuration overrides.
"""

import pytest
import os
import tempfile
import sys
import argparse
from unittest.mock import patch, MagicMock, call
from io import StringIO

from pylithics.app import (
    PyLithicsApplication,
    main,
    parse_arguments,
    validate_arguments,
    setup_logging,
    create_output_directories,
    _handle_single_image_processing,
    _handle_batch_processing,
    _display_results_summary
)


@pytest.mark.unit
class TestPyLithicsApplication:
    """Test the main PyLithicsApplication class."""

    def test_application_initialization_default(self):
        """Test application initialization with default parameters."""
        app = PyLithicsApplication()

        assert app.config_file is None
        assert app.output_dir is None
        assert app.verbose is False
        assert app.debug is False

    def test_application_initialization_with_params(self):
        """Test application initialization with custom parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "test_config.yaml")
            with open(config_file, 'w') as f:
                f.write("test: config\n")

            app = PyLithicsApplication(
                config_file=config_file,
                output_dir=temp_dir,
                verbose=True,
                debug=True
            )

            assert app.config_file == config_file
            assert app.output_dir == temp_dir
            assert app.verbose is True
            assert app.debug is True

    def test_application_run_single_image(self):
        """Test application run method with single image processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            app = PyLithicsApplication(output_dir=temp_dir)

            with patch('pylithics.app.analyze_single_image') as mock_analyze:
                mock_analyze.return_value = True

                result = app.run(
                    mode='single',
                    image_path='/path/to/image.png',
                    image_id='test_image',
                    scale_value=10.0
                )

                assert result is True
                mock_analyze.assert_called_once_with(
                    '/path/to/image.png', 'test_image', 10.0, temp_dir, None
                )

    def test_application_run_single_image_with_config(self):
        """Test application run method with single image and custom config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "config.yaml")
            with open(config_file, 'w') as f:
                f.write("arrow_detection:\n  enabled: false\n")

            app = PyLithicsApplication(config_file=config_file, output_dir=temp_dir)

            with patch('pylithics.app.analyze_single_image') as mock_analyze:
                mock_analyze.return_value = True

                result = app.run(
                    mode='single',
                    image_path='/path/to/image.png',
                    image_id='test_image',
                    scale_value=15.0
                )

                assert result is True
                mock_analyze.assert_called_once_with(
                    '/path/to/image.png', 'test_image', 15.0, temp_dir, config_file
                )

    def test_application_run_batch_processing(self):
        """Test application run method with batch processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_file = os.path.join(temp_dir, "metadata.csv")
            with open(metadata_file, 'w') as f:
                f.write("image_id,scale\ntest1.png,10.0\ntest2.png,12.0\n")

            app = PyLithicsApplication(output_dir=temp_dir)

            with patch('pylithics.app.batch_process_images') as mock_batch:
                mock_batch.return_value = [True, True]

                result = app.run(
                    mode='batch',
                    data_dir=temp_dir,
                    metadata_file=metadata_file
                )

                assert result == [True, True]
                mock_batch.assert_called_once_with(temp_dir, metadata_file, None)

    def test_application_run_invalid_mode(self):
        """Test application run method with invalid mode."""
        app = PyLithicsApplication()

        with patch('pylithics.app.logging') as mock_logging:
            result = app.run(mode='invalid_mode')

            assert result is False
            mock_logging.error.assert_called()

    def test_application_run_missing_required_params(self):
        """Test application run method with missing required parameters."""
        app = PyLithicsApplication()

        # Single mode without required params
        with patch('pylithics.app.logging') as mock_logging:
            result = app.run(mode='single')  # Missing image_path, image_id, scale_value

            assert result is False
            mock_logging.error.assert_called()

        # Batch mode without required params
        with patch('pylithics.app.logging') as mock_logging:
            result = app.run(mode='batch')  # Missing data_dir, metadata_file

            assert result is False
            mock_logging.error.assert_called()

    def test_application_setup_logging_verbose(self):
        """Test application logging setup in verbose mode."""
        app = PyLithicsApplication(verbose=True)

        with patch('pylithics.app.setup_logging') as mock_setup_log:
            app._setup_logging()
            mock_setup_log.assert_called_once_with(verbose=True, debug=False)

    def test_application_setup_logging_debug(self):
        """Test application logging setup in debug mode."""
        app = PyLithicsApplication(debug=True)

        with patch('pylithics.app.setup_logging') as mock_setup_log:
            app._setup_logging()
            mock_setup_log.assert_called_once_with(verbose=False, debug=True)

    def test_application_create_output_directories(self):
        """Test application output directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "outputs")
            app = PyLithicsApplication(output_dir=output_dir)

            with patch('pylithics.app.create_output_directories') as mock_create_dirs:
                app._create_output_directories()
                mock_create_dirs.assert_called_once_with(output_dir)


@pytest.mark.unit
class TestParseArguments:
    """Test command-line argument parsing."""

    def test_parse_arguments_single_mode_minimal(self):
        """Test parsing arguments for single image mode with minimal parameters."""
        args = parse_arguments([
            'single',
            '--image-path', '/path/to/image.png',
            '--image-id', 'test_image',
            '--scale', '10.0'
        ])

        assert args.mode == 'single'
        assert args.image_path == '/path/to/image.png'
        assert args.image_id == 'test_image'
        assert args.scale == 10.0
        assert args.output_dir == './processed'  # Default
        assert args.config_file is None
        assert args.verbose is False
        assert args.debug is False

    def test_parse_arguments_single_mode_full(self):
        """Test parsing arguments for single image mode with all parameters."""
        args = parse_arguments([
            'single',
            '--image-path', '/path/to/image.png',
            '--image-id', 'test_image',
            '--scale', '15.5',
            '--output-dir', '/custom/output',
            '--config-file', '/path/to/config.yaml',
            '--verbose',
            '--debug'
        ])

        assert args.mode == 'single'
        assert args.image_path == '/path/to/image.png'
        assert args.image_id == 'test_image'
        assert args.scale == 15.5
        assert args.output_dir == '/custom/output'
        assert args.config_file == '/path/to/config.yaml'
        assert args.verbose is True
        assert args.debug is True

    def test_parse_arguments_batch_mode_minimal(self):
        """Test parsing arguments for batch mode with minimal parameters."""
        args = parse_arguments([
            'batch',
            '--data-dir', '/path/to/data',
            '--metadata-file', '/path/to/metadata.csv'
        ])

        assert args.mode == 'batch'
        assert args.data_dir == '/path/to/data'
        assert args.metadata_file == '/path/to/metadata.csv'
        assert args.output_dir == './processed'  # Default
        assert args.config_file is None

    def test_parse_arguments_batch_mode_full(self):
        """Test parsing arguments for batch mode with all parameters."""
        args = parse_arguments([
            'batch',
            '--data-dir', '/path/to/data',
            '--metadata-file', '/path/to/metadata.csv',
            '--output-dir', '/custom/output',
            '--config-file', '/path/to/config.yaml',
            '--verbose',
            '--debug'
        ])

        assert args.mode == 'batch'
        assert args.data_dir == '/path/to/data'
        assert args.metadata_file == '/path/to/metadata.csv'
        assert args.output_dir == '/custom/output'
        assert args.config_file == '/path/to/config.yaml'
        assert args.verbose is True
        assert args.debug is True

    def test_parse_arguments_help(self):
        """Test parsing help argument."""
        with patch('sys.exit') as mock_exit:
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                try:
                    parse_arguments(['--help'])
                except SystemExit:
                    pass

                # Help should be displayed
                output = mock_stdout.getvalue()
                assert 'usage:' in output.lower() or 'PyLithics' in output

    def test_parse_arguments_invalid_mode(self):
        """Test parsing with invalid mode."""
        with patch('sys.stderr', new_callable=StringIO):
            with patch('sys.exit') as mock_exit:
                try:
                    parse_arguments(['invalid_mode'])
                except SystemExit:
                    pass
                mock_exit.assert_called()

    def test_parse_arguments_missing_required_single(self):
        """Test parsing single mode with missing required arguments."""
        with patch('sys.stderr', new_callable=StringIO):
            with patch('sys.exit') as mock_exit:
                try:
                    parse_arguments(['single'])  # Missing required args
                except SystemExit:
                    pass
                mock_exit.assert_called()

    def test_parse_arguments_missing_required_batch(self):
        """Test parsing batch mode with missing required arguments."""
        with patch('sys.stderr', new_callable=StringIO):
            with patch('sys.exit') as mock_exit:
                try:
                    parse_arguments(['batch', '--data-dir', '/path'])  # Missing metadata-file
                except SystemExit:
                    pass
                mock_exit.assert_called()

    def test_parse_arguments_invalid_scale_type(self):
        """Test parsing with invalid scale value type."""
        with patch('sys.stderr', new_callable=StringIO):
            with patch('sys.exit') as mock_exit:
                try:
                    parse_arguments([
                        'single',
                        '--image-path', '/path/to/image.png',
                        '--image-id', 'test',
                        '--scale', 'not_a_number'
                    ])
                except SystemExit:
                    pass
                mock_exit.assert_called()


@pytest.mark.unit
class TestValidateArguments:
    """Test argument validation."""

    def test_validate_arguments_single_mode_valid(self):
        """Test validation of valid single mode arguments."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test image file
            image_path = os.path.join(temp_dir, "test_image.png")
            with open(image_path, 'w') as f:
                f.write("dummy image")

            args = argparse.Namespace(
                mode='single',
                image_path=image_path,
                image_id='test_image',
                scale=10.0,
                output_dir=temp_dir,
                config_file=None
            )

            assert validate_arguments(args) is True

    def test_validate_arguments_single_mode_invalid_image_path(self):
        """Test validation with invalid image path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            args = argparse.Namespace(
                mode='single',
                image_path='/nonexistent/image.png',
                image_id='test_image',
                scale=10.0,
                output_dir=temp_dir,
                config_file=None
            )

            with patch('pylithics.app.logging') as mock_logging:
                assert validate_arguments(args) is False
                mock_logging.error.assert_called()

    def test_validate_arguments_single_mode_invalid_scale(self):
        """Test validation with invalid scale value."""
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "test_image.png")
            with open(image_path, 'w') as f:
                f.write("dummy image")

            args = argparse.Namespace(
                mode='single',
                image_path=image_path,
                image_id='test_image',
                scale=-5.0,  # Invalid negative scale
                output_dir=temp_dir,
                config_file=None
            )

            with patch('pylithics.app.logging') as mock_logging:
                assert validate_arguments(args) is False
                mock_logging.error.assert_called()

    def test_validate_arguments_single_mode_empty_image_id(self):
        """Test validation with empty image ID."""
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "test_image.png")
            with open(image_path, 'w') as f:
                f.write("dummy image")

            args = argparse.Namespace(
                mode='single',
                image_path=image_path,
                image_id='',  # Empty image ID
                scale=10.0,
                output_dir=temp_dir,
                config_file=None
            )

            with patch('pylithics.app.logging') as mock_logging:
                assert validate_arguments(args) is False
                mock_logging.error.assert_called()

    def test_validate_arguments_batch_mode_valid(self):
        """Test validation of valid batch mode arguments."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create data directory and metadata file
            data_dir = os.path.join(temp_dir, "data")
            os.makedirs(data_dir)

            metadata_file = os.path.join(temp_dir, "metadata.csv")
            with open(metadata_file, 'w') as f:
                f.write("image_id,scale\ntest1.png,10.0\n")

            args = argparse.Namespace(
                mode='batch',
                data_dir=data_dir,
                metadata_file=metadata_file,
                output_dir=temp_dir,
                config_file=None
            )

            assert validate_arguments(args) is True

    def test_validate_arguments_batch_mode_invalid_data_dir(self):
        """Test validation with invalid data directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_file = os.path.join(temp_dir, "metadata.csv")
            with open(metadata_file, 'w') as f:
                f.write("image_id,scale\ntest1.png,10.0\n")

            args = argparse.Namespace(
                mode='batch',
                data_dir='/nonexistent/directory',
                metadata_file=metadata_file,
                output_dir=temp_dir,
                config_file=None
            )

            with patch('pylithics.app.logging') as mock_logging:
                assert validate_arguments(args) is False
                mock_logging.error.assert_called()

    def test_validate_arguments_batch_mode_invalid_metadata_file(self):
        """Test validation with invalid metadata file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data")
            os.makedirs(data_dir)

            args = argparse.Namespace(
                mode='batch',
                data_dir=data_dir,
                metadata_file='/nonexistent/metadata.csv',
                output_dir=temp_dir,
                config_file=None
            )

            with patch('pylithics.app.logging') as mock_logging:
                assert validate_arguments(args) is False
                mock_logging.error.assert_called()

    def test_validate_arguments_invalid_config_file(self):
        """Test validation with invalid config file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "test_image.png")
            with open(image_path, 'w') as f:
                f.write("dummy image")

            args = argparse.Namespace(
                mode='single',
                image_path=image_path,
                image_id='test_image',
                scale=10.0,
                output_dir=temp_dir,
                config_file='/nonexistent/config.yaml'  # Invalid config file
            )

            with patch('pylithics.app.logging') as mock_logging:
                assert validate_arguments(args) is False
                mock_logging.error.assert_called()

    def test_validate_arguments_valid_config_file(self):
        """Test validation with valid config file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "test_image.png")
            with open(image_path, 'w') as f:
                f.write("dummy image")

            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                f.write("test: config\n")

            args = argparse.Namespace(
                mode='single',
                image_path=image_path,
                image_id='test_image',
                scale=10.0,
                output_dir=temp_dir,
                config_file=config_path
            )

            assert validate_arguments(args) is True


@pytest.mark.unit
class TestSetupLogging:
    """Test logging setup functionality."""

    def test_setup_logging_default(self):
        """Test logging setup with default parameters."""
        with patch('logging.basicConfig') as mock_basic_config, \
             patch('logging.getLogger') as mock_get_logger:

            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            setup_logging()

            mock_basic_config.assert_called_once()
            # Check that INFO level is set by default
            call_kwargs = mock_basic_config.call_args[1]
            assert call_kwargs['level'] >= 20  # INFO level or higher

    def test_setup_logging_verbose(self):
        """Test logging setup in verbose mode."""
        with patch('logging.basicConfig') as mock_basic_config:
            setup_logging(verbose=True)

            mock_basic_config.assert_called_once()
            call_kwargs = mock_basic_config.call_args[1]
            assert call_kwargs['level'] == 10  # DEBUG level

    def test_setup_logging_debug(self):
        """Test logging setup in debug mode."""
        with patch('logging.basicConfig') as mock_basic_config:
            setup_logging(debug=True)

            mock_basic_config.assert_called_once()
            call_kwargs = mock_basic_config.call_args[1]
            assert call_kwargs['level'] == 10  # DEBUG level

    def test_setup_logging_custom_format(self):
        """Test logging setup with custom format."""
        with patch('logging.basicConfig') as mock_basic_config:
            setup_logging(verbose=True)

            mock_basic_config.assert_called_once()
            call_kwargs = mock_basic_config.call_args[1]
            assert 'format' in call_kwargs
            # Should include timestamp, level, and message
            format_str = call_kwargs['format']
            assert 'levelname' in format_str or 'message' in format_str


@pytest.mark.unit
class TestCreateOutputDirectories:
    """Test output directory creation."""

    def test_create_output_directories_new(self):
        """Test creating output directories that don't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "new_output")

            # Directory doesn't exist initially
            assert not os.path.exists(output_dir)

            create_output_directories(output_dir)

            # Directory should now exist
            assert os.path.exists(output_dir)
            assert os.path.isdir(output_dir)

    def test_create_output_directories_existing(self):
        """Test creating output directories that already exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Directory already exists
            assert os.path.exists(temp_dir)

            # Should not raise error
            create_output_directories(temp_dir)

            # Directory should still exist
            assert os.path.exists(temp_dir)

    def test_create_output_directories_nested(self):
        """Test creating nested output directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = os.path.join(temp_dir, "level1", "level2", "output")

            # Nested path doesn't exist
            assert not os.path.exists(nested_dir)

            create_output_directories(nested_dir)

            # All levels should be created
            assert os.path.exists(nested_dir)
            assert os.path.isdir(nested_dir)

    def test_create_output_directories_permission_error(self):
        """Test handling of permission errors."""
        # Try to create directory in root (should fail without permissions)
        restricted_path = "/root/pylithics_test"

        with patch('pylithics.app.logging') as mock_logging:
            with patch('os.makedirs', side_effect=PermissionError("Permission denied")):
                create_output_directories(restricted_path)

                # Should log error
                mock_logging.error.assert_called()


@pytest.mark.unit
class TestHandleSingleImageProcessing:
    """Test single image processing handler."""

    def test_handle_single_image_processing_success(self):
        """Test successful single image processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "test.png")
            with open(image_path, 'w') as f:
                f.write("dummy")

            with patch('pylithics.app.analyze_single_image') as mock_analyze:
                mock_analyze.return_value = True

                result = _handle_single_image_processing(
                    image_path, 'test_image', 10.0, temp_dir, None
                )

                assert result is True
                mock_analyze.assert_called_once()

    def test_handle_single_image_processing_failure(self):
        """Test single image processing failure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('pylithics.app.analyze_single_image') as mock_analyze, \
                 patch('pylithics.app.logging') as mock_logging:

                mock_analyze.return_value = False

                result = _handle_single_image_processing(
                    '/nonexistent/image.png', 'test_image', 10.0, temp_dir, None
                )

                assert result is False
                mock_logging.error.assert_called()

    def test_handle_single_image_processing_exception(self):
        """Test single image processing with exception."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('pylithics.app.analyze_single_image') as mock_analyze, \
                 patch('pylithics.app.logging') as mock_logging:

                mock_analyze.side_effect = Exception("Processing error")

                result = _handle_single_image_processing(
                    'dummy_path', 'test_image', 10.0, temp_dir, None
                )

                assert result is False
                mock_logging.error.assert_called()


@pytest.mark.unit
class TestHandleBatchProcessing:
    """Test batch processing handler."""

    def test_handle_batch_processing_success(self):
        """Test successful batch processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_file = os.path.join(temp_dir, "metadata.csv")
            with open(metadata_file, 'w') as f:
                f.write("image_id,scale\ntest1.png,10.0\ntest2.png,12.0\n")

            with patch('pylithics.app.batch_process_images') as mock_batch:
                mock_batch.return_value = [True, True]

                results = _handle_batch_processing(temp_dir, metadata_file, None)

                assert results == [True, True]
                mock_batch.assert_called_once()

    def test_handle_batch_processing_partial_success(self):
        """Test batch processing with partial success."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_file = os.path.join(temp_dir, "metadata.csv")
            with open(metadata_file, 'w') as f:
                f.write("image_id,scale\ntest1.png,10.0\ntest2.png,12.0\n")

            with patch('pylithics.app.batch_process_images') as mock_batch, \
                 patch('pylithics.app.logging') as mock_logging:

                mock_batch.return_value = [True, False]  # One success, one failure

                results = _handle_batch_processing(temp_dir, metadata_file, None)

                assert results == [True, False]
                # Should log summary with mixed results

    def test_handle_batch_processing_exception(self):
        """Test batch processing with exception."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('pylithics.app.batch_process_images') as mock_batch, \
                 patch('pylithics.app.logging') as mock_logging:

                mock_batch.side_effect = Exception("Batch processing error")

                results = _handle_batch_processing(temp_dir, 'dummy_metadata', None)

                assert results == []
                mock_logging.error.assert_called()


@pytest.mark.unit
class TestDisplayResultsSummary:
    """Test results summary display."""

    def test_display_results_summary_all_success(self):
        """Test results summary with all successful results."""
        results = [True, True, True]

        with patch('pylithics.app.logging') as mock_logging:
            _display_results_summary(results, 'batch')

            # Should log success summary
            mock_logging.info.assert_called()
            info_calls = [call.args[0] for call in mock_logging.info.call_args_list]
            summary_message = ' '.join(info_calls)
            assert '3' in summary_message and 'successful' in summary_message.lower()

    def test_display_results_summary_mixed_results(self):
        """Test results summary with mixed results."""
        results = [True, False, True, False]

        with patch('pylithics.app.logging') as mock_logging:
            _display_results_summary(results, 'batch')

            # Should log mixed results summary
            mock_logging.info.assert_called()
            info_calls = [call.args[0] for call in mock_logging.info.call_args_list]
            summary_message = ' '.join(info_calls)
            assert '2' in summary_message  # Should mention success count
            assert 'failed' in summary_message.lower() or 'error' in summary_message.lower()

    def test_display_results_summary_all_failure(self):
        """Test results summary with all failed results."""
        results = [False, False, False]

        with patch('pylithics.app.logging') as mock_logging:
            _display_results_summary(results, 'batch')

            # Should log failure summary
            mock_logging.error.assert_called() or mock_logging.warning.assert_called()

    def test_display_results_summary_empty_results(self):
        """Test results summary with empty results."""
        results = []

        with patch('pylithics.app.logging') as mock_logging:
            _display_results_summary(results, 'batch')

            # Should log about no results
            mock_logging.warning.assert_called()

    def test_display_results_summary_single_mode(self):
        """Test results summary for single image mode."""
        with patch('pylithics.app.logging') as mock_logging:
            _display_results_summary(True, 'single')

            # Should log single image success
            mock_logging.info.assert_called()

            _display_results_summary(False, 'single')

            # Should log single image failure
            mock_logging.error.assert_called()


@pytest.mark.integration
class TestMainFunction:
    """Test the main entry point function."""

    def test_main_single_mode_success(self):
        """Test main function with successful single mode processing."""
        test_args = [
            'pylithics',
            'single',
            '--image-path', '/path/to/image.png',
            '--image-id', 'test_image',
            '--scale', '10.0'
        ]

        with patch('sys.argv', test_args), \
             patch('pylithics.app.validate_arguments') as mock_validate, \
             patch('pylithics.app.PyLithicsApplication') as mock_app_class:

            mock_validate.return_value = True
            mock_app = MagicMock()
            mock_app.run.return_value = [True, True, False]  # Mixed results
            mock_app_class.return_value = mock_app

            result = main()

            assert result == 0  # Success exit code (some processed successfully)
            mock_app.run.assert_called_once()

    def test_main_validation_failure(self):
        """Test main function with argument validation failure."""
        test_args = [
            'pylithics',
            'single',
            '--image-path', '/nonexistent/image.png',
            '--image-id', 'test',
            '--scale', '10.0'
        ]

        with patch('sys.argv', test_args), \
             patch('pylithics.app.validate_arguments') as mock_validate:

            mock_validate.return_value = False

            result = main()

            assert result == 1  # Error exit code

    def test_main_processing_failure(self):
        """Test main function with processing failure."""
        test_args = [
            'pylithics',
            'single',
            '--image-path', '/path/to/image.png',
            '--image-id', 'test_image',
            '--scale', '10.0'
        ]

        with patch('sys.argv', test_args), \
             patch('pylithics.app.validate_arguments') as mock_validate, \
             patch('pylithics.app.PyLithicsApplication') as mock_app_class:

            mock_validate.return_value = True
            mock_app = MagicMock()
            mock_app.run.return_value = False
            mock_app_class.return_value = mock_app

            result = main()

            assert result == 1  # Error exit code

    def test_main_exception_handling(self):
        """Test main function exception handling."""
        test_args = [
            'pylithics',
            'single',
            '--image-path', '/path/to/image.png',
            '--image-id', 'test_image',
            '--scale', '10.0'
        ]

        with patch('sys.argv', test_args), \
             patch('pylithics.app.validate_arguments') as mock_validate, \
             patch('pylithics.app.PyLithicsApplication') as mock_app_class, \
             patch('pylithics.app.logging') as mock_logging:

            mock_validate.return_value = True
            mock_app = MagicMock()
            mock_app.run.side_effect = Exception("Unexpected error")
            mock_app_class.return_value = mock_app

            result = main()

            assert result == 1  # Error exit code
            mock_logging.error.assert_called()

    def test_main_help_display(self):
        """Test main function with help argument."""
        test_args = ['pylithics', '--help']

        with patch('sys.argv', test_args), \
             patch('sys.exit') as mock_exit:

            try:
                main()
            except SystemExit:
                pass

            # Help should cause exit
            mock_exit.assert_called()


@pytest.mark.integration
class TestCLIIntegrationScenarios:
    """Test CLI integration with realistic scenarios."""

    def test_cli_single_image_workflow(self):
        """Test complete CLI workflow for single image processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test image
            image_path = os.path.join(temp_dir, "test_artifact.png")
            with open(image_path, 'w') as f:
                f.write("dummy image content")

            # Create config file
            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                f.write("""
arrow_detection:
  enabled: true
  reference_dpi: 300.0
thresholding:
  method: otsu
contour_filtering:
  min_area: 100.0
""")

            test_args = [
                'pylithics',
                'single',
                '--image-path', image_path,
                '--image-id', 'test_artifact',
                '--scale', '15.5',
                '--output-dir', temp_dir,
                '--config-file', config_path,
                '--verbose'
            ]

            with patch('sys.argv', test_args), \
                 patch('pylithics.app.analyze_single_image') as mock_analyze:

                mock_analyze.return_value = True

                result = main()

                assert result == 0
                mock_analyze.assert_called_once_with(
                    image_path, 'test_artifact', 15.5, temp_dir, config_path
                )

    def test_cli_batch_processing_workflow(self):
        """Test complete CLI workflow for batch processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create data directory structure
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Create test images
            for i in range(3):
                image_path = os.path.join(images_dir, f"artifact_{i}.png")
                with open(image_path, 'w') as f:
                    f.write(f"dummy image {i}")

            # Create metadata file
            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write("image_id,scale\n")
                for i in range(3):
                    f.write(f"artifact_{i}.png,{10.0 + i}\n")

            # Create output directory
            output_dir = os.path.join(temp_dir, "outputs")

            test_args = [
                'pylithics',
                'batch',
                '--data-dir', data_dir,
                '--metadata-file', metadata_path,
                '--output-dir', output_dir,
                '--debug'
            ]

            with patch('sys.argv', test_args), \
                 patch('pylithics.app.batch_process_images') as mock_batch:

                mock_batch.return_value = [True, True, False]  # Mixed results

                result = main()

                assert result == 0  # Success (some processed)
                mock_batch.assert_called_once_with(data_dir, metadata_path, None)

    def test_cli_config_override_workflow(self):
        """Test CLI workflow with configuration overrides."""
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "test.png")
            with open(image_path, 'w') as f:
                f.write("dummy")

            # Test different config scenarios
            configs = [
                {
                    'name': 'arrows_disabled',
                    'content': 'arrow_detection:\n  enabled: false\n'
                },
                {
                    'name': 'high_dpi',
                    'content': 'arrow_detection:\n  reference_dpi: 600.0\nthresholding:\n  method: adaptive\n'
                },
                {
                    'name': 'minimal_areas',
                    'content': 'contour_filtering:\n  min_area: 25.0\n'
                }
            ]

            for config_info in configs:
                config_path = os.path.join(temp_dir, f"{config_info['name']}.yaml")
                with open(config_path, 'w') as f:
                    f.write(config_info['content'])

                test_args = [
                    'pylithics',
                    'single',
                    '--image-path', image_path,
                    '--image-id', f"test_{config_info['name']}",
                    '--scale', '12.0',
                    '--output-dir', temp_dir,
                    '--config-file', config_path
                ]

                with patch('sys.argv', test_args), \
                     patch('pylithics.app.analyze_single_image') as mock_analyze:

                    mock_analyze.return_value = True

                    result = main()

                    assert result == 0
                    mock_analyze.assert_called_with(
                        image_path, f"test_{config_info['name']}", 12.0, temp_dir, config_path
                    )

    def test_cli_error_handling_workflow(self):
        """Test CLI error handling in various scenarios."""
        error_scenarios = [
            {
                'name': 'nonexistent_image',
                'args': [
                    'pylithics', 'single',
                    '--image-path', '/nonexistent/image.png',
                    '--image-id', 'test',
                    '--scale', '10.0'
                ],
                'expected_exit': 1
            },
            {
                'name': 'invalid_scale',
                'args': [
                    'pylithics', 'single',
                    '--image-path', 'dummy.png',
                    '--image-id', 'test',
                    '--scale', '-5.0'
                ],
                'expected_exit': 1
            },
            {
                'name': 'missing_metadata',
                'args': [
                    'pylithics', 'batch',
                    '--data-dir', '/tmp',
                    '--metadata-file', '/nonexistent/metadata.csv'
                ],
                'expected_exit': 1
            }
        ]

        for scenario in error_scenarios:
            with patch('sys.argv', scenario['args']):
                result = main()
                assert result == scenario['expected_exit'], f"Failed scenario: {scenario['name']}"

    def test_cli_verbose_and_debug_modes(self):
        """Test CLI verbose and debug logging modes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "test.png")
            with open(image_path, 'w') as f:
                f.write("dummy")

            # Test verbose mode
            test_args_verbose = [
                'pylithics',
                'single',
                '--image-path', image_path,
                '--image-id', 'test_verbose',
                '--scale', '10.0',
                '--verbose'
            ]

            with patch('sys.argv', test_args_verbose), \
                 patch('pylithics.app.setup_logging') as mock_setup_log, \
                 patch('pylithics.app.analyze_single_image') as mock_analyze:

                mock_analyze.return_value = True

                result = main()

                assert result == 0
                # Verify verbose logging was set up
                mock_setup_log.assert_called_with(verbose=True, debug=False)

            # Test debug mode
            test_args_debug = [
                'pylithics',
                'single',
                '--image-path', image_path,
                '--image-id', 'test_debug',
                '--scale', '10.0',
                '--debug'
            ]

            with patch('sys.argv', test_args_debug), \
                 patch('pylithics.app.setup_logging') as mock_setup_log, \
                 patch('pylithics.app.analyze_single_image') as mock_analyze:

                mock_analyze.return_value = True

                result = main()

                assert result == 0
                # Verify debug logging was set up
                mock_setup_log.assert_called_with(verbose=False, debug=True)


@pytest.mark.integration
class TestCLIRealWorldUsage:
    """Test CLI with realistic archaeological data processing scenarios."""

    def test_cli_archaeological_blade_processing(self):
        """Test CLI processing of archaeological blade tool."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Simulate blade tool image
            image_path = os.path.join(temp_dir, "blade_tool_001.png")
            with open(image_path, 'w') as f:
                f.write("blade tool image data")

            # Archaeological-specific config
            config_path = os.path.join(temp_dir, "archaeological_config.yaml")
            with open(config_path, 'w') as f:
                f.write("""
# Archaeological blade tool processing configuration
thresholding:
  method: adaptive  # Better for varying lighting
  max_value: 255

arrow_detection:
  enabled: true
  reference_dpi: 300.0
  min_area_scale_factor: 0.6  # Smaller arrows on blades
  debug_enabled: false

contour_filtering:
  min_area: 75.0  # Filter small noise
  exclude_border: true

normalization:
  enabled: true
  method: minmax
  clip_values: [5, 250]  # Handle shadows/highlights

morphological_closing:
  enabled: true
  kernel_size: 3  # Fill small gaps in scars
""")

            test_args = [
                'pylithics',
                'single',
                '--image-path', image_path,
                '--image-id', 'blade_tool_001',
                '--scale', '18.5',  # mm scale
                '--output-dir', temp_dir,
                '--config-file', config_path,
                '--verbose'
            ]

            with patch('sys.argv', test_args), \
                 patch('pylithics.app.analyze_single_image') as mock_analyze:

                mock_analyze.return_value = True

                result = main()

                assert result == 0
                mock_analyze.assert_called_once_with(
                    image_path, 'blade_tool_001', 18.5, temp_dir, config_path
                )

    def test_cli_batch_archaeological_survey(self):
        """Test CLI batch processing of archaeological survey data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create survey data structure
            survey_dir = os.path.join(temp_dir, "survey_2024")
            images_dir = os.path.join(survey_dir, "images")
            os.makedirs(images_dir)

            # Create sample artifact images
            artifacts = [
                ("blade_001.png", "22.0"),
                ("core_002.png", "35.5"),
                ("scraper_003.png", "18.2"),
                ("biface_004.png", "28.7"),
                ("flake_005.png", "15.0")
            ]

            metadata_content = "image_id,scale_id,scale,site_context,artifact_type\n"
            for i, (filename, scale) in enumerate(artifacts):
                image_path = os.path.join(images_dir, filename)
                with open(image_path, 'w') as f:
                    f.write(f"archaeological artifact {i}")

                artifact_type = filename.split('_')[0]
                metadata_content += f"{filename},scale_{i+1},{scale},site_A,{artifact_type}\n"

            metadata_path = os.path.join(survey_dir, "artifact_metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            # Survey-specific configuration
            survey_config_path = os.path.join(temp_dir, "survey_config.yaml")
            with open(survey_config_path, 'w') as f:
                f.write("""
# Archaeological survey batch processing config
logging:
  level: INFO
  log_to_file: true
  log_file: survey_processing.log

thresholding:
  method: otsu  # Consistent for batch processing

arrow_detection:
  enabled: true
  reference_dpi: 300.0

contour_filtering:
  min_area: 50.0  # Consistent minimum for survey

# Enable all spatial analysis for survey
symmetry_analysis:
  enabled: true

voronoi_analysis:
  enabled: true
  padding_factor: 0.05

lateral_analysis:
  enabled: true
""")

            test_args = [
                'pylithics',
                'batch',
                '--data-dir', survey_dir,
                '--metadata-file', metadata_path,
                '--output-dir', temp_dir,
                '--config-file', survey_config_path,
                '--debug'
            ]

            with patch('sys.argv', test_args), \
                 patch('pylithics.app.batch_process_images') as mock_batch:

                # Simulate mixed success (realistic for field data)
                mock_batch.return_value = [True, True, False, True, True]

                result = main()

                assert result == 0  # Success overall
                mock_batch.assert_called_once_with(survey_dir, metadata_path, survey_config_path)

    def test_cli_high_resolution_artifact_processing(self):
        """Test CLI processing of high-resolution artifact image."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # High-res artifact image
            hr_image_path = os.path.join(temp_dir, "high_res_biface.tiff")
            with open(hr_image_path, 'w') as f:
                f.write("high resolution biface image")

            # High-resolution specific config
            hr_config_path = os.path.join(temp_dir, "high_res_config.yaml")
            with open(hr_config_path, 'w') as f:
                f.write("""
# High-resolution image processing configuration
thresholding:
  method: adaptive
  max_value: 255

normalization:
  enabled: true
  method: clahe  # Better for high-res detail enhancement

grayscale_conversion:
  enabled: true
  method: clahe

arrow_detection:
  enabled: true
  reference_dpi: 600.0  # High DPI
  min_area_scale_factor: 0.5  # Detect smaller features
  min_defect_depth_scale_factor: 0.6
  debug_enabled: true  # Enable for quality control

contour_filtering:
  min_area: 25.0  # Lower threshold for high-res detail

morphological_closing:
  enabled: true
  kernel_size: 5  # Larger kernel for high-res
""")

            test_args = [
                'pylithics',
                'single',
                '--image-path', hr_image_path,
                '--image-id', 'high_res_biface_001',
                '--scale', '45.2',  # Large artifact
                '--output-dir', temp_dir,
                '--config-file', hr_config_path,
                '--debug'
            ]

            with patch('sys.argv', test_args), \
                 patch('pylithics.app.analyze_single_image') as mock_analyze:

                mock_analyze.return_value = True

                result = main()

                assert result == 0
                mock_analyze.assert_called_once_with(
                    hr_image_path, 'high_res_biface_001', 45.2, temp_dir, hr_config_path
                )

    def test_cli_multi_site_comparative_study(self):
        """Test CLI for multi-site comparative archaeological study."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multi-site structure
            sites = ['site_a', 'site_b', 'site_c']

            for site in sites:
                site_dir = os.path.join(temp_dir, site)
                images_dir = os.path.join(site_dir, "images")
                os.makedirs(images_dir)

                # Create site-specific artifacts
                for i in range(3):
                    artifact_name = f"{site}_artifact_{i:02d}.png"
                    image_path = os.path.join(images_dir, artifact_name)
                    with open(image_path, 'w') as f:
                        f.write(f"{site} artifact {i}")

                # Site-specific metadata
                metadata_path = os.path.join(site_dir, "metadata.csv")
                with open(metadata_path, 'w') as f:
                    f.write("image_id,scale,site_context,excavation_unit\n")
                    for i in range(3):
                        artifact_name = f"{site}_artifact_{i:02d}.png"
                        scale = 12.0 + i * 2.5
                        f.write(f"{artifact_name},{scale},{site},unit_{i}\n")

            # Comparative study config
            comp_config_path = os.path.join(temp_dir, "comparative_config.yaml")
            with open(comp_config_path, 'w') as f:
                f.write("""
# Multi-site comparative study configuration
logging:
  level: INFO
  log_to_file: true

# Standardized processing for comparison
thresholding:
  method: otsu  # Consistent across sites

arrow_detection:
  enabled: true
  reference_dpi: 300.0

# Enable all analyses for comprehensive comparison
symmetry_analysis:
  enabled: true

voronoi_analysis:
  enabled: true

lateral_analysis:
  enabled: true

contour_filtering:
  min_area: 75.0  # Consistent standard
""")

            # Process each site
            for site in sites:
                site_dir = os.path.join(temp_dir, site)
                metadata_path = os.path.join(site_dir, "metadata.csv")
                output_dir = os.path.join(temp_dir, f"outputs_{site}")

                test_args = [
                    'pylithics',
                    'batch',
                    '--data-dir', site_dir,
                    '--metadata-file', metadata_path,
                    '--output-dir', output_dir,
                    '--config-file', comp_config_path,
                    '--verbose'
                ]

                with patch('sys.argv', test_args), \
                     patch('pylithics.app.batch_process_images') as mock_batch:

                    mock_batch.return_value = [True, True, True]

                    result = main()

                    assert result == 0
                    mock_batch.assert_called_with(site_dir, metadata_path, comp_config_path)


@pytest.mark.performance
class TestCLIPerformance:
    """Test CLI performance characteristics."""

    def test_cli_batch_processing_performance(self):
        """Test CLI performance with large batch processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create large batch simulation
            data_dir = os.path.join(temp_dir, "large_batch")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Simulate many images
            num_images = 20
            metadata_content = "image_id,scale\n"

            for i in range(num_images):
                image_name = f"artifact_{i:03d}.png"
                image_path = os.path.join(images_dir, image_name)
                with open(image_path, 'w') as f:
                    f.write(f"artifact {i}")
                metadata_content += f"{image_name},{10.0 + i * 0.5}\n"

            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            test_args = [
                'pylithics',
                'batch',
                '--data-dir', data_dir,
                '--metadata-file', metadata_path,
                '--output-dir', temp_dir
            ]

            with patch('sys.argv', test_args), \
                 patch('pylithics.app.batch_process_images') as mock_batch:

                # Simulate processing results
                results = [True] * (num_images - 2) + [False] * 2  # Mostly successful
                mock_batch.return_value = results

                import time
                start_time = time.time()

                result = main()

                end_time = time.time()
                processing_time = end_time - start_time

                assert result == 0
                # CLI overhead should be minimal
                assert processing_time < 5.0  # 5 seconds max for CLI operations

    def test_cli_memory_usage(self):
        """Test CLI memory usage characteristics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "test.png")
            with open(image_path, 'w') as f:
                f.write("test image")

            test_args = [
                'pylithics',
                'single',
                '--image-path', image_path,
                '--image-id', 'memory_test',
                '--scale', '10.0',
                '--output-dir', temp_dir
            ]

            with patch('sys.argv', test_args), \
                 patch('pylithics.app.analyze_single_image') as mock_analyze:

                mock_analyze.return_value = True

                import psutil
                import os

                process = psutil.Process(os.getpid())
                initial_memory = process.memory_info().rss

                result = main()

                final_memory = process.memory_info().rss
                memory_increase = final_memory - initial_memory

                assert result == 0
                # CLI memory overhead should be reasonable
                assert memory_increase < 50 * 1024 * 1024  # 50MB max for CLI()
            mock_app.run.return_value = True
            mock_app_class.return_value = mock_app

            result = main()

            assert result == 0  # Success exit code
            mock_app.run.assert_called_once()

    def test_main_batch_mode_success(self):
        """Test main function with successful batch mode processing."""
        test_args = [
            'pylithics',
            'batch',
            '--data-dir', '/path/to/data',
            '--metadata-file', '/path/to/metadata.csv'
        ]

        with patch('sys.argv', test_args), \
             patch('pylithics.app.validate_arguments') as mock_validate, \
             patch('pylithics.app.PyLithicsApplication') as mock_app_class:

            mock_validate.return_value = True
            mock_app = MagicMock