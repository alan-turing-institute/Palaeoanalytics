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
    create_argument_parser
)


@pytest.mark.unit
class TestPyLithicsApplication:
    """Test the main PyLithicsApplication class."""

    def test_application_initialization_default(self):
        """Test application initialization with default parameters."""
        app = PyLithicsApplication()

        # Test basic initialization - check attributes that actually exist
        assert hasattr(app, 'config_manager')
        assert app is not None

    def test_application_initialization_with_config(self):
        """Test application initialization with custom config file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "test_config.yaml")
            with open(config_file, 'w') as f:
                f.write("test: config\n")

            app = PyLithicsApplication(config_file=config_file)

            assert app is not None
            assert hasattr(app, 'config_manager')

    def test_application_run_batch_analysis(self):
        """Test application run_batch_analysis method."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create basic directory structure
            data_dir = temp_dir
            images_dir = os.path.join(data_dir, 'images')
            os.makedirs(images_dir)

            metadata_file = os.path.join(temp_dir, "metadata.csv")
            with open(metadata_file, 'w') as f:
                f.write("image_id,scale\ntest1.png,10.0\ntest2.png,12.0\n")

            app = PyLithicsApplication()

            with patch('pylithics.app.read_metadata') as mock_read_meta, \
                 patch.object(app, 'process_single_image') as mock_process:

                mock_read_meta.return_value = [
                    {'image_id': 'test1.png', 'scale': '10.0'},
                    {'image_id': 'test2.png', 'scale': '12.0'}
                ]
                mock_process.return_value = True

                results = app.run_batch_analysis(data_dir, metadata_file)

                assert isinstance(results, dict)
                assert 'success' in results
                assert 'total_images' in results

    def test_application_process_single_image(self):
        """Test application process_single_image method."""
        with tempfile.TemporaryDirectory() as temp_dir:
            app = PyLithicsApplication()

            with patch('pylithics.app.execute_preprocessing_pipeline') as mock_preprocess, \
                 patch('pylithics.app.verify_image_dpi_and_scale') as mock_verify, \
                 patch('pylithics.app.process_and_save_contours') as mock_process:

                mock_preprocess.return_value = MagicMock()  # Mock processed image
                mock_verify.return_value = 0.1  # Mock conversion factor
                mock_process.return_value = None  # Void function

                result = app.process_single_image(
                    'test_image', 10.0, temp_dir, temp_dir
                )

                assert isinstance(result, bool)

    def test_application_validate_inputs(self):
        """Test application validate_inputs method."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test metadata file
            metadata_file = os.path.join(temp_dir, "metadata.csv")
            with open(metadata_file, 'w') as f:
                f.write("image_id,scale\ntest1.png,10.0\n")

            # Create images directory
            images_dir = os.path.join(temp_dir, 'images')
            os.makedirs(images_dir)

            app = PyLithicsApplication()

            result = app.validate_inputs(temp_dir, metadata_file)
            assert isinstance(result, bool)

    def test_application_setup_logging(self):
        """Test application setup_logging method."""
        app = PyLithicsApplication()

        # Should not raise an exception
        app.setup_logging()

    def test_application_update_configuration(self):
        """Test application update_configuration method."""
        app = PyLithicsApplication()

        # Should not raise an exception
        app.update_configuration(**{'logging.level': 'DEBUG'})


@pytest.mark.unit
class TestCreateArgumentParser:
    """Test the argument parser creation."""

    def test_create_argument_parser_basic(self):
        """Test basic argument parser creation."""
        parser = create_argument_parser()

        assert isinstance(parser, argparse.ArgumentParser)
        assert parser.prog == 'PyLithics'

    def test_argument_parser_help_options(self):
        """Test that parser has help options."""
        parser = create_argument_parser()

        # Test help works without crashing
        help_text = parser.format_help()
        assert 'PyLithics' in help_text
        assert 'usage:' in help_text.lower()

    def test_argument_parser_data_dir_argument(self):
        """Test that parser accepts data_dir argument."""
        parser = create_argument_parser()

        # Parse valid arguments
        args = parser.parse_args(['--data_dir', '/test/path', '--meta_file', '/test/meta.csv'])

        assert args.data_dir == '/test/path'
        assert args.meta_file == '/test/meta.csv'

    def test_argument_parser_optional_arguments(self):
        """Test parser with optional arguments."""
        parser = create_argument_parser()

        # Test with optional arguments
        args = parser.parse_args([
            '--data_dir', '/test/path',
            '--meta_file', '/test/meta.csv',
            '--threshold_method', 'otsu',
            '--log_level', 'DEBUG'
        ])

        assert args.threshold_method == 'otsu'
        assert args.log_level == 'DEBUG'

    def test_argument_parser_missing_required(self):
        """Test parser with missing required arguments."""
        parser = create_argument_parser()

        with pytest.raises(SystemExit):
            # Missing required arguments should cause SystemExit
            parser.parse_args([])

    def test_argument_parser_invalid_choices(self):
        """Test parser with invalid choice values."""
        parser = create_argument_parser()

        with pytest.raises(SystemExit):
            # Invalid threshold method should cause SystemExit
            parser.parse_args([
                '--data_dir', '/test/path',
                '--meta_file', '/test/meta.csv',
                '--threshold_method', 'invalid_method'
            ])


@pytest.mark.integration
class TestMainFunction:
    """Test the main entry point function."""

    def test_main_with_valid_arguments(self):
        """Test main function with valid arguments."""
        test_args = [
            'pylithics',
            '--data_dir', '/test/data',
            '--meta_file', '/test/metadata.csv'
        ]

        with patch('sys.argv', test_args), \
             patch.object(PyLithicsApplication, 'validate_inputs') as mock_validate, \
             patch.object(PyLithicsApplication, 'run_batch_analysis') as mock_run:

            mock_validate.return_value = True
            mock_run.return_value = {
                'success': True,
                'processed_successfully': 2,
                'total_images': 2,
                'failed_images': []
            }

            result = main()

            assert result == 0  # Success exit code

    def test_main_with_help_argument(self):
        """Test main function with help argument."""
        test_args = ['pylithics', '--help']

        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit) as excinfo:
                main()
            # Help should exit with code 0
            assert excinfo.value.code == 0

    def test_main_with_invalid_arguments(self):
        """Test main function with invalid arguments."""
        test_args = [
            'pylithics',
            '--invalid_arg', 'value'
        ]

        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit) as excinfo:
                main()
            # Invalid args should exit with non-zero code
            assert excinfo.value.code != 0

    def test_main_validation_failure(self):
        """Test main function when input validation fails."""
        test_args = [
            'pylithics',
            '--data_dir', '/nonexistent/path',
            '--meta_file', '/nonexistent/metadata.csv'
        ]

        with patch('sys.argv', test_args), \
             patch.object(PyLithicsApplication, 'validate_inputs') as mock_validate:

            mock_validate.return_value = False

            result = main()

            assert result == 1  # Error exit code

    def test_main_processing_failure(self):
        """Test main function when processing fails."""
        test_args = [
            'pylithics',
            '--data_dir', '/test/data',
            '--meta_file', '/test/metadata.csv'
        ]

        with patch('sys.argv', test_args), \
             patch.object(PyLithicsApplication, 'validate_inputs') as mock_validate, \
             patch.object(PyLithicsApplication, 'run_batch_analysis') as mock_run:

            mock_validate.return_value = True
            mock_run.return_value = {
                'success': False,
                'error': 'Processing failed'
            }

            result = main()

            assert result == 1  # Error exit code

    def test_main_exception_handling(self):
        """Test main function exception handling."""
        test_args = [
            'pylithics',
            '--data_dir', '/test/data',
            '--meta_file', '/test/metadata.csv'
        ]

        with patch('sys.argv', test_args), \
             patch.object(PyLithicsApplication, 'validate_inputs') as mock_validate:

            mock_validate.side_effect = Exception("Unexpected error")

            result = main()

            assert result == 1  # Error exit code

    def test_main_partial_success(self):
        """Test main function with partial processing success."""
        test_args = [
            'pylithics',
            '--data_dir', '/test/data',
            '--meta_file', '/test/metadata.csv'
        ]

        with patch('sys.argv', test_args), \
             patch.object(PyLithicsApplication, 'validate_inputs') as mock_validate, \
             patch.object(PyLithicsApplication, 'run_batch_analysis') as mock_run:

            mock_validate.return_value = True
            mock_run.return_value = {
                'success': True,
                'processed_successfully': 2,
                'total_images': 3,
                'failed_images': ['failed_image.png']
            }

            result = main()

            assert result == 0  # Still success if some processed


@pytest.mark.integration
class TestCLIIntegrationScenarios:
    """Test CLI integration with realistic scenarios."""

    def test_cli_with_config_file(self):
        """Test CLI with custom configuration file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config file
            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                f.write("""
arrow_detection:
  enabled: true
  reference_dpi: 300.0
thresholding:
  method: otsu
""")

            # Create basic data structure
            data_dir = temp_dir
            images_dir = os.path.join(data_dir, 'images')
            os.makedirs(images_dir)

            metadata_path = os.path.join(temp_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write("image_id,scale\ntest1.png,10.0\n")

            test_args = [
                'pylithics',
                '--data_dir', data_dir,
                '--meta_file', metadata_path,
                '--config_file', config_path
            ]

            with patch('sys.argv', test_args), \
                 patch.object(PyLithicsApplication, 'run_batch_analysis') as mock_run:

                mock_run.return_value = {
                    'success': True,
                    'processed_successfully': 1,
                    'total_images': 1,
                    'failed_images': []
                }

                result = main()

                assert result == 0

    def test_cli_with_threshold_method_override(self):
        """Test CLI with threshold method override."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create basic data structure
            data_dir = temp_dir
            images_dir = os.path.join(data_dir, 'images')
            os.makedirs(images_dir)

            metadata_path = os.path.join(temp_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write("image_id,scale\ntest1.png,10.0\n")

            test_args = [
                'pylithics',
                '--data_dir', data_dir,
                '--meta_file', metadata_path,
                '--threshold_method', 'adaptive',
                '--log_level', 'DEBUG'
            ]

            with patch('sys.argv', test_args), \
                 patch.object(PyLithicsApplication, 'run_batch_analysis') as mock_run:

                mock_run.return_value = {
                    'success': True,
                    'processed_successfully': 1,
                    'total_images': 1,
                    'failed_images': []
                }

                result = main()

                assert result == 0

    def test_cli_with_arrow_detection_disabled(self):
        """Test CLI with arrow detection disabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create basic data structure
            data_dir = temp_dir
            images_dir = os.path.join(data_dir, 'images')
            os.makedirs(images_dir)

            metadata_path = os.path.join(temp_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write("image_id,scale\ntest1.png,10.0\n")

            test_args = [
                'pylithics',
                '--data_dir', data_dir,
                '--meta_file', metadata_path,
                '--disable_arrow_detection'
            ]

            with patch('sys.argv', test_args), \
                 patch.object(PyLithicsApplication, 'run_batch_analysis') as mock_run:

                mock_run.return_value = {
                    'success': True,
                    'processed_successfully': 1,
                    'total_images': 1,
                    'failed_images': []
                }

                result = main()

                assert result == 0

    def test_cli_with_debug_mode(self):
        """Test CLI with debug mode enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create basic data structure
            data_dir = temp_dir
            images_dir = os.path.join(data_dir, 'images')
            os.makedirs(images_dir)

            metadata_path = os.path.join(temp_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write("image_id,scale\ntest1.png,10.0\n")

            test_args = [
                'pylithics',
                '--data_dir', data_dir,
                '--meta_file', metadata_path,
                '--arrow_debug',
                '--log_level', 'DEBUG'
            ]

            with patch('sys.argv', test_args), \
                 patch.object(PyLithicsApplication, 'run_batch_analysis') as mock_run:

                mock_run.return_value = {
                    'success': True,
                    'processed_successfully': 1,
                    'total_images': 1,
                    'failed_images': []
                }

                result = main()

                assert result == 0

    def test_cli_error_handling_workflow(self):
        """Test CLI error handling in various scenarios."""
        # Test with nonexistent data directory
        test_args = [
            'pylithics',
            '--data_dir', '/nonexistent/directory',
            '--meta_file', '/nonexistent/metadata.csv'
        ]

        with patch('sys.argv', test_args):
            result = main()
            assert result == 1  # Should fail gracefully

    def test_cli_batch_processing_workflow(self):
        """Test complete CLI batch processing workflow."""
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

            test_args = [
                'pylithics',
                '--data_dir', data_dir,
                '--meta_file', metadata_path,
                '--log_level', 'INFO'
            ]

            with patch('sys.argv', test_args), \
                 patch.object(PyLithicsApplication, 'run_batch_analysis') as mock_run:

                mock_run.return_value = {
                    'success': True,
                    'processed_successfully': 3,
                    'total_images': 3,
                    'failed_images': []
                }

                result = main()

                assert result == 0


@pytest.mark.unit
class TestApplicationConfiguration:
    """Test application configuration handling."""

    def test_application_with_custom_config(self):
        """Test application initialization with custom configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "custom_config.yaml")
            with open(config_file, 'w') as f:
                f.write("""
arrow_detection:
  enabled: false
thresholding:
  method: simple
  threshold_value: 100
logging:
  level: WARNING
""")

            app = PyLithicsApplication(config_file=config_file)

            # Should initialize without errors
            assert app is not None
            assert hasattr(app, 'config_manager')

    def test_application_update_configuration(self):
        """Test runtime configuration updates."""
        app = PyLithicsApplication()

        # Test configuration update
        app.update_configuration(**{
            'logging.level': 'DEBUG',
            'arrow_detection.enabled': False
        })

        # Should not raise an exception
        assert app is not None

    def test_application_configuration_validation(self):
        """Test configuration validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create invalid config file
            config_file = os.path.join(temp_dir, "invalid_config.yaml")
            with open(config_file, 'w') as f:
                f.write("invalid: yaml: content: [")

            # Should handle invalid config gracefully
            app = PyLithicsApplication(config_file=config_file)
            assert app is not None


@pytest.mark.performance
class TestCLIPerformance:
    """Test CLI performance characteristics."""

    def test_cli_startup_performance(self):
        """Test CLI startup time."""
        import time

        test_args = [
            'pylithics',
            '--help'
        ]

        start_time = time.time()

        with patch('sys.argv', test_args):
            try:
                main()
            except SystemExit:
                pass

        end_time = time.time()
        startup_time = end_time - start_time

        # Should start up quickly
        assert startup_time < 5.0  # 5 seconds max

    def test_argument_parser_performance(self):
        """Test argument parser performance."""
        import time

        parser = create_argument_parser()

        # Test with complex arguments
        args = [
            '--data_dir', '/test/path',
            '--meta_file', '/test/meta.csv',
            '--threshold_method', 'adaptive',
            '--log_level', 'DEBUG',
            '--disable_arrow_detection',
            '--arrow_debug'
        ]

        start_time = time.time()

        parsed_args = parser.parse_args(args)

        end_time = time.time()
        parse_time = end_time - start_time

        # Should parse quickly
        assert parse_time < 1.0  # 1 second max
        assert parsed_args.data_dir == '/test/path'


@pytest.mark.integration
class TestCLIRealWorldScenarios:
    """Test CLI with realistic archaeological scenarios."""

    def test_cli_archaeological_workflow(self):
        """Test CLI with archaeological processing workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create archaeological data structure
            survey_dir = os.path.join(temp_dir, "survey_2024")
            images_dir = os.path.join(survey_dir, "images")
            os.makedirs(images_dir)

            # Create sample artifact images
            artifacts = ["blade_001.png", "core_002.png", "scraper_003.png"]
            for artifact in artifacts:
                image_path = os.path.join(images_dir, artifact)
                with open(image_path, 'w') as f:
                    f.write(f"archaeological artifact {artifact}")

            # Create metadata
            metadata_path = os.path.join(survey_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write("image_id,scale\n")
                for i, artifact in enumerate(artifacts):
                    f.write(f"{artifact},{15.0 + i * 2.5}\n")

            # Create archaeological config
            config_path = os.path.join(temp_dir, "archaeological_config.yaml")
            with open(config_path, 'w') as f:
                f.write("""
thresholding:
  method: otsu
arrow_detection:
  enabled: true
  reference_dpi: 300.0
logging:
  level: INFO
""")

            test_args = [
                'pylithics',
                '--data_dir', survey_dir,
                '--meta_file', metadata_path,
                '--config_file', config_path,
                '--log_level', 'INFO'
            ]

            with patch('sys.argv', test_args), \
                 patch.object(PyLithicsApplication, 'run_batch_analysis') as mock_run:

                mock_run.return_value = {
                    'success': True,
                    'processed_successfully': 3,
                    'total_images': 3,
                    'failed_images': []
                }

                result = main()

                assert result == 0