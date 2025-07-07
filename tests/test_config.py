"""
PyLithics Configuration Management Tests
=======================================

Tests for the configuration loading, validation, and management system.
Covers config file loading, default values, validation, and runtime updates.
"""

import pytest
import os
import tempfile
import yaml
from unittest.mock import patch, mock_open

from pylithics.image_processing.config import (
    ConfigurationManager,
    get_config_manager,
    clear_config_cache,
    load_preprocessing_config,
    get_contour_filtering_config,
    get_thresholding_config,
    get_morphological_config,
    get_logging_config,
    get_normalization_config,
    get_grayscale_config,
    get_arrow_detection_config
)


@pytest.mark.unit
class TestConfigurationManager:
    """Test the ConfigurationManager class."""

    def _setup_test_config(self, test_config_dict):
        """Helper method to set up a temporary config file and initialize config manager."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config_dict, f)
            temp_config_file = f.name

        clear_config_cache()
        from pylithics.image_processing.config import _config_manager
        import pylithics.image_processing.config as config_module
        config_module._config_manager = None

        get_config_manager(temp_config_file)
        return temp_config_file

    def _reset_config_manager(self):
        """Helper method to reset config manager to defaults."""
        clear_config_cache()
        from pylithics.image_processing.config import _config_manager
        import pylithics.image_processing.config as config_module
        config_module._config_manager = None

    def test_init_with_valid_config_file(self, sample_config_file):
        """Test initialization with a valid config file."""
        config_manager = ConfigurationManager(sample_config_file)

        assert config_manager._config is not None
        assert config_manager._config_file == sample_config_file
        assert 'thresholding' in config_manager.config
        assert 'logging' in config_manager.config

    def test_init_with_nonexistent_config_file(self):
        """Test initialization with a non-existent config file falls back to defaults."""
        with patch('pylithics.image_processing.config.logging') as mock_logging:
            config_manager = ConfigurationManager('/nonexistent/config.yaml')

            # Should fall back to default config
            assert config_manager._config is not None
            assert 'thresholding' in config_manager.config
            mock_logging.error.assert_called()

    def test_init_with_invalid_yaml(self, test_data_dir):
        """Test initialization with invalid YAML content."""
        invalid_config_path = os.path.join(test_data_dir, "invalid_config.yaml")
        with open(invalid_config_path, 'w') as f:
            f.write("invalid: yaml: content: [unclosed")

        with patch('pylithics.image_processing.config.logging') as mock_logging:
            config_manager = ConfigurationManager(invalid_config_path)

            # Should fall back to default config
            assert config_manager._config is not None
            mock_logging.error.assert_called()

    def test_config_property(self, sample_config_file, sample_config):
        """Test the config property returns the loaded configuration."""
        config_manager = ConfigurationManager(sample_config_file)

        config = config_manager.config
        assert isinstance(config, dict)
        assert config['thresholding']['method'] == sample_config['thresholding']['method']

    def test_get_section_existing(self, sample_config_file):
        """Test getting an existing configuration section."""
        config_manager = ConfigurationManager(sample_config_file)

        thresholding_config = config_manager.get_section('thresholding')
        assert isinstance(thresholding_config, dict)
        assert 'method' in thresholding_config
        assert 'threshold_value' in thresholding_config

    def test_get_section_nonexistent(self, sample_config_file):
        """Test getting a non-existent configuration section returns defaults."""
        config_manager = ConfigurationManager(sample_config_file)

        nonexistent_config = config_manager.get_section('nonexistent_section')
        assert isinstance(nonexistent_config, dict)
        # Should return empty dict or defaults

    def test_get_value_existing(self, sample_config_file):
        """Test getting an existing configuration value."""
        config_manager = ConfigurationManager(sample_config_file)

        method = config_manager.get_value('thresholding', 'method')
        assert method == 'simple'

        threshold_value = config_manager.get_value('thresholding', 'threshold_value')
        assert threshold_value == 127

    def test_get_value_nonexistent_with_default(self, sample_config_file):
        """Test getting a non-existent value returns the default."""
        config_manager = ConfigurationManager(sample_config_file)

        value = config_manager.get_value('thresholding', 'nonexistent_key', 'default_value')
        assert value == 'default_value'

    def test_get_value_nonexistent_section(self, sample_config_file):
        """Test getting a value from a non-existent section."""
        config_manager = ConfigurationManager(sample_config_file)

        value = config_manager.get_value('nonexistent_section', 'some_key', 'default')
        assert value == 'default'

    def test_update_value_existing_section(self, sample_config_file):
        """Test updating a value in an existing section."""
        config_manager = ConfigurationManager(sample_config_file)

        original_value = config_manager.get_value('thresholding', 'method')
        assert original_value == 'simple'

        config_manager.update_value('thresholding', 'method', 'otsu')
        updated_value = config_manager.get_value('thresholding', 'method')
        assert updated_value == 'otsu'

    def test_update_value_new_section(self, sample_config_file):
        """Test updating a value creates a new section if needed."""
        config_manager = ConfigurationManager(sample_config_file)

        config_manager.update_value('new_section', 'new_key', 'new_value')
        value = config_manager.get_value('new_section', 'new_key')
        assert value == 'new_value'

    def test_validate_config_missing_sections(self, test_data_dir):
        """Test configuration validation with missing sections."""
        incomplete_config = {
            'thresholding': {'method': 'simple'}
            # Missing other required sections
        }

        config_path = os.path.join(test_data_dir, "incomplete_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(incomplete_config, f)

        with patch('pylithics.image_processing.config.logging') as mock_logging:
            config_manager = ConfigurationManager(config_path)

            # Should add default sections for missing ones
            assert 'normalization' in config_manager.config
            assert 'logging' in config_manager.config
            mock_logging.warning.assert_called()


@pytest.mark.unit
class TestGlobalConfigManager:
    """Test the global configuration manager functions."""

    def test_get_config_manager_singleton(self, sample_config_file):
        """Test that get_config_manager returns the same instance."""
        # Clear any existing instance
        clear_config_cache()

        manager1 = get_config_manager(sample_config_file)
        manager2 = get_config_manager()  # Should return same instance

        assert manager1 is manager2

    def test_get_config_manager_new_file(self, sample_config_file, test_data_dir):
        """Test that get_config_manager creates new instance with different file."""
        # Clear any existing instance
        clear_config_cache()

        manager1 = get_config_manager(sample_config_file)

        # Create a different config file
        different_config = {'thresholding': {'method': 'otsu'}}
        different_config_path = os.path.join(test_data_dir, "different_config.yaml")
        with open(different_config_path, 'w') as f:
            yaml.dump(different_config, f)

        # This should still return the same instance (singleton behavior)
        manager2 = get_config_manager(different_config_path)
        assert manager1 is manager2


@pytest.mark.unit
class TestConfigFunctions:
    """Test the individual configuration getter functions."""

    def test_load_preprocessing_config_with_file(self, sample_config_file):
        """Test loading preprocessing config with a file."""
        config = load_preprocessing_config(sample_config_file)

        assert isinstance(config, dict)
        assert 'thresholding' in config
        assert 'normalization' in config

    def test_load_preprocessing_config_without_file(self):
        """Test loading preprocessing config without a file uses global manager."""
        clear_config_cache()
        config = load_preprocessing_config()

        assert isinstance(config, dict)
        assert 'thresholding' in config

    def test_get_contour_filtering_config_with_config(self):
        """Test getting contour filtering config with provided config dict."""
        test_config = {
            'contour_filtering': {
                'min_area': 100.0,
                'exclude_border': False
            }
        }

        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            temp_config_file = f.name

        try:
            # Clear cache and set up config manager with test config
            clear_config_cache()
            # Reset the global config manager
            from pylithics.image_processing.config import _config_manager
            import pylithics.image_processing.config as config_module
            config_module._config_manager = None

            # Initialize with test config
            get_config_manager(temp_config_file)

            # Now call without arguments
            config = get_contour_filtering_config()

            assert config['min_area'] == 100.0
            assert config['exclude_border'] is False
        finally:
            os.unlink(temp_config_file)

    def test_get_contour_filtering_config_without_config(self):
        """Test getting contour filtering config without provided config."""
        clear_config_cache()
        config = get_contour_filtering_config()

        assert isinstance(config, dict)
        assert 'min_area' in config
        assert 'exclude_border' in config

    def test_get_contour_filtering_config_defaults(self):
        """Test contour filtering config falls back to defaults."""
        # Clear cache and reset config manager to use defaults
        clear_config_cache()
        from pylithics.image_processing.config import _config_manager
        import pylithics.image_processing.config as config_module
        config_module._config_manager = None

        # Call without any config file (should use defaults)
        config = get_contour_filtering_config()

        assert config['min_area'] == 50.0
        assert config['exclude_border'] is True

    def test_get_thresholding_config_with_config(self):
        """Test getting thresholding config with provided config dict."""
        test_config = {
            'thresholding': {
                'method': 'adaptive',
                'threshold_value': 100,
                'max_value': 200
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            temp_config_file = f.name

        try:
            clear_config_cache()
            from pylithics.image_processing.config import _config_manager
            import pylithics.image_processing.config as config_module
            config_module._config_manager = None

            get_config_manager(temp_config_file)
            config = get_thresholding_config()

            assert config['method'] == 'adaptive'
            assert config['threshold_value'] == 100
            assert config['max_value'] == 200
        finally:
            os.unlink(temp_config_file)

    def test_get_thresholding_config_defaults(self):
        """Test thresholding config falls back to defaults."""
        clear_config_cache()
        from pylithics.image_processing.config import _config_manager
        import pylithics.image_processing.config as config_module
        config_module._config_manager = None

        config = get_thresholding_config()

        assert config['method'] == 'simple'
        assert config['threshold_value'] == 127
        assert config['max_value'] == 255

    def test_get_morphological_config(self):
        """Test getting morphological processing config."""
        test_config = {
            'morphological_closing': {
                'enabled': False,
                'kernel_size': 5
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            temp_config_file = f.name

        try:
            clear_config_cache()
            from pylithics.image_processing.config import _config_manager
            import pylithics.image_processing.config as config_module
            config_module._config_manager = None

            get_config_manager(temp_config_file)
            config = get_morphological_config()

            assert config['enabled'] is False
            assert config['kernel_size'] == 5
        finally:
            os.unlink(temp_config_file)

        assert config['enabled'] is False
        assert config['kernel_size'] == 5

    def test_get_logging_config(self):
        """Test getting logging config."""
        test_config = {
            'logging': {
                'level': 'DEBUG',
                'log_to_file': False,
                'log_file': 'custom.log'
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            temp_config_file = f.name

        try:
            clear_config_cache()
            from pylithics.image_processing.config import _config_manager
            import pylithics.image_processing.config as config_module
            config_module._config_manager = None

            get_config_manager(temp_config_file)
            config = get_logging_config()

            assert config['level'] == 'DEBUG'
            assert config['log_to_file'] is False
            assert config['log_file'] == 'custom.log'
        finally:
            os.unlink(temp_config_file)

    def test_get_normalization_config(self):
        """Test getting normalization config."""
        test_config = {
            'normalization': {
                'enabled': False,
                'method': 'zscore',
                'clip_values': [10, 200]
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            temp_config_file = f.name

        try:
            clear_config_cache()
            from pylithics.image_processing.config import _config_manager
            import pylithics.image_processing.config as config_module
            config_module._config_manager = None

            get_config_manager(temp_config_file)
            config = get_normalization_config()

            assert config['enabled'] is False
            assert config['method'] == 'zscore'
            assert config['clip_values'] == [10, 200]
        finally:
            os.unlink(temp_config_file)

    def test_get_grayscale_config(self):
        """Test getting grayscale conversion config."""
        test_config = {
            'grayscale_conversion': {
                'enabled': False,
                'method': 'clahe'
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            temp_config_file = f.name

        try:
            clear_config_cache()
            from pylithics.image_processing.config import _config_manager
            import pylithics.image_processing.config as config_module
            config_module._config_manager = None

            get_config_manager(temp_config_file)
            config = get_grayscale_config()

            assert config['enabled'] is False
            assert config['method'] == 'clahe'
        finally:
            os.unlink(temp_config_file)

    def test_get_arrow_detection_config_with_config(self):
        """Test getting arrow detection config with provided config dict."""
        test_config = {
            'arrow_detection': {
                'enabled': False,
                'reference_dpi': 150.0,
                'min_area_scale_factor': 0.5,
                'debug_enabled': True
            }
        }

        config = get_arrow_detection_config(test_config)

        assert config['enabled'] is False
        assert config['reference_dpi'] == 150.0
        assert config['min_area_scale_factor'] == 0.5
        assert config['debug_enabled'] is True

    def test_get_arrow_detection_config_defaults(self):
        """Test arrow detection config falls back to defaults."""
        config = get_arrow_detection_config()

        assert config['enabled'] is True
        assert config['reference_dpi'] == 300.0
        assert config['min_area_scale_factor'] == 0.7
        assert config['min_defect_depth_scale_factor'] == 0.8
        assert config['min_triangle_height_scale_factor'] == 0.8
        assert config['debug_enabled'] is False

    def test_get_arrow_detection_config_none_config(self):
        """Test arrow detection config with None config."""
        config = get_arrow_detection_config(None)

        # Should load from global config manager
        assert isinstance(config, dict)
        assert 'enabled' in config
        assert 'reference_dpi' in config


@pytest.mark.unit
class TestConfigValidation:
    """Test configuration validation and error handling."""

    def test_determine_config_path_absolute(self):
        """Test determining config path with absolute path."""
        config_manager = ConfigurationManager()
        absolute_path = "/absolute/path/to/config.yaml"
        config_manager._config_file = absolute_path

        determined_path = config_manager._determine_config_path()
        assert determined_path == absolute_path

    def test_determine_config_path_environment_variable(self):
        """Test determining config path from environment variable."""
        env_path = "/env/path/to/config.yaml"

        with patch.dict(os.environ, {'PYLITHICS_CONFIG': env_path}):
            config_manager = ConfigurationManager()
            config_manager._config_file = None

            determined_path = config_manager._determine_config_path()
            assert determined_path == env_path

    def test_get_default_section_unknown_section(self):
        """Test getting default config for unknown section."""
        config_manager = ConfigurationManager()

        default_section = config_manager._get_default_section('unknown_section')
        assert isinstance(default_section, dict)
        assert len(default_section) == 0  # Should return empty dict

    def test_clear_config_cache(self):
        """Test clearing the configuration cache."""
        # This mainly tests that the function runs without error
        clear_config_cache()

        # After clearing cache, getting configs should work
        config = get_contour_filtering_config()
        assert isinstance(config, dict)


@pytest.mark.unit
class TestConfigEnvironmentVariables:
    """Test configuration behavior with environment variables."""

    def test_config_path_from_environment(self, sample_config_file):
        """Test loading config path from PYLITHICS_CONFIG environment variable."""
        with patch.dict(os.environ, {'PYLITHICS_CONFIG': sample_config_file}):
            config_manager = ConfigurationManager()

            # Should load the config from environment variable
            assert config_manager._config is not None
            assert 'thresholding' in config_manager.config


@pytest.mark.integration
class TestConfigIntegration:
    """Integration tests for configuration management."""

    def test_config_file_integration_workflow(self, test_data_dir):
        """Test complete workflow of creating, loading, and updating config."""
        # Create initial config
        initial_config = {
            'thresholding': {'method': 'simple', 'threshold_value': 100},
            'logging': {'level': 'INFO'}
        }

        config_path = os.path.join(test_data_dir, "workflow_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(initial_config, f)

        # Load config
        config_manager = ConfigurationManager(config_path)
        assert config_manager.get_value('thresholding', 'method') == 'simple'

        # Update config
        config_manager.update_value('thresholding', 'method', 'otsu')
        assert config_manager.get_value('thresholding', 'method') == 'otsu'

        # Verify section handling
        morphological_config = config_manager.get_section('morphological_closing')
        assert isinstance(morphological_config, dict)

    def test_config_manager_with_partial_config(self, test_data_dir):
        """Test config manager behavior with partially defined config."""
        partial_config = {
            'thresholding': {'method': 'otsu'},
            # Missing other sections
        }

        config_path = os.path.join(test_data_dir, "partial_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(partial_config, f)

        with patch('pylithics.image_processing.config.logging'):
            config_manager = ConfigurationManager(config_path)

            # Should have the specified value
            assert config_manager.get_value('thresholding', 'method') == 'otsu'

            # Should have defaults for missing sections
            normalization_config = config_manager.get_section('normalization')
            assert isinstance(normalization_config, dict)