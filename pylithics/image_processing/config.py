"""
Configuration Management for PyLithics
======================================

Validation, caching, and error handling for the configuration management system.
"""

import os
import logging
import yaml
from typing import Dict, Any, Optional
try:
    from importlib import resources
except ImportError:
    import importlib_resources as resources


class ConfigurationManager:
    """
    Centralized configuration manager with validation and caching.
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the configuration manager.

        Parameters
        ----------
        config_file : str, optional
            Path to configuration file. If None, uses default locations.
        """
        self._config = None
        self._config_file = config_file
        self.load_config()

    def load_config(self) -> None:
        """Load configuration from file with validation."""
        config_path = self._determine_config_path()

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)

            self._validate_config()
            logging.info(f"Successfully loaded configuration from {config_path}")

        except FileNotFoundError:
            logging.error(f"Configuration file {config_path} not found")
            self._config = self._get_default_config()
        except yaml.YAMLError as e:
            logging.error(f"Failed to parse YAML file {config_path}: {e}")
            self._config = self._get_default_config()
        except Exception as e:
            logging.error(f"Unexpected error loading config: {e}")
            self._config = self._get_default_config()

    def _determine_config_path(self) -> str:
        """Determine the configuration file path."""
        if self._config_file and os.path.isabs(self._config_file):
            return self._config_file
        elif os.getenv('PYLITHICS_CONFIG'):
            return os.getenv('PYLITHICS_CONFIG')
        else:
            # Use absolute path to avoid relative path warnings
            config_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            return os.path.join(config_dir, 'config', 'config.yaml')

    def _validate_config(self) -> None:
        """Validate the loaded configuration."""
        required_sections = [
            'thresholding', 'normalization', 'grayscale_conversion',
            'morphological_closing', 'logging', 'contour_filtering'
        ]

        for section in required_sections:
            if section not in self._config:
                logging.warning(f"Missing configuration section: {section}")
                self._config[section] = self._get_default_section(section)

    def _get_default_section(self, section: str) -> Dict[str, Any]:
        """Get default configuration for a section."""
        defaults = {
            'thresholding': {
                'method': 'simple',
                'threshold_value': 127,
                'max_value': 255
            },
            'normalization': {
                'enabled': True,
                'method': 'minmax',
                'clip_values': [0, 255]
            },
            'grayscale_conversion': {
                'enabled': True,
                'method': 'standard'
            },
            'morphological_closing': {
                'enabled': True,
                'kernel_size': 3
            },
            'logging': {
                'level': 'INFO',
                'log_to_file': True,
                'log_file': 'pylithics/data/processed/pylithics.log'
            },
            'contour_filtering': {
                'min_area': 50.0,
                'exclude_border': True
            }
        }
        return defaults.get(section, {})

    def _get_default_config(self) -> Dict[str, Any]:
        """Get complete default configuration."""
        return {
            'thresholding': self._get_default_section('thresholding'),
            'normalization': self._get_default_section('normalization'),
            'grayscale_conversion': self._get_default_section('grayscale_conversion'),
            'morphological_closing': self._get_default_section('morphological_closing'),
            'logging': self._get_default_section('logging'),
            'contour_filtering': self._get_default_section('contour_filtering')
        }

    @property
    def config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return self._config

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get a specific configuration section."""
        return self._config.get(section, self._get_default_section(section))

    def get_value(self, section: str, key: str, default: Any = None) -> Any:
        """Get a specific configuration value."""
        section_config = self.get_section(section)
        return section_config.get(key, default)

    def update_value(self, section: str, key: str, value: Any) -> None:
        """Update a configuration value at runtime."""
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value
        logging.info(f"Updated config: {section}.{key} = {value}")


# Global configuration manager instance
_config_manager = None

def get_config_manager(config_file: Optional[str] = None) -> ConfigurationManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager(config_file)
    return _config_manager


# Backward compatibility functions
def load_preprocessing_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration settings from a YAML file (backward compatibility)."""
    return get_config_manager(config_file).config


def get_contour_filtering_config(
    config: Optional[Dict] = None
) -> Dict[str, Any]:
    """Get contour filtering configuration with defaults."""
    if config is None:
        config = get_config_manager().config
    return config.get('contour_filtering', {
        'min_area': 50.0,
        'exclude_border': True,
    })


def get_arrow_detection_config(config: Optional[Dict] = None) -> Dict[str, Any]:
    """Get arrow detection configuration with defaults."""
    if config is None:
        config = get_config_manager().config

    return config.get('arrow_detection', {
        'enabled': True,
        'reference_dpi': 300.0,
        'min_area_scale_factor': 0.7,
        'min_defect_depth_scale_factor': 0.8,
        'min_triangle_height_scale_factor': 0.8,
        'debug_enabled': False,
        'show_arrow_lines': False
    })


def get_surface_classification_config(
    config: Optional[Dict] = None
) -> Dict[str, Any]:
    """Get surface classification configuration with defaults."""
    if config is None:
        config = get_config_manager().config
    return config.get('surface_classification', {
        'enabled': True,
        'tolerance': 0.1,
        'classification_rules': {
            'dorsal_area_threshold': 0.6,
            'platform_aspect_ratio_max': 0.3,
            'lateral_area_threshold': 0.1,
        }
    })


def get_cortex_detection_config(config: Optional[Dict] = None) -> Dict[str, Any]:
    """Get cortex detection configuration with defaults."""
    if config is None:
        config = get_config_manager().config
    return config.get('cortex_detection', {
        'enabled': True,
        'stippling_density_threshold': 0.2, # Minimum density of small dots/stipples per 1000 pixels
                                            # Higher values = more restrictive cortex detection
                                            # Lower values = more sensitive cortex detection
                                            # Range: 0.1-1.0, typical: 0.2
        'texture_variance_threshold': 100,  # Minimum texture roughness/irregularity for cortex
                                            # Higher values = requires more textured surfaces
                                            # Lower values = detects smoother surfaces as cortex
                                            # Range: 50-500, typical: 100
        'edge_density_threshold': 0.05      # Minimum proportion of edge pixels from stippling
                                            # Higher values = requires more detailed edge patterns
                                            # Lower values = accepts simpler edge patterns
                                            # Range: 0.01-0.2, typical: 0.05
    })


def get_arrow_integration_config(config: Optional[Dict] = None) -> Dict[str, Any]:
    """Get arrow integration configuration with defaults."""
    if config is None:
        config = get_config_manager().config
    return config.get('arrow_integration', {
        'min_candidate_area': 1.0,
        'min_solidity': 0.4,
        'max_solidity': 0.9,
        'area_match_tolerance': 1.0
    })


def get_scar_complexity_config(config: Optional[Dict] = None) -> Dict[str, Any]:
    """Get scar complexity configuration with defaults."""
    if config is None:
        config = get_config_manager().config
    return config.get('scar_complexity', {
        'enabled': True,
        'distance_threshold': 5.0
    })


def get_scale_calibration_config(
    config: Optional[Dict] = None
) -> Dict[str, Any]:
    """Get scale calibration configuration with defaults."""
    if config is None:
        config = get_config_manager().config
    return config.get('scale_calibration', {
        'enabled': True,
        'debug_output': False,
    })


def clear_config_cache() -> None:
    """Reset the global config manager (useful for testing)."""
    global _config_manager
    _config_manager = None
