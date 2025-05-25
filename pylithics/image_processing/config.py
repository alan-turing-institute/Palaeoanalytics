"""
Configuration management for PyLithics image processing.

This module handles loading and parsing of configuration files,
providing a centralized interface for all configuration needs.
"""

import os
import logging
import yaml
from pkg_resources import resource_filename


def load_preprocessing_config(config_file=None):
    """
    Load configuration settings from a YAML file.

    Parameters
    ----------
    config_file : str, optional
        Path to configuration file. If None, uses default locations in order:
        1. PYLITHICS_CONFIG environment variable
        2. pylithics/config/config.yaml (default)

    Returns
    -------
    dict or None
        Configuration dictionary, or None if loading fails

    Raises
    ------
    FileNotFoundError
        If configuration file cannot be found
    yaml.YAMLError
        If YAML parsing fails
    """
    # Determine config file path
    if config_file and os.path.isabs(config_file):
        config_path = config_file
    elif os.getenv('PYLITHICS_CONFIG'):
        config_path = os.getenv('PYLITHICS_CONFIG')
    else:
        # Default: use the config.yaml in the project root
        config_path = resource_filename(__name__, '../config/config.yaml')

    logging.info("Loading configuration from: %s", config_path)

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logging.info("Successfully loaded configuration from %s", config_path)
        return config

    except FileNotFoundError:
        logging.error("Configuration file %s not found", config_path)
        return None

    except yaml.YAMLError as yaml_error:
        logging.error("Failed to parse YAML file %s: %s", config_path, yaml_error)
        return None

    except OSError as os_error:
        logging.error("Failed to load config file %s due to OS error: %s", config_path, os_error)
        return None


def get_contour_filtering_config(config=None):
    """
    Get contour filtering configuration with defaults.

    Parameters
    ----------
    config : dict, optional
        Full configuration dictionary. If None, loads default config.

    Returns
    -------
    dict
        Contour filtering configuration with keys:
        - min_area: minimum contour area in pixels (default: 50.0)
        - exclude_border: whether to exclude border-touching contours (default: True)
    """
    if config is None:
        config = load_preprocessing_config()

    if config is None:
        logging.warning("Could not load configuration, using defaults for contour filtering")
        return {
            'min_area': 50.0,
            'exclude_border': True
        }

    contour_config = config.get('contour_filtering', {})
    return {
        'min_area': contour_config.get('min_area', 50.0),
        'exclude_border': contour_config.get('exclude_border', True)
    }


def get_thresholding_config(config=None):
    """
    Get thresholding configuration with defaults.

    Parameters
    ----------
    config : dict, optional
        Full configuration dictionary. If None, loads default config.

    Returns
    -------
    dict
        Thresholding configuration
    """
    if config is None:
        config = load_preprocessing_config()

    if config is None:
        return {
            'method': 'simple',
            'threshold_value': 127,
            'max_value': 255
        }

    return config.get('thresholding', {
        'method': 'simple',
        'threshold_value': 127,
        'max_value': 255
    })


def get_morphological_config(config=None):
    """
    Get morphological processing configuration.

    Parameters
    ----------
    config : dict, optional
        Full configuration dictionary. If None, loads default config.

    Returns
    -------
    dict
        Morphological processing configuration
    """
    if config is None:
        config = load_preprocessing_config()

    if config is None:
        return {
            'enabled': True,
            'kernel_size': 3
        }

    return config.get('morphological_closing', {
        'enabled': True,
        'kernel_size': 3
    })


def get_logging_config(config=None):
    """
    Get logging configuration.

    Parameters
    ----------
    config : dict, optional
        Full configuration dictionary. If None, loads default config.

    Returns
    -------
    dict
        Logging configuration
    """
    if config is None:
        config = load_preprocessing_config()

    if config is None:
        return {
            'level': 'INFO',
            'log_to_file': True,
            'log_file': 'logs/pylithics.log'
        }

    return config.get('logging', {
        'level': 'INFO',
        'log_to_file': True,
        'log_file': 'logs/pylithics.log'
    })


def get_normalization_config(config=None):
    """
    Get normalization configuration.

    Parameters
    ----------
    config : dict, optional
        Full configuration dictionary. If None, loads default config.

    Returns
    -------
    dict
        Normalization configuration
    """
    if config is None:
        config = load_preprocessing_config()

    if config is None:
        return {
            'enabled': True,
            'method': 'minmax',
            'clip_values': [0, 255]
        }

    return config.get('normalization', {
        'enabled': True,
        'method': 'minmax',
        'clip_values': [0, 255]
    })


def get_grayscale_config(config=None):
    """
    Get grayscale conversion configuration.

    Parameters
    ----------
    config : dict, optional
        Full configuration dictionary. If None, loads default config.

    Returns
    -------
    dict
        Grayscale conversion configuration
    """
    if config is None:
        config = load_preprocessing_config()

    if config is None:
        return {
            'enabled': True,
            'method': 'standard'
        }

    return config.get('grayscale_conversion', {
        'enabled': True,
        'method': 'standard'
    })
