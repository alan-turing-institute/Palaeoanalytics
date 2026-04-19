"""Tests for the configuration management system."""

import os
import tempfile
from unittest.mock import patch

import pytest
import yaml

import pylithics.image_processing.config as config_module
from pylithics.image_processing.config import (
    ConfigurationManager,
    clear_config_cache,
    get_arrow_detection_config,
    get_arrow_integration_config,
    get_config_manager,
    get_contour_filtering_config,
    get_cortex_detection_config,
    get_scale_calibration_config,
    get_scar_complexity_config,
    get_surface_classification_config,
    load_preprocessing_config,
)


# ---------------------------------------------------------------------------
# ConfigurationManager core behavior
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConfigurationManager:
    """Core behaviors of ConfigurationManager."""

    def test_loads_from_valid_file(self, sample_config_file):
        manager = ConfigurationManager(sample_config_file)
        assert manager._config_file == sample_config_file
        assert manager.config["thresholding"]["method"] == "simple"
        assert "logging" in manager.config

    def test_falls_back_to_defaults_on_missing_file(self):
        with patch("pylithics.image_processing.config.logging") as mock_logging:
            manager = ConfigurationManager("/nonexistent/config.yaml")

        assert "thresholding" in manager.config
        assert manager.config["thresholding"]["method"] == "simple"
        mock_logging.error.assert_called()

    def test_falls_back_to_defaults_on_invalid_yaml(self, tmp_path):
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("invalid: yaml: content: [unclosed")

        with patch("pylithics.image_processing.config.logging") as mock_logging:
            manager = ConfigurationManager(str(bad_yaml))

        assert "thresholding" in manager.config
        mock_logging.error.assert_called()

    def test_missing_required_sections_get_filled_with_defaults(self, tmp_path):
        partial = tmp_path / "partial.yaml"
        partial.write_text(yaml.dump({"thresholding": {"method": "otsu"}}))

        with patch("pylithics.image_processing.config.logging") as mock_logging:
            manager = ConfigurationManager(str(partial))

        assert manager.get_value("thresholding", "method") == "otsu"
        # Required sections that were missing should be filled in
        for section in ("normalization", "logging", "contour_filtering"):
            assert section in manager.config
        mock_logging.warning.assert_called()

    def test_get_value_returns_default_when_missing(self, sample_config_file):
        manager = ConfigurationManager(sample_config_file)
        assert manager.get_value("thresholding", "missing_key", "fallback") == "fallback"
        assert manager.get_value("missing_section", "key", 42) == 42

    def test_update_value_persists_in_memory(self, sample_config_file):
        manager = ConfigurationManager(sample_config_file)
        manager.update_value("thresholding", "method", "otsu")
        manager.update_value("brand_new_section", "key", "value")

        assert manager.get_value("thresholding", "method") == "otsu"
        assert manager.get_value("brand_new_section", "key") == "value"


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGlobalConfigManager:
    """Behavior of the module-level ConfigurationManager singleton."""

    def test_singleton_is_cached(self, sample_config_file):
        clear_config_cache()
        first = get_config_manager(sample_config_file)
        second = get_config_manager()  # no path -> reuses cached instance
        assert first is second

    def test_clear_cache_resets_singleton(self, sample_config_file):
        clear_config_cache()
        first = get_config_manager(sample_config_file)

        clear_config_cache()
        second = get_config_manager(sample_config_file)
        assert first is not second

    def test_config_path_from_environment_variable(self, sample_config_file):
        with patch.dict(os.environ, {"PYLITHICS_CONFIG": sample_config_file}):
            manager = ConfigurationManager()
        assert "thresholding" in manager.config


# ---------------------------------------------------------------------------
# Section getter functions (parameterized across all eight getters)
# ---------------------------------------------------------------------------


# (getter_function, section_name, override_payload)
#
# The override_payload is what we write under the section when we want to prove
# that a user-supplied config is respected. Each getter must round-trip it.
GETTER_CASES = [
    (
        get_contour_filtering_config,
        "contour_filtering",
        {"min_area": 100.0, "exclude_border": False},
    ),
    (
        get_arrow_detection_config,
        "arrow_detection",
        {"enabled": False, "reference_dpi": 150.0},
    ),
    (
        get_surface_classification_config,
        "surface_classification",
        {"enabled": False, "tolerance": 0.5},
    ),
    (
        get_cortex_detection_config,
        "cortex_detection",
        {"enabled": False, "stippling_density_threshold": 0.9},
    ),
    (
        get_arrow_integration_config,
        "arrow_integration",
        {"min_candidate_area": 5.0, "min_solidity": 0.6},
    ),
    (
        get_scar_complexity_config,
        "scar_complexity",
        {"enabled": False, "distance_threshold": 20.0},
    ),
    (
        get_scale_calibration_config,
        "scale_calibration",
        {"enabled": False, "debug_output": True},
    ),
]


@pytest.mark.unit
class TestSectionGetters:
    """Each section getter must honor overrides and fall back to defaults."""

    @pytest.mark.parametrize("getter, section, override", GETTER_CASES)
    def test_returns_values_from_provided_config(self, getter, section, override):
        result = getter({section: override})
        for key, value in override.items():
            assert result[key] == value

    @pytest.mark.parametrize("getter, section, override", GETTER_CASES)
    def test_returns_defaults_when_section_missing(self, getter, section, override):
        # Config has every section EXCEPT the one under test
        result = getter({"unrelated_section": {"foo": "bar"}})
        assert isinstance(result, dict)
        # Defaults vary per getter; we assert the result is non-empty
        # and distinct from the override (i.e. really came from defaults).
        assert result
        assert result != override

    @pytest.mark.parametrize("getter", [g for g, _, _ in GETTER_CASES])
    def test_falls_back_to_global_manager_when_config_none(self, getter):
        clear_config_cache()
        config_module._config_manager = None
        result = getter(None)
        assert isinstance(result, dict)
        assert result  # default dicts are non-empty for every getter


# ---------------------------------------------------------------------------
# load_preprocessing_config thin wrapper
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoadPreprocessingConfig:

    def test_with_explicit_file(self, sample_config_file):
        config = load_preprocessing_config(sample_config_file)
        assert config["thresholding"]["method"] == "simple"
        assert "normalization" in config

    def test_without_file_uses_global_manager(self):
        clear_config_cache()
        config = load_preprocessing_config()
        assert "thresholding" in config


# ---------------------------------------------------------------------------
# Integration: file-on-disk round trip
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_full_workflow_from_disk(tmp_path):
    """Write -> load -> update -> re-read via ConfigurationManager."""
    initial = {
        "thresholding": {"method": "simple", "threshold_value": 100},
        "logging": {"level": "INFO"},
    }
    config_path = tmp_path / "workflow_config.yaml"
    config_path.write_text(yaml.dump(initial))

    manager = ConfigurationManager(str(config_path))
    assert manager.get_value("thresholding", "method") == "simple"

    manager.update_value("thresholding", "method", "otsu")
    assert manager.get_value("thresholding", "method") == "otsu"

    # Validation should have filled in required sections
    assert isinstance(manager.get_section("morphological_closing"), dict)
