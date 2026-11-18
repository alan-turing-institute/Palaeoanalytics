"""
PyLithics Cortex Detection Tests
================================

Tests for cortex detection functionality including texture analysis, 
stippling detection, and cortex/scar differentiation.
"""

import pytest
import numpy as np
from unittest.mock import patch

from pylithics.image_processing.modules.cortex_detection import (
    detect_cortex_in_child_contours,
    _detect_cortex_texture,
    calculate_total_cortex_metrics
)
from pylithics.image_processing.config import get_cortex_detection_config

CONFIG_PATH = 'pylithics.image_processing.modules.cortex_detection.get_cortex_detection_config'
SQUARE = [[10, 10], [20, 10], [20, 20], [10, 20]]


@pytest.mark.unit
class TestCortexDetection:
    """Test cortex detection functionality."""

    def test_cortex_detection_disabled(self):
        """Test that cortex detection is skipped when disabled in config."""
        # Mock configuration with cortex detection disabled
        with patch(CONFIG_PATH) as mock_config:
            mock_config.return_value = {'enabled': False}
            
            # Sample metrics
            metrics = [
                {'parent': 'parent 1', 'scar': 'parent 1', 'area': 1000},
                {'parent': 'parent 1', 'scar': 'scar 1', 'area': 100, 'contour': SQUARE}
            ]
            
            # Mock inverted image
            inverted_image = np.zeros((100, 100), dtype=np.uint8)
            
            # Run cortex detection
            result = detect_cortex_in_child_contours(metrics, inverted_image)
            
            # Should return original metrics unchanged
            assert result == metrics
            assert len(result) == 2

    def test_cortex_detection_enabled_no_cortex(self):
        """Test cortex detection when enabled but no cortex is found."""
        # Mock configuration with cortex detection enabled
        with patch(CONFIG_PATH) as mock_config:
            mock_config.return_value = {
                'enabled': True,
                'stippling_density_threshold': 0.2,
                'texture_variance_threshold': 100,
                'edge_density_threshold': 0.05
            }
            
            # Sample metrics with child contour - need surface type for parent
            metrics = [
                {'parent': 'parent 1', 'scar': 'parent 1', 'area': 1000, 'surface_type': 'Dorsal'},
                {'parent': 'parent 1', 'scar': 'scar 1', 'area': 100, 'contour': SQUARE}
            ]
            
            # Create simple inverted image (no complex texture)
            inverted_image = np.zeros((100, 100), dtype=np.uint8)
            inverted_image[10:21, 10:21] = 255  # Simple filled rectangle
            
            # Run cortex detection
            result = detect_cortex_in_child_contours(metrics, inverted_image)
            
            # Should have 2 items: 1 parent + 1 child labeled as scar
            assert len(result) == 2
            
            # Find the child contour
            child = next(r for r in result if r['parent'] != r['scar'])
            assert child['scar'] == 'scar 1'
            assert child.get('is_cortex', False) == False

    def test_cortex_texture_detection_simple(self):  
        """Test the texture analysis function with simple patterns."""
        # Create a simple contour
        contour = np.array([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=np.int32)
        
        # Create image with no texture (should not be cortex)
        inverted_image = np.zeros((100, 100), dtype=np.uint8)
        inverted_image[10:51, 10:51] = 255  # Solid filled area
        
        config = {
            'stippling_density_threshold': 0.2,
            'texture_variance_threshold': 100,
            'edge_density_threshold': 0.05
        }
        
        # Should not detect cortex in simple filled area
        result = _detect_cortex_texture(contour, inverted_image, config)
        assert result == False

    def test_one_piece_of_ink_in_a_small_scar_is_not_cortex(self):
        """An arrow is one small piece of ink; alone it used to pass the density test."""
        contour = np.array([[10, 10], [40, 10], [40, 40], [10, 40]], dtype=np.int32)
        image = np.zeros((60, 60), dtype=np.uint8)
        image[22:26, 18:28] = 255          # one mark of 40 px
        config = {'stippling_density_threshold': 0.1, 'texture_variance_threshold': 10,
                  'edge_density_threshold': 0.01}
        assert _detect_cortex_texture(contour, image, config) is False

    def test_two_pieces_are_not_cortex_and_three_are(self):
        contour = np.array([[10, 10], [40, 10], [40, 40], [10, 40]], dtype=np.int32)
        config = {'stippling_density_threshold': 0.1, 'texture_variance_threshold': 10,
                  'edge_density_threshold': 0.01}
        two = np.zeros((60, 60), dtype=np.uint8)
        two[15:18, 15:18] = 255
        two[30:33, 30:33] = 255
        three = two.copy()
        three[15:18, 30:33] = 255
        assert _detect_cortex_texture(contour, two, config) is False
        assert _detect_cortex_texture(contour, three, config) is True
        assert _detect_cortex_texture(contour, two, {**config, 'min_stipple_pieces': 1}) is True

    def test_cortex_texture_detection_complex(self):
        """Test texture analysis with complex stippled pattern."""
        # Create a contour
        contour = np.array([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=np.int32)
        
        # Create image with stippled texture pattern
        inverted_image = np.zeros((100, 100), dtype=np.uint8)
        
        # Add random stippling pattern
        np.random.seed(42)  # For reproducible results
        for i in range(10, 50, 3):
            for j in range(10, 50, 3):
                if np.random.random() > 0.5:
                    # Add small stipples
                    inverted_image[i:i+2, j:j+2] = 255
        
        config = {
            'stippling_density_threshold': 0.1,  # Lower threshold for testing
            'texture_variance_threshold': 50,    # Lower threshold for testing  
            'edge_density_threshold': 0.02       # Lower threshold for testing
        }
        
        # Should detect cortex in stippled pattern
        result = _detect_cortex_texture(contour, inverted_image, config)
        assert result == True

    def test_calculate_total_cortex_metrics(self):
        """Test aggregate cortex metrics calculation."""
        metrics = [
            {'parent': 'parent 1', 'scar': 'parent 1', 'area': 1000},
            {'is_cortex': True, 'cortex_area': 150, 'scar': 'cortex 1'},
            {'is_cortex': True, 'cortex_area': 250, 'scar': 'cortex 2'},
            {'is_cortex': False, 'scar': 'scar 1'}
        ]
        
        result = calculate_total_cortex_metrics(metrics)
        
        assert result['total_cortex_area'] == 400
        assert result['cortex_count'] == 2
        assert result['average_cortex_size'] == 200.0

    def test_cortex_config_defaults(self):
        """Test that cortex detection config returns proper defaults."""
        config = get_cortex_detection_config()
        
        assert 'enabled' in config
        assert 'stippling_density_threshold' in config
        assert 'texture_variance_threshold' in config
        assert 'edge_density_threshold' in config
        
        # Check default values
        assert config['enabled'] == True
        assert config['stippling_density_threshold'] == 0.2
        assert config['texture_variance_threshold'] == 100
        assert config['edge_density_threshold'] == 0.05


@pytest.mark.integration
class TestCortexDetectionIntegration:
    """Integration tests for cortex detection with real image processing."""

    def test_cortex_detection_pipeline_integration(self):
        """Test cortex detection within the full analysis pipeline."""
        # This would be an integration test with actual image processing
        # For now, we'll test the basic pipeline integration
        
        with patch(CONFIG_PATH) as mock_config:
            mock_config.return_value = {
                'enabled': True, 'stippling_density_threshold': 0.2,
                'texture_variance_threshold': 100, 'edge_density_threshold': 0.05,
            }
            
            # Create sample data that simulates the pipeline
            metrics = [
                {
                    'parent': 'parent 1', 
                    'scar': 'parent 1', 
                    'area': 10000,
                    'surface_type': 'Dorsal'
                },
                {
                    'parent': 'parent 1', 
                    'scar': 'scar 1', 
                    'area': 500,
                    'contour': [[20, 20], [80, 20], [80, 80], [20, 80]]
                }
            ]
            
            # Simple test image
            inverted_image = np.zeros((100, 100), dtype=np.uint8)
            inverted_image[20:81, 20:81] = 255
            
            result = detect_cortex_in_child_contours(metrics, inverted_image)
            
            # Should process without errors
            assert len(result) >= 2
            # Parent should remain unchanged
            parent = next(r for r in result if r['parent'] == r['scar'])
            assert parent['surface_type'] == 'Dorsal'

# ---------------------------------------------------------------------------
# The letter C
# ---------------------------------------------------------------------------

import cv2  # noqa: E402

from pylithics.image_processing.modules import cortex_detection as cortex_module  # noqa: E402

READER_PATH = 'pylithics.image_processing.modules.cortex_detection._load_reader'


def _scar_with_mark():
    """A 100 px square scar outline on a binary image, with one small mark inside it."""
    ink = np.zeros((140, 140), np.uint8)
    cv2.rectangle(ink, (20, 20), (120, 120), 255, 2)
    cv2.ellipse(ink, (70, 70), (10, 14), 0, 40, 320, 255, 3)     # a drawn C
    contour = np.array([[20, 20], [120, 20], [120, 120], [20, 120]], np.int32)
    return contour, ink


def _reader(text, score):
    """A stand-in for RapidOCR's recognizer: it always reads ``text`` at ``score``."""
    def read(image, use_det=True, use_cls=True, use_rec=True):
        assert use_det is False                         # a lone glyph: recognizer only
        return [[text, score]], None
    return read


def _config(**letter):
    return {'letter': {**cortex_module.DEFAULT_LETTER, **letter}}


@pytest.mark.unit
class TestCortexLetter:
    """A scar that holds the letter C is cortex: some illustrators mark cortex so."""

    def test_a_scar_holding_a_c_is_cortex(self):
        contour, ink = _scar_with_mark()
        with patch(READER_PATH, return_value=_reader('C', 0.93)):
            assert cortex_module._letter_in_scar(contour, ink, 1, _config())

    def test_a_lower_case_c_counts_too(self):
        contour, ink = _scar_with_mark()
        with patch(READER_PATH, return_value=_reader('c', 0.9)):
            assert cortex_module._letter_in_scar(contour, ink, 1, _config())

    def test_other_letters_count_only_when_listed(self):
        contour, ink = _scar_with_mark()
        with patch(READER_PATH, return_value=_reader('B', 0.95)):
            assert not cortex_module._letter_in_scar(contour, ink, 1, _config())
            assert cortex_module._letter_in_scar(contour, ink, 1, _config(letters=['C', 'B']))

    def test_an_arrow_read_as_a_numeral_is_not_cortex(self):
        contour, ink = _scar_with_mark()
        with patch(READER_PATH, return_value=_reader('1', 0.95)):
            assert not cortex_module._letter_in_scar(contour, ink, 1, _config())

    def test_a_weak_read_does_not_count(self):
        contour, ink = _scar_with_mark()
        with patch(READER_PATH, return_value=_reader('C', 0.7)):
            assert not cortex_module._letter_in_scar(contour, ink, 1, _config())

    def test_without_the_reader_the_test_is_skipped(self):
        contour, ink = _scar_with_mark()
        with patch(READER_PATH, return_value=None):
            assert not cortex_module._letter_in_scar(contour, ink, 1, _config())

    def test_switched_off_the_reader_is_not_asked(self):
        contour, ink = _scar_with_mark()
        with patch(READER_PATH) as load:
            assert not cortex_module._letter_in_scar(contour, ink, 1, _config(enabled=False))
            load.assert_not_called()

    def test_the_lettered_scar_is_labelled_cortex_by_letter(self):
        contour, ink = _scar_with_mark()
        child = {'parent': 'p', 'scar': 's', 'contour': contour.tolist(), 'area': 10000}
        config = {'min_stipple_pieces': 3, **_config()}
        with patch(READER_PATH, return_value=_reader('C', 0.93)):
            cortex, rest = cortex_module._classify_cortex_children([child], ink, config)
        assert cortex == [child] and rest == []
        assert child['cortex_source'] == 'letter'


@pytest.mark.unit
def test_a_dotted_override_reaches_a_nested_setting():
    from pylithics.image_processing.config import ConfigurationManager
    manager = ConfigurationManager()
    manager.update_value('cortex_detection', 'letter.enabled', False)
    assert manager.config['cortex_detection']['letter']['enabled'] is False
    assert 'letter.enabled' not in manager.config['cortex_detection']
