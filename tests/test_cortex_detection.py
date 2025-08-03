"""
PyLithics Cortex Detection Tests
================================

Tests for cortex detection functionality including texture analysis, 
stippling detection, and cortex/scar differentiation.
"""

import pytest
import numpy as np
import cv2
import tempfile
import os
from unittest.mock import patch, MagicMock

from pylithics.image_processing.modules.cortex_detection import (
    detect_cortex_in_child_contours,
    _detect_cortex_texture,
    calculate_total_cortex_metrics
)
from pylithics.image_processing.config import get_cortex_detection_config


@pytest.mark.unit
class TestCortexDetection:
    """Test cortex detection functionality."""

    def test_cortex_detection_disabled(self):
        """Test that cortex detection is skipped when disabled in config."""
        # Mock configuration with cortex detection disabled
        with patch('pylithics.image_processing.modules.cortex_detection.get_cortex_detection_config') as mock_config:
            mock_config.return_value = {'enabled': False}
            
            # Sample metrics
            metrics = [
                {'parent': 'parent 1', 'scar': 'parent 1', 'area': 1000},
                {'parent': 'parent 1', 'scar': 'scar 1', 'area': 100, 'contour': [[10, 10], [20, 10], [20, 20], [10, 20]]}
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
        with patch('pylithics.image_processing.modules.cortex_detection.get_cortex_detection_config') as mock_config:
            mock_config.return_value = {
                'enabled': True,
                'stippling_density_threshold': 0.2,
                'texture_variance_threshold': 100,
                'edge_density_threshold': 0.05
            }
            
            # Sample metrics with child contour - need surface type for parent
            metrics = [
                {'parent': 'parent 1', 'scar': 'parent 1', 'area': 1000, 'surface_type': 'Dorsal'},
                {'parent': 'parent 1', 'scar': 'scar 1', 'area': 100, 'contour': [[10, 10], [20, 10], [20, 20], [10, 20]]}
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
        
        with patch('pylithics.image_processing.modules.cortex_detection.get_cortex_detection_config') as mock_config:
            mock_config.return_value = {'enabled': True, 'stippling_density_threshold': 0.2, 'texture_variance_threshold': 100, 'edge_density_threshold': 0.05}
            
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