#!/usr/bin/env python3
"""
Test script for scale calibration feature.
"""

import os
import sys
import logging

# Add parent directory to path to import pylithics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pylithics.image_processing.modules.scale_calibration import (
    detect_scale_bar,
    calculate_conversion_factor,
    get_calibration_factor
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_scale_detection():
    """Test scale bar detection on provided examples."""

    # Path to scale examples
    scale_dir = ".claude/visual_examples/features/scale_calibration"

    # Test configuration
    config = {
        'debug_output': True
    }

    # Test each scale image
    scale_files = [
        "sc_001.png",  # Segmented scale bar
        "sc_004.png",  # Simple line with white center
        "sc_005.png",  # Thick bar with white center
        "sc_006.png",  # Vertical scale bar
        "sc_007.png",  # Line with center tick
        "sc_008.png",  # Line with multiple ticks
    ]

    print("\n" + "="*60)
    print("SCALE BAR DETECTION TEST")
    print("="*60)

    for scale_file in scale_files:
        scale_path = os.path.join(scale_dir, scale_file)

        if not os.path.exists(scale_path):
            print(f"\n❌ File not found: {scale_path}")
            continue

        print(f"\n📏 Testing: {scale_file}")
        print("-" * 40)

        # Detect scale bar
        result = detect_scale_bar(scale_path, config)

        if result:
            scale_pixels, confidence = result
            print(f"✅ Scale detected: {scale_pixels} pixels")
            print(f"   Confidence: {confidence:.2f}")

            # Test conversion calculation
            test_scale_mm = 50  # Assume 50mm scale
            pixels_per_mm = calculate_conversion_factor(scale_pixels, test_scale_mm)
            print(f"   Conversion: {pixels_per_mm:.3f} pixels/mm")
            print(f"   (1mm = {pixels_per_mm:.1f} pixels)")
        else:
            print(f"❌ Scale detection failed")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60 + "\n")

def test_calibration_fallback():
    """Test the calibration fallback system."""

    print("\n" + "="*60)
    print("CALIBRATION FALLBACK TEST")
    print("="*60)

    # Test configuration
    config = {
        'scale_calibration': {
            'enabled': True,
            'fallback_to_dpi': True,
            'fallback_to_pixels': True,
            'debug_output': False
        }
    }

    # Test cases
    test_cases = [
        {
            'name': 'With scale bar',
            'image_path': 'test_image.png',
            'scale_data': {
                'scale_id': 'sc_004.png',
                'scale': '50'
            }
        },
        {
            'name': 'No scale bar (DPI fallback)',
            'image_path': 'test_image.png',
            'scale_data': {}
        },
        {
            'name': 'Invalid scale',
            'image_path': 'test_image.png',
            'scale_data': {
                'scale_id': 'nonexistent.png',
                'scale': '50'
            }
        }
    ]

    for test_case in test_cases:
        print(f"\n📋 Test: {test_case['name']}")
        print("-" * 40)

        # Note: This will fail for non-existent images, but shows the logic
        try:
            factor, method = get_calibration_factor(
                test_case['image_path'],
                test_case['scale_data'],
                config
            )

            if factor:
                print(f"✅ Calibration method: {method}")
                print(f"   Conversion factor: {factor:.3f} pixels/mm")
            else:
                print(f"⚠️  No calibration available")
                print(f"   Method: {method}")
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n" + "="*60)
    print("FALLBACK TEST COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    # Run tests
    test_scale_detection()
    # test_calibration_fallback()  # Commented out as it needs real images

    print("\n✨ All tests completed!")