import unittest
from pylithics.image_processing.measurement import Measurement
from unittest.mock import patch

class TestMeasurement(unittest.TestCase):

    def test_conversion_to_millimeters_with_valid_dpi(self):
        """Test conversion from pixels to millimeters with a valid DPI."""
        measurement = Measurement(500, dpi=300)  # 500 pixels, 300 DPI
        millimeters = measurement.to_millimeters()
        self.assertAlmostEqual(millimeters, (500 / 300) * 25.4)

    def test_conversion_without_dpi_returns_pixels(self):
        """Test that if no DPI is provided, the measurement is returned in pixels."""
        measurement = Measurement(500)  # 500 pixels, no DPI
        pixels = measurement.to_millimeters()
        self.assertEqual(pixels, 500)

    def test_conversion_with_invalid_dpi_returns_pixels(self):
        """Test that invalid DPI (zero or negative) returns measurement in pixels."""
        measurement = Measurement(500, dpi=0)  # Invalid DPI
        pixels = measurement.to_millimeters()
        self.assertEqual(pixels, 500)

        measurement = Measurement(500, dpi=-100)  # Negative DPI
        pixels = measurement.to_millimeters()
        self.assertEqual(pixels, 500)

    @patch('pylithics.image_processing.measurement.logging.warning')
    def test_logging_when_no_dpi_provided(self, mock_warning):
        """Test that a warning is logged when no DPI is provided."""
        measurement = Measurement(500)
        measurement.to_millimeters()
        mock_warning.assert_called_once_with('No DPI provided. Returning measurement in pixels: 500px')

    @patch('pylithics.image_processing.measurement.logging.error')
    def test_logging_for_invalid_dpi(self, mock_error):
        """Test that an error is logged when an invalid DPI is provided."""
        measurement = Measurement(500, dpi=0)
        measurement.to_millimeters()
        mock_error.assert_called_once_with('Invalid DPI (0) provided. Returning measurement in pixels: 500px')

    def test_is_scaled(self):
        """Test the is_scaled() method to check if a valid DPI is present."""
        measurement = Measurement(500, dpi=300)
        self.assertTrue(measurement.is_scaled())

        measurement = Measurement(500)
        self.assertFalse(measurement.is_scaled())

        measurement = Measurement(500, dpi=0)
        self.assertFalse(measurement.is_scaled())

if __name__ == '__main__':
    unittest.main()
