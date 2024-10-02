import unittest
from unittest.mock import patch, mock_open
from pylithics.image_processing.importer import validate_image_scale_dpi, import_images

class TestImporter(unittest.TestCase):

    @patch('pylithics.image_processing.importer.os.path.exists')
    @patch('pylithics.image_processing.importer.Image.open')
    def test_validate_image_scale_dpi(self, mock_image_open, mock_path_exists):
        # Mock the os.path.exists to return True
        mock_path_exists.return_value = True

        # Mock the image info
        mock_img = mock_image_open.return_value.__enter__.return_value
        mock_img.info = {'dpi': (300, 300)}

        # Test DPI validation success
        result = validate_image_scale_dpi('test_image.jpg', 300)
        self.assertTrue(result)

        # Test DPI validation failure
        result = validate_image_scale_dpi('test_image.jpg', 150)
        self.assertFalse(result)

    # Additional tests can be written here for edge cases and error handling
