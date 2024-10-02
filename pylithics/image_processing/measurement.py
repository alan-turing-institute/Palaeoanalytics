import logging

# Load logging configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class Measurement:
    def __init__(self, pixels, dpi=None):
        """
        Initializes a measurement object.

        :param pixels: Measurement in pixels.
        :param dpi: Dots per inch for the scale. If None, it will default to pixels.
        """
        self.pixels = pixels
        self.dpi = dpi

    def to_millimeters(self):
        """
        Converts the pixel measurement to millimeters based on the provided DPI.
        If DPI is None or invalid, the measurement is returned in pixels.
        """
        if self.dpi is None:
            logging.warning(f"No DPI provided. Returning measurement in pixels: {self.pixels}px")
            return self.pixels

        if self.dpi <= 0:
            logging.error(f"Invalid DPI ({self.dpi}) provided. Returning measurement in pixels: {self.pixels}px")
            return self.pixels

        # Convert pixels to millimeters
        millimeters = (self.pixels / self.dpi) * 25.4
        logging.info(f"Converted {self.pixels}px to {millimeters}mm using DPI: {self.dpi}")
        return millimeters

    def is_scaled(self):
        """
        Returns True if a valid DPI is present, indicating that the measurement is scaled.
        """
        return self.dpi is not None and self.dpi > 0
