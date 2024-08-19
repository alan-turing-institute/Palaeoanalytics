import cv2
import numpy as np
import pandas as pd
import os

def pixels_to_mm(area_pixels, dpi):
    # Convert DPI to dots per millimeter
    dots_per_mm = dpi / 25.4

    # Convert area from pixels to square millimeters
    area_mm = area_pixels / dots_per_mm**2

    return area_mm

# Read the image
image_path = 'data/images/1.png'  # Replace with the path to your input image
image_name = os.path.splitext(os.path.basename(image_path))[0]  # Extract image name without extension
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur to smooth the image
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Threshold the blurred image to create a binary image
_, binary_image = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

# Find connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8, ltype=cv2.CV_32S)

# Initialize lists to store contour labels, areas, and centroids
contour_labels = []
contour_areas = []
centroid_points = []

# Create a copy of the original image
image_with_contours = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)

# Image properties
dpi = 400  # Example DPI value, replace with the actual DPI of your image
image_width_mm = 210  # Example width of the image in millimeters
image_height_mm = 297  # Example height of the image in millimeters

# Iterate through each label
for label in range(1, num_labels):
    # Calculate the area of the contour
    area = stats[label, cv2.CC_STAT_AREA]

    # If the area is greater than 10, label the contour - adjust to eliminate contours under a certain area.
    if area > 10:
        # Draw contour on the image
        mask = (labels == label).astype(np.uint8)
        contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_with_contours, contour, -1, (0, 255, 0), 2)

        # Get bounding box of the contour
        x, y, w, h = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP], stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]
        cx, cy = x + w // 2, y + h // 2  # Center of the bounding box

        # Append contour label, area, and centroid to the lists
        contour_labels.append(f"Contour {label}: Area {pixels_to_mm(area, dpi):.2f} mm^2")
        contour_areas.append(area)
        centroid_points.append((cx, cy))

        # Put text with contour label and area at the center of the bounding box
        cv2.putText(image_with_contours, f"Contour {label}: Area {pixels_to_mm(area, dpi):.2f} mm^2", (cx - 100, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

# Add red dots at the centroids of the contours
for centroid in centroid_points:
    cv2.circle(image_with_contours, centroid, 7, (0, 0, 255), -1)

# Create a DataFrame from the contour labels and areas
contour_df = pd.DataFrame({'Contour Label': contour_labels, 'Area (Pixels)': contour_areas})

# Export the DataFrame to a CSV file with the same name as the input image
csv_filename = f'{image_name}_contour_areas_mm.csv'
contour_df.to_csv(csv_filename, index=False)

# Display original image
cv2.imshow('Original Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Display image with contours, labels, and centroids
cv2.imshow('Image with Contours, Labels, and Centroids', image_with_contours)
cv2.waitKey(0)

# Save the image with contours
output_image_path = f'{image_name}_with_contours.jpg'
cv2.imwrite(output_image_path, image_with_contours)

cv2.destroyAllWindows()
