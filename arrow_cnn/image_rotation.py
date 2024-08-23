import cv2
import os
import numpy as np

def rotate_image(image, angle):
    """
    Rotate the image by the specified angle while ensuring the entire image fits within the new canvas.

    Parameters:
    image (numpy.ndarray): The input image to be rotated.
    angle (float): The angle to rotate the image.

    Returns:
    numpy.ndarray: The rotated image with adjusted canvas size to avoid clipping.
    """
    height, width = image.shape[:2]
    image_center = (width // 2, height // 2)

    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])

    # Compute the new bounding dimensions of the image after rotation
    new_width = int(height * abs_sin + width * abs_cos)
    new_height = int(height * abs_cos + width * abs_sin)

    # Adjust the rotation matrix to account for the new image size
    rotation_matrix[0, 2] += (new_width / 2) - image_center[0]
    rotation_matrix[1, 2] += (new_height / 2) - image_center[1]

    # Perform the rotation with a white border to handle empty areas
    rotated_image = cv2.warpAffine(
        image, rotation_matrix, (new_width, new_height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
    )
    return rotated_image

def save_rotated_images(input_image_path, output_dir, base_filename):
    """
    Rotate the input image by 1 degree increments and save each rotated image.

    Parameters:
    input_image_path (str): Path to the input image.
    output_dir (str): Directory to save the rotated images.
    base_filename (str): Base filename for the output images.
    """
    # Load the input image
    image = cv2.imread(input_image_path)

    # Rotate the image from 1 to 360 degrees and save each one
    for angle in range(0, 360):
        rotated_image = rotate_image(image, angle)
        output_filename = f"{base_filename}_rotated_{angle}deg.png"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, rotated_image)

def process_multiple_images(input_dir, output_dir):
    """
    Process all images in the input directory, rotating each one by 1 degree increments and saving the results.

    Parameters:
    input_dir (str): Directory containing input images.
    output_dir (str): Directory to save the rotated images.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Supported image file extensions
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

    # Iterate through each image in the input directory
    for filename in os.listdir(input_dir):
        if any(filename.lower().endswith(ext) for ext in supported_extensions):
            input_image_path = os.path.join(input_dir, filename)
            base_filename = os.path.splitext(filename)[0]  # Get the base filename without extension
            save_rotated_images(input_image_path, output_dir, base_filename)

    print("Rotation and saving of all images completed.")

if __name__ == "__main__":
    # Set your input directory path and output directory here
    INPUT_DIR = 'arrow_templates/non_arrow'  # Example: 'images/original'
    OUTPUT_DIR = 'arrow_templates/non_arrow_rotated'  # Example: 'images/rotated'

    # Run the function to process and rotate all images in the input directory
    process_multiple_images(INPUT_DIR, OUTPUT_DIR)
