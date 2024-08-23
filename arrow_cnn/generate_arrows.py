import cv2
import os
import numpy as np
import random

def random_rotation(image):
    """
    Apply a random rotation to the input image.

    Parameters:
    image (numpy.ndarray): The input image to be rotated.

    Returns:
    numpy.ndarray: The rotated image with adjusted canvas size to avoid clipping.
    """
    height, width = image.shape[:2]
    angle = random.uniform(1, 360)  # Random rotation between 1 and 360 degrees
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

def random_resizing(image):
    """
    Apply a random resizing to the input image.

    Parameters:
    image (numpy.ndarray): The input image to be resized.

    Returns:
    numpy.ndarray: The resized image.
    """
    scale = random.uniform(0.2, 2.0)  # Random scaling between 20% and 200%
    new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    return resized_image

def augment_image(image):
    """
    Apply a series of augmentations (rotation and resizing) to the input image.

    Parameters:
    image (numpy.ndarray): The input image to be augmented.

    Returns:
    numpy.ndarray: The augmented image.
    """
    rotated_image = random_rotation(image)
    augmented_image = random_resizing(rotated_image)
    return augmented_image

def main(input_dir, train_output_dir, test_output_dir, num_augmentations=10):
    """
    Main function to perform image augmentation and save the augmented images.

    Parameters:
    input_dir (str): Path to the directory with original images.
    train_output_dir (str): Path to the directory to save augmented training images.
    test_output_dir (str): Path to the directory to save augmented test images.
    num_augmentations (int): Number of augmentations to perform per image. Default is 10.
    """
    # Ensure the output directories exist
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    # Supported image file extensions
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

    # Initialize counters for the naming conventions
    image_counter = 1
    test_counter = 1

    # Collect all image filenames in the input directory
    image_filenames = [
        f for f in os.listdir(input_dir)
        if any(f.lower().endswith(ext) for ext in supported_extensions)
    ]

    # Shuffle the filenames to ensure randomness when selecting the 20% test images
    random.shuffle(image_filenames)

    # Calculate the number of test images (20% of total images)
    num_test_images = int(len(image_filenames) * 0.2)

    # Iterate through each image in the input directory
    for idx, filename in enumerate(image_filenames):
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)

        # Determine if the image should be part of the test set or training set
        if idx < num_test_images:
            prefix = 'arrow_test'
            counter = test_counter
            output_dir = test_output_dir
        else:
            prefix = 'arrow'
            counter = image_counter
            output_dir = train_output_dir

        # Perform multiple augmentations per image and save each one
        for aug_idx in range(num_augmentations):
            augmented_image = augment_image(image)
            augmented_filename = f"{prefix}{counter}{os.path.splitext(filename)[1]}"
            augmented_image_path = os.path.join(output_dir, augmented_filename)

            # Save the augmented image
            cv2.imwrite(augmented_image_path, augmented_image)

            # Increment the counter for each saved augmentation
            counter += 1

        # Update the image and test counters
        if idx < num_test_images:
            test_counter = counter
        else:
            image_counter = counter

    print("Image augmentation and renaming completed.")

if __name__ == "__main__":
    # Set your input and output directories here
    INPUT_DIR = 'arrow_templates/arrow_rotated'
    TRAIN_OUTPUT_DIR = 'train/arrows'  # Example: 'images/augmented/train'
    TEST_OUTPUT_DIR = 'test/arrows'  # Example: 'images/augmented/test'

    # Run the main function with the specified directories
    main(INPUT_DIR, TRAIN_OUTPUT_DIR, TEST_OUTPUT_DIR)
