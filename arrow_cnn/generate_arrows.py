import cv2
import os
import random
import argparse
from tqdm import tqdm  # Import tqdm for progress bar

def process_image(image, angle):
    """
    Rotate the image by the specified angle and apply random resizing.

    Parameters:
    image (numpy.ndarray): The input image to be processed.
    angle (float): The angle to rotate the image.

    Returns:
    numpy.ndarray: The processed image (rotated and resized).
    """
    height, width = image.shape[:2]
    image_center = (width // 2, height // 2)

    # Rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0) # change angle of rotation
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])

    # New bounding box size after rotation
    new_width = int(height * abs_sin + width * abs_cos)
    new_height = int(height * abs_cos + width * abs_sin)

    # Adjust matrix to account for the new image size
    rotation_matrix[0, 2] += (new_width / 2) - image_center[0]
    rotation_matrix[1, 2] += (new_height / 2) - image_center[1]

    # Rotate with white padding for empty areas
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(255, 255, 255))

    # Random scaling between 20% and 200%
    scale = random.uniform(0.2, 2.0)
    new_size = (max(int(rotated_image.shape[1] * scale), 1), max(int(rotated_image.shape[0] * scale), 1))

    # Resize the image
    resized_image = cv2.resize(rotated_image, new_size, interpolation=cv2.INTER_LINEAR)

    return resized_image

def process_images(input_dir, output_dir1, output_dir2, percentage):
    """
    Process and distribute images between two directories based on the percentage split.

    Parameters:
    input_dir (str): Directory containing input images.
    output_dir1 (str): Directory to save the majority percentage of images.
    output_dir2 (str): Directory to save the minority percentage of images.
    percentage (float): Percentage of images to save to output_dir1.
    """
    # Create output directories if they don't exist
    os.makedirs(output_dir1, exist_ok=True)
    os.makedirs(output_dir2, exist_ok=True)

    # Get supported image file extensions
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

    # Collect all image files from input directory
    image_filenames = [
        filename for filename in os.listdir(input_dir)
        if any(filename.lower().endswith(ext) for ext in supported_extensions)
    ]

    # Store all variations (rotations and resizes) of all images
    all_variations = []

    for filename in tqdm(image_filenames, desc="Processing images"):
        input_image_path = os.path.join(input_dir, filename)
        base_filename = os.path.splitext(filename)[0]

        # Load the image
        image = cv2.imread(input_image_path)
        if image is None:
            print(f"Error loading image {input_image_path}")
            continue

        # Generate 360 variations (rotated versions) for each image
        all_variations.extend([(image, base_filename, angle) for angle in range(0,360, 15)]) # change angle rotations

    # Shuffle the variations list
    random.shuffle(all_variations)

    # Determine the split based on the percentage
    num_variations_output1 = int(len(all_variations) * (percentage / 100.0))

    # Save the shuffled variations to output directories
    for i, (image, base_filename, angle) in tqdm(enumerate(all_variations), total=len(all_variations), desc="Saving images"):
        processed_image = process_image(image, angle)
        output_filename = f"{base_filename}_rotated_{angle}deg_resized.png"

        if i < num_variations_output1:
            output_path = os.path.join(output_dir1, output_filename)
        else:
            output_path = os.path.join(output_dir2, output_filename)

        cv2.imwrite(output_path, processed_image)

    print(f"Rotation, resizing, and saving of all image variations completed with {percentage}% going to {output_dir1} and the remaining {100 - percentage}% going to {output_dir2}.")

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Rotate and resize images, and distribute them between two output directories."
    )
# Define the command-line arguments
    parser.add_argument('-i', '--input', type=str, required=True, help="Directory containing the input images.")
    parser.add_argument('-o1', '--output1', type=str, required=True, help="Directory to save the training set of images.")
    parser.add_argument('-o2', '--output2', type=str, required=True, help="Directory to save the test set of images.")
    parser.add_argument(
        '-p', '--percentage', type=float, default=80.0,
        help=(
            "Percentage of images to save to output1 (the training set). "
            "Remaining images will automatically go to output2 (the test set). Default is an 80%% : 20%% split."
        )
    )

    # Parse the arguments and run the image processing function
    args = parser.parse_args()
    process_images(args.input, args.output1, args.output2, args.percentage)
