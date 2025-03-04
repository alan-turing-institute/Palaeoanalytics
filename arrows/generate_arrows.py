import cv2
import os
import argparse
import random
from tqdm import tqdm

def process_image(image, angle):
    """
    Rotate the image by the specified angle in the opposite direction,
    without applying random resizing.

    This function rotates the input image by the negative of the given angle,
    adjusting the output image dimensions and applying white padding where needed.

    Parameters:
        image (numpy.ndarray): The input image.
        angle (float): The rotation angle in degrees.

    Returns:
        numpy.ndarray: The rotated image with white padding.
    """
    height, width = image.shape[:2]
    image_center = (width // 2, height // 2)

    # Compute the rotation matrix using the negative of the given angle to reverse direction.
    rotation_matrix = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])

    # Compute the new dimensions of the image after rotation.
    new_width = int(height * abs_sin + width * abs_cos)
    new_height = int(height * abs_cos + width * abs_sin)

    # Adjust the rotation matrix to take into account the translation.
    rotation_matrix[0, 2] += (new_width / 2) - image_center[0]
    rotation_matrix[1, 2] += (new_height / 2) - image_center[1]

    # Perform the rotation with white border padding.
    rotated_image = cv2.warpAffine(
        image, rotation_matrix, (new_width, new_height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    return rotated_image

def compute_bounding_box(image):
    """
    Compute a bounding box for the arrow in the image using contour detection.

    Assumes the arrow is dark on a white background. This function converts the image to grayscale,
    applies a threshold to detect the arrow, and finds the largest contour. The bounding box is computed
    for that contour.

    Parameters:
        image (numpy.ndarray): The processed (rotated) image.

    Returns:
        tuple: Bounding box as (x, y, w, h) in absolute pixel coordinates.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return x, y, w, h
    else:
        return 0, 0, image.shape[1], image.shape[0]

def convert_bbox_to_yolo(bbox, img_dims):
    """
    Convert bounding box coordinates from absolute pixel values to YOLO format (normalized).

    YOLO format: <x_center> <y_center> <width> <height>, where all values are normalized by the image dimensions.

    Parameters:
        bbox (tuple): Bounding box in (x, y, w, h) format.
        img_dims (tuple): Dimensions of the image in (width, height).

    Returns:
        tuple: Normalized bounding box as (x_center, y_center, norm_w, norm_h).
    """
    x, y, w, h = bbox
    img_width, img_height = img_dims

    x_center = x + w / 2.0
    y_center = y + h / 2.0

    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    w_norm = w / img_width
    h_norm = h / img_height

    return x_center_norm, y_center_norm, w_norm, h_norm

def process_images(input_dir, train_img_dir, train_ann_dir, val_img_dir, val_ann_dir, test_img_dir, test_ann_dir,
                   train_ratio, val_ratio):
    """
    Generate 360 rotated variations (one for each degree) of each input arrow image and split them
    into training, validation, and test sets. For each rotated image, compute the YOLO-format annotation
    containing normalized bounding box coordinates, a class label (0 for arrow), and the rotation angle.

    Splitting is done on a per-base-image level: for each original image, its 360 rotations are randomly
    shuffled and then partitioned into training, validation, and test sets according to the specified ratios.
    This ensures that each base image contributes to all splits.

    Annotation format per image:
        0 x_center y_center bbox_width bbox_height angle
    where the bounding box values are normalized.

    Parameters:
        input_dir (str): Directory containing input arrow images.
        train_img_dir (str): Directory to save training images.
        train_ann_dir (str): Directory to save training annotation files.
        val_img_dir (str): Directory to save validation images.
        val_ann_dir (str): Directory to save validation annotation files.
        test_img_dir (str): Directory to save test images.
        test_ann_dir (str): Directory to save test annotation files.
        train_ratio (float): Proportion of rotations to assign to training (e.g., 0.7).
        val_ratio (float): Proportion of rotations to assign to validation (e.g., 0.15).
        Note: Test ratio is implicitly (1 - train_ratio - val_ratio).
    """
    # Create output directories if they don't exist.
    for directory in [train_img_dir, train_ann_dir, val_img_dir, val_ann_dir, test_img_dir, test_ann_dir]:
        os.makedirs(directory, exist_ok=True)

    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_filenames = [filename for filename in os.listdir(input_dir)
                       if any(filename.lower().endswith(ext) for ext in supported_extensions)]

    for filename in tqdm(image_filenames, desc="Processing base images"):
        input_image_path = os.path.join(input_dir, filename)
        base_filename = os.path.splitext(filename)[0]
        image = cv2.imread(input_image_path)
        if image is None:
            print(f"Error loading image {input_image_path}")
            continue

        # Create list of all rotation angles (0-359).
        rotations = list(range(0, 360))
        random.shuffle(rotations)

        # Calculate split counts per base image.
        total = len(rotations)  # Should be 360.
        n_train = max(1, int(total * train_ratio))
        n_val = max(1, int(total * val_ratio))
        n_test = total - n_train - n_val

        # Ensure at least one image per split.
        if n_test < 1:
            n_test = 1
            if n_train > n_val:
                n_train -= 1
            else:
                n_val -= 1

        train_angles = rotations[:n_train]
        val_angles = rotations[n_train:n_train + n_val]
        test_angles = rotations[n_train + n_val:]

        # Process and save images for each split.
        for angle in train_angles:
            processed_image = process_image(image, angle)
            bbox = compute_bounding_box(processed_image)
            proc_height, proc_width = processed_image.shape[:2]
            x_center, y_center, norm_w, norm_h = convert_bbox_to_yolo(bbox, (proc_width, proc_height))
            annotation_line = f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f} {angle}\n"
            out_filename = f"{base_filename}_{angle}.png"
            cv2.imwrite(os.path.join(train_img_dir, out_filename), processed_image)
            with open(os.path.join(train_ann_dir, f"{base_filename}_{angle}.txt"), "w") as f:
                f.write(annotation_line)

        for angle in val_angles:
            processed_image = process_image(image, angle)
            bbox = compute_bounding_box(processed_image)
            proc_height, proc_width = processed_image.shape[:2]
            x_center, y_center, norm_w, norm_h = convert_bbox_to_yolo(bbox, (proc_width, proc_height))
            annotation_line = f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f} {angle}\n"
            out_filename = f"{base_filename}_{angle}.png"
            cv2.imwrite(os.path.join(val_img_dir, out_filename), processed_image)
            with open(os.path.join(val_ann_dir, f"{base_filename}_{angle}.txt"), "w") as f:
                f.write(annotation_line)

        for angle in test_angles:
            processed_image = process_image(image, angle)
            bbox = compute_bounding_box(processed_image)
            proc_height, proc_width = processed_image.shape[:2]
            x_center, y_center, norm_w, norm_h = convert_bbox_to_yolo(bbox, (proc_width, proc_height))
            annotation_line = f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f} {angle}\n"
            out_filename = f"{base_filename}_{angle}.png"
            cv2.imwrite(os.path.join(test_img_dir, out_filename), processed_image)
            with open(os.path.join(test_ann_dir, f"{base_filename}_{angle}.txt"), "w") as f:
                f.write(annotation_line)

    print("Processing complete.")
    print(f"Training images saved in: {train_img_dir}")
    print(f"Training annotations saved in: {train_ann_dir}")
    print(f"Validation images saved in: {val_img_dir}")
    print(f"Validation annotations saved in: {val_ann_dir}")
    print(f"Test images saved in: {test_img_dir}")
    print(f"Test annotations saved in: {test_ann_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rotate arrow images (0-359°), compute YOLO-format bounding box annotations (with angle), "
                    "and split outputs into training, validation, and test sets."
    )
    parser.add_argument('-i', '--input', type=str, required=True, help="Directory containing input arrow images.")
    parser.add_argument('-tri', '--train_images', type=str, required=True, help="Directory to save training images.")
    parser.add_argument('-tra', '--train_annotations', type=str, required=True, help="Directory to save training annotation files.")
    parser.add_argument('-vi', '--val_images', type=str, required=True, help="Directory to save validation images.")
    parser.add_argument('-va', '--val_annotations', type=str, required=True, help="Directory to save validation annotation files.")
    parser.add_argument('-tei', '--test_images', type=str, required=True, help="Directory to save test images.")
    parser.add_argument('-tea', '--test_annotations', type=str, required=True, help="Directory to save test annotation files.")
    parser.add_argument('-tr', '--train_ratio', type=float, default=0.7, help="Proportion of rotations for training (default: 0.7).")
    parser.add_argument('-vr', '--val_ratio', type=float, default=0.15, help="Proportion of rotations for validation (default: 0.15).")
    # Test ratio is computed as 1 - (train_ratio + val_ratio).

    args = parser.parse_args()
    process_images(args.input, args.train_images, args.train_annotations,
                   args.val_images, args.val_annotations,
                   args.test_images, args.test_annotations,
                   args.train_ratio, args.val_ratio)
