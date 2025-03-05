import os
import cv2
import argparse
import numpy as np

# Global variables to store keypoints
points = []  # will store tuples (x, y)
image = None
clone = None

def mouse_callback(event, x, y, flags, param):
    global points, image
    # When left mouse button is clicked, record the point.
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:
            points.append((x, y))
            # Draw a circle where the user clicked.
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Annotate Arrow", image)
        else:
            print("Already have two keypoints (tip and tail). Press 'r' to reset.")

def normalize_point(pt, width, height):
    """
    Normalize a point's coordinates relative to image dimensions.

    Parameters:
        pt (tuple): (x, y) coordinate.
        width (int): Image width.
        height (int): Image height.

    Returns:
        tuple: Normalized (x, y) between 0 and 1.
    """
    return (pt[0] / width, pt[1] / height)

def annotate_image(image_path, output_annotation_path):
    """
    Annotate one arrow image by clicking its tip and tail.
    Saves the annotation to a file in the format:
        0 x_tip y_tip x_tail y_tail
    where coordinates are normalized.

    Parameters:
        image_path (str): Path to the arrow image.
        output_annotation_path (str): Path to save the annotation text file.
    """
    global points, image, clone
    points = []  # Reset keypoints
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return False

    # Convert image to RGB for display (optional)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    clone = image.copy()

    cv2.namedWindow("Annotate Arrow")
    cv2.setMouseCallback("Annotate Arrow", mouse_callback)

    print("Instructions:")
    print("1. Left-click on the arrow TIP (the pointed end).")
    print("2. Left-click on the arrow TAIL (the other end).")
    print("Press 's' to save the annotation, 'r' to reset, or 'q' to skip this image.")

    while True:
        cv2.imshow("Annotate Arrow", image)
        key = cv2.waitKey(1) & 0xFF

        # Save annotation if 's' is pressed and two keypoints are selected.
        if key == ord("s") and len(points) == 2:
            height, width = clone.shape[:2]
            tip_norm = normalize_point(points[0], width, height)
            tail_norm = normalize_point(points[1], width, height)
            # Write annotation file in the format: 0 x_tip y_tip x_tail y_tail
            with open(output_annotation_path, "w") as f:
                f.write(f"0 {tip_norm[0]:.6f} {tip_norm[1]:.6f} {tail_norm[0]:.6f} {tail_norm[1]:.6f}\n")
            print(f"Annotation saved to {output_annotation_path}")
            break
        # Reset if 'r' is pressed.
        elif key == ord("r"):
            points = []
            image = clone.copy()
            print("Annotation reset. Please click the points again.")
        # Skip the image if 'q' is pressed.
        elif key == ord("q"):
            print("Skipping this image.")
            break

    cv2.destroyAllWindows()
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Annotate arrow images with keypoints (tip and tail) for YOLO training."
    )
    parser.add_argument('-i','--input_dir', type=str, required=True,
                        help="Directory containing arrow images.")
    parser.add_argument('-o','--output_dir', type=str, required=True,
                        help="Directory to save annotation files.")
    args = parser.parse_args()

    # Ensure output directory exists.
    os.makedirs(args.output_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.png')])
    if not image_files:
        print("No PNG images found in the input directory.")
        return

    for img_file in image_files:
        image_path = os.path.join(args.input_dir, img_file)
        output_annotation_path = os.path.join(args.output_dir, img_file.replace('.png', '.txt'))
        print(f"\nAnnotating image: {image_path}")
        success = annotate_image(image_path, output_annotation_path)
        if not success:
            continue

if __name__ == "__main__":
    main()
