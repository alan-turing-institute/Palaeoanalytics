#!/usr/bin/env python3
"""
Train a modified YOLOv7-tiny model for arrow keypoint detection with letterboxed inputs.

This script:
  - Loads arrow images and their corresponding keypoint annotations (normalized) in the format:
      0 x_tip y_tip x_tail y_tail angle
    (The extra angle is used only for computing a validation metric.)
  - Applies letterbox pre‑processing to preserve image aspect ratio.
  - Adjusts the keypoint annotations to match the letterboxed images.
  - Trains a YOLOv7-tiny–style model that uses adaptive average pooling and a fully connected
    regression head to predict 4 keypoint coordinates.
  - Computes a mean absolute error (MAE) metric for the predicted arrow angle.
  - Produces composite visualization images showing the original letterboxed image (top)
    and the annotated prediction (bottom) with red graphics (a red dot at the tail and an arrowed line at the tip)
    and with red text showing the transformed (clockwise) predicted angle.
  - Implements learning rate scheduling (ReduceLROnPlateau) and early stopping.

Usage Example:
  python arrows/YOLO_arrow_train.py \
    --train_images arrows/images/train \
    --train_annotations arrows/arrow_annotations/train \
    --val_images arrows/images/val \
    --val_annotations arrows/arrow_annotations/val \
    --num_epochs 50 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --vis_dir YOLO_visualizations \
    --img_size 640 \
    --patience 5
"""

import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import argparse
import time
import math
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Setup logging with timestamp and log level.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def letterbox(image: np.ndarray, target_size: int):
    """
    Resize an image to fit within target_size x target_size while preserving the original aspect ratio,
    and pad with white pixels.

    Args:
        image (np.ndarray): Input image.
        target_size (int): Desired output dimension (width and height).

    Returns:
        tuple: (letterboxed_image, scale, pad_w, pad_h)
            letterboxed_image: The resized and padded image.
            scale: Scale factor used for resizing.
            pad_w: Horizontal padding (left side).
            pad_h: Vertical padding (top side).
    """
    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    canvas = np.full((target_size, target_size, 3), 255, dtype=np.uint8)
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    canvas[pad_h:pad_h+new_h, pad_w:pad_w+new_w, :] = resized
    return canvas, scale, pad_w, pad_h

def resize_image(image: np.ndarray, target_size: int):
    """
    Resize image to target_size x target_size.

    Args:
        image (np.ndarray): Input image.
        target_size (int): Desired output dimension.

    Returns:
        np.ndarray: The resized image.
    """
    return cv2.resize(image, (target_size, target_size))

class ArrowDataset(Dataset):
    def __init__(self, images_dir: str, annotations_dir: str, img_size: int = 640, transform=None):
        """
        Custom dataset for arrow images and keypoint annotations.

        The expected annotation file format is:
            0 x_tip y_tip x_tail y_tail angle
        Coordinates are normalized relative to the original image dimensions.
        The extra angle value is preserved for computing a metric but is not used as a regression target.

        Args:
            images_dir (str): Directory containing arrow images.
            annotations_dir (str): Directory containing annotation files.
            img_size (int): Target letterboxed image size (img_size x img_size).
            transform (callable, optional): Additional transform to apply on the image.
        """
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.img_size = img_size
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])

    def __len__(self) -> int:
        """Return the total number of images in the dataset."""
        return len(self.image_files)

    def __getitem__(self, idx: int):
        """
        Load an image and its annotation, apply letterbox pre‑processing,
        adjust the keypoint coordinates, and return them as tensors.

        Returns:
            tuple: (image_tensor, target_tensor)
                - image_tensor: Tensor of shape (3, img_size, img_size) with values normalized to [0,1].
                - target_tensor: Tensor of shape (6,) containing [cls, x_tip, y_tip, x_tail, y_tail, angle].
        """
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error loading image {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        # Apply letterbox resizing to preserve aspect ratio.
        letterboxed, scale, pad_w, pad_h = letterbox(image, self.img_size)
        letterboxed = letterboxed.astype(np.float32) / 255.0

        # Read the corresponding annotation.
        ann_path = os.path.join(self.annotations_dir, self.image_files[idx].replace('.png', '.txt'))
        with open(ann_path, 'r') as f:
            line = f.readline().strip().split()
        cls_label = int(line[0])
        if len(line) >= 6:
            x_tip_norm, y_tip_norm, x_tail_norm, y_tail_norm, angle = map(float, line[1:6])
        else:
            x_tip_norm, y_tip_norm, x_tail_norm, y_tail_norm = map(float, line[1:5])
            angle = 0.0

        # Convert normalized coordinates to absolute pixel coordinates in the original image.
        tip_abs = (x_tip_norm * orig_w, y_tip_norm * orig_h)
        tail_abs = (x_tail_norm * orig_w, y_tail_norm * orig_h)
        # Map these points to the letterboxed image coordinate system.
        tip_lb = (pad_w + tip_abs[0] * scale, pad_h + tip_abs[1] * scale)
        tail_lb = (pad_w + tail_abs[0] * scale, pad_h + tail_abs[1] * scale)
        # Re-normalize the coordinates with respect to the target (letterboxed) image dimensions.
        tip_norm_lb = (tip_lb[0] / self.img_size, tip_lb[1] / self.img_size)
        tail_norm_lb = (tail_lb[0] / self.img_size, tail_lb[1] / self.img_size)
        # Create the target vector.
        target = np.array([cls_label, tip_norm_lb[0], tip_norm_lb[1],
                           tail_norm_lb[0], tail_norm_lb[1], angle], dtype=np.float32)

        if self.transform:
            letterboxed = self.transform(letterboxed)
        image_tensor = torch.from_numpy(letterboxed.transpose(2, 0, 1))
        target_tensor = torch.from_numpy(target)
        return image_tensor, target_tensor

class YOLOv7Tiny(nn.Module):
    def __init__(self):
        """
        A modified YOLOv7-tiny model for keypoint detection.

        The model consists of a convolutional backbone, followed by an adaptive pooling layer
        to preserve spatial resolution, and a fully connected layer to regress the 4 keypoint coordinates.
        """
        super(YOLOv7Tiny, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Use adaptive average pooling to obtain a fixed spatial dimension.
        self.adapt_pool = nn.AdaptiveAvgPool2d((8, 8))
        # Fully connected layer to predict 4 keypoint values: [x_tip, y_tip, x_tail, y_tail].
        self.fc = nn.Linear(64 * 8 * 8, 4)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch, 3, H, W).

        Returns:
            dict: Dictionary containing:
                - "pred": The raw prediction output from the FC layer.
                - "keypoints": The predicted keypoint coordinates (shape: (batch, 4)).
        """
        features = self.backbone(x)
        pooled = self.adapt_pool(features)
        flattened = pooled.view(pooled.size(0), -1)
        out = self.fc(flattened)
        return {"pred": out, "keypoints": out}

    def modify_for_angle(self):
        """
        Placeholder for any modifications needed for angle handling.
        For this model, keypoint outputs are used to compute the arrow angle.
        """
        pass

class YOLOLoss(nn.Module):
    def __init__(self):
        """
        Mean Squared Error loss for keypoint regression.
        """
        super(YOLOLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, outputs: dict, targets: torch.Tensor):
        """
        Compute loss between predicted and true keypoints.

        Args:
            outputs (dict): Dictionary with model outputs; expects "keypoints" key.
            targets (torch.Tensor): Ground truth targets with shape (batch, 6),
                                    where indices 1:5 are [x_tip, y_tip, x_tail, y_tail].

        Returns:
            torch.Tensor: The computed MSE loss.
        """
        pred_kps = outputs["keypoints"]
        true_kps = targets[:, 1:5]
        loss = self.mse(pred_kps, true_kps)
        return loss

def compute_angle_from_keypoints(kps: np.ndarray) -> float:
    """
    Compute the arrow angle in degrees based solely on the tail's position relative to the center.

    In this design, the arrow tip is fixed (pointing toward the center, assumed to be at (0.5, 0.5)
    in normalized coordinates), so the tail's location determines the arrow’s orientation.

    Standard computation:
      angle = atan2(y_tip - y_tail, x_tip - x_tail)
    Then the transformation is applied so that an arrow with its tail on the positive y-axis becomes 0°.

    Args:
        kps (np.ndarray): Array of 4 normalized values: [x_tip, y_tip, x_tail, y_tail].

    Returns:
        float: Transformed angle in degrees (0 to 360) computed from the tail's position relative to the center.
    """
    # Define the center in normalized coordinates.
    center = (0.5, 0.5)
    # Use only the tail's coordinates.
    x_tail, y_tail = kps[2], kps[3]
    # Compute the vector from tail to center.
    vec_x = center[0] - x_tail
    vec_y = center[1] - y_tail
    angle_rad = math.atan2(vec_y, vec_x)
    angle_deg = (math.degrees(angle_rad) + 360) % 360
    # Apply transformation: adjust so that an arrow with its tail on the positive y-axis becomes 0°.
    transformed_angle = (270 + angle_deg) % 360
    return transformed_angle

def visualize_predictions(model: nn.Module, val_loader: DataLoader, vis_dir: str, num_images: int = 5):
    """
    Generate composite visualization images for a few validation samples.

    Each composite image consists of two vertically stacked parts:
      - Top: The original letterboxed image.
      - Bottom: The same image with predicted keypoints annotated:
             * A red arrowed line is drawn from the tail to the tip.
             * A red dot is drawn at the tail.
             * Red text shows the predicted (transformed) angle.

    Args:
        model (nn.Module): Trained model.
        val_loader (DataLoader): Validation dataset loader.
        vis_dir (str): Directory where the composite images will be saved.
        num_images (int): Number of composite images to generate.
    """
    import math
    os.makedirs(vis_dir, exist_ok=True)
    model.eval()
    images_visualized = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = model(images)
            pred_kps = outputs["keypoints"].cpu().numpy()  # shape: (batch, 4)
            for i in range(images.size(0)):
                # Retrieve the original letterboxed image.
                orig_img = images[i].cpu().numpy().transpose(1, 2, 0)
                orig_img = (orig_img * 255).astype(np.uint8)
                orig_img = np.ascontiguousarray(orig_img)
                # Convert normalized keypoints to absolute coordinates.
                kp = pred_kps[i]  # [x_tip, y_tip, x_tail, y_tail]
                abs_kp = np.array(kp) * vis_img_size  # vis_img_size equals the fixed img_size
                # Compute the predicted transformed angle.
                pred_angle = compute_angle_from_keypoints(kp)
                # Create an annotated copy of the image.
                annotated = orig_img.copy()
                h, w, _ = annotated.shape
                # Extract absolute coordinates for keypoints.
                x_tip, y_tip, x_tail, y_tail = abs_kp.astype(int)
                # Draw a red dot at the tail.
                cv2.circle(annotated, (x_tail, y_tail), 5, (0, 0, 255), -1)
                # Draw a red arrowed line from tail to tip.
                cv2.arrowedLine(annotated, (x_tail, y_tail), (x_tip, y_tip), (0, 0, 255), 2, tipLength=0.1)
                # Add red text displaying the predicted angle.
                text = f"Predicted Angle: {pred_angle:.2f}"
                cv2.putText(annotated, text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)
                # Create a composite image: original on top, annotated on bottom.
                composite = cv2.vconcat([orig_img, annotated])
                out_path = os.path.join(vis_dir, f"vis_{images_visualized}.png")
                cv2.imwrite(out_path, composite)
                images_visualized += 1
                if images_visualized >= num_images:
                    return

def train_yolo_arrow(train_images_dir: str, train_annotations_dir: str, val_images_dir: str, val_annotations_dir: str,
                     num_epochs: int, batch_size: int, learning_rate: float, vis_dir: str, img_size: int, patience: int):
    """
    Train a modified YOLOv7-tiny model for arrow keypoint detection using letterboxed images.

    This function:
      - Loads the training and validation datasets (with letterbox applied).
      - Trains the model using MSE loss for keypoint regression.
      - Uses ReduceLROnPlateau for learning rate scheduling and implements early stopping.
      - Computes an angle metric (mean absolute error in degrees) on the validation set.
      - Saves composite visualization images for inspection.
      - Writes a training summary to a text file.

    Args:
        train_images_dir (str): Directory containing training images.
        train_annotations_dir (str): Directory containing training annotation files.
        val_images_dir (str): Directory containing validation images.
        val_annotations_dir (str): Directory containing validation annotation files.
        num_epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        learning_rate (float): Initial learning rate.
        vis_dir (str): Directory to save composite visualization images.
        img_size (int): Fixed letterboxed image size used for training.
        patience (int): Early stopping patience (epochs with no improvement).
    """
    global vis_img_size
    vis_img_size = img_size  # For converting normalized coordinates to absolute in visualization.

    # Initialize datasets.
    train_dataset = ArrowDataset(train_images_dir, train_annotations_dir, img_size=img_size)
    val_dataset = ArrowDataset(val_images_dir, val_annotations_dir, img_size=img_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLOv7Tiny()
    model.modify_for_angle()  # Placeholder for any modifications.
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = YOLOLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, verbose=True)

    best_val_loss = float('inf')
    best_epoch = 0
    epochs_since_improvement = 0
    start_time = time.time()

    def compute_angle_error(loader: DataLoader) -> float:
        """
        Compute the mean absolute error (MAE) for the predicted arrow angle on a dataset loader.

        Args:
            loader (DataLoader): DataLoader for a dataset (e.g., validation set).

        Returns:
            float: Mean absolute error in degrees.
        """
        total_error = 0.0
        count = 0
        model.eval()
        with torch.no_grad():
            for imgs, tgts in loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                pred_kps = outputs["keypoints"].cpu().numpy()
                for i in range(imgs.size(0)):
                    pred_angle = compute_angle_from_keypoints(pred_kps[i])
                    true_angle = tgts[i, 5].item()
                    total_error += abs(pred_angle - true_angle)
                    count += 1
        model.train()
        return total_error / count if count > 0 else 0.0

    # Training loop.
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training"):
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        logging.info(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation"):
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        logging.info(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "yolov7_tiny_arrow_best.pth")
            logging.info(f"Epoch {epoch+1}: Best model saved with validation loss {best_val_loss:.4f}")
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= patience:
            logging.info(f"Early stopping triggered after {patience} epochs with no improvement.")
            break

        # Compute and log the mean absolute angle error.
        angle_mae = compute_angle_error(val_loader)
        logging.info(f"Epoch [{epoch+1}] Mean Absolute Angle Error: {angle_mae:.2f} degrees")

        visualize_predictions(model, val_loader, os.path.join(vis_dir, f"epoch_{epoch+1}"), num_images=5)

    total_training_time = time.time() - start_time
    logging.info("Training complete.")

    # Write training summary to a text file.
    summary_path = "training_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Training Summary\n")
        f.write("================\n")
        f.write(f"Model: YOLOv7-tiny modified for keypoint detection\n")
        f.write(f"Epochs: {epoch+1}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Image size: {img_size}x{img_size}\n")
        f.write(f"Training images: {len(train_dataset)}\n")
        f.write(f"Validation images: {len(val_dataset)}\n")
        f.write(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}\n")
        f.write(f"Total training time: {total_training_time:.2f} seconds\n")
    logging.info(f"Training summary saved to {summary_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Train a modified YOLOv7-tiny model for arrow keypoint detection with letterboxed inputs. "
                    "Outputs composite visualizations (original on top, predictions on bottom with red markers and text) "
                    "and logs a metric for angle error."
    )
    parser.add_argument('-tri', '--train_images', type=str, required=True, help="Directory for training images.")
    parser.add_argument('-tra', '--train_annotations', type=str, required=True, help="Directory for training annotations.")
    parser.add_argument('-vi', '--val_images', type=str, required=True, help="Directory for validation images.")
    parser.add_argument('-va', '--val_annotations', type=str, required=True, help="Directory for validation annotations.")
    parser.add_argument('-ne', '--num_epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('-bs', '--batch_size', type=int, default=16, help="Training batch size.")
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help="Learning rate.")
    parser.add_argument('-vd', '--vis_dir', type=str, default="YOLO_visualizations", help="Directory to save composite visualization images.")
    parser.add_argument('-is', '--img_size', type=int, default=640, help="Fixed image size (letterboxed) for training.")
    parser.add_argument('-p', '--patience', type=int, default=5, help="Early stopping patience (epochs with no improvement).")

    args = parser.parse_args()
    train_yolo_arrow(args.train_images, args.train_annotations,
                     args.val_images, args.val_annotations,
                     args.num_epochs, args.batch_size, args.learning_rate,
                     args.vis_dir, args.img_size, args.patience)

if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
