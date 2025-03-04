import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import argparse
import time
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def resize_image(image, target_size):
    """
    Resize image to target_size x target_size.

    Parameters:
        image (numpy.ndarray): Input image.
        target_size (int): Desired output size for both width and height.

    Returns:
        numpy.ndarray: Resized image.
    """
    return cv2.resize(image, (target_size, target_size))

class ArrowDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, img_size=640, transform=None):
        """
        Custom dataset for arrow images and YOLO-format annotations.

        Parameters:
            images_dir (str): Directory containing images.
            annotations_dir (str): Directory containing corresponding annotation files.
            img_size (int): Target image size (img_size x img_size).
            transform (callable, optional): Additional transform to apply on the image.
        """
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.img_size = img_size
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error loading image {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = resize_image(image, self.img_size)
        image = image.astype(np.float32) / 255.0  # Normalize to [0,1]

        ann_path = os.path.join(self.annotations_dir, self.image_files[idx].replace('.png', '.txt'))
        with open(ann_path, 'r') as f:
            line = f.readline().strip().split()
        cls_label = int(line[0])
        x_center, y_center, bbox_w, bbox_h, angle = map(float, line[1:])
        target = np.array([cls_label, x_center, y_center, bbox_w, bbox_h, angle], dtype=np.float32)

        if self.transform:
            image = self.transform(image)
        image = torch.from_numpy(image.transpose(2, 0, 1))  # (C, H, W)
        target = torch.from_numpy(target)
        return image, target

class YOLOv7Tiny(nn.Module):
    def __init__(self):
        super(YOLOv7Tiny, self).__init__()
        # Dummy backbone and detection head for illustration.
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Detection head outputs 6 channels: [tx, ty, tw, th, obj_score, angle]
        self.det_head = nn.Conv2d(16, 6, kernel_size=1)

    def forward(self, x):
        features = self.backbone(x)
        out = self.det_head(features)  # Shape: (batch, 6, H, W)
        # Aggregate spatial dimensions via mean (dummy implementation).
        out = out.mean(dim=[2,3])  # (batch, 6)
        return {"pred": out, "angle": out[:, 5]}

    def modify_for_angle(self):
        # Dummy function; assume the detection head already outputs an angle.
        pass

class YOLOLoss(nn.Module):
    def __init__(self):
        super(YOLOLoss, self).__init__()
        self.mse = nn.MSELoss()
        # In a full implementation, include losses for bbox regression, objectness, and classification.

    def forward(self, outputs, targets):
        # Dummy loss: only compute loss on angle regression.
        pred_angle = outputs["angle"]
        true_angle = targets[:, 5]
        loss_angle = self.mse(pred_angle, true_angle)
        return loss_angle

def visualize_predictions(model, val_loader, vis_dir, num_images=5):
    """
    Run inference on a few validation images, overlay the predicted angle on the images,
    and save the visualizations.

    Parameters:
        model (nn.Module): Trained model.
        val_loader (DataLoader): Validation DataLoader.
        vis_dir (str): Directory to save visualization images.
        num_images (int): Number of images to visualize.
    """
    os.makedirs(vis_dir, exist_ok=True)
    model.eval()
    images_visualized = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            outputs = model(images)
            pred_angles = outputs["angle"].cpu().numpy()
            for i in range(images.size(0)):
                img = images[i].cpu().numpy().transpose(1, 2, 0)
                img = (img * 255).astype(np.uint8)
                img = np.ascontiguousarray(img)
                angle_text = f"Predicted Angle: {pred_angles[i]:.2f}"
                cv2.putText(img, angle_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)
                out_path = os.path.join(vis_dir, f"vis_{images_visualized}.png")
                cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                images_visualized += 1
                if images_visualized >= num_images:
                    return

def train_yolo_arrow(train_images_dir, train_annotations_dir, val_images_dir, val_annotations_dir,
                     num_epochs, batch_size, learning_rate, vis_dir, img_size):
    """
    Train the modified YOLOv7-tiny model for arrow detection with angle regression.
    Image resizing to fixed dimensions is performed in the dataset loader.

    Parameters:
        train_images_dir (str): Directory for training images.
        train_annotations_dir (str): Directory for training annotations.
        val_images_dir (str): Directory for validation images.
        val_annotations_dir (str): Directory for validation annotations.
        num_epochs (int): Number of epochs.
        batch_size (int): Batch size.
        learning_rate (float): Learning rate.
        vis_dir (str): Directory to save visualization images.
        img_size (int): Fixed image size (img_size x img_size) for training.
    """
    train_dataset = ArrowDataset(train_images_dir, train_annotations_dir, img_size=img_size)
    val_dataset = ArrowDataset(val_images_dir, val_annotations_dir, img_size=img_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLOv7Tiny()
    model.modify_for_angle()
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = YOLOLoss()

    best_val_loss = float('inf')
    best_epoch = 0
    start_time = time.time()

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

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "yolov7_tiny_arrow_best.pth")
            logging.info(f"Epoch {epoch+1}: Best model saved with validation loss {best_val_loss:.4f}")

        visualize_predictions(model, val_loader, os.path.join(vis_dir, f"epoch_{epoch+1}"), num_images=5)

    total_training_time = time.time() - start_time
    logging.info("Training complete.")

    # Write training summary to a text file.
    summary_path = "training_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Training Summary\n")
        f.write("================\n")
        f.write(f"Model: YOLOv7-tiny with angle regression head\n")
        f.write(f"Epochs: {num_epochs}\n")
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
        description="Train a modified YOLOv7-tiny model for arrow detection with angle regression. "
                    "Images are resized to a fixed size in the training pipeline."
    )
    parser.add_argument('-tri', '--train_images', type=str, required=True, help="Directory for training images.")
    parser.add_argument('-tra', '--train_annotations', type=str, required=True, help="Directory for training annotations.")
    parser.add_argument('-vi', '--val_images', type=str, required=True, help="Directory for validation images.")
    parser.add_argument('-va', '--val_annotations', type=str, required=True, help="Directory for validation annotations.")
    parser.add_argument('-ne', '--num_epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('-bs', '--batch_size', type=int, default=16, help="Training batch size.")
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help="Learning rate.")
    parser.add_argument('-vd', '--vis_dir', type=str, default="YOLO_visualizations", help="Directory to save validation visualizations.")
    parser.add_argument('-is', '--img_size', type=int, default=640, help="Fixed image size (img_size x img_size) for training.")

    args = parser.parse_args()
    train_yolo_arrow(args.train_images, args.train_annotations,
                     args.val_images, args.val_annotations,
                     args.num_epochs, args.batch_size, args.learning_rate,
                     args.vis_dir, args.img_size)

if __name__ == "__main__":
    main()
