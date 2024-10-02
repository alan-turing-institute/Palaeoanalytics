import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random

# Ensure correct environment settings for encoding
os.environ['LC_ALL'] = 'en_US.UTF-8'
os.environ['LANG'] = 'en_US.UTF-8'

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# If using CUDA, set the seed for all GPU operations as well
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)  # For multi-GPU setups

# Define the CNN model for arrow detection
class ArrowDetectionCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(ArrowDetectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)  # Adjusted to match the output of conv4
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor for fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def check_class_mapping(dataset):
    """Check if the dataset class-to-index mapping is correct."""
    class_to_idx = dataset.class_to_idx
    if 'arrows' not in class_to_idx or 'non_arrows' not in class_to_idx:
        raise ValueError("Error: 'arrows' and 'non_arrows' folders are missing in the dataset.")
    print(f"Class mapping: {class_to_idx}")

    if class_to_idx['arrows'] != 0:
        raise ValueError("Error: 'arrows' class should be mapped to label 0.")
    if class_to_idx['non_arrows'] != 1:
        raise ValueError("Error: 'non_arrows' class should be mapped to label 1.")


# Helper function to unnormalize and display a grid of images
def imshow_grid(imgs, titles, num_cols=3):
    fig, axes = plt.subplots(1, num_cols, figsize=(15, 5))
    for i in range(num_cols):
        img = imgs[i] / 2 + 0.5  # Unnormalize the image
        np_img = img.numpy()  # Convert to numpy for plotting
        axes[i].imshow(np.transpose(np_img, (1, 2, 0)))  # Rearrange the dimensions for display
        axes[i].set_title(titles[i])
        axes[i].axis('off')
    plt.show()


def evaluate_model(model, dataloader, criterion, device):
    """Evaluate the model on the given dataset (train or test)"""
    model.eval()  # Set model to evaluation mode
    total = 0
    correct = 0
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(dataloader)
    return avg_loss, accuracy


def calculate_class_weights(dataset):
    """Calculate class weights for imbalanced datasets."""
    class_counts = [0, 0]  # [arrows, non_arrows]
    for _, label in dataset.imgs:
        class_counts[label] += 1

    # Calculate weights inversely proportional to the frequency of each class
    total_samples = sum(class_counts)
    class_weights = [total_samples / count for count in class_counts]

    return torch.tensor(class_weights).to(device), class_counts  # Return class_weights and class_counts


if __name__ == '__main__':
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up paths for your data
    TRAIN_DIR = 'train'
    TEST_DIR = 'test'

    # Data preprocessing and loading with augmentation for training
    transform_train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),  # Add data augmentation
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the training dataset
    print("Loading the training dataset...")
    try:
        train_dataset = torchvision.datasets.ImageFolder(root=TRAIN_DIR, transform=transform_train)
        print(f"Number of images in the training dataset: {len(train_dataset)}")
        if len(train_dataset) == 0:
            raise ValueError("Error: No images found in the training dataset.")
    except Exception as e:
        print(f"Error loading training dataset: {e}")
        exit(1)

    # Check class mappings in training set
    check_class_mapping(train_dataset)

    # DataLoader for batching
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Load the test dataset
    print("Loading the test dataset...")
    try:
        test_dataset = torchvision.datasets.ImageFolder(root=TEST_DIR, transform=transform_test)
        print(f"Number of images in the test dataset: {len(test_dataset)}")
        if len(test_dataset) == 0:
            raise ValueError("Error: No images found in the test dataset.")
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        exit(1)

    # Check class mappings in test set
    check_class_mapping(test_dataset)

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Calculate class weights to handle imbalance and get class counts
    class_weights, class_counts = calculate_class_weights(train_dataset)

    # Initialize the CNN
    model = ArrowDetectionCNN(num_classes=2).to(device)

    # Loss function with class weights to address imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Early stopping parameters
    patience = 3  # Number of epochs to wait for improvement
    best_loss = float('inf')
    patience_counter = 0

    # Run for multiple epochs
    num_epochs = 20  # Set to multiple epochs for training
    train_losses = []  # Store training losses
    test_losses = []  # Store test losses
    train_accuracies = []  # Store training accuracies
    test_accuracies = []  # Store test accuracies

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        print(f"Starting epoch {epoch + 1}/{num_epochs}")

        for i, (images, labels) in enumerate(train_loader):
            print(f"Epoch {epoch + 1}/{num_epochs}: Processing training batch {i + 1} of {len(train_loader)}")
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Evaluate on training and test data
        train_loss, train_accuracy = evaluate_model(model, train_loader, criterion, device)
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")
        print(f"Epoch {epoch + 1}/{num_epochs} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

        # Early stopping logic
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0  # Reset counter if improvement
            torch.save(model.state_dict(), 'generalized_model.pth')  # Save model with best performance based on validation/test data during training
        else:
            patience_counter += 1
            print(f"Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered. Training stopped.")
                break

    # Save the final model after training
    model_filename = 'last_epoch_model.pth'
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved as '{model_filename}'")

    # Testing the model with visual output (only showing 3 images in a grid)
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    max_images_to_show = 3  # Only show a maximum of 3 images
    images_shown = 0  # Counter to track number of images displayed

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect images and predictions for display
            if images_shown < max_images_to_show:
                imgs_to_show = []
                titles = []
                for j in range(min(max_images_to_show - images_shown, images.size(0))):  # Display up to 3 images
                    img = images[j].cpu()  # Move image to CPU for displaying
                    predicted_label = "arrows" if predicted[j].item() == 0 else "non_arrows"
                    true_label = "arrows" if labels[j].item() == 0 else "non_arrows"
                    title = f"Predicted: {predicted_label}, True: {true_label}"
                    imgs_to_show.append(img)
                    titles.append(title)
                    images_shown += 1

                imshow_grid(imgs_to_show, titles, num_cols=len(imgs_to_show))

            if images_shown >= max_images_to_show:
                break  # Stop after showing the maximum number of images

    # Print the overall accuracy of the model
    accuracy = 100 * correct / total
    print(f'Accuracy of the trained model on test images: {accuracy:.2f}%')


    # Summary of the training/testing process
    summary = f"""
    Training Summary:
    -------------------
    1. **Dataset Description**:
        - Number of training images: {len(train_dataset)}
        - Number of test images: {len(test_dataset)}
        - Class distribution in training set:
            - Arrows: {class_counts[0]} images
            - Non-arrows: {class_counts[1]} images

    2. **Class Imbalance Handling**:
        - Applied class weights to handle class imbalance:
            - Weight for arrows: {class_weights[0]:.4f}
            - Weight for non-arrows: {class_weights[1]:.4f}
        - Weights are based on the frequency of each class in the training dataset, and used in the loss function to balance learning between minority and majority classes.

    3. **Training Performance**:
        - Number of epochs: {epoch+1}
        - Training Losses per epoch: {train_losses}
        - Test Losses per epoch: {test_losses}
        - Training Accuracies per epoch: {train_accuracies}
        - Test Accuracies per epoch: {test_accuracies}
        - Final Test Accuracy: {accuracy:.2f}%

    4. **Early Stopping**:
        - Early stopping was triggered after no improvement in test loss for {patience} consecutive epochs.

    5. **Models Saved**:
        - Best generalized model (based on test loss): 'generalized_model.pth'
        - Final model after last epoch: 'last_epoch_model.pth'
    """

    # Print the summary to the console
    print(summary)

    # Save the summary to a text file
    with open('training_summary.txt', 'w') as f:
        f.write(summary)
    print("Training summary saved to 'training_summary.txt'")
