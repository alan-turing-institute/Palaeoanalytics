import os
import locale
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Set locale environment variables
os.environ['LC_ALL'] = 'en_US.UTF-8'
os.environ['LANG'] = 'en_US.UTF-8'
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

# Define the CNN model for arrow detection (same as the one used in training)
class ArrowDetectionCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(ArrowDetectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def imshow(img, title):
    """Helper function to display an image with a title."""
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Path to the test data directory and the saved model file
    TEST_DIR = 'test'
    MODEL_PATH = 'arrow_detection_cnn.pth'
    CLASS_NAMES = ['non_arrows', 'arrows']

    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the test dataset
    test_dataset = torchvision.datasets.ImageFolder(root=TEST_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Initialize the model and load the trained weights
    model = ArrowDetectionCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()  # Set the model to evaluation mode

    # Evaluate the model on the test set
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store predictions and labels for visualization
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

    # Save predictions and labels for later visualization
    results_path = 'test_results.pth'
    torch.save({'predictions': all_preds, 'labels': all_labels}, results_path)
    print(f"Results saved to {results_path}")

    # Visualize some test images with their predicted and actual labels
    # Uncomment the block below to visualize without rerunning the entire script

    # %%
    # Visualization Block - can be run independently after the above code is executed

    # Load saved predictions and labels
    results = torch.load(results_path)
    all_preds = results['predictions']
    all_labels = results['labels']

    # Load the test dataset for visualization purposes
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    # Get model predictions
    outputs = model(images.to(device))
    _, predicted = torch.max(outputs, 1)

    # Convert back to CPU for visualization
    images, labels, predicted = images.cpu(), labels.cpu(), predicted.cpu()

    # Display a few images with their predicted and actual labels
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 3))
    for i in range(5):
        ax = axes[i]
        ax.imshow(np.transpose(images[i].numpy() / 2 + 0.5, (1, 2, 0)))  # Unnormalize and transpose to display
        ax.set_title(f'Pred: {CLASS_NAMES[predicted[i]]}\nActual: {CLASS_NAMES[labels[i]]}')
        ax.axis('off')

    plt.show()
