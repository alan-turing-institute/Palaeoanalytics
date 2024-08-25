import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import cv2
from PIL import Image
import numpy as np

os.environ['LC_ALL'] = 'en_US.UTF-8'
os.environ['LANG'] = 'en_US.UTF-8'

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
        x = x.view(x.size(0), -1)  # Ensure batch size remains consistent
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up paths for your data (Assuming you have 'train' and 'test' folders with subfolders for each class)
    TRAIN_DIR = 'train'
    TEST_DIR = 'test'

    # Data preprocessing and loading
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.ImageFolder(root=TRAIN_DIR, transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(root=TEST_DIR, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Initialize the model, loss function, and optimizer
    model = ArrowDetectionCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}")
                running_loss = 0.0

    print("Training completed")
    torch.save(model.state_dict(), "arrow_detection_cnn.pth")

    # Evaluate the model on the test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

    # Function to load arrow templates for contour matching
    def load_templates(template_dir):
        templates = []
        for filename in os.listdir(template_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                template_path = os.path.join(template_dir, filename)
                template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                _, binary_template = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY)
                templates.append(binary_template)
        return templates

    arrow_templates = load_templates('arrow_templates/')

    # Function to match detected arrows with templates
    def match_template(input_image, templates):
        detected_contours = []
        for template in templates:
            result = cv2.matchTemplate(input_image, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > 0.8:  # Threshold to consider a match
                detected_contours.append((max_loc, template.shape))
        return detected_contours

    # Function to perform detection using the CNN and refine it with contour matching
    def detect_arrows(image_path, model, templates):
        model.eval()
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)

        if predicted.item() == 1:  # Assuming '1' represents arrows
            print("Arrow detected by CNN")
            image_np = np.array(Image.open(image_path).convert('L'))
            _, binary_image = cv2.threshold(image_np, 127, 255, cv2.THRESH_BINARY)
            contours = match_template(binary_image, templates)
            if contours:
                print("Contour match found")
                return contours
            else:
                print("No contour match found, false positive")
                return None
        else:
            print("No arrow detected")
            return None

    # Example usage
    # Assuming 'image.jpg' is the test image and 'arrow_templates/' contains the arrow templates
    detected_contours = detect_arrows('image.jpg', model, arrow_templates)
    if detected_contours:
        print("Arrows detected and refined with contour matching")
    else:
        print("No arrows detected or false positives removed")
