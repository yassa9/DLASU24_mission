import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torchvision.models import Inception_V3_Weights

# Basic setup
start_time = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"- Current Device Used: {device}")

# Paths to data directories
train_dir = '../oral_data/Training'
val_dir = '../oral_data/Validation'
test_dir = '../oral_data/Testing'

# Initial transformations for calculating mean and std
initial_transforms = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor()
])

# Load the training dataset with the initial transforms
train_dataset = ImageFolder(root=train_dir, transform=initial_transforms)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=4)

# Function to calculate mean and std through my GPU
def calculate_mean_std(loader, device):
    mean = torch.zeros(3).to(device)
    std = torch.zeros(3).to(device)
    total_images_count = 0

    for images, _ in loader:
        images = images.to(device)
        images_count_in_batch = images.size(0)
        # img_cnt in batch, number of channels = 3, flattening
        images = images.view(images_count_in_batch, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += images_count_in_batch

    mean /= total_images_count
    std /= total_images_count
    return mean, std

# Calculate mean and std for the training dataset on GPU
mean, std = calculate_mean_std(train_loader, device)
print('- Calculated Mean:', mean)
print('- Calculated Std:', std)

# Data augmentation transformations
class RandomChannelShift:
    def __init__(self, shift_value):
        self.shift_value = shift_value

    def __call__(self, tensor):
        # Generate a random shift for each channel
        shifts = torch.empty(3).uniform_(-self.shift_value, self.shift_value).to(tensor.device)
        tensor = tensor + shifts.view(-1, 1, 1)
        # Clip the values to ensure they remain between 0 and 1
        tensor = torch.clamp(tensor, 0, 1)
        return tensor

train_transforms = transforms.Compose([
    transforms.Resize((299, 299)),  # Resize to the required input size
    transforms.RandomRotation(25),  # Rotate the image by up to 25 degrees
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.RandomResizedCrop(299, scale=(0.8, 1.0)),  # Randomly resize and crop
    transforms.ColorJitter(brightness=(0.5, 1.0)),  # Adjust brightness
    transforms.RandomAffine(degrees=0, shear=20),  # Apply shear transformation
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Width and height shifts
    transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),  # Random zoom
    #RandomChannelShift(0.05),  # Apply channel shift
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=mean, std=std),  # Normalize pixel values to range [0, 1]
])

# Validation and test transforms (no augmentation, just resizing and normalization)
val_test_transforms = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# Some Hyperparameters
loader_batch_size = 16
num_workers = 0
learning_rate = 0.001
num_epochs = 10

# Loading datasets => transformers
train_dataset = ImageFolder(root=train_dir, transform=train_transforms)
val_dataset = ImageFolder(root=val_dir, transform=val_test_transforms)
test_dataset = ImageFolder(root=test_dir, transform=val_test_transforms)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=loader_batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=loader_batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=loader_batch_size, shuffle=False, num_workers=num_workers)

# Loading the Inception model
model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)

# Modifing model's final layer to match the number of classes in training dataset
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Define loss function and optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Lists to store losses and accuracy
train_losses = []
val_losses = []
val_accuracies = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs, aux_outputs = model(inputs)
        loss1 = loss_func(outputs, labels)
        loss2 = loss_func(aux_outputs, labels)
        loss = loss1 + 0.4 * loss2

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Validation phase
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    corrects = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)

    val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(val_loss)  # Save validation loss
    val_acc = corrects.double() / len(val_loader.dataset)
    val_accuracies.append(val_acc.item())  # Save validation accuracy
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

print("Training complete")

# Plot the training and validation loss
plt.figure(figsize=(12, 5))

# Subplot for Training and Validation Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Subplot for Validation Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()

# Adjust layout and save the plots
plt.tight_layout()

# Save the figure to a file
loss_accuracy_plot_path = 'training_validation_loss_accuracy.png'
plt.savefig(loss_accuracy_plot_path)
print(f"Plot saved to {loss_accuracy_plot_path}")


# Save the model after the last epoch
model_save_path = 'oral_inception_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")


# Duration Tracking
end_time = time.time()
elapsed_time = end_time - start_time

hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)

print(f"- Duration: {hours:02}:{minutes:02}:{seconds:02}")
