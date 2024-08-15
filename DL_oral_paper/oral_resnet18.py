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
from torchvision.models import ResNet18_Weights

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
    transforms.Resize((224, 224)),  # Resize to fit ResNet input size
    transforms.ToTensor()
])

# Some Hyperparameters
loader_batch_size = 16
num_workers = 0
learning_rate = 0.001
num_epochs = 20
patience = 3

# Load the training dataset with the initial transforms
train_dataset = ImageFolder(root=train_dir, transform=initial_transforms)
train_loader = DataLoader(train_dataset, batch_size=loader_batch_size,
                          shuffle=False, num_workers=num_workers)

# Function to calculate mean and std through GPU
def calculate_mean_std(loader, device):
    mean = torch.zeros(3).to(device)
    std = torch.zeros(3).to(device)
    total_images_count = 0

    for images, _ in loader:
        images = images.to(device)
        images_count_in_batch = images.size(0)
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
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for ResNet input size
    transforms.RandomRotation(25),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=(0.5, 1.0)),
    transforms.RandomAffine(degrees=0, shear=20),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# Load datasets with the transformations
train_dataset = ImageFolder(root=train_dir, transform=train_transforms)
val_dataset = ImageFolder(root=val_dir, transform=val_test_transforms)
test_dataset = ImageFolder(root=test_dir, transform=val_test_transforms)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=loader_batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=loader_batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=loader_batch_size, shuffle=False, num_workers=num_workers)

# Load the ResNet model
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# Modify the final layer to match the number of classes in the training dataset
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
best_val_loss = float('inf')
early_stopping_counter = 0

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = loss_func(outputs, labels)

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
    val_losses.append(val_loss)
    val_acc = corrects.double() / len(val_loader.dataset)
    val_accuracies.append(val_acc.item())
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    # Early Stopping Check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
        best_model_wts = model.state_dict()
        torch.save(best_model_wts, 'best_resnet18_model.pth')
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print("Early stopping triggered!")
            model.load_state_dict(best_model_wts)
            break

print("Training complete")

# Test phase
model.eval()
test_loss = 0.0
corrects = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        test_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        corrects += torch.sum(preds == labels.data)

test_loss = test_loss / len(test_loader.dataset)
test_acc = corrects.double() / len(test_loader.dataset)
print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

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
loss_accuracy_plot_path = 'training_validation_loss_acc_resnet.png'
plt.savefig(loss_accuracy_plot_path)
print(f"Plot saved to {loss_accuracy_plot_path}")

# Save the model after the last epoch
model_save_path = 'resnet18_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Duration Tracking
end_time = time.time()
elapsed_time = end_time - start_time

hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)

print(f"- Duration: {hours:02}:{minutes:02}:{seconds:02}")
