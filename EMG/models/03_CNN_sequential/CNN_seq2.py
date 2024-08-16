import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device Used: {device}")

# start time tracking
start_time = time.time()

# loading data
X_train = np.load('../../../X_train_padding.npy')
y_train = np.load('../../../y_train_padding.npy')

# convert into tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)

print(f"X_train_tensor is on {X_train_tensor.device}")
print(f"y_train_tensor is on {y_train_tensor.device}")

print(f"Data shape: {X_train_tensor.shape}")
print(f"Labels shape: {y_train_tensor.shape}")

##################################################
##################################################

# model architecture
class CNN_SequentialModel(nn.Module):
    def __init__(self, input_channels, output_size):
        super(CNN_SequentialModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, output_size)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))

        # swap dimensions for Linear layer: [batch_size, time_steps, features]
        x = x.permute(0, 2, 1)
        x = self.dropout(self.leaky_relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# initializing the model
input_channels = X_train_tensor.shape[2]  # number of features (channels)
output_size = y_train_tensor.shape[2]  # number of outputs per time step
model = CNN_SequentialModel(input_channels, output_size).to(device)

##################################################
##################################################

# hyperparameters
num_epochs = 10
batch_size = 32
learning_rate = 1e-4
weight_decay = 1e-5 
patience = 3

# loss function and optimizer
loss_func = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# split the data into train and test sets
train_ratio = 0.8
dataset_size = len(X_train_tensor)
train_size = int(train_ratio * dataset_size)
test_size = dataset_size - train_size

# reshape input data for CNN: [samples, channels, time_steps]
X_train_tensor = X_train_tensor.permute(0, 2, 1)  # [samples, features, time_steps]

# creating a dataset
dataset = TensorDataset(X_train_tensor, y_train_tensor)

# splitting the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# creating DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# lists to store loss values
train_losses = []
test_losses = []

# early stopping variables
best_test_loss = float('inf')
epochs_no_improve = 0

##################################################
##################################################

# training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = loss_func(outputs, y_batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}")

    # evaluating the model
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = loss_func(outputs, y_batch)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    test_losses.append(avg_test_loss)
    print(f"Test Loss: {avg_test_loss:.4f}")

    # early stopping  checking
    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print("Early stopping triggered")
            break

##################################################
##################################################

# plotting
plt.figure()
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Test Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('../../plots/cnn_seq_loss_plot.png')
plt.show()

model_path = 'CNN_seq.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved: {model_path}")

##################################################
##################################################

# script duration tracking
end_time = time.time()
elapsed_time = end_time - start_time

hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)

print(f"Total time taken: {hours:02}:{minutes:02}:{seconds:02}")
