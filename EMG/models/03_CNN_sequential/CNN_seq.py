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
class CNN_Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNN_Model, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        self.relu = nn.ReLU()
        
        # dynamically calculate the size of the flattened layer
        self._to_linear = None
        self.convs(torch.randn(1, input_size, 12246))  # Dummy forward pass to calculate the flattened size
        
        self.fc1 = nn.Linear(self._to_linear, 256)
        self.fc2 = nn.Linear(256, output_size * 12246)  # Predict for each time step
        
    def convs(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        if self._to_linear is None:
            self._to_linear = x.numel()
        return x

    def forward(self, x):
        x = x.transpose(1, 2)  # From [batch_size, seq_len, input_size] to [batch_size, input_size, seq_len]
        x = self.convs(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Reshape to match the output format: [batch_size, seq_len, output_size]
        x = x.view(x.size(0), 12246, -1)
        
        return x

# initializing the model
input_size = X_train_tensor.shape[2]  # Assuming shape is (batch_size, sequence_length, features)
output_size = y_train_tensor.shape[2]  # Output should match the label dimension 14
model = CNN_Model(input_size, output_size).to(device)

##################################################
##################################################

# hyperparameters
num_epochs = 10
batch_size = 4
learning_rate = 0.001

# loss function and optimizer
loss_func = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# split the data into train and test sets
train_ratio = 0.8
dataset_size = len(X_train_tensor)
train_size = int(train_ratio * dataset_size)
test_size = dataset_size - train_size

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

##################################################
##################################################

# training loop
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()

        outputs = model(X_batch)
        loss = loss_func(outputs, y_batch)

        loss.backward()
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

    epoch_end_time = time.time()
    epoch_elapsed_time = epoch_end_time - epoch_start_time

    hours = int(epoch_elapsed_time // 3600)
    minutes = int((epoch_elapsed_time % 3600) // 60)
    seconds = int(epoch_elapsed_time % 60)

    print(f"Total time taken: {hours:02}:{minutes:02}:{seconds:02}")

##################################################
##################################################

# plotting the loss curves
plt.figure()
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Test Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('../../plots/CNN_loss_plot.png')
plt.show()

# save the model
model_path = 'CNN_model.pth'
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

