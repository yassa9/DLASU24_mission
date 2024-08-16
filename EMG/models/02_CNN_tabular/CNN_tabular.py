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
X_train = np.load('../../../X_train_tabular.npy')
y_train = np.load('../../../y_train_tabular.npy')

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
class cnn_model(nn.Module):
    def __init__(self, input_channels, output_size):
        super(cnn_model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(32, 64)
        self.dropout = nn.Dropout(0.3)  # reduced dropout rate
        self.fc2 = nn.Linear(64, output_size)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = torch.mean(x, dim=2)  # global average pooling
        x = self.dropout(self.leaky_relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# initializing the model
input_channels = X_train_tensor.shape[1]  # number of features (channels)
output_size = y_train_tensor.shape[1]
model = cnn_model(input_channels, output_size).to(device)

##################################################
##################################################

# hyperparameters
num_epochs = 10
batch_size = 256
learning_rate = 0.001
weight_decay = 1e-4  # L2 regularization

# loss function and optimizer
loss_func = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# split the data into train and test sets
train_ratio = 0.8
dataset_size = len(X_train_tensor)
train_size = int(train_ratio * dataset_size)
test_size = dataset_size - train_size

# reshape input data for CNN: [samples, channels, time_steps=1]
X_train_tensor = X_train_tensor.view(-1, input_channels, 1)

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
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()

        # input is already permuted [batch_size, channels, time_steps]
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
            # input is already permuted [batch_size, channels, time_steps]
            outputs = model(X_batch)
            loss = loss_func(outputs, y_batch)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    test_losses.append(avg_test_loss)
    print(f"Test Loss: {avg_test_loss:.4f}")

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
plt.savefig('../../plots/cnn_loss_plot.png')
plt.show()

# save the model
model_path = 'cnn_model.pth'
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
