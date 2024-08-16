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
class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(CNN_LSTM_Model, self).__init__()
        
        # CNN layers with padding to maintain sequence length
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # LSTM layers
        self.lstm = nn.LSTM(128, hidden_size, num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        x = x.permute(0, 2, 1)  # Switch to (batch_size, input_size, sequence_length) for CNN
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        
        x = x.permute(0, 2, 1)  # Switch back to (batch_size, sequence_length, channels) for LSTM
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
        out, _ = self.lstm(x, (h_0, c_0))  # LSTM expects inputs of shape (batch_size, seq_len, input_size)
        out = self.fc(out)
        return out

# initializing the model
input_size = X_train_tensor.shape[2]  # Features per timestep
hidden_size = 128
num_layers = 2
output_size = y_train_tensor.shape[2]  # Output features per timestep
model = CNN_LSTM_Model(input_size, hidden_size, output_size, num_layers).to(device)

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
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_batch)
        loss = loss_func(outputs, y_batch)

        # Backward pass and optimization
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
plt.savefig('../../plots/CNN_LSTM_loss_plot.png')
plt.show()

# save the model
model_path = 'CNN_LSTM_model.pth'
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

