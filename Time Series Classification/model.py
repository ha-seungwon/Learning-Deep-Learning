import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, num_classes)
        self.linear = torch.nn.Linear(250, num_classes)  # Add a linear layer for classification

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = x.view(x.size(0), -1)  # Flatten before fully connected layers
        x = self.linear(x)  # Apply the linear layer for classification
        return x

class Conv1DModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Conv1DModel, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128, 64)  # Adjust input size based on pooling
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        print(x.size())
        x = x.permute(0, 2, 1)  # Permute for Conv1d input format (batch, channels, sequence length)
        print(x.size())
        x = self.conv1(x)
        print(x.size())
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten before fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        return x



class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Get the output from the last time step
        return out