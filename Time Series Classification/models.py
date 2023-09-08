import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, num_classes)
        self.linear = torch.nn.Linear(1250, num_classes)  # Add a linear layer for classification

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = x.view(x.size(0), -1)  # Flatten before fully connected layers
        x = self.linear(x)  # Apply the linear layer for classification
        return x

class GCNModel2(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes,timesteps):
        super(GCNModel2, self).__init__()
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.conv3 = GCNConv(hidden_size, hidden_size)
        self.conv4 = GCNConv(hidden_size, hidden_size)
        self.conv5 = GCNConv(hidden_size, num_classes)
        self.linear = torch.nn.Linear(25*timesteps, num_classes)  # Add a linear layer for classification

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = self.conv4(x, edge_index)
        x = self.conv5(x, edge_index)
        x = x.view(x.size(0), -1)  # Flatten before fully connected layers
        x = self.linear(x)  # Apply the linear layer for classification
        return x
class ConvAutoencoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose1d(32, input_size, kernel_size=3, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.permute(0, 2, 1)
        return x

    import torch.nn as nn

class AutoConvWithGenerator(nn.Module):
    def __init__(self, input_size, latent_size):
        super(AutoConvWithGenerator, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_size, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(16, input_size, kernel_size=2, stride=2)
        )

        # Generator (After Encoder)
        self.generator = nn.Sequential(
            nn.Linear(latent_size, 64),  # Adjust the layer sizes as needed
            nn.ReLU(),
            nn.Linear(64, input_size)
        )

        self.latent_size = latent_size

    def forward(self, x):
        # Encoder
        x = self.encoder(x)

        # Generator
        latent_vector = torch.randn(x.size(0), self.latent_size).to(x.device)
        generated_output = self.generator(latent_vector)

        # Decoder
        x = self.decoder(x)

        return x, generated_output



class Conv1DModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Conv1DModel, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(50*64, 64)  # Adjust input size based on pooling
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Permute for Conv1d input format (batch, channels, sequence length)
        x = self.conv1(x)
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
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=5)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)

        out = self.fc(out[:, -1, :])  # Get the output from the last time step
        return out




class AutoConvWithGenerator(nn.Module):
    def __init__(self, input_size, latent_size, num_classes,edge_index,device):
        super(AutoConvWithGenerator, self).__init__()
        self.device = device
        self.edge_index = edge_index
        # Encoder

        self.encoder = nn.Sequential(
            nn.Conv1d(input_size, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(16, input_size, kernel_size=2, stride=2)
        )

        # Generator (After Encoder)
        self.generator = nn.Sequential(
            nn.Linear(latent_size, 64),  # Adjust the layer sizes as needed
            nn.ReLU(),
            nn.Linear(64, latent_size)  # Adjust to match the input size
        )
        self.gcn1 = GCNConv(latent_size, latent_size)
        self.gcn2 = GCNConv(latent_size, latent_size)
        self.gcn3 = GCNConv(latent_size, latent_size)
        self.gcn4 = GCNConv(latent_size, latent_size)






        self.latent_size = latent_size
        self.linear = torch.nn.Linear(latent_size, num_classes)


    def forward(self, x):
        x = x.permute(0, 2, 1)

        # Encoder
        encoded = self.encoder(x)

        # Generator
        latent_vector = torch.randn(x.size(0), self.latent_size).to(x.device)
        #generated_output = self.generator(latent_vector)

        generated_output = self.gcn1(latent_vector, self.edge_index.to(x.device))
        generated_output = self.gcn2(generated_output, self.edge_index.to(x.device))
        generated_output = self.gcn3(generated_output, self.edge_index.to(x.device))
        generated_output = self.gcn4(generated_output, self.edge_index.to(x.device))

        generator_label_output = self.linear(generated_output)

        # Decoder
        decoded = self.decoder(encoded)
        decoded = decoded.permute(0, 2, 1)  # Adjust output permutation if needed

        return decoded, generator_label_output

