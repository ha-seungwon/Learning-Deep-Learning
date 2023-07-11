import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor, Normalize, Compose
import deeplabv3
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import dataloader
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from torch.utils.data import DataLoader

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Create an instance of your model
model = deeplabv3.model_load()

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the device
model = model.to(device)

# Define your training loop
def train_model(model, dataloader, criterion, optimizer, device, num_epochs):
    model.train()  # Set the model to train mode

    for epoch in range(num_epochs):
        running_loss = 0.0

        # Iterate over the training dataset
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            print("outputs",outputs.size())
            print("labels",labels.size())
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update the running loss
            running_loss += loss.item() * inputs.size(0)

        # Compute the epoch loss
        epoch_loss = running_loss / len(dataloader.dataset)

        # Print the loss for this epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")



# Define your image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Custom collate function
def custom_collate(batch):
    images = [transform(item[0]) for item in batch]
    masks = [transform(item[1]) for item in batch]
    return torch.stack(images, dim=0), torch.stack(masks, dim=0)

# Create a dataset using VOCSegmentation
dataset = VOCSegmentation("C:/Users/USER/Desktop/연구실/data", transform=None)

# Create a DataLoader with custom collate function
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)

num_epochs = 10
train_model(model, dataloader, criterion, optimizer, device, num_epochs)
