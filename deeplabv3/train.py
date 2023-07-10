import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor, Normalize, Compose
import deeplabv3
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import dataloader

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



dataset=dataloader.CocoDataset("/Users/haseung-won/Desktop/학교/연구실/data/coco_minitrain_25k")
num_epochs = 10
train_model(model, dataset, criterion, optimizer, device, num_epochs)
