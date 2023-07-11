import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor, Normalize, Compose
import deeplabv3
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import dataloader
from pascal import VOCSegmentation
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

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
dataset = VOCSegmentation("data/VOCdevkit")

# Create a DataLoader with custom collate function
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)

# Calculate IoU
def calculate_iou(predicted_mask, target_mask):
    intersection = torch.logical_and(predicted_mask, target_mask).sum()
    union = torch.logical_or(predicted_mask, target_mask).sum()
    iou = intersection.float() / union.float()
    return iou

# Define your training loop
def train_model(model, criterion, optimizer, device, epochs):
    model.train()  # Set the model to train mode

    for epoch in range(epochs):
        running_loss = 0.0
        running_iou = 0.0

        # Wrap the dataloader with tqdm and add a progress bar
        with tqdm(dataloader, unit="batch") as t:
            t.set_description(f"Epoch [{epoch+1}/{epochs}]")

            # Iterate over the training dataset
            for inputs, labels in t:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Calculate IoU
                predicted_masks = torch.argmax(outputs, dim=1)
                iou = calculate_iou(predicted_masks, labels)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Update the running loss and IoU
                running_loss += loss.item() * inputs.size(0)
                running_iou += iou.item() * inputs.size(0)

                # Update the progress bar description
                t.set_postfix(loss=loss.item(), iou=iou.item())

        # Compute the epoch loss and IoU
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_iou = running_iou / len(dataloader.dataset)

        # Print the loss and IoU for this epoch
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, IoU: {epoch_iou:.4f}")

    return model

