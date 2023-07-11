import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
from PIL import Image
from scipy.io import loadmat
from torch.autograd import Variable
from torchvision import transforms

import deeplabv3
from pascal import VOCSegmentation
from utils import AverageMeter, inter_and_union



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


# Create a dataset using VOCSegmentation
dataset = VOCSegmentation('C:/Users/USER/Desktop/연구실/data/VOCdevkit',
        train=True, crop_size=513)

dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=True,
        pin_memory=True, num_workers=4)

# Calculate IoU
def calculate_iou(predicted_mask, target_mask):
    intersection = torch.logical_and(predicted_mask, target_mask).sum()
    union = torch.logical_or(predicted_mask, target_mask).sum()
    iou = intersection.float() / union.float()
    return iou
from tqdm import tqdm
# Define your training loop
def train_model(model, criterion, optimizer, device, epochs):
    model.train()  # Set the model to train mode

    for epoch in range(epochs):
        running_loss = 0.0
        running_iou = 0.0

        # Wrap the dataloader with tqdm and add a progress bar
        with tqdm(dataset_loader, unit="batch") as t:
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
        epoch_loss = running_loss / len(dataset_loader.dataset)
        epoch_iou = running_iou / len(dataset_loader.dataset)

        # Print the loss and IoU for this epoch
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, IoU: {epoch_iou:.4f}")

    return model

