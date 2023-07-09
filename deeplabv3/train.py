import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor, Normalize, Compose
import deeplabv3
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

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


class ImageSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load the image and mask using PIL
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Convert to grayscale

        # Apply transformations to the image
        if self.transform is not None:
            image = self.transform(image)

        # Apply transformations to the mask without normalization
        if self.transform is not None:
            mask_transform = Compose([
                ToTensor()  # Converts PIL image to torch.Tensor
            ])
            mask = mask_transform(mask)

        return image, mask.squeeze(0).long()

# Load the CSV file containing image and mask paths
df = pd.read_csv('./archive/meta_data.csv')
image_path = df['image'].tolist()
mask_path = df['mask'].tolist()

# Prepend the directory path to the file names
image_dir = 'archive/Forest Segmented/Forest Segmented/images/'
mask_dir = 'archive/Forest Segmented/Forest Segmented/masks/'
image_path = [image_dir + path for path in image_path]
mask_path = [mask_dir + path for path in mask_path]

transform = Compose([
    ToTensor(),  # Converts PIL image to torch.Tensor
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
])

# Create an instance of the dataset
dataset = ImageSegmentationDataset(image_path, mask_path, transform=transform)

# Create a data loader for training
batch_size = 16
shuffle = True
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

num_epochs = 10
train_model(model, train_dataloader, criterion, optimizer, device, num_epochs)
