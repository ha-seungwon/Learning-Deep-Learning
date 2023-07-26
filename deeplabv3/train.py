import torch
from torchvision import transforms
from tqdm import tqdm
from torch.autograd import Variable
from pascal import VOCSegmentation
import warnings
import wandb
import arguments as args


warnings.filterwarnings("ignore")

# Create a dataset using VOCSegmentation
dataset = VOCSegmentation('C:/Users/dongj/Desktop/haseungwon/data/VOCdevkit', train=True, crop_size=513)

dataset_loader = torch.utils.data.DataLoader(
    dataset, batch_size=8, shuffle=True, pin_memory=True
)

# Calculate IoU
def calculate_iou(predicted_mask, target_mask):
    intersection = torch.logical_and(predicted_mask, target_mask).sum()
    union = torch.logical_or(predicted_mask, target_mask).sum()
    iou = intersection.float() / union.float()
    iou = iou * 100
    return iou

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your training loop
def train_model(model, criterion, optimizer, epochs):
    wandb.init(project="GNN-Segmentation-seungwon")
    model.train()  # Set the model to train mode

    for epoch in range(epochs):
        running_loss = 0.0
        running_iou = 0.0

        # Wrap the dataloader with tqdm and add a progress bar
        with tqdm(dataset_loader, unit="batch") as t:
            t.set_description(f"Epoch [{epoch + 1}/{epochs}]")

            # Iterate over the training dataset
            for inputs, labels in t:

                # Zero the gradients
                optimizer.zero_grad()

                inputs = Variable(inputs.to(device))
                labels = Variable(labels.to(device)).long()
                outputs = model(inputs).float()

                loss = criterion(outputs, labels)
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
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, IoU: {epoch_iou:.4f}")
        wandb.log({"iou": epoch_iou, "loss": epoch_loss})

    wandb.finish()
    return model
