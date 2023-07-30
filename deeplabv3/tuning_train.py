import torch
from torchvision import transforms
from tqdm import tqdm
from torch.autograd import Variable
from pascal import VOCSegmentation
import warnings
import wandb
import arguments as args
import numpy as np
from utils import AverageMeter, inter_and_union

warnings.filterwarnings("ignore")

# Create a dataset using VOCSegmentation
dataset = VOCSegmentation('C:/Users/dongj/Desktop/haseungwon/data/VOCdevkit', train=True, crop_size=513)

valid_dataset = VOCSegmentation('data/VOCdevkit',train=False, crop_size=513)


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
    miou_buf=[]
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

                #iou = calculate_iou(predicted_masks, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Update the running loss and IoU
                running_loss += loss.item() * inputs.size(0)
                #running_iou += iou.item() * inputs.size(0)

                # Update the progress bar description
                t.set_postfix(loss=loss.item())

                # validation with test dataset
        with torch.no_grad():
            model.eval()
            inter_meter = AverageMeter()
            union_meter = AverageMeter()

            for i in tqdm(range(len(valid_dataset))):
                valid_inputs, valid_target = valid_dataset[i]
                valid_inputs = Variable(valid_inputs.cuda())
                valid_outputs = model(valid_inputs.unsqueeze(0))
                _, valid_pred = torch.max(valid_outputs, 1)
                valid_pred = valid_pred.data.cpu().numpy().squeeze().astype(np.uint8)
                mask = valid_target.numpy().astype(np.uint8)

                inter, union = inter_and_union(valid_pred, mask, len(valid_dataset.CLASSES))
                inter_meter.update(inter)
                union_meter.update(union)

            iou = inter_meter.sum / (union_meter.sum + 1e-10)
            miou = iou.mean()
            miou_buf.append(miou)


        # Compute the epoch loss and IoU
        epoch_loss = running_loss / len(dataset_loader.dataset)
        print('epoch: {0}\t'
            'loss: {2:.4f}\t'
            'miou: {3:.2f}'.format(
            epoch, epoch_loss, miou * 100))


        
        #epoch_iou = running_iou / len(dataset_loader.dataset)

        # Print the loss and IoU for this epoch
        #print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, IoU: {epoch_iou:.4f}")
        wandb.log({"iou": miou, "loss": epoch_loss})

    wandb.finish()
    return max(miou_buf)
