import torch
import torch.nn.functional as F
import numpy as np
import deeplabv3_gcn
from pascal import VOCSegmentation
# Step 1: Load the trained model and set it to evaluation mode
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'path/to/your/trained/model.pth'  # Update with the path to your trained model
model = deeplabv3_gcn.model_load()  # Replace 'YourModelClass' with the actual class of your model
model.load_state_dict(torch.load(model_name, map_location=device))
model.to(device)
model.eval()
dataset = VOCSegmentation('C:/Users/dongj/Desktop/haseungwon/data/VOCdevkit', train=True, crop_size=513)
dataset_loader = torch.utils.data.DataLoader(
    dataset, batch_size=8, shuffle=True, pin_memory=True
)


# Step 2: Define a function to calculate the Intersection over Union (IoU) for each class
def calculate_iou(outputs, targets, num_classes):
    ious = []
    for cls in range(num_classes):
        pred = (outputs == cls)
        target = (targets == cls)
        intersection = (pred & target).sum().float()
        union = (pred | target).sum().float()
        iou = (intersection / (union + 1e-8)).item()  # Adding a small epsilon to avoid division by zero
        ious.append(iou)
    return ious

# Step 3: Test the model with the dataset and calculate IoU for each class
iou_per_class = np.zeros(len(VOCSegmentation.CLASSES))
num_samples = len(dataset)

with torch.no_grad():
    for i, (input_image, target_mask) in enumerate(dataset_loader):
        input_image = input_image.to(device)
        target_mask = target_mask.to(device)

        # Forward pass through the model
        output_mask = model(input_image)

        # Get the predicted classes by taking the argmax along the channel dimension
        _, predicted_classes = torch.max(output_mask, dim=1)

        # Calculate IoU for each class for this batch
        iou_per_batch = calculate_iou(predicted_classes.cpu(), target_mask.cpu(), len(VOCSegmentation.CLASSES))

        # Accumulate IoU per class
        iou_per_class += np.array(iou_per_batch)

# Average IoU across all samples
iou_per_class /= num_samples

# Print IoU for each class
for i, cls in enumerate(VOCSegmentation.CLASSES):
    print(f"IoU for class '{cls}': {iou_per_class[i]}")
