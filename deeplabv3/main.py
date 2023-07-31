import deeplabv3
import train
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import deeplabv3
import deeplabv3_gcn
warnings.filterwarnings("ignore")
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms
from argument import args



if __name__ == '__main__':
    mp.freeze_support()

    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    print(args.model)
    # Create an instance of your model
    if args.model == 'deeplabv3_gcn':
        model = deeplabv3_gcn.model_load('sage', [3], [2])
    else:
        model = deeplabv3.model_load()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    print(args.lr)
    # Set the device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model to the device
    model = model.to(device)

    # Define data augmentation transformation
    transform = transforms.Compose([
        transforms.RandomResizedCrop(513, scale=(0.5, 2.0)),
        transforms.RandomHorizontalFlip(),
    ])

    # Define the learning rate policy (poly learning rate policy)
    max_iter = args.epochs * len(train.dataset_loader)  # Total number of training iterations
    scheduler = LambdaLR(optimizer, lr_lambda=lambda iter: (1 - iter / max_iter) ** 0.9)

    # Code for creating and training the model
    trained_model = train.train_model(model, criterion, optimizer, scheduler, transform, device, args.epochs, max_iter, args.initial_lr)
