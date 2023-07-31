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
import arguments as args
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms
import deeplabv3_gcn_encoder
import tuning_train

if __name__ == '__main__':
    mp.freeze_support()

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Create an instance of your model

    

    model_atrous_rate_list=[0,1,3]
    model_gcnn_rate_list=[[8,16],[2,4,8,16],[3,6,9]]

    for atrous in model_atrous_rate_list:
        for gcn in model_gcnn_rate_list:
            if atrous ==0:

                model = deeplabv3_gcn_encoder.model_load('sage', [], gcn)
            else:
                model= deeplabv3_gcn_encoder.model_load('sage', atrous, gcn)

            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

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
            max_miou = tuning_train.train_model(model, criterion, optimizer, scheduler, transform, device, args.epochs, max_iter, args.initial_lr)
            print(f'atrous: {atrous}, gnc: {gcn} model train max_miou: {max_miou}')
