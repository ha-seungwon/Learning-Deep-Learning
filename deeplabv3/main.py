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

if __name__ == '__main__':
    mp.freeze_support()
    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Create an instance of your model
    if args.model == 'deeplabv3_gcn':
        model = deeplabv3_gcn.model_load('sage',[3],[2])
    else:
        model = deeplabv3.model_load()
    

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)


    # Set the device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Move the model to the device
    model = model.to(device)


    # Code for creating and training the model
    trained_model = train.train_model(model, criterion, optimizer, device, args.epochs)
