import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from keras.utils import to_categorical
import models
import random
import os
from arguments import args
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)


seed_everything(42)  # Seed 고정

data=pd.read_csv('../PAMAP2_Dataset/train.csv')
test_data=pd.read_csv('../PAMAP2_Dataset/test.csv')

data = data.fillna(0)
test_data = test_data.fillna(0)
X = data.drop(columns=['activityID'])
y = data['activityID']

test_X= test_data.drop(columns=['activityID'])
test_y= test_data['activityID']

num_classes =25

timesteps = 100
step_size = 20

X_train = []
y_train = []

X_test = []
y_test = []

# 좌우로 10개의 타임스텝을 겹치게 하려면 step_size를 10으로 설정

for i in range(0, len(X) - timesteps + 1, step_size):
    X_sequence = X.iloc[i:i + timesteps, :].values
    y_label = y.iloc[i + timesteps - 1]  # 시퀀스의 마지막 레이블 가져오기
    X_train.append(X_sequence)
    y_train.append(y_label)

valid_index_range = len(test_X) - timesteps + 1

for i in range(0, valid_index_range, step_size):
    X_sequence = test_X.iloc[i:i + timesteps, :].values
    y_label = test_y.iloc[i + timesteps - 1]  # 시퀀스의 마지막 레이블 가져오기
    X_test.append(X_sequence)
    y_test.append(y_label)


X_train_lstm = np.array(X_train)
y_train_lstm = np.array(y_train)

X_test_lstm = np.array(X_test)
y_test_lstm = np.array(y_test)


# Now you can proceed with multi-label encoding
y_train_encoded = np.array([to_categorical(labels, num_classes=num_classes) for labels in y_train])
y_train_encoded_flattened = y_train_encoded.reshape(-1, num_classes)

y_test_encoded = np.array([to_categorical(labels, num_classes=num_classes) for labels in y_test])
y_test_encoded_flattened = y_test_encoded.reshape(-1, num_classes)

# Assuming X_train_lstm, y_train_encoded_flattened are already prepared
# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_encoded_flattened_tensor = torch.tensor(y_train_encoded_flattened, dtype=torch.float32)


X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_encoded_flattened_tensor = torch.tensor(y_test_encoded_flattened, dtype=torch.float32)
# Create a custom dataset
class CustomDataset(Dataset):
    def __init__(self, X, y, edge_index):
        self.X = X
        self.y = y
        self.edge_index = edge_index

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
print("X_train_lstm",X_train_lstm.shape)
num_nodes = X_train_lstm.shape[1]

# Create an adjacency matrix with edges connecting each node to its neighbors
adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

# Connect each node to its immediate neighbors (adjust as needed)
for i in range(num_nodes):
    if i > 0:
        adjacency_matrix[i, i - 1] = 1.0
    if i < num_nodes - 1:
        adjacency_matrix[i, i + 1] = 1.0

# Convert the adjacency matrix to a sparse tensor
edge_index = torch.tensor(np.array(np.where(adjacency_matrix == 1)), dtype=torch.long).to(device)

# Create instances of custom dataset
custom_dataset = CustomDataset(X_train_tensor, y_train_encoded_flattened_tensor,edge_index)


# Create data loaders
batch_size = 64
train_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# Assuming you have a validation dataset named 'val_dataset'
val_dataset = CustomDataset(X_test_tensor, y_test_encoded_flattened_tensor, edge_index)  # X_val_tensor, y_val_encoded_flattened_tensor, edge_index_val은 검증 데이터에 대한 텐서와 엣지 인덱스입니다.
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Initialize model and hyperparameters
input_size = X_train_tensor.shape[2]
hidden_size = 30
num_classes = 25

model_name=args.model_name

if model_name=='LSTM':
    model=models.LSTMModel(input_size, hidden_size, num_classes).to(device)
    lr = 0.00001
elif model_name=='Conv1D':
    model=models.Conv1DModel(input_size, num_classes).to(device)
    lr = 0.0001
elif model_name=='GCN':
    model=models.GCNModel(input_size,hidden_size, num_classes).to(device)
    edge_index=custom_dataset.edge_index
    edge_index=edge_index.to(device)
    lr = 0.0001
elif model_name=='GCN2':
    model=models.GCNModel2(input_size,hidden_size, num_classes).to(device)
    edge_index=custom_dataset.edge_index
    edge_index=edge_index.to(device)
    lr = 0.0001
elif model_name=="AutoConv":
    lr = 0.0001
    latent_size = 16  # Adjust as needed
    model = models.ConvAutoencoder(input_size, latent_size).to(device)
elif model_name=='Generate':
    lr = 0.001
    latent_size = 16  # Adjust as needed
    model = models.AutoConvWithGenerator(input_size, latent_size,num_classes).to(device)

print("model_name : ",model_name,"input_size : ",input_size,"hidden_size : ",hidden_size,"num_classes : ",num_classes)
criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr=lr)

from tqdm import tqdm  # Import tqdm library


def some_custom_loss(generator_label_output,target):
    # Define your custom loss here
    # For example, you can use Mean Squared Error (MSE) loss
    mse_loss = nn.MSELoss()
    loss = mse_loss(generator_label_output, target)  # Define 'target' according to your task
    return loss


# Training loop
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}] Training", leave=False)

    for inputs, labels in train_loader_tqdm:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        if 'GCN' in model_name:
            outputs = model(inputs,edge_index)
            loss = criterion(outputs, labels)
        elif model_name=="AutoConv":
            target = inputs[:, -1, :]  # Use the last time step as the target for autoencoder
            reconstructed_outputs = model(inputs)
            # Select the last time step from reconstructed_outputs
            reconstructed_outputs = reconstructed_outputs[:, -1, :]
            loss = criterion(reconstructed_outputs, target)
        elif model_name == "Generate":
            target = inputs[:, -1, :]  # Use the last time step as the target for autoencoder

            # 모델의 출력 중에서 decoder_outputs와 generator_label_output을 얻습니다.
            decoder_outputs, generator_label_output = model(inputs)

            # Select the last time step from decoder_outputs
            decoder_outputs = decoder_outputs[:, -1, :]

            # Autoencoder 부분의 손실 계산 (예: 평균 제곱 오차)
            label_loss = criterion(generator_label_output, labels)

            # Custom loss 함수인 some_custom_loss를 이용해 추가적인 손실 계산
            custom_loss = some_custom_loss(decoder_outputs, target)

            # Autoencoder 손실과 추가적인 손실을 합하여 총 손실 계산
            loss = label_loss + custom_loss

        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)




        loss.backward()
        optimizer.step()

        train_loader_tqdm.set_postfix(loss=loss.item())

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if 'GCN' in model_name:
                outputs = model(inputs, edge_index)
                loss = criterion(outputs, labels)
            elif model_name == "AutoConv":
                target = inputs[:, -1, :]  # Use the last time step as the target for autoencoder
                reconstructed_outputs = model(inputs)
                # Select the last time step from reconstructed_outputs
                reconstructed_outputs = reconstructed_outputs[:, -1, :]
                loss = criterion(reconstructed_outputs, target)
            elif model_name == "Generate":
                target = inputs[:, -1, :]  # Use the last time step as the target for autoencoder

                # 모델의 출력 중에서 decoder_outputs와 generator_label_output을 얻습니다.
                decoder_outputs, generator_label_output = model(inputs)

                # Select the last time step from decoder_outputs
                decoder_outputs = decoder_outputs[:, -1, :]

                # Autoencoder 부분의 손실 계산 (예: 평균 제곱 오차)
                label_loss = criterion(generator_label_output, labels)

                # Custom loss 함수인 some_custom_loss를 이용해 추가적인 손실 계산
                custom_loss = some_custom_loss(decoder_outputs, target)

                # Autoencoder 손실과 추가적인 손실을 합하여 총 손실 계산
                loss = label_loss + custom_loss
                print(f"Label Loss: {label_loss:.4f}, Custom Loss: {custom_loss:.4f}")

            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {num_params}")
# Save the model
torch.save(model.state_dict(), f"model/{model_name}_timeseries_model.pth")
print("Model saved successfully.")

