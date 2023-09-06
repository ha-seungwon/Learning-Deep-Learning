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

df_list=[]
for subject_id in range(1,10,1):

    dir_csv = f'../PAMAP2_Dataset/Protocol_csv/subject10{str(subject_id)}.csv'

    # Define column names based on the provided structure
    column_names = [
        'timestamp',
        'activityID',
        'heart_rate',
    ]

    imu_sensors = [
        'temperature',
        'acceleration16g_x', 'acceleration16g_y', 'acceleration16g_z',
        'acceleration6g_x', 'acceleration6g_y', 'acceleration6g_z',
        'gyroscope_x', 'gyroscope_y', 'gyroscope_z',
        'magnetometer_x', 'magnetometer_y', 'magnetometer_z',
        'orientation_1', 'orientation_2', 'orientation_3','orientation_4'
    ]

    imu_parts = ['hand', 'chest', 'ankle']

    # Add column names for IMU hand, chest, and ankle data
    for part in imu_parts:
        for sensor in imu_sensors:
            column_names.append(f'IMU_{part}_{sensor}')
    # Read the CSV file using pandas
    df = pd.read_csv(dir_csv, names=column_names)
    df_list.append(df)

    # Now df contains the data from the CSV file with appropriate column names


# 데이터 병합
data = pd.concat(df_list, ignore_index=True)

subject_ids = []  # Subject ID 값을 저장할 빈 리스트성

for subject_id in range(1, 10, 1):
    subject_ids += [subject_id] * len(df_list[subject_id - 1])  # Subject ID 반복하여 리스트에 추가

data['subjectID'] = subject_ids  # Subject ID 컬럼을 데이터프레임에 추가


data = data.fillna(0)
X = data.drop(columns=['activityID'])
y = data['activityID']
num_classes =25


# 시퀀스 길이 설정
timesteps = 10

X_train= []
y_train = []

for i in range(len(X) - timesteps + 1):
    X_sequence = X.iloc[i:i + timesteps, :].values
    y_label = y.iloc[i + timesteps - 1]  # Get a single label for the sequence
    X_train.append(X_sequence)
    y_train.append(y_label)





X_train_lstm = np.array(X_train)
y_train_lstm = np.array(y_train)
# Now you can proceed with multi-label encoding
y_train_encoded = np.array([to_categorical(labels, num_classes=num_classes) for labels in y_train])
y_train_encoded_flattened = y_train_encoded.reshape(-1, num_classes)



# Assuming X_train_lstm, y_train_encoded_flattened are already prepared
# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_encoded_flattened_tensor = torch.tensor(y_train_encoded_flattened, dtype=torch.float32)

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
# num_nodes = X_train_lstm.shape[1]
#
# # Create an adjacency matrix with edges connecting each node to its neighbors
# adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
#
# # Connect each node to its immediate neighbors (adjust as needed)
# for i in range(num_nodes):
#     if i > 0:
#         adjacency_matrix[i, i - 1] = 1.0
#     if i < num_nodes - 1:
#         adjacency_matrix[i, i + 1] = 1.0
#
# # Convert the adjacency matrix to a sparse tensor
# edge_index = torch.tensor(np.array(np.where(adjacency_matrix == 1)), dtype=torch.long).to(device)


edge_index=[]
# Create instances of custom dataset
custom_dataset = CustomDataset(X_train_tensor, y_train_encoded_flattened_tensor,edge_index)

# Split the data into training and validation sets
train_size = int(0.8 * len(custom_dataset))
val_size = len(custom_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(custom_dataset, [train_size, val_size])

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



# Initialize model and hyperparameters
input_size = X_train_tensor.shape[2]
hidden_size = 50
num_classes = 25

model_name=args.model_name

model=models.MyModel(input_size,hidden_size,num_classes).to(device)

print("model_name : ",model_name,"input_size : ",input_size,"hidden_size : ",hidden_size,"num_classes : ",num_classes)
criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

from tqdm import tqdm  # Import tqdm library

# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}] Training", leave=False)
    lstm_outputs=[]
    for inputs, labels in train_loader_tqdm:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        if 'GCN' in model_name:
            outputs = model(inputs,edge_index)
        elif 'LSTM' in model_name:
            outputs,lstm_output = model(inputs)
        else:
            outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loader_tqdm.set_postfix(loss=loss.item())
        lstm_outputs.append(lstm_output)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if 'GCN' in model_name:
                outputs = model(inputs, edge_index)
            else:
                outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")



num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {num_params}")


print("lstms : ",lstm_outputs)
