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
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)


seed_everything(42)  # Seed 고정

df_list=[]
for subject_id in range(1,10,1):

    dir_csv = f'./PAMAP2_Dataset/Protocol_csv/subject10{str(subject_id)}.csv'

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
    print(column_names)
    print(len(column_names))
    # Read the CSV file using pandas
    df = pd.read_csv(dir_csv, names=column_names)
    df_list.append(df)

    # Now df contains the data from the CSV file with appropriate column names





# 데이터 병합
data = pd.concat(df_list, ignore_index=True)


data = data.fillna(0)
X = data.drop(columns=['activityID'])
y = data['activityID']
num_classes =25
n_features = X.shape[1]  # Number of columns in your feature matrix X
print(num_classes)




# 시퀀스 길이 설정
timesteps = 10

X_train_lstm = []
y_train_lstm = []

for i in range(len(X) - timesteps + 1):
    X_sequence = X.iloc[i:i + timesteps, :].values
    y_label = y.iloc[i + timesteps - 1]  # Get a single label for the sequence
    X_train_lstm.append(X_sequence)
    y_train_lstm.append(y_label)





X_train_lstm = np.array(X_train_lstm)
y_train_lstm = np.array(y_train_lstm)
# Now you can proceed with multi-label encoding
y_train_encoded = np.array([to_categorical(labels, num_classes=num_classes) for labels in y_train_lstm])
y_train_encoded_flattened = y_train_encoded.reshape(-1, num_classes)



# Assuming X_train_lstm, y_train_encoded_flattened are already prepared
# Convert numpy arrays to PyTorch tensors
X_train_lstm_tensor = torch.tensor(X_train_lstm, dtype=torch.float32)
y_train_encoded_flattened_tensor = torch.tensor(y_train_encoded_flattened, dtype=torch.float32)

# Create a custom dataset
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create instances of custom dataset
custom_dataset = CustomDataset(X_train_lstm_tensor, y_train_encoded_flattened_tensor)

# Split the data into training and validation sets
train_size = int(0.8 * len(custom_dataset))
val_size = len(custom_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(custom_dataset, [train_size, val_size])

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define the LSTM model


# Initialize model and hyperparameters
input_size = X_train_lstm_tensor.shape[2]
hidden_size = 50
num_classes = 25

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = LSTMModel(input_size, hidden_size, num_classes).to(device)

model_name='Conv1D'

if model_name=='LSTM':
    model=models.LSTMModel(input_size, hidden_size, num_classes).to(device)
elif model_name=='Conv1D':
    model=models.Conv1DModel(input_size, num_classes).to(device)


criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

from tqdm import tqdm  # Import tqdm library

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}] Training", leave=False)

    for inputs, labels in train_loader_tqdm:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
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

