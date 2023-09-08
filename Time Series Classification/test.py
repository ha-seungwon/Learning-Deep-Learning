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

data=pd.read_csv('../PAMAP2_Dataset/test.csv')


data = data.fillna(0)


X = data.drop(columns=['activityID'])
y = data['activityID']
num_classes =25





# 시퀀스 길이 설정
timesteps = 100
step_size = 20

X_test = []
y_test = []


for i in range(0, len(X) - timesteps + 1, step_size):
    X_sequence = X.iloc[i:i + timesteps, :].values
    y_label = y.iloc[i + timesteps - 1]  # Get a single label for the sequence
    X_test.append(X_sequence)
    y_test.append(y_label)







X_test = np.array(X_test)
y_test = np.array(y_test)
# Now you can proceed with multi-label encoding
y_test_encoded = np.array([to_categorical(labels, num_classes=num_classes) for labels in y_test])
y_test_encoded_flattened = y_test_encoded.reshape(-1, num_classes)



# Assuming X_train_lstm, y_train_encoded_flattened are already prepared
# Convert numpy arrays to PyTorch tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_encoded_flattened_tensor = torch.tensor(y_test_encoded_flattened, dtype=torch.float32)
class CustomDataset(Dataset):
    def __init__(self, X, y, edge_index):
        self.X = X
        self.y = y
        self.edge_index = edge_index

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

num_nodes = X_test.shape[1]

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



# Create instances of cus
# Create instances of custom dataset
custom_dataset = CustomDataset(X_test_tensor, y_test_encoded_flattened_tensor,edge_index)
batch_size = 64
test_dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)
# Initialize model and hyperparameters
input_size = X_test_tensor.shape[2]
hidden_size = 30
num_classes = 25

print("input_size",input_size,"hidden_size",hidden_size,"num_classes",num_classes,"batch_size",batch_size)
model_name=args.model_name
print("model_name",model_name)
if model_name=='Conv1D':
    model=models.Conv1DModel(input_size, num_classes).to(device)
elif model_name=='LSTM':
    model = models.LSTMModel(input_size, hidden_size, num_classes).to(device)
elif model_name=='GCN':
    model = models.GCNModel(input_size, hidden_size, num_classes).to(device)
elif model_name=='GCN2':
    model = models.GCNModel2(input_size, hidden_size, num_classes).to(device)
elif model_name == "AutoConv":
    latent_size = 16  # Adjust as needed
    model = models.ConvAutoencoder(input_size, latent_size).to(device)


edge_index=custom_dataset.edge_index.to(device)
# Load the trained model parameters
model.load_state_dict(torch.load(f'./model/{model_name}_timeseries_model.pth'))
print("Model load complete")
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {num_params}")
# 모델 평가를 위한 함수 정의
def evaluate_model(model, dataloader, device):
    model.eval()
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 모델에 입력 데이터 전달하여 예측 수행
            if 'GCN' in model_name:
                outputs = model(inputs,edge_index)
            else:
                outputs = model(inputs)

            # 예측값 중 가장 높은 확률을 가진 클래스 선택
            predicted = torch.argmax(outputs, dim=1)  # 모델의 예측값 (크기: batch_size,)
            # labels는 one-hot 인코딩된 형태이므로, argmax로 다시 변환하여 크기를 (batch_size,)로 맞춘다.
            labels = torch.argmax(labels, dim=1)  # 실제 레이블 (크기: batch_size,)
            correct_predictions += (predicted == labels).sum().item()

            # 정확하게 예측한 샘플 수 계산
            total_samples += labels.size(0)

    # 정확도 계산
    accuracy = correct_predictions / total_samples
    return accuracy
def evaluate_autoencoder(model, dataloader, device):
    # Define the evaluation logic for the autoencoder
    autoencoder_criterion = nn.MSELoss()
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            target = inputs[:, -1, :]
            outputs = model(inputs)
            outputs = outputs[:, -1, :]
            loss = autoencoder_criterion(outputs, target)
            total_loss += loss.item() * inputs.size(0)
            num_samples += inputs.size(0)

    average_loss = total_loss / num_samples
    return average_loss
# 모델 평가 수

if model_name=='AutoConv':
    # Evaluate the autoencoder
    autoencoder_loss = evaluate_autoencoder(model, test_dataloader, device)
    print(f"Autoencoder Loss: {autoencoder_loss:.4f}")
else:
    accuracy = evaluate_model(model, test_dataloader, device)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
