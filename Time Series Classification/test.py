import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
from torch import nn
import torch
test_df_list=[]
subject_id_list=[1,5,6,8,9]
for subject_id in subject_id_list:

    dir_csv = f'./PAMAP2_Dataset/Optional_csv/subject10{str(subject_id)}.csv'

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
    test_df_list.append(df)
# Preprocessing on df_list[1]
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMModel, self).__init__()
        self.conv1= nn.Conv1d(input_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.lstm = nn.LSTM(hidden_size, num_classes, batch_first=True)
        self.fc = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Get the output from the last time step
        return out

for i in test_df_list:
    eval_data = i.fillna(0)
    X_eval = eval_data.drop(columns=['activityID'])
    y_eval = eval_data['activityID']
    n_features = X_eval.shape[1]  # Number of columns in your feature matrix X
    # Prepare sequences and labels for evaluation
    X_eval_lstm = []
    y_eval_lstm = []
    timesteps = 10
    num_classes =25
    for i in range(len(X_eval) - timesteps + 1):
        X_sequence = X_eval.iloc[i:i + timesteps, :].values
        y_label = y_eval.iloc[i + timesteps - 1]  # Get a single label for the sequence
        X_eval_lstm.append(X_sequence)
        y_eval_lstm.append(y_label)

    X_eval_lstm = np.array(X_eval_lstm)
    y_eval_lstm = np.array(y_eval_lstm)

    # One-hot encode the evaluation labels
    y_eval_encoded = np.array([to_categorical(label, num_classes=num_classes) for label in y_eval_lstm])
    y_eval_encoded_flattened = y_eval_encoded.reshape(-1, num_classes)

    # Initialize model and hyperparameters
    input_size = X_eval_lstm.shape[2]
    hidden_size = 53
    num_classes = 25

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(input_size, hidden_size, num_classes).to(device)

    # Load the trained model parameters
    model.load_state_dict(torch.load('./model/lstm_timeseries_model.pth'))
    print("Model load complete")

    # Evaluate the model on the evaluation data
    model.eval()
    X_eval_lstm = torch.from_numpy(X_eval_lstm).float().to(device)
    y_eval_encoded_flattened = torch.from_numpy(y_eval_encoded_flattened).float().to(device)
    with torch.no_grad():
        y_pred = model(X_eval_lstm)
        eval_loss = nn.BCEWithLogitsLoss()(y_pred, y_eval_encoded_flattened)
        eval_accuracy = ((y_pred > 0.5) == (y_eval_encoded_flattened > 0.5)).float().mean()



    print("Evaluation Loss:", eval_loss)
    print("Evaluation Accuracy:", eval_accuracy)