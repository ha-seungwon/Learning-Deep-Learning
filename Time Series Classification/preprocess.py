import pandas as pd
import numpy as np
import torch
import os
import random


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)


seed_everything(42)  # Seed 고정

df_list=[]

for subject_id in range(1,10,1):

    dir_csv = f'/Users/haseung-won/Desktop/학교/연구실/time_series_data/PAMAP2_Dataset/Protocol_csv/subject10{str(subject_id)}.csv'

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


train_df = pd.DataFrame()  # 훈련 데이터프레임 초기화
test_df = pd.DataFrame()   # 테스트 데이터프레임 초기화

for subject_id in range(1, 10):
    subject_data = data[data['subjectID'] == subject_id]
    num_samples = len(subject_data)
    split_index = int(0.8 * num_samples)  # 마지막 20% 위치 계산
    train_df = pd.concat([train_df, subject_data[:split_index]], ignore_index=True)
    test_df = pd.concat([test_df, subject_data[split_index:]], ignore_index=True)



train_df.to_csv( '/Users/haseung-won/Desktop/학교/연구실/time_series_data/PAMAP2_Dataset/train.csv', index=False)
test_df.to_csv('/Users/haseung-won/Desktop/학교/연구실/time_series_data/PAMAP2_Dataset/test.csv',index=False)