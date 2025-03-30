import torch.nn as nn
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

class TimeDataset(Dataset):
    def __init__(self, xlsx_name_list, label_feature = ['反应时间'], sequence_length = 1500):
        """
        Args:
            dataframe (pd.DataFrame): 输入数据，其中包括 period_num, Time_stamp, label 和其他特征
        """
        self.xlsx_name_list = xlsx_name_list
        self.features = ['Event_type_眼动形式', 'Average_pupil_diameter_平均瞳孔直径', '0_direction_眼跳方向', 'Average_velocity_平均眼跳时间', 'Peak_velocity_眼跳峰值速度', '0_amplitude_眼跳幅度'] 
        self.sequence_length = sequence_length
        
        self.num_features = len(self.features)
        self.label_feature = label_feature
        
    def __len__(self):
        # 数据集的长度
        return len(self.xlsx_name_list)
    
    def __getitem__(self, idx):
        # 获取指定 period_num 的数据
        xlsx_name = self.xlsx_name_list[idx]
        input_data = pd.read_excel(xlsx_name, sheet_name='Sheet1')
        input_data = input_data.fillna(method='ffill') # 如果有nan 等于 上面没有 nan 的value 
        # 获取输入特征 X，形状为 (1500, 7)
        X = input_data[self.features].values[:self.sequence_length, :].astype(np.float32)  # 取出特征
        X = torch.tensor(X)  # 转换为 Tensor
        # X = standardize_channel_wise(X)
        # 获取标签 Y，形状为 (1,)
        Y = input_data[self.label_feature].iloc[0]  # 每个 period_num 的标签是一样的，取第一个即可
        # Y = torch.tensor([Y], dtype=torch.float32)  # 转换为 Tensor
        Y = torch.tensor(Y, dtype=torch.float32)
        # Y = torch.log1p(Y)
        return X, Y
    
