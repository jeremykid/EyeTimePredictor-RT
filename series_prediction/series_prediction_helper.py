import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from transformers import AutoformerConfig, AutoformerForPrediction

class SeriesPredictionModule(pl.LightningModule):
    def __init__(self, config, lr = 0.001):
        super().__init__()
        # Hyperparameters
        self.lr = lr
        self.config = config
        self.model = AutoformerForPrediction(self.config).to(self.device)
        # self.device = device
        
    def forward(self, batch):
        past_values, future_values = batch['past_values'], batch['future_values']
        past_values = past_values.to(self.device)
        future_values = future_values.to(self.device)

        past_time_features = torch.zeros(
            past_values.size(0),
            self.config.context_length,
            0
        ).to(self.device)

        past_observed_mask = torch.ones_like(past_values).to(self.device)

        future_time_features = torch.zeros(
            past_values.size(0),
            self.config.prediction_length,
            0
        ).to(self.device)
        
        outputs = self.model(
            past_values=past_values,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask,
            future_time_features=future_time_features,
            future_values=future_values
        )        
        return outputs
        
    def training_step(self, batch, batch_idx):

        outputs = self(batch)
        loss = outputs.loss        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=1e-3
        )
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
            ),
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        past_values, future_values = batch['past_values'], batch['future_values']
        past_values = past_values.to(self.device)
        future_values = future_values.to(self.device)

        past_time_features = torch.zeros(
            past_values.size(0),
            self.config.context_length,
            0
        ).to(self.device)

        past_observed_mask = torch.ones_like(past_values).to(self.device)

        future_time_features = torch.zeros(
            past_values.size(0),
            self.config.prediction_length,
            0
        ).to(self.device)
        
        # 1) 用 generate 采样
        outputs = self.model.generate(
            past_values=past_values,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask,
            future_time_features=future_time_features
        )
        # outputs.sequences: (B, num_samples, pred_len, C)

        # 2) 聚合为点预测（这里举例用均值）
        mean_pred = outputs.sequences.mean(dim=1)   # (B, pred_len, C)
        # 调整维度到 (B, C, pred_len) 以便跟 ground truth 对齐
        mean_pred = mean_pred.permute(0, 2, 1)

        # 3) ground truth
        gt = batch['future_values']                 # (B, C, pred_len)
        return {
            'pred_seq': mean_pred.detach().cpu(),
            'gt_seq':   gt.detach().cpu()
        }   
    
class SeriesPredictionDataModule(pl.LightningDataModule):
    def __init__(self, train_files, val_files, batch_size = 8, num_workers = 2):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_files = train_files
        self.val_files = val_files
        
    def setup(self, stage=None):        
        self.train_ds = TimeSeriesDataset(self.train_files, input_len=300, output_len=1201)
        self.val_ds = TimeSeriesDataset(self.val_files, input_len=300, output_len=1201)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers
        )    

def ts_collate_fn(batch):
    """
    batch: list of dict, {'past_values': Tensor(C,input_len), 'future_values': Tensor(C,output_len)}
    return dict，(B, C, input_len) / (B, C, output_len)
    """
    past = torch.stack([item['past_values']  for item in batch], dim=0)
    future = torch.stack([item['future_values'] for item in batch], dim=0)
    return {
        'past_values':  past,
        'future_values': future
    }

class TimeSeriesDataset(Dataset):
    def __init__(self, files_list, input_len, output_len, transform=None):
        """
        df_list: List of pandas.DataFrame, each of shape (C, T)
        input_len: int, input_length (e.g. 300)
        output_len: int,output_length (e.g. 1200)
        transform: option， (x, y) for extra change
        """
        self.files_list = files_list
        self.input_len = input_len
        self.output_len = output_len
        self.transform = transform

    def __len__(self):
        return len(self.files_list)

    def read_xlsx_file(self, xlsx_file):
        input_data = pd.read_excel(xlsx_file, sheet_name='Sheet1')
        # input_data.loc[input_data['Event_type'] == 'Saccade', 'Event_type'] = 0
        # input_data.loc[input_data['Event_type'] != 0, 'Event_type'] = 1
        if 'Average_pupil_diameter' in input_data.columns:
            input_data = input_data[['Average_pupil_diameter', '1_direction', 'Average_velocity', 'Peak_velocity', '1_amplitude']]
        elif 'Average_velocity_平均眼跳时间' in input_data.columns:
            input_data = input_data[['Average_pupil_diameter_平均瞳孔直径', '0_direction_眼跳方向', 'Average_velocity_平均眼跳时间', 'Peak_velocity_眼跳峰值速度', '0_amplitude_眼跳幅度']]
        elif 'Average_velocity_平均眼跳速度' in input_data.columns:
            input_data = input_data[['Average_pupil_diameter_平均瞳孔直径', '0_direction_眼跳方向', 'Average_velocity_平均眼跳速度', 'Peak_velocity_眼跳峰值速度', '0_amplitude_眼跳幅度']]
        else:
            print (input_data.columns.tolist())
        return input_data
    
    def __getitem__(self, idx):
        file = self.files_list[idx]
        df = self.read_xlsx_file(file)

        arr = df.values.astype(np.float32)  # (C, T)
        x_np = arr[:self.input_len]                     # (C, input_len)
        y_np = arr[self.input_len:]  # (C, output_len)

        x = torch.tensor(x_np, dtype=torch.float32)
        y = torch.tensor(y_np, dtype=torch.float32)
        
        if self.transform:
            x, y = self.transform(x, y)
        return {
            'past_values': x,
            'future_values': y
        }
