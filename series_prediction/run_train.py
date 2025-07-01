import pandas as pd
from sklearn.model_selection import train_test_split
from series_prediction_helper import TimeSeriesDataset, SeriesPredictionModule, SeriesPredictionDataModule
import glob
from transformers import AutoformerConfig, AutoformerForPrediction
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(
    project='Eye_series_prediction',
    name='autoformer_300'
)

xlsx_files = glob.glob(f"./series_prediction_data/*/*.xlsx")

train_files, val_files = train_test_split(xlsx_files, test_size=0.2, shuffle=True, random_state=42)

print(f"Train ({len(train_files)})")
print(f"Test  ({len(val_files)})")

prediction_length = 1201
context_length = 300
# lags_sequence = [1, 2, 5, 10, 20, 50, 100]
num_time_features = 0

# # 3. 配置 Autoformer
config = AutoformerConfig(
    # 必填：预测和上下文长度
    prediction_length=prediction_length,  
    context_length=context_length,          

    # 多通道输入（这里是 5 维）
    input_size=5,                           # target 变量的维度（多元时序）
    scaling=True,                           # 是否对 target 做内部缩放 :contentReference[oaicite:0]{index=0}

    # 时序特征和滞后作为额外协变量
    lags_sequence=[0],            
    num_time_features=0,

    # 如果有静态分类/实数特征，这里可以配置；本例中都设为 0
    num_dynamic_real_features=0,
    num_static_categorical_features=0,
    num_static_real_features=0,
    cardinality=[],
    embedding_dimension=[],

    # Transformer 架构参数
    d_model=64,                             # 每层 hidden size
    encoder_layers=3,                       # Encoder 层数
    decoder_layers=3,                       # Decoder 层数
    encoder_attention_heads=4,              
    decoder_attention_heads=4,
    encoder_ffn_dim=128,                    # Feed-forward 层中间维度
    decoder_ffn_dim=128,
    dropout=0.1,                            # 全连接层 dropout
    attention_dropout=0.1,                  # 注意力权重 dropout
    activation_dropout=0.1,                 # 激活函数后 dropout
    activation_function="gelu",             # 激活函数 :contentReference[oaicite:1]{index=1}

    # 推理阶段并行采样数
    num_parallel_samples=100,
)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

# Callbacks
checkpoint_cb = ModelCheckpoint(
    filename='series-{epoch:02d}-{val_nll:.4f}',
    save_top_k=1, monitor='val_loss', mode='min'
)
earlystop_cb = EarlyStopping(monitor='val_loss', patience=10, mode='min')
devices = [0]
trainer = pl.Trainer(
    max_epochs=2000,
    # accelerator='cpu',    # use CPU
    # devices=1,            
    precision='32',      
    accelerator='gpu',    
    devices=devices,  # use GPU 0 and GPU 1
    # strategy=DDPStrategy(find_unused_parameters=True), #allow unused parameters
    callbacks=[checkpoint_cb, earlystop_cb],
    logger=wandb_logger
)

# Data & Model instantiation
dm = SeriesPredictionDataModule(train_files, val_files, num_workers = 0)
model = SeriesPredictionModule(config)

# Run training
trainer.fit(model, dm)
