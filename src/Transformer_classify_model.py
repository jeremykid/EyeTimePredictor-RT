import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        x = x + self.pe[:x.size(0), :].unsqueeze(1)
        return x

class TransformerClassifier(nn.Module):
    def __init__(
        self,
        feature_dim: int,       # channel
        d_model: int = 64,      # Transformer d_model
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_classes: int = 4,
        pooling: str = "last"   # "last" or "mean"
    ):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooling = pooling
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, src, src_key_padding_mask):
        """
        Args:
            src: Tensor, shape (seq_len, batch, feature_dim)
            src_key_padding_mask: BoolTensor, (batch, seq_len)
        Returns:
            logits: (batch, num_classes)
        """
        # 1) project + position encoder -> (seq_len, batch, d_model)
        x = self.input_proj(src)
        x = self.pos_enc(x)

        # 2) TransformerEncoder -> same (seq_len, batch, d_model)
        #    ignore padding step
        x = self.transformer(
            x,
            src_key_padding_mask=src_key_padding_mask
        )

        # 3) global pooling
        if self.pooling == "last":
            xb = x.transpose(0,1)
            lengths = (~src_key_padding_mask).sum(dim=1) - 1  # zero-based idx
            # gather last step
            idx = lengths.view(-1,1,1).expand(-1,1, x.size(2))
            feat = xb.gather(1, idx).squeeze(1)  # (batch, d_model)
        else:  # mean pooling
            xb = x.transpose(0,1)               # (batch, seq_len, d_model)
            mask = ~src_key_padding_mask.unsqueeze(2)  # (batch, seq_len, 1)
            summed = (xb * mask).sum(dim=1)
            counts = mask.sum(dim=1)
            feat = summed / counts

        # 4) classify
        logits = self.classifier(feat)
        return logits
