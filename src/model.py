import torch
import torch.nn as nn

class LSTMSequenceLabeler(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, num_layers=2, dropout_rate=0.3):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout_rate)
        
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        # x: [batch_size, 1500, 6]
        lstm_out, _ = self.lstm(x)  # [batch_size, 1500, hidden_dim*2]
        
        # Apply Layer Normalization
        norm_out = self.layer_norm(lstm_out)
        
        # Apply Dropout
        dropped_out = self.dropout(norm_out)
        
        # Classifier
        logits = self.classifier(dropped_out).squeeze(-1)  # [batch_size, 1500]

        return logits
    
def get_patch_tst_model(num_targets = 2, context_length=1500):
    config = PatchTSTConfig(
        num_input_channels=7,
        num_targets=num_targets,
        context_length=context_length,
        patch_length=50,
        stride=25,
        use_cls_token=True,
    )
    model = PatchTSTForClassification(config=config).to(device)