import torch
import torch.nn as nn

class SiameseLSTM(nn.Module):
    def __init__(self, config, input_dim: int, context_dim: int):
        super(SiameseLSTM, self).__init__()
        self.config = config

        # 1. The Twin LSTMs (Process History)
        # They share weights because "Player A" and "Player B" are symmetric concepts
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=config.model.hidden_size,
            num_layers=config.model.num_layers,
            batch_first=True,
            dropout=config.model.dropout if config.model.num_layers > 1 else 0
        )

        # 2. Fusion Layer
        # Concatenate: (LSTM_Out_A) + (LSTM_Out_B) + (Context_Features)
        # LSTM output size is hidden_size (we take the last state)
        fusion_input_dim = (config.model.hidden_size * 2) + context_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(32, 1) # Output logit
        )

    def forward(self, seq_a, seq_b, context):
        # seq_a/b shape: (Batch, Seq_Len, Features)
        # context shape: (Batch, Context_Features)

        # Process Player A
        # output is (Batch, Seq_Len, Hidden), hidden is tuple (h_n, c_n)
        # We want the final hidden state of the last layer: h_n[-1]
        _, (h_n_a, _) = self.lstm(seq_a)
        emb_a = h_n_a[-1] # Shape: (Batch, Hidden)

        # Process Player B (Shared Weights)
        _, (h_n_b, _) = self.lstm(seq_b)
        emb_b = h_n_b[-1] # Shape: (Batch, Hidden)

        # Concatenate everything
        combined = torch.cat([emb_a, emb_b, context], dim=1)

        # Predict
        return self.fusion(combined)
