import torch
import torch.nn as nn
from src.config import ProjectConfig, LSTMArch

class SiameseLSTM(nn.Module):
    def __init__(self, config: ProjectConfig, input_dim: int, context_dim: int) -> None:
        super(SiameseLSTM, self).__init__()
        self.config: ProjectConfig = config
        
        # Route specifically to the LSTM architecture configs
        lstm_cfg: LSTMArch = config.models.lstm.architecture

        self.lstm: nn.LSTM = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_cfg.hidden_size,
            num_layers=lstm_cfg.num_layers,
            batch_first=True,
            dropout=lstm_cfg.dropout if lstm_cfg.num_layers > 1 else 0
        )

        fusion_input_dim: int = (lstm_cfg.hidden_size * 2) + context_dim

        self.fusion: nn.Sequential = nn.Sequential(
            nn.Linear(fusion_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(lstm_cfg.dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(lstm_cfg.dropout),
            nn.Linear(32, 1) # Output logit
        )

    def forward(self, seq_a: torch.Tensor, seq_b: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # Note: h_n shape is (num_layers, batch, hidden_size)
        _, (h_n_a, _) = self.lstm(seq_a)
        emb_a: torch.Tensor = h_n_a[-1]  # Get last layer's hidden state

        _, (h_n_b, _) = self.lstm(seq_b)
        emb_b: torch.Tensor = h_n_b[-1]

        combined: torch.Tensor = torch.cat([emb_a, emb_b, context], dim=1)
        return self.fusion(combined)