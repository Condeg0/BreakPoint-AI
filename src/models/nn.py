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

        # Shared attention over all hidden states (consistent with Siamese design)
        self.attention: nn.Linear = nn.Linear(lstm_cfg.hidden_size, 1)

        fusion_input_dim: int = (lstm_cfg.hidden_size * 2) + context_dim
        fd: int = lstm_cfg.fusion_dim

        self.fusion: nn.Sequential = nn.Sequential(
            nn.Linear(fusion_input_dim, fd),
            nn.ReLU(),
            nn.Dropout(lstm_cfg.dropout),
            nn.Linear(fd, fd // 2),
            nn.ReLU(),
            nn.Dropout(lstm_cfg.dropout),
            nn.Linear(fd // 2, 1)
        )

    def _attend(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """Weighted sum of LSTM hidden states via learned attention."""
        scores: torch.Tensor = self.attention(lstm_out)          # (batch, seq_len, 1)
        weights: torch.Tensor = torch.softmax(scores, dim=1)     # (batch, seq_len, 1)
        return (weights * lstm_out).sum(dim=1)                   # (batch, hidden_size)

    def forward(self, seq_a: torch.Tensor, seq_b: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        out_a, _ = self.lstm(seq_a)   # (batch, seq_len, hidden_size)
        emb_a: torch.Tensor = self._attend(out_a)

        out_b, _ = self.lstm(seq_b)
        emb_b: torch.Tensor = self._attend(out_b)

        combined: torch.Tensor = torch.cat([emb_a, emb_b, context], dim=1)
        return self.fusion(combined)