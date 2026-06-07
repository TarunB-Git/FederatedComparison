#!/usr/bin/env python3
import torch
import torch.nn as nn


class NumericGRUEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input_proj(x)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, h_n = self.gru(packed)
        else:
            _, h_n = self.gru(x)
        return self.dropout(h_n[-1])


class NumericLSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input_proj(x)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(x)
        return self.dropout(h_n[-1])


class NumericTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        layers: int = 2,
        dropout: float = 0.2,
        nhead: int = 8,
        max_len: int = 1024,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, hidden_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.max_len = max_len

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")
        x = self.input_proj(x) + self.pos_embedding[:, :seq_len]
        key_padding_mask = None
        if lengths is not None:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, seq_len)
            key_padding_mask = positions >= lengths.unsqueeze(1)
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)
        if lengths is None:
            out = h[:, -1]
        else:
            idx = (lengths - 1).clamp(min=0)
            out = h[torch.arange(bsz, device=h.device), idx]
        return self.dropout(out)


class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128, dropout: float = 0.2, depth: int = 1):
        super().__init__()
        layers: list[nn.Module] = []
        cur_dim = in_dim
        n_hidden = max(1, int(depth))
        for i in range(n_hidden):
            layers.append(nn.Linear(cur_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            cur_dim = hidden_dim
        layers.append(nn.Linear(cur_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UnifiedActionOutcomeHead(nn.Module):
    def __init__(self, in_dim: int, action_dim: int, predict_win: bool = True):
        super().__init__()
        self.action_dim = action_dim
        self.predict_win = predict_win
        out_dim = action_dim + (1 if predict_win else 0)
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        out = self.proj(x)
        action_logits = out[:, : self.action_dim]
        if not self.predict_win:
            return action_logits, None
        win_logit = out[:, self.action_dim]
        return action_logits, win_logit


def build_encoder(model_name: str, input_dim: int, hidden_dim: int = 256, layers: int = 2, dropout: float = 0.2):
    model_name = model_name.lower()
    if model_name == "gru":
        return NumericGRUEncoder(input_dim=input_dim, hidden_dim=hidden_dim, layers=layers, dropout=dropout)
    if model_name == "lstm":
        return NumericLSTMEncoder(input_dim=input_dim, hidden_dim=hidden_dim, layers=layers, dropout=dropout)
    if model_name == "transformer":
        nhead = 8
        if hidden_dim % nhead != 0:
            for cand in [4, 2, 1]:
                if hidden_dim % cand == 0:
                    nhead = cand
                    break
        return NumericTransformerEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            layers=layers,
            dropout=dropout,
            nhead=nhead,
        )
    raise ValueError(f"Unsupported model_name: {model_name}")
