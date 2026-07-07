import torch
import torch.nn as nn

class PricePredictModel(nn.Module):
    """Predicts the next transaction price from item information and historical transaction data."""
    def __init__(
        self,
        num_items,
        item_emb_dim=16,
        seq_input_dim=2,       # [log_price, log_volume]
        seq_hidden_dim=32,
        numeric_dim=4,         # month_sin, month_cos, dow_sin, dow_cos, log_volume(current)
        hidden_dims=(64, 32),
        dropout=0.1,
        rnn_layers=1,
    ):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, item_emb_dim)
        self.seq_encoder = nn.GRU(
            input_size=seq_input_dim,
            hidden_size=seq_hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
        )
 
        input_dim = item_emb_dim + seq_hidden_dim + numeric_dim
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)
 
    def forward(self, item_ids, seq_input, seq_lengths, numeric_features):
        """
        다음 거래 가격을 예측한다.

        Args:
            item_ids: (B,)
            seq_input: (B, L, 2)
            seq_lengths: (B,)
            numeric_features: (B, 4)

        Returns:
            pred_log_price: (B,)
        """ 
        item_emb = self.item_embedding(item_ids)  # (B, item_emb_dim)
 
        packed = nn.utils.rnn.pack_padded_sequence(
            seq_input, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.seq_encoder(packed)
        seq_feat = h_n[-1]  # last layer hidden state, (B, seq_hidden_dim)
 
        x = torch.cat([item_emb, seq_feat, numeric_features], dim=1)
        out = self.mlp(x)
        return out.squeeze(-1)  # (B,) -> log1p(price)

