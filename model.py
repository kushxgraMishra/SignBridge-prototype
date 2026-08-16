import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_dim)
        attn_weights = torch.softmax(self.attn(lstm_output), dim=1)  # (batch, seq_len, 1)
        context = torch.sum(attn_weights * lstm_output, dim=1)       # (batch, hidden_dim)
        return context, attn_weights


class SignBiLSTM(nn.Module):
    def __init__(self, input_size=258, hidden_size=128, num_layers=2, num_classes=76, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = Attention(hidden_size * 2)  # *2 because bidirectional
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)                 # (batch, seq_len, hidden_size*2)
        context, attn_weights = self.attention(lstm_out)
        out = self.dropout(context)
        out = self.fc(out)                          # (batch, num_classes)
        return out


if __name__ == "__main__":
    # Quick sanity check
    model = SignBiLSTM(input_size=258, hidden_size=128, num_layers=2, num_classes=76)
    dummy_input = torch.randn(4, 33, 258)  # batch=4, seq_len=33, features=258
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # should be (4, 76)