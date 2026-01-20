
from torch import nn

class FFN(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout):
        super().__init__()

        self.Relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model) or (..., d_model)
        return self.fc2(self.dropout(self.relu(self.fc1(x))))
    