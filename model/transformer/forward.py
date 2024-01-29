import torch.nn as nn


class Forward(nn.Module):
    def __init__(self, dim, dropout=0.):
        super().__init__()
        dimx = int(dim * 4)
        dim_out = dim
        self.gelu = nn.Sequential(
            nn.Linear(dim, dimx),
            nn.Dropout(0.1),
            nn.GELU()
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(dimx, dim_out)
        self.fc.weight.data.fill_(0)
        self.fc.bias.data.fill_(0)

    def forward(self, x):
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
