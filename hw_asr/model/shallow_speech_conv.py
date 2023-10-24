import torch
from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel

class FF(nn.Module):
    def __init__(self, n_feats, n_hidden, dropout=0.05):
        super().__init__()
        self.net = Sequential(
            nn.Hardtanh(0, 20),
            nn.Linear(n_feats, n_hidden, bias=True),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
class ShallowSpeechConv(BaseModel):
    def __init__(self, n_feats, n_class, n_hidden=512, **batch):
        super().__init__(n_feats, n_class, **batch)
        self.n_hidden = n_hidden
        self.conv = Sequential(
            nn.Conv2d(1, 8, (21, 6), stride=(2, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, (11, 6), stride=(2, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.recurrent = nn.GRU(((((n_feats - 21) // 2 + 1) - 11) // 2 + 1) * 8, n_hidden, batch_first=True, bidirectional=True, num_layers=2)
        self.bn = nn.BatchNorm1d(n_hidden * 2)
        self.final = Sequential(
            nn.ReLU(),
            FF(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_class)
        )

    def forward(self, spectrogram, **batch):
        x = self.conv(spectrogram.unsqueeze(dim=1))
        x, _ = self.recurrent(x.flatten(start_dim=1, end_dim=2).transpose(1, 2))
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        return {"logits": self.final(x[:, :, :self.n_hidden] + x[:, :, self.n_hidden:])}

    def transform_input_lengths(self, input_lengths):
        return input_lengths - 6 + 1 - 6 + 1
