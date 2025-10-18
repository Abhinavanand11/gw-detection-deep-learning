import torch
from torch import nn
from torch.nn import functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        if out_channels != in_channels or stride > 1:
            self.x_transform = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.x_transform = nn.Identity()

        self.body = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        x = F.relu(self.body(x) + self.x_transform(x))
        return x


class ResNet54(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ResBlock(2, 8),
            ResBlock(8, 8),
            ResBlock(8, 8),
            ResBlock(8, 8),
            ResBlock(8, 16, stride=2),
            ResBlock(16, 16),
            ResBlock(16, 16),
            ResBlock(16, 32, stride=2),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 64, stride=2),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64, stride=2),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64, stride=2),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 16),
            ResBlock(16, 16),
            ResBlock(16, 16),
        )
        self.cls_head = nn.Sequential(
            nn.Conv1d(16, 32, 64), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 2, 1), nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.cls_head(x).squeeze(2)


class ResNet54Double(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ResBlock(2, 16),
            ResBlock(16, 16),
            ResBlock(16, 16),
            ResBlock(16, 16),
            ResBlock(16, 32, stride=2),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 64, stride=2),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 128, stride=2),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 128, stride=2),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
        )
        self.cls_head = nn.Sequential(
            nn.Conv1d(32, 64, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 2, 1), nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.cls_head(x).squeeze(2)
class CNN(nn.Module):
    """
    CNN architecture from the gravitational wave detection script.
    Input: (batch, detectors, 2048) where detectors=2 for coherent search
    Output: (batch, 2) with class probabilities
    """
    def __init__(self, detectors=2):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            # Shapes commented as in original
            nn.BatchNorm1d(detectors),                          # (1*detectors) x 2048
            nn.Conv1d(detectors, 8*detectors, 33),              # (8*detectors) x 2016
            nn.ELU(),                                           # (8*detectors) x 2016
            nn.Conv1d(8*detectors, 8*detectors, 32),            # (8*detectors) x 1985
            nn.ELU(),                                           # (8*detectors) x 1985
            nn.Conv1d(8*detectors, 8*detectors, 17),            # (8*detectors) x 1969
            nn.ELU(),                                           # (8*detectors) x 1969
            nn.Conv1d(8*detectors, 8*detectors, 16),            # (8*detectors) x 1954
            nn.MaxPool1d(4),                                    # (8*detectors) x 488
            nn.ELU(),                                           # (8*detectors) x 488
            nn.Conv1d(8*detectors, 8*detectors, 17),            # (8*detectors) x 472
            nn.ELU(),                                           # (8*detectors) x 472
            nn.Conv1d(8*detectors, 16*detectors, 16),           # (16*detectors) x 457
            nn.ELU(),                                           # (16*detectors) x 457
            nn.Conv1d(16*detectors, 16*detectors, 9),           # (16*detectors) x 449
            nn.ELU(),                                           # (16*detectors) x 449
            nn.Conv1d(16*detectors, 16*detectors, 8),           # (16*detectors) x 442
            nn.MaxPool1d(3),                                    # (16*detectors) x 147
            nn.ELU(),                                           # (16*detectors) x 147
            nn.Conv1d(16*detectors, 16*detectors, 9),           # (16*detectors) x 139
            nn.ELU(),                                           # (16*detectors) x 139
            nn.Conv1d(16*detectors, 32*detectors, 8),           # (32*detectors) x 132
            nn.ELU(),                                           # (32*detectors) x 132
            nn.Conv1d(32*detectors, 32*detectors, 9),           # (32*detectors) x 124
            nn.ELU(),                                           # (32*detectors) x 124
            nn.Conv1d(32*detectors, 32*detectors, 8),           # (32*detectors) x 117
            nn.MaxPool1d(2),                                    # (32*detectors) x 58
            nn.ELU(),                                           # (32*detectors) x 58
            nn.Flatten(),                                       # 1856*detectors
        )
        self.cls_head = nn.Sequential(
            nn.Linear(1856*detectors, 64*detectors),            # 64*detectors
            nn.Dropout(p=0.5),                                  # 64*detectors
            nn.ELU(),                                           # 64*detectors
            nn.Linear(64*detectors, 64*detectors),              # 64*detectors
            nn.Dropout(p=0.5),                                  # 64*detectors
            nn.ELU(),                                           # 64*detectors
            nn.Linear(64*detectors, 2),                         # 2
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        return self.cls_head(x)

