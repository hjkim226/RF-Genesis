"""
Lightweight PointNet implementation for radar point cloud tasks.

Designed as a clean, dependency-light stub for the SMAL/SMIL downstream validation phase.
Based on the original PointNet paper (Qi et al. 2017) but heavily simplified.

Supports:
- Classification (domain, action, etc.)
- Regression (pose parameters, etc.)

Input: (B, N, C)  — typically C=6 for [x,y,z,R,energy,V]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    def __init__(self, in_channels: int = 6, feature_dim: int = 1024):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, feature_dim, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, C)
        returns global feature (B, feature_dim)
        """
        x = x.transpose(2, 1)           # (B, C, N)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))     # (B, feature_dim, N)

        x = torch.max(x, 2)[0]          # global max pooling -> (B, feature_dim)
        return x


class PointNetClassifier(nn.Module):
    """PointNet for classification tasks (domain, action, etc.)."""

    def __init__(self, in_channels: int = 6, num_classes: int = 4, feature_dim: int = 1024, dropout: float = 0.3):
        super().__init__()
        self.encoder = PointNetEncoder(in_channels, feature_dim)
        self.fc1 = nn.Linear(feature_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        x = F.relu(self.bn1(self.fc1(feat)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class PointNetRegressor(nn.Module):
    """PointNet for regression tasks (e.g. pose parameters)."""

    def __init__(self, in_channels: int = 6, output_dim: int = 75, feature_dim: int = 1024):
        super().__init__()
        self.encoder = PointNetEncoder(in_channels, feature_dim)
        self.fc1 = nn.Linear(feature_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        x = F.relu(self.bn1(self.fc1(feat)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x
