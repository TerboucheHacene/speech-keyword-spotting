from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F


class M5(nn.Module):
    """M5 model definition, composed of a features extraction module, followed by
    an MLP model

    Parameters
    ----------
    depth : int
        Number of input channels
    num_classes : int
        Number of classes
    stride : int
        Stride for the first convolutional layer
    n_channels : int
        Number of channels for the first convolutional layer

    """

    def __init__(
        self,
        depth: int = 1,
        num_classes: int = 35,
        stride: int = 16,
        n_channels: int = 32,
        **kwargs,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(depth, n_channels, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channels)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channels, n_channels, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channels)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channels, 2 * n_channels, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channels)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channels, 2 * n_channels, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channels)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channels, num_classes)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--depth", type=int, default=1)
        parser.add_argument("--num_classes", type=int, default=35)
        parser.add_argument("--stride", type=int, default=16)
        parser.add_argument("--n_channels", type=int, default=32)
        return parser

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return x.squeeze()  # (batch_size, num_classes)
