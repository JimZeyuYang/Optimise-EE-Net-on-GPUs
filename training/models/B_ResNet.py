import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BranchyNet import BranchyNet

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = F.relu((self.bn1(self.conv1(input))))
        input = F.relu((self.bn2(self.conv2(input))))
        input = input + shortcut
        return F.relu((input))

class B_ResNet110_CIFAR10(BranchyNet):
    def _make_layer(self, channels, num_blocks, downsample):
        layers = []
        if downsample:
            layers.append(ResBlock(self.in_channels, channels, True))
            self.in_channels = channels
            num_blocks = num_blocks - 1
        for _ in range (num_blocks):
            layers.append(ResBlock(self.in_channels, channels, False))
            self.in_channels = channels
        return nn.Sequential(*layers)

    def _build_backbone(self):
        self.in_channels = 16
        bb1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
        )
        self.backbone.append(bb1)

        bb2 = self._make_layer(16, num_blocks=18, downsample=False)
        self.backbone.append(bb2)

        bbf = nn.Sequential(
            self._make_layer(32, num_blocks=18, downsample=True),
            self._make_layer(64, num_blocks=18, downsample=True),
            nn.AvgPool2d(4, stride=2),
            nn.Flatten()
        )
        self.backbone.append(bbf)

    def _build_exits(self):
        ee1 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.LocalResponseNorm(3, alpha=5e-05, beta=0.75),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(28800, 10, bias=False),
        )
        self.exits.append(ee1)

        ee2 = nn.Sequential(
            ResBlock(16, 16, False),
            nn.Flatten(),
            nn.Linear(14400, 10, bias=False),
        )
        self.exits.append(ee2)

        eef = nn.Sequential(
            nn.Linear(576, 10, bias=False),
        )
        self.exits.append(eef)