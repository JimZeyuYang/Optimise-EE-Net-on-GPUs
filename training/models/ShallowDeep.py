import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BranchyNet import BranchyNet



class S_VGG16_CIFAR10(BranchyNet):
    def _build_backbone(self):
        self.exit_loss_weights = [0.5, 0.5, 0.7, 0.7, 0.9, 0.9, 2]
        self.in_channels = 16
        bb1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.backbone.append(bb1)

        bb2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.backbone.append(bb2)

        bb3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )
        self.backbone.append(bb3)

        bb4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.backbone.append(bb4)

        bb5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )
        self.backbone.append(bb5)

        bb6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.backbone.append(bb6)

        bbf = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),

            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
        )
        self.backbone.append(bbf)

    def _build_exits(self):
        ee1 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(4096, 10, bias=False),
        )
        self.exits.append(ee1)

        ee2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(2048, 10, bias=False),
        )
        self.exits.append(ee2)

        ee3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(4096, 10, bias=False),
        )
        self.exits.append(ee3)

        ee4 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(1024, 10, bias=False),
        )
        self.exits.append(ee4)

        ee5 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(2048, 10, bias=False),
        )
        self.exits.append(ee5)

        ee6 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(512, 10, bias=False),
        )
        self.exits.append(ee6)

        eef = nn.Sequential(
            nn.Linear(512, 10, bias=False),
        )
        self.exits.append(eef)



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

class S_ResNet56_CIFAR10(BranchyNet):
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
        self.exit_loss_weights = [0.5, 0.5, 0.7, 0.7, 0.9, 0.9, 2]
        self.in_channels = 16
        bb1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            self._make_layer(16, num_blocks=4, downsample=False)
        )
        self.backbone.append(bb1)

        bb2 = self._make_layer(16, num_blocks=4, downsample=False)
        self.backbone.append(bb2)

        bb3 = nn.Sequential(
            self._make_layer(16, num_blocks=1, downsample=False),
            self._make_layer(32, num_blocks=3, downsample=True)
        )
        self.backbone.append(bb3)

        bb4 = self._make_layer(32, num_blocks=4, downsample=False)
        self.backbone.append(bb4)

        bb5 = nn.Sequential(
            self._make_layer(32, num_blocks=2, downsample=False),
            self._make_layer(64, num_blocks=2, downsample=True)
        )
        self.backbone.append(bb5)

        bb6 = self._make_layer(64, num_blocks=4, downsample=False)
        self.backbone.append(bb6)

        bbf = nn.Sequential(
            self._make_layer(64, num_blocks=3, downsample=False),
            nn.AvgPool2d(4, stride=2),
            nn.Flatten()
        )
        self.backbone.append(bbf)

    def _build_exits(self):
        ee1 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(3600, 10, bias=False),
        )
        self.exits.append(ee1)

        ee2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(3600, 10, bias=False),
        )
        self.exits.append(ee2)

        ee3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(1568, 10, bias=False),
        )
        self.exits.append(ee3)

        ee4 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(1568, 10, bias=False),
        )
        self.exits.append(ee4)

        ee5 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(1024, 10, bias=False),
        )
        self.exits.append(ee5)

        ee6 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(1024, 10, bias=False),
        )
        self.exits.append(ee6)

        eef = nn.Sequential(
            nn.Linear(576, 10, bias=False),
        )
        self.exits.append(eef)