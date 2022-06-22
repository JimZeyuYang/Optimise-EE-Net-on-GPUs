import torch
import torch.nn as nn
from models.BranchyNet import BranchyNet

class B_VGG_ImageNet(BranchyNet):
    def _build_backbone(self):
        bb1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )
        self.backbone.append(bb1)

        bb2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.backbone.append(bb2)

        bbf = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),

            nn.Flatten(),
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
        )
        self.backbone.append(bbf)

    def _build_exits(self):
        ee1 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),

            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),

            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),

            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),

            nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),

            nn.Flatten(),
            nn.Linear(12544, 1000, bias=False),
        )
        self.exits.append(ee1)

        ee2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),

            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),

            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),

            nn.Flatten(),
            nn.Linear(9408, 1000, bias=False),
        )
        self.exits.append(ee2)

        eef = nn.Sequential(
            nn.Linear(4096, 1000, bias=False),
        )
        self.exits.append(eef)

class B_VGG11_ImageNet(BranchyNet):
    def _build_backbone(self):
        bb1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.backbone.append(bb1)

        bbf = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=3, ceil_mode=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(4, stride=4, ceil_mode=True),

            nn.Flatten(),
            nn.Linear(4608, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
        )
        self.backbone.append(bbf)

    def _build_exits(self):
        ee1 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),

            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),

            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),

            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(4, stride=4, ceil_mode=True),

            nn.Flatten(),
            nn.Linear(512, 1000, bias=False),
        )
        self.exits.append(ee1)

        eef = nn.Sequential(
            nn.Linear(4096, 1000, bias=False),
        )
        self.exits.append(eef)