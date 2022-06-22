import torch
import torch.nn as nn
from models.BranchyNet import BranchyNet

class B_Alexnet_CIFAR10(BranchyNet):
    def _build_backbone(self):
        bb1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            nn.LocalResponseNorm(3, alpha=5e-05, beta=0.75),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            nn.LocalResponseNorm(3, alpha=5e-05, beta=0.75),
        )
        self.backbone.append(bb1)

        bb2 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )
        self.backbone.append(bb2)

        bbf = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.5, inplace=False),
        )
        self.backbone.append(bbf)

    def _build_exits(self):
        ee1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            nn.Flatten(),
            nn.Linear(512, 10, bias=False),
        )
        self.exits.append(ee1)

        ee2 = nn.Sequential(
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            nn.Conv2d(96, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            nn.Flatten(),
            nn.Linear(128, 10, bias=False),
        )
        self.exits.append(ee2)

        eef = nn.Sequential(
            nn.Linear(128, 10, bias=False),
        )
        self.exits.append(eef)



class B_AlexnetRedesigned_CIFAR10(BranchyNet):
    def _build_backbone(self):
        bb1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            nn.LocalResponseNorm(3, alpha=5e-05, beta=0.75),
        )
        self.backbone.append(bb1)

        bb2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            nn.LocalResponseNorm(3, alpha=5e-05, beta=0.75),
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )
        self.backbone.append(bb2)

        bbf = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.5, inplace=False),
        )
        self.backbone.append(bbf)

    def _build_exits(self):
        ee1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            nn.Flatten(),
            nn.Linear(512, 10, bias=False),
        )
        self.exits.append(ee1)

        ee2 = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=3, ceil_mode=True),
            nn.Flatten(),
            nn.Linear(288, 10, bias=False),
        )
        self.exits.append(ee2)

        eef = nn.Sequential(
            nn.Linear(128, 10, bias=False),
        )
        self.exits.append(eef)



class B_Alexnet_ImageNet(BranchyNet):
    def _build_backbone(self):
        bb1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            nn.LocalResponseNorm(3, alpha=5e-05, beta=0.75),
        )
        self.backbone.append(bb1)

        bb2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            nn.LocalResponseNorm(3, alpha=5e-05, beta=0.75),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )
        self.backbone.append(bb2)

        bbf = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            nn.Flatten(),
            nn.Linear(9216, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5, inplace=False),
        )
        self.backbone.append(bbf)

    def _build_exits(self):
        ee1 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            nn.LocalResponseNorm(3, alpha=5e-05, beta=0.75),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            nn.Flatten(),
            nn.Linear(6912, 1000, bias=False),
        )
        self.exits.append(ee1)

        ee2 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            nn.Flatten(),
            nn.Linear(4608, 1000, bias=False),
        )
        self.exits.append(ee2)

        eef = nn.Sequential(
            nn.Linear(4096, 1000, bias=False),
        )
        self.exits.append(eef)