import torch
import torch.nn as nn
from models.BranchyNet import BranchyNet

class B_Lenet_MNIST(BranchyNet):
    def _build_backbone(self):
        #Starting conv2d layer
        bb1 = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=5, stride=1, padding=3),
            nn.MaxPool2d(2, stride=2), #ksize, stride
            nn.ReLU(True)
        )
        self.backbone.append(bb1)
        
        #remaining backbone
        bbf = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size=5, stride=1, padding=3),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.ReLU(True),
            nn.Conv2d(10, 20, kernel_size=5, stride=1, padding=3),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(720,84)
        )
        self.backbone.append(bbf)

    def _build_exits(self): #adding early exits/branches
        #early exit 1
        ee1 = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(640,10, bias=False)
        )
        self.exits.append(ee1)

        #final exit
        eeF = nn.Sequential(
            nn.Linear(84,10, bias=False),
        )
        self.exits.append(eeF)



class B_LenetRedesigned_MNIST(BranchyNet):
    def _build_backbone(self):
        #Starting conv2d layer
        bb1 = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2), #ksize, stride
        )
        self.backbone.append(bb1)
        
        #remaining backbone
        bbf = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(10, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Flatten(),
            nn.Linear(320,84)
        )
        self.backbone.append(bbf)

    def _build_exits(self): #adding early exits/branches
        #early exit 1
        ee1 = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Flatten(),
            nn.Linear(490, 10, bias=False)
        )
        self.exits.append(ee1)

        #final exit
        eeF = nn.Sequential(
            nn.Linear(84,10, bias=False),
        )
        self.exits.append(eeF)




class B_LenetNarrow1_MNIST(BranchyNet):
    def _build_backbone(self):
        #Starting conv2d layer
        bb1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2), #ksize, stride
        )
        self.backbone.append(bb1)
        
        #remaining backbone
        bbf = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Flatten(),
        )
        self.backbone.append(bbf)

    def _build_exits(self): #adding early exits/branches
        #early exit 1
        ee1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(6272,10, bias=False),
        )
        self.exits.append(ee1)

        #final exit
        eeF = nn.Sequential(
            nn.Linear(6272,10, bias=False),
        )
        self.exits.append(eeF)

class B_LenetNarrow2_MNIST(BranchyNet):
    def _build_backbone(self):
        #Starting conv2d layer
        bb1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2), #ksize, stride
        )
        self.backbone.append(bb1)
        
        #remaining backbone
        bbf = nn.Sequential(
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Flatten(),
        )
        self.backbone.append(bbf)

    def _build_exits(self): #adding early exits/branches
        #early exit 1
        ee1 = nn.Sequential(
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(3920,10, bias=False)
        )
        self.exits.append(ee1)

        #final exit
        eeF = nn.Sequential(
            nn.Linear(3920,10, bias=False),
        )
        self.exits.append(eeF)

class B_LenetMassiveLayer_ImageNet(BranchyNet):
    def _build_backbone(self):
        #Starting conv2d layer
        bb1 = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            
        )
        self.backbone.append(bb1)
        
        #remaining backbone
        bbf = nn.Sequential(
            nn.Conv2d(256, 5, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(5, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 5, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(5, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(8, stride=8),
            nn.Flatten(),
        )
        self.backbone.append(bbf)

    def _build_exits(self): #adding early exits/branches
        #early exit 1
        ee1 = nn.Sequential(
            nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(8, stride=8),
            nn.Flatten(),
            nn.Linear(2352,1000, bias=False)
        )
        self.exits.append(ee1)

        #final exit
        eeF = nn.Sequential(
            nn.Linear(2352,1000, bias=False),
        )
        self.exits.append(eeF)