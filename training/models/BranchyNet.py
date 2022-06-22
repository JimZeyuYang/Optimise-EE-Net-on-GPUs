import torch
import torch.nn as nn

class BranchyNet(nn.Module):
    def __init__(self, exit_threshold=[0.5, 0.5, 0]):
        super(BranchyNet, self).__init__()

        self.fast_inference_mode = False

        self.exit_threshold = [torch.tensor([exit_threshold[0]], dtype=torch.float32), 
                               torch.tensor([exit_threshold[1]], dtype=torch.float32),
                               torch.tensor([exit_threshold[2]], dtype=torch.float32)]
        
        self.backbone = nn.ModuleList()
        self.exits = nn.ModuleList()
        self.exit_loss_weights = [0.6, 0.4, 0.3]

        self._build_backbone()
        self._build_exits()
    
    def _build_backbone(self):
        print('Error, the parent class')

    def _build_exits(self):
        print('Error, the parent class')

    def exit_criterion(self, x, t):
        with torch.no_grad():
            pk = nn.functional.softmax(x, dim=-1)
            entr = -torch.sum(pk * torch.log(pk))
            return entr < t
            
    def exit_criterion_top1(self, x, t):
        with torch.no_grad():
            pk = nn.functional.softmax(x, dim=-1)
            top1 = torch.max(pk) #x)
            return top1 > t

    @torch.jit.unused #decorator to skip jit comp
    def _forward_training(self, x):
        #TODO make jit compatible - not urgent
        #broken because returning list()
        res = []
        for bb, ee in zip(self.backbone, self.exits):
            x = bb(x)
            res.append(ee(x))
        return res

    def forward(self, x):
        if self.fast_inference_mode:
            for i, (bb, ee) in enumerate(zip(self.backbone, self.exits)):
                x = bb(x)
                res = ee(x) 
                if self.exit_criterion_top1(res, self.exit_threshold[i]):
                    # print("EE fired")
                    return res
                return res

        else: #used for training
            #calculate all exits
            return self._forward_training(x)

    def set_fast_inf_mode(self, mode=True):
        if mode:
            self.eval()
        self.fast_inference_mode = mode