#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as modelzoo

# non-depracated solution
from torch.hub import load_state_dict_from_url

# from modules.bn import InPlaceABNSync as BatchNorm2d

resnet18_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_chan, out_chan, stride)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.conv2 = conv3x3(out_chan, out_chan)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chan),
                )

    def forward(self, x):
        residual = self.conv1(x)
        residual = F.relu(self.bn1(residual))
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = shortcut + residual
        out = self.relu(out)
        return out

# Create a layer of basic blocks
def create_layer_basic(in_chan, out_chan, bnum, stride=1):
    layers = [BasicBlock(in_chan, out_chan, stride=stride)]
    for i in range(bnum-1):
        layers.append(BasicBlock(out_chan, out_chan, stride=1))
    return nn.Sequential(*layers)

class Resnet18(nn.Module):
    def __init__(self, pretrained=True):
        super(Resnet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = create_layer_basic(64, 64, bnum=2, stride=1)
        self.layer2 = create_layer_basic(64, 128, bnum=2, stride=2)
        self.layer3 = create_layer_basic(128, 256, bnum=2, stride=2)
        self.layer4 = create_layer_basic(256, 512, bnum=2, stride=2)
        
        # will raise when no internet acess
        self.init_weight()

    def init_weight(self):
        # Modern non-deprecated model loader
        state_dict = load_state_dict_from_url(resnet18_url, progress=True)
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            if 'fc' in k:
                continue
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)

    def get_params(self):
        wd_params, nowd_params = [], []

        for name, module in self.named_modules():

            if isinstance(module, (nn.Linear, nn.Conv2d)):

                wd_params.append(module.weight)

                if not module.bias is None:

                    nowd_params.append(module.bias)

            elif isinstance(module,  nn.BatchNorm2d):

                nowd_params += list(module.parameters())

        return wd_params, nowd_params 


if __name__ == "__main__":
    has_cuda = torch.cuda.is_available()

    device = torch.device("cuda") if has_cuda else torch.device("cpu")

    if not has_cuda:
        print("Warning, cuda is not supported on your device.")

    net = Resnet18(pretrained=False).to(device)
    net.eval()

    x = torch.randn(16, 3, 224, 224, device=device)

    with torch.no_grad():
        feat8, feat16, feat32 = net(x)

    print(f"feat8 size:  {feat8.size()}")
    print(f"feat16 size: {feat16.size()}")
    print(f"feat32 size: {feat32.size()}")

    wd_params, nowd_params = net.get_params()

    print(f"Wd Params Size: {len(wd_params)}")
    print(f"Nowd Params Size: {len(nowd_params)}")
