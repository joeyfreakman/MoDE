import torch
import torch.nn as nn
import torchvision
from .pretrained_resnets import FiLMResNet18Policy
from .convnext import ConvNextv2


class Conv_pc(nn.Module):
    def __init__(self,):
        super(Conv_pc, self).__init__()

        self.pc_static = ConvNextv2(pretrained=False)
        self.pc_gripper = ConvNextv2(pretrained=False)

    # For point clouds of each camera view, first 3 dimensions stand for xyz, other 3 dimensions stand for rgb
    def forward(self, x):

        static_tokens = self.pc_static(x['pc_static'])
        gripper_tokens = self.pc_gripper(x['pc_gripper'])

        cam_features = torch.stack([static_tokens, gripper_tokens], dim=1)

        return cam_features
