import torch
import torch.nn as nn
import torchvision
from .pretrained_resnets import FiLMResNet50Policy
from .convnext import ConvNextv2, FiLMConvNeXtV2Policy


class Conv_pc_rgb(nn.Module):
    def __init__(self):
        super(Conv_pc_rgb, self).__init__()

        # self.pc_static = ConvNextv2(pretrained=False)
        # self.pc_gripper = ConvNextv2(pretrained=False)

        self.pc_static = FiLMResNet50Policy(condition_dim=512, pretrained=False)
        self.pc_gripper = FiLMResNet50Policy(condition_dim=512, pretrained=False)

        self.rgb_static = FiLMResNet50Policy(condition_dim=512, pretrained=True)
        self.rgb_gripper = FiLMResNet50Policy(condition_dim=512, pretrained=True)

    # For point clouds of each camera view, first 3 dimensions stand for xyz, other 3 dimensions stand for rgb
    def forward(self, x):

        lang_emb = x['latent_goal']

        static_tokens = self.rgb_static(x['rgb_static'], lang_emb)
        gripper_tokens = self.rgb_gripper(x['rgb_gripper'], lang_emb)

        # static_inputs = self.preprocessing_static(x['pc_static'])
        # gripper_inputs = self.preprocessing_gripper(x['pc_gripper'])

        pc_static_tokens = self.pc_static(x['pc_static'], lang_emb)
        pc_gripper_tokens = self.pc_gripper(x['pc_gripper'], lang_emb)

        obs_tokens = {'rgb_tokens': torch.stack([static_tokens, gripper_tokens], dim=1),
                      'pc_tokens': torch.stack([pc_static_tokens, pc_gripper_tokens], dim=1)}

        return obs_tokens
