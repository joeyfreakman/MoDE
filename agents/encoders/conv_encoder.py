import torch
import torch.nn as nn
import torchvision
from .pretrained_resnets import FiLMResNet50Policy
from .convnext import ConvNextv2, FiLMConvNeXtV2Policy


class Conv_pc_rgb(nn.Module):
    def __init__(self, fuse_type='add'):
        super(Conv_pc_rgb, self).__init__()

        # self.pc_static = ConvNextv2(pretrained=False)
        # self.pc_gripper = ConvNextv2(pretrained=False)

        self.pc_static = FiLMResNet50Policy(condition_dim=512, pretrained=False)
        self.pc_gripper = FiLMResNet50Policy(condition_dim=512, pretrained=False)

        self.rgb_static = FiLMResNet50Policy(condition_dim=512, pretrained=True)
        self.rgb_gripper = FiLMResNet50Policy(condition_dim=512, pretrained=True)

        self.fuse_type = fuse_type

    # For point clouds of each camera view, first 3 dimensions stand for xyz, other 3 dimensions stand for rgb
    def forward(self, x):

        lang_emb = x['latent_goal']

        static_tokens = self.rgb_static(x['rgb_static'], lang_emb)
        gripper_tokens = self.rgb_gripper(x['rgb_gripper'], lang_emb)

        pc_static_tokens = self.pc_static(x['pc_static'], lang_emb)
        pc_gripper_tokens = self.pc_gripper(x['pc_gripper'], lang_emb)

        if self.fuse_type == 'add':

            cam_features = torch.stack([static_tokens+pc_static_tokens, gripper_tokens+pc_gripper_tokens], dim=1)

        elif self.fuse_type == 'cat':

            cam_features = torch.stack([static_tokens, pc_static_tokens, gripper_tokens, pc_gripper_tokens], dim=1)

        return cam_features


class Conv_rgb(nn.Module):
    def __init__(self,):
        super(Conv_rgb, self).__init__()

        # self.pc_static = ConvNextv2(pretrained=False)
        # self.pc_gripper = ConvNextv2(pretrained=False)

        self.rgb_static = FiLMResNet50Policy(condition_dim=512, pretrained=True)
        self.rgb_gripper = FiLMResNet50Policy(condition_dim=512, pretrained=True)

    # For point clouds of each camera view, first 3 dimensions stand for xyz, other 3 dimensions stand for rgb
    def forward(self, x):

        lang_emb = x['latent_goal']

        static_tokens = self.rgb_static(x['rgb_static'], lang_emb)
        gripper_tokens = self.rgb_gripper(x['rgb_gripper'], lang_emb)

        cam_features = torch.stack([static_tokens, gripper_tokens], dim=1)

        return cam_features


class Conv_pc(nn.Module):
    def __init__(self,):
        super(Conv_pc, self).__init__()

        # self.pc_static = ConvNextv2(pretrained=False)
        # self.pc_gripper = ConvNextv2(pretrained=False)

        self.pc_static = FiLMResNet50Policy(condition_dim=512, pretrained=False)
        self.pc_gripper = FiLMResNet50Policy(condition_dim=512, pretrained=False)

    # For point clouds of each camera view, first 3 dimensions stand for xyz, other 3 dimensions stand for rgb
    def forward(self, x):

        lang_emb = x['latent_goal']

        static_tokens = self.pc_static(x['pc_static'], lang_emb)
        gripper_tokens = self.pc_gripper(x['pc_gripper'], lang_emb)

        cam_features = torch.stack([static_tokens, gripper_tokens], dim=1)

        return cam_features


class Convnext_pc(nn.Module):
    def __init__(self,):
        super(Convnext_pc, self).__init__()

        self.pc_static = FiLMConvNeXtV2Policy(condition_dim=512, pretrained=False, model_size='tiny')
        self.pc_gripper = FiLMConvNeXtV2Policy(condition_dim=512, pretrained=False, model_size='tiny')

    # For point clouds of each camera view, first 3 dimensions stand for xyz, other 3 dimensions stand for rgb
    def forward(self, x):

        lang_emb = x['latent_goal']

        static_tokens = self.pc_static(x['pc_static'], lang_emb)
        gripper_tokens = self.pc_gripper(x['pc_gripper'], lang_emb)

        cam_features = torch.stack([static_tokens, gripper_tokens], dim=1)

        return cam_features
