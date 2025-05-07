import torch
import torch.nn as nn
import torchvision
from .pretrained_resnets import FiLMResNet50Policy
from .convnext import ConvNextv2, FiLMConvNeXtV2Policy


class Conv_pc_rgb(nn.Module):
    def __init__(self, fuse_type='add'):
        super(Conv_pc_rgb, self).__init__()

        self.pc_static = ConvNextv2(pretrained=False)
        self.pc_gripper = ConvNextv2(pretrained=False)

        # self.pc_static = FiLMResNet50Policy(condition_dim=512, pretrained=True)
        # self.pc_gripper = FiLMResNet50Policy(condition_dim=512, pretrained=True)

        # Add a learnable preprocessing layer to adapt xyz to a more RGB-like distribution
        # self.preprocessing_static = nn.Sequential(
        #     nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.BatchNorm2d(3),
        #     nn.ReLU()
        # )
        # self.preprocessing_gripper = nn.Sequential(
        #     nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.BatchNorm2d(3),
        #     nn.ReLU()
        # )

        self.rgb_static = FiLMResNet50Policy(condition_dim=512, pretrained=True)
        self.rgb_gripper = FiLMResNet50Policy(condition_dim=512, pretrained=True)

        self.fuse_type = fuse_type

    # For point clouds of each camera view, first 3 dimensions stand for xyz, other 3 dimensions stand for rgb
    def forward(self, x):

        lang_emb = x['latent_goal']

        static_tokens = self.rgb_static(x['rgb_static'], lang_emb)
        gripper_tokens = self.rgb_gripper(x['rgb_gripper'], lang_emb)

        # static_inputs = self.preprocessing_static(x['pc_static'])
        # gripper_inputs = self.preprocessing_gripper(x['pc_gripper'])

        pc_static_tokens = self.pc_static(x['pc_static'])
        pc_gripper_tokens = self.pc_gripper(x['pc_gripper'])

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
    def __init__(self, pretrained=False):
        super(Conv_pc, self).__init__()

        # self.pc_static = ConvNextv2(pretrained=False)
        # self.pc_gripper = ConvNextv2(pretrained=False)

        self.pc_static = FiLMResNet50Policy(condition_dim=512, pretrained=pretrained)
        self.pc_gripper = FiLMResNet50Policy(condition_dim=512, pretrained=pretrained)

        self.pretrained = pretrained

        if pretrained:
            # Add a learnable preprocessing layer to adapt xyz to a more RGB-like distribution
            self.preprocessing_static = nn.Sequential(
                nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(3),
                nn.ReLU()
            )
            self.preprocessing_gripper = nn.Sequential(
                nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(3),
                nn.ReLU()
            )

    # For point clouds of each camera view, first 3 dimensions stand for xyz, other 3 dimensions stand for rgb
    def forward(self, x):

        lang_emb = x['latent_goal']

        if self.pretrained:
            static_inputs = self.preprocessing_static(x['pc_static'])
            gripper_inputs = self.preprocessing_gripper(x['pc_gripper'])
        else:
            static_inputs = x['pc_static']
            gripper_inputs = x['pc_gripper']

        static_tokens = self.pc_static(static_inputs, lang_emb)
        gripper_tokens = self.pc_gripper(gripper_inputs, lang_emb)

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
