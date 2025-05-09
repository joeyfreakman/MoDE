import torch
import torch.nn as nn
import torchvision
from .pretrained_resnets import FiLMResNet50Policy
from .convnext import ConvNextv2, FiLMConvNeXtV2Policy


class FiLMLayer(nn.Module):
    def __init__(self, num_features, condition_dim):
        super(FiLMLayer, self).__init__()
        self.num_features = num_features
        self.condition_dim = condition_dim
        self.gamma = nn.Linear(condition_dim, num_features)
        self.beta = nn.Linear(condition_dim, num_features)

        # Zero initialization
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, condition):
        gamma = self.gamma(condition)
        beta = self.beta(condition)
        x = (1 + gamma) * x + beta  # Using (1 + gamma) to start with identity transform
        return x.contiguous()


class Conv_fusion(nn.Module):
    def __init__(self, fuse_type='add'):
        super(Conv_fusion, self).__init__()

        self.pc_static = FiLMResNet50Policy(condition_dim=512, pretrained=False)
        self.pc_gripper = FiLMResNet50Policy(condition_dim=512, pretrained=False)

        self.rgb_static = FiLMResNet50Policy(condition_dim=512, pretrained=True)
        self.rgb_gripper = FiLMResNet50Policy(condition_dim=512, pretrained=True)

        self.fuse_type = fuse_type

        if fuse_type == 'pc_cond_rgb':
            self.film_pc_static = FiLMLayer(num_features=2048, condition_dim=2048)
            self.film_pc_gripper = FiLMLayer(num_features=2048, condition_dim=2048)
        elif fuse_type == 'rgb_cond_pc':
            self.film_rgb_static = FiLMLayer(num_features=2048, condition_dim=2048)
            self.film_rgb_gripper = FiLMLayer(num_features=2048, condition_dim=2048)
        elif fuse_type == 'cross_cond':
            self.film_pc_static = FiLMLayer(num_features=2048, condition_dim=2048)
            self.film_pc_gripper = FiLMLayer(num_features=2048, condition_dim=2048)

            self.film_rgb_static = FiLMLayer(num_features=2048, condition_dim=2048)
            self.film_rgb_gripper = FiLMLayer(num_features=2048, condition_dim=2048)
    # For point clouds of each camera view, first 3 dimensions stand for xyz, other 3 dimensions stand for rgb
    def forward(self, x):

        lang_emb = x['latent_goal']

        static_tokens = self.rgb_static(x['rgb_static'], lang_emb)
        gripper_tokens = self.rgb_gripper(x['rgb_gripper'], lang_emb)

        pc_static_tokens = self.pc_static(x['pc_static'], lang_emb)
        pc_gripper_tokens = self.pc_gripper(x['pc_gripper'], lang_emb)

        if self.fuse_type == 'pc_cond_rgb':
            pc_static_tokens = self.film_pc_static(pc_static_tokens, static_tokens)
            pc_gripper_tokens = self.film_pc_gripper(pc_gripper_tokens, gripper_tokens)

            cam_features = torch.stack([static_tokens, pc_static_tokens, gripper_tokens, pc_gripper_tokens], dim=1)

        elif self.fuse_type == 'rgb_cond_pc':
            static_tokens = self.film_rgb_static(static_tokens, pc_static_tokens)
            gripper_tokens = self.film_rgb_gripper(gripper_tokens, pc_gripper_tokens)

            cam_features = torch.stack([static_tokens, pc_static_tokens, gripper_tokens, pc_gripper_tokens], dim=1)

        elif self.fuse_type == 'cross_cond':
            cond_pc_static_tokens = self.film_pc_static(pc_static_tokens, static_tokens)
            cond_pc_gripper_tokens = self.film_pc_gripper(pc_gripper_tokens, gripper_tokens)

            cond_static_tokens = self.film_rgb_static(static_tokens, pc_static_tokens)
            cond_gripper_tokens = self.film_rgb_gripper(gripper_tokens, pc_gripper_tokens)

            cam_features = torch.stack([cond_static_tokens, cond_pc_static_tokens, cond_gripper_tokens, cond_pc_gripper_tokens], dim=1)

        elif self.fuse_type == 'add':
            cam_features = torch.stack([static_tokens+pc_static_tokens, gripper_tokens+pc_gripper_tokens], dim=1)
        elif self.fuse_type == 'cat':
            cam_features = torch.stack([static_tokens, pc_static_tokens, gripper_tokens, pc_gripper_tokens], dim=1)

        return cam_features


class Res_fusion(nn.Module):
    def __init__(self, fuse_type=None):
        super(Res_fusion, self).__init__()

        self.fuse_type = fuse_type

        if fuse_type == 'pc_cond_rgb':

            self.pc_static = FiLMResNet50Policy(condition_dim=2048, pretrained=False)
            self.pc_gripper = FiLMResNet50Policy(condition_dim=2048, pretrained=False)

            self.rgb_static = FiLMResNet50Policy(condition_dim=512, pretrained=True)
            self.rgb_gripper = FiLMResNet50Policy(condition_dim=512, pretrained=True)

        elif fuse_type == 'rgb_cond_pc':
            self.pc_static = FiLMResNet50Policy(condition_dim=512, pretrained=False)
            self.pc_gripper = FiLMResNet50Policy(condition_dim=512, pretrained=False)

            self.rgb_static = FiLMResNet50Policy(condition_dim=2048, pretrained=True)
            self.rgb_gripper = FiLMResNet50Policy(condition_dim=2048, pretrained=True)

    # For point clouds of each camera view, first 3 dimensions stand for xyz, other 3 dimensions stand for rgb
    def forward(self, x):

        lang_emb = x['latent_goal']

        if self.fuse_type == 'pc_cond_rgb':

            static_tokens = self.rgb_static(x['rgb_static'], lang_emb)
            gripper_tokens = self.rgb_gripper(x['rgb_gripper'], lang_emb)

            pc_static_tokens = self.pc_static(x['pc_static'], static_tokens)
            pc_gripper_tokens = self.pc_gripper(x['pc_gripper'], gripper_tokens)

            cam_features = torch.stack([static_tokens, pc_static_tokens, gripper_tokens, pc_gripper_tokens], dim=1)

        elif self.fuse_type == 'rgb_cond_pc':
            pc_static_tokens = self.pc_static(x['pc_static'], lang_emb)
            pc_gripper_tokens = self.pc_gripper(x['pc_gripper'], lang_emb)

            static_tokens = self.rgb_static(x['rgb_static'], pc_static_tokens)
            gripper_tokens = self.rgb_gripper(x['rgb_gripper'], pc_gripper_tokens)

            cam_features = torch.stack([static_tokens, pc_static_tokens, gripper_tokens, pc_gripper_tokens], dim=1)

        return cam_features
