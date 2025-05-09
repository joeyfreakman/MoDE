import torch
import torch.nn as nn
import torchvision
from timm import create_model


class DaViTFeatureExtractor(nn.Module):
    def __init__(self, model_name='davit_tiny', pretrained=True):
        """
        Initialize a DaViT model for feature extraction (without classification head)

        Args:
            model_name: The name of the DaViT model ('davit_tiny', 'davit_small', 'davit_base')
            pretrained: Whether to load pretrained weights (ImageNet-1K)
        """
        super().__init__()
        # Create the model without the classification head by setting num_classes=0
        self.model = create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove the classifier head
        )

    def forward(self, x):
        """
        Extract features from the input images

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]

        Returns:
            Features from the backbone
        """
        features = self.model(x)
        return features


class Dinov2(nn.Module):
    def __init__(self, model_name='vit_small_patch14_dinov2', pretrained=True, img_size=224):

        super().__init__()
        # Create the model without the classification head by setting num_classes=0
        self.model = create_model(
            model_name,
            pretrained=pretrained,
            img_size=img_size,
            num_classes=0,  # Remove the classifier head
        )

    def forward(self, x):
        features = self.model(x)
        return features


class Dinov2_rgb(nn.Module):
    def __init__(self,):
        super(Dinov2_rgb, self).__init__()

        self.rgb_static = Dinov2(model_name='vit_small_patch14_dinov2', pretrained=True, img_size=224)
        self.rgb_gripper = Dinov2(model_name='vit_small_patch14_dinov2', pretrained=True, img_size=112)

    # For point clouds of each camera view, first 3 dimensions stand for xyz, other 3 dimensions stand for rgb
    def forward(self, x):

        lang_emb = x['latent_goal']

        static_tokens = self.rgb_static(x['rgb_static'])
        gripper_tokens = self.rgb_gripper(x['rgb_gripper'])

        cam_features = torch.stack([static_tokens, gripper_tokens], dim=1)

        return cam_features


class Davit_rgb(nn.Module):
    def __init__(self,):
        super(Davit_rgb, self).__init__()

        self.rgb_static = DaViTFeatureExtractor(model_name='davit_tiny', pretrained=True)
        self.rgb_gripper = DaViTFeatureExtractor(model_name='davit_tiny', pretrained=True)

    # For point clouds of each camera view, first 3 dimensions stand for xyz, other 3 dimensions stand for rgb
    def forward(self, x):

        lang_emb = x['latent_goal']

        static_tokens = self.rgb_static(x['rgb_static'])
        gripper_tokens = self.rgb_gripper(x['rgb_gripper'])

        cam_features = torch.stack([static_tokens, gripper_tokens], dim=1)

        return cam_features


class Davit_pc_rgb(nn.Module):
    def __init__(self,):
        super(Davit_pc_rgb, self).__init__()

        self.pc_static = DaViTFeatureExtractor(model_name='davit_tiny', pretrained=False)
        self.pc_gripper = DaViTFeatureExtractor(model_name='davit_tiny', pretrained=False)

        self.rgb_static = DaViTFeatureExtractor(model_name='davit_tiny', pretrained=True)
        self.rgb_gripper = DaViTFeatureExtractor(model_name='davit_tiny', pretrained=True)

    # For point clouds of each camera view, first 3 dimensions stand for xyz, other 3 dimensions stand for rgb
    def forward(self, x):

        lang_emb = x['latent_goal']

        static_tokens = self.rgb_static(x['rgb_static'])
        gripper_tokens = self.rgb_gripper(x['rgb_gripper'])

        pc_static_tokens = self.pc_static(x['pc_static'])
        pc_gripper_tokens = self.pc_gripper(x['pc_gripper'])

        cam_features = torch.stack([static_tokens, pc_static_tokens, gripper_tokens, pc_gripper_tokens], dim=1)

        return cam_features