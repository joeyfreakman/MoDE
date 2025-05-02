import torch
import torch.nn as nn
import einops
from timm import create_model


class ConvNextv2(nn.Module):
    def __init__(self, pretrained=True):
        super(ConvNextv2, self).__init__()

        self.convnext = create_model('convnextv2_tiny', pretrained=pretrained, num_classes=0)

    def forward(self, x):

        x = self.convnext(x)

        return x

