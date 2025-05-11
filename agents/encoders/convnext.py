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
        gamma = self.gamma(condition).unsqueeze(2).unsqueeze(3)
        beta = self.beta(condition).unsqueeze(2).unsqueeze(3)
        x = (1 + gamma) * x + beta  # Using (1 + gamma) to start with identity transform
        return x.contiguous()


class FiLMConvNeXtV2Policy(nn.Module):
    def __init__(self, condition_dim, pretrained=True, model_size='tiny'):
        super(FiLMConvNeXtV2Policy, self).__init__()
        # Load pretrained ConvNeXtV2
        model_name = f'convnextv2_{model_size}'
        self.convnext = create_model(model_name, pretrained=pretrained, num_classes=0)

        # ConvNeXtV2 feature dimensions for different stages
        # These depend on the model size, here are the values for 'base'
        if model_size == 'base':
            feature_dims = [128, 256, 512, 1024]
        elif model_size == 'tiny':
            feature_dims = [96, 192, 384, 768]
        elif model_size == 'small':
            feature_dims = [96, 192, 384, 768]
        elif model_size == 'large':
            feature_dims = [192, 384, 768, 1536]
        else:  # Default to 'base'
            feature_dims = [128, 256, 512, 1024]

        # Add FiLM layers after each stage
        self.film1 = FiLMLayer(feature_dims[0], condition_dim)
        self.film2 = FiLMLayer(feature_dims[1], condition_dim)
        self.film3 = FiLMLayer(feature_dims[2], condition_dim)
        self.film4 = FiLMLayer(feature_dims[3], condition_dim)

    def forward(self, x, condition):
        if len(condition.shape) == 3:
            condition = condition.squeeze(1)

        # ConvNeXtV2 stem
        x = self.convnext.stem(x)

        # Stage 1
        x = self.convnext.stages[0](x)
        x = self.film1(x, condition)

        # Stage 2
        x = self.convnext.stages[1](x)
        x = self.film2(x, condition)

        # Stage 3
        x = self.convnext.stages[2](x)
        x = self.film3(x, condition)

        # Stage 4
        x = self.convnext.stages[3](x)
        x = self.film4(x, condition)

        # Global pooling and flatten
        x = self.convnext.head.global_pool(x)
        x = x.flatten(1)

        return x  # Return latent features