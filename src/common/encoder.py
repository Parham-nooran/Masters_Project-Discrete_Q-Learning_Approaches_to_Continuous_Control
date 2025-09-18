import torch
import torch.nn as nn
from src.common.networks import LayerNormMLP


class VisionEncoder(nn.Module):
    """Vision encoder based on DrQ-v2 architecture."""

    def __init__(self, config, shape=84):
        super().__init__()
        self.shape = shape

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate conv output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, shape, shape)
            conv_output_size = self.conv(dummy_input).shape[1]

        self.mlp = LayerNormMLP(
            [conv_output_size, config.layer_size_bottleneck], activate_final=True
        )

    def forward(self, x):
        if not x.is_cuda and next(self.parameters()).is_cuda:
            x = x.cuda()
        # Handle both normalized [0,1] and pixel [0,255] inputs
        if x.max() > 1.0:
            x = x / 255.0 - 0.5  # Normalize to [-0.5, 0.5] for [0,255] inputs
        else:
            x = x - 0.5  # Assume already normalized to [0,1], shift to [-0.5, 0.5]

        x = self.conv(x)
        return self.mlp(x)
