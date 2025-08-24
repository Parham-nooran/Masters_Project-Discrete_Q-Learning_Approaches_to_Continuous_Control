import torch.nn as nn
import torch.nn.functional as F


class LayerNormAndResidualMLP(nn.Module):
    """MLP with layer norm and residual connections."""

    def __init__(self, size, num_blocks=1):
        super().__init__()
        self.input_norm = nn.LayerNorm(size)
        self.blocks = nn.ModuleList([ResidualBlock(size) for _ in range(num_blocks)])

    def forward(self, x):
        x = self.input_norm(x)
        for block in self.blocks:
            x = block(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with layer norm."""

    def __init__(self, size):
        super().__init__()
        self.linear = nn.Linear(size, size)
        self.layer_norm = nn.LayerNorm(size)

    def forward(self, x):
        return F.elu(self.layer_norm(x + self.linear(F.elu(x))))


class LayerNormMLP(nn.Module):
    """MLP with layer normalization."""

    def __init__(self, sizes, activate_final=False):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2 or activate_final:
                self.layer_norms.append(nn.LayerNorm(sizes[i + 1]))

        self.activate_final = activate_final

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1 or self.activate_final:
                x = self.layer_norms[i](x)
                x = F.elu(x)
        return x
