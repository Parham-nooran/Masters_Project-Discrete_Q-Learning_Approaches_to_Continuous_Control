import torch
import torch.nn as nn
from src.cqn.networks import ImgChLayerNorm
import src.cqn.utils as utils


class MultiViewCNNEncoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()


        if len(obs_shape) == 3:
            self.num_views = 1
            channels = obs_shape[0]
            self.repr_dim = 256 * 5 * 5  # for 84x84 input

            conv_net = nn.Sequential(
                nn.Conv2d(channels, 32, 4, stride=2, padding=1),
                ImgChLayerNorm(32),
                nn.SiLU(),
                nn.Conv2d(32, 64, 4, stride=2, padding=1),
                ImgChLayerNorm(64),
                nn.SiLU(),
                nn.Conv2d(64, 128, 4, stride=2, padding=1),
                ImgChLayerNorm(128),
                nn.SiLU(),
                nn.Conv2d(128, 256, 4, stride=2, padding=1),
                ImgChLayerNorm(256),
                nn.SiLU(),
            )
            self.conv_nets = nn.ModuleList([conv_net])
        else:
            # Multi-view case (RLBench)
            # assert len(obs_shape) == 4
            self.num_views = obs_shape[0]
            self.repr_dim = self.num_views * 256 * 5 * 5  # for 84,84. hard-coded

            self.conv_nets = nn.ModuleList()
            for _ in range(self.num_views):
                conv_net = nn.Sequential(
                    nn.Conv2d(obs_shape[1], 32, 4, stride=2, padding=1),
                    ImgChLayerNorm(32),
                    nn.SiLU(),
                    nn.Conv2d(32, 64, 4, stride=2, padding=1),
                    ImgChLayerNorm(64),
                    nn.SiLU(),
                    nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    ImgChLayerNorm(128),
                    nn.SiLU(),
                    nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    ImgChLayerNorm(256),
                    nn.SiLU(),
                )
                self.conv_nets.append(conv_net)

        self.apply(utils.weight_init)

    def forward(self, obs: torch.Tensor):
        obs = obs / 255.0 - 0.5

        if self.num_views == 1:
            # Single view: obs shape is [B, C, H, W]
            h = self.conv_nets[0](obs)
            h = h.view(h.shape[0], -1)
            return h
        else:
            # Multi-view: obs shape is [B, num_views, C, H, W]
            hs = []
            for v in range(self.num_views):
                h = self.conv_nets[v](obs[:, v])
                h = h.view(h.shape[0], -1)
                hs.append(h)
            h = torch.cat(hs, -1)
            return h