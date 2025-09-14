import torch
import numpy as np
import torch.optim as optim
from src.common.encoder import VisionEncoder

"""
Implementing pure bang bang without growing
"""


class BangBangAgent:
    def __init__(self, config, obs_shape, action_spec):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_shape = obs_shape
        self.action_spec = action_spec
        self.action_min = torch.tensor(action_spec["low"], dtype=torch.float32)
        self.action_max = torch.tensor(action_spec["high"], dtype=torch.float32)
        self.action_dim = len(self.action_min)
        self.action_bins = [
            torch.tensor(
                [self.action_min[dim], 0.0, self.action_max[dim]], device=self.device
            )
            for dim in range(self.action_dim)
        ]
        self.action_bins = torch.stack(self.action_bins)

    def initialize_network(self):
        if self.config.use_pixels:
            self.encoder = VisionEncoder(self.config, self.config.num_pixels).to(
                self.device
            )
            encoder_output_size = self.config.layer_size_bottleneck
            self.encoder_optimizer = optim.Adam(
                self.encoder.parameters(), lr=self.config.learning_rate
            )
        else:
            self.encoder = None
            encoder_output_size = np.prod(self.obs_shape)
        output_size = self.action_dim * self.config.num_bins
        self.q_network = BangBangCritic(
            self.config,
            encoder_output_size,
        )
