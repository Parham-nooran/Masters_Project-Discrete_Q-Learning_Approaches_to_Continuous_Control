import torch
import numpy as np
import torch.optim as optim

from src.common.networks import LayerNormMLP
from src.common.replay_buffer import PrioritizedReplayBuffer
from src.common.encoder import VisionEncoder
from typing import Dict
from src.bangbang.bernoulli_policy import BernoulliPolicy
from src.common.logger import Logger
from src.common.utils import get_batch_components


class BangBangAgent(Logger):
    """Bang-Bang Control Agent implementing the paper's core ideas."""

    def __init__(self, config, obs_shape: tuple, action_spec: dict, working_dir="."):
        super().__init__(working_dir)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_spec = action_spec
        self.action_dim = len(action_spec["low"])
        self.action_scale = torch.tensor(
            (action_spec["high"] - action_spec["low"]) / 2.0, device=self.device
        )
        self.action_bias = torch.tensor(
            (action_spec["high"] + action_spec["low"]) / 2.0, device=self.device
        )
        if config.use_pixels:
            self.encoder = VisionEncoder(config, config.num_pixels).to(self.device)
            self.encoder_output_size = config.layer_size_bottleneck
            self.encoder_optimizer = optim.Adam(
                self.encoder.parameters(), lr=config.learning_rate
            )
        else:
            self.encoder = None
            self.encoder_output_size = np.prod(obs_shape)
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=config.max_replay_size,
            alpha=config.priority_exponent,
            beta=config.importance_sampling_exponent,
            n_step=config.adder_n_step,
            discount=config.discount,
        )
        self.replay_buffer.device = self.device
        self.training_step = 0
        self.last_obs = None
        self.policy = BernoulliPolicy(
            self.encoder_output_size, self.action_dim, config.layer_size_network
        ).to(self.device)
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(), lr=config.learning_rate
        )
        self.value_function = LayerNormMLP(
            [self.encoder_output_size] + config.layer_size_network + [1],
            activate_final=False,
        ).to(self.device)
        self.value_optimizer = optim.Adam(
            self.value_function.parameters(), lr=config.learning_rate
        )

    def _encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Shared observation encoding logic."""
        if self.encoder:
            return self.encoder(obs)
        else:
            return obs.flatten(1)

    def observe_first(self, obs: torch.Tensor):
        """Store first observation."""
        self.last_obs = obs

    def select_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        """Select bang-bang action."""
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.device)

        if len(obs.shape) == 1 or (self.encoder and len(obs.shape) == 3):
            obs = obs.unsqueeze(0)

        with torch.no_grad():
            if self.encoder:
                encoded_obs = self.encoder(obs)
            else:
                encoded_obs = obs.flatten(1)

            action, _ = self.policy.get_action(encoded_obs, deterministic)

            scaled_action = action * self.action_scale + self.action_bias

        return scaled_action.squeeze(0)

    def observe(
        self, action: torch.Tensor, reward: float, next_obs: torch.Tensor, done: bool
    ):
        """Store transition in replay buffer."""
        if self.last_obs is not None:
            normalized_action = (action - self.action_bias) / self.action_scale
            binary_action = (normalized_action > 0).float()

            self.replay_buffer.add(self.last_obs, binary_action, reward, next_obs, done)

        self.last_obs = next_obs.detach()

    def update(self) -> Dict[str, float]:
        """Update policy using importance-weighted policy gradient."""
        if len(self.replay_buffer) < self.config.min_replay_size:
            return {}
        obs, actions, rewards, next_obs, dones, discounts, weights, indices = \
            get_batch_components(self.replay_buffer, self.config.batch_size, self.device)

        if self.encoder:
            obs_encoded = self.encoder(obs)
        else:
            obs_encoded = obs.flatten(1)

        logits = self.policy(obs_encoded)
        probs = torch.sigmoid(logits)
        dist = torch.distributions.Bernoulli(probs)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        returns = rewards
        policy_loss = -(log_probs * returns * weights).mean()

        self.policy_optimizer.zero_grad()
        if self.encoder:
            self.encoder_optimizer.zero_grad()

        policy_loss.backward()

        if getattr(self.config, "clip_gradients", False):
            clip_norm = getattr(self.config, "clip_gradients_norm", 40.0)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), clip_norm)
            if self.encoder:
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip_norm)

        self.policy_optimizer.step()
        if self.encoder:
            self.encoder_optimizer.step()

        priorities = torch.abs(returns).detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, priorities)

        self.training_step += 1

        return {
            "policy_loss": policy_loss.item(),
            "mean_return": returns.mean().item(),
            "mean_prob": probs.mean().item(),
        }

    def save_checkpoint(self, path: str, episode: int):
        """Save agent checkpoint."""
        checkpoint = {
            "episode": episode,
            "policy_state_dict": self.policy.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "config": self.config,
            "training_step": self.training_step,
            "action_scale": self.action_scale.cpu(),
            "action_bias": self.action_bias.cpu(),
        }

        if self.encoder:
            checkpoint["encoder_state_dict"] = self.encoder.state_dict()
            checkpoint["encoder_optimizer_state_dict"] = (
                self.encoder_optimizer.state_dict()
            )

        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> int:
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        self.training_step = checkpoint.get("training_step", 0)

        if self.encoder and "encoder_state_dict" in checkpoint:
            self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
            self.encoder_optimizer.load_state_dict(
                checkpoint["encoder_optimizer_state_dict"]
            )

        self.action_scale = checkpoint["action_scale"].to(self.device)
        self.action_bias = checkpoint["action_bias"].to(self.device)

        self.logger.info(f"Checkpoint loaded: {path}")
        return checkpoint["episode"]
