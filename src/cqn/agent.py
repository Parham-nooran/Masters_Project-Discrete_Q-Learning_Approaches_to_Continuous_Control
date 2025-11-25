"""
Coarse-to-Fine Q-Network Agent for continuous control.

This implementation follows the CQN paper's approach using distributional RL (C51)
with hierarchical action discretization.
"""

from typing import Dict, Tuple

import numpy as np
import torch

from src.common.device_utils import (
    get_device,
    save_checkpoint as save_ckpt,
    load_checkpoint as load_ckpt
)
from src.cqn.networks import CQNNetwork
from src.cqn.replay_buffer import ReplayBuffer


class CQNAgent:
    """
    Coarse-to-fine Q-Network agent using distributional RL.

    Implements hierarchical action discretization where the agent iteratively
    refines action selection from coarse to fine levels using C51 distributional
    value estimation.
    """

    def __init__(self, config, obs_shape: Tuple, action_spec: Dict):
        """
        Initialize CQN agent.

        Args:
            config: Configuration object with hyperparameters.
            obs_shape: Shape of observations.
            action_spec: Dictionary with 'low' and 'high' action bounds.
        """
        self.config = config
        self.obs_shape = obs_shape
        self.action_spec = action_spec
        self.device, self.is_tpu, self.use_amp = get_device()

        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize network, optimizer, and replay buffer."""
        self.num_levels = self.config.num_levels
        self.num_bins = self.config.num_bins
        self.action_dim = len(self.action_spec["low"])

        self.initial_low = torch.tensor(
            self.action_spec["low"], dtype=torch.float32, device=self.device
        )
        self.initial_high = torch.tensor(
            self.action_spec["high"], dtype=torch.float32, device=self.device
        )

        self.network = CQNNetwork(
            self.config,
            self.obs_shape,
            self.action_dim,
            self.num_levels,
            self.num_bins,
            self.config.num_atoms,
            self.config.v_min,
            self.config.v_max
        ).to(self.device)

        self.target_network = CQNNetwork(
            self.config,
            self.obs_shape,
            self.action_dim,
            self.num_levels,
            self.num_bins,
            self.config.num_atoms,
            self.config.v_min,
            self.config.v_max
        ).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()

        if self.network.encoder is not None:
            self.encoder_opt = torch.optim.Adam(
                self.network.encoder.parameters(),
                lr=self.config.lr
            )
        else:
            self.encoder_opt = None

        self.critic_opt = torch.optim.Adam(
            self.network.critic.parameters(),
            lr=self.config.lr
        )

        self.replay_buffer = ReplayBuffer(
            capacity=self.config.replay_buffer_size,
            obs_shape=self.obs_shape,
            action_shape=(self.action_dim,),
            device=self.device
        )

        self._initialize_training_state()

    def _initialize_training_state(self) -> None:
        """Initialize training state variables."""
        self.training_steps = 0
        self.num_expl_steps = self.config.num_seed_frames // self.config.action_repeat

    def act(self, obs: np.ndarray, step: int, eval_mode: bool = False) -> np.ndarray:
        """
        Select action using hierarchical coarse-to-fine strategy.

        Args:
            obs: Observation array.
            step: Current training step.
            eval_mode: Whether in evaluation mode.

        Returns:
            Continuous action array.
        """
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if len(obs.shape) == len(self.obs_shape):
            obs = obs.unsqueeze(0)

        with torch.no_grad():
            if self.network.encoder is not None:
                obs_features = self.network.encoder(obs)
            else:
                obs_features = obs.flatten(1)

            stddev = self._get_stddev(step)
            action, _ = self.target_network.critic.get_action(obs_features)

            if eval_mode:
                action = action
            else:
                stddev_tensor = torch.ones_like(action) * stddev
                noise = torch.randn_like(action) * stddev_tensor
                action = action + noise

                if step < self.num_expl_steps:
                    action.uniform_(-1.0, 1.0)

            action = self._encode_decode_action(action)

        return action.cpu().numpy()[0]

    def _get_stddev(self, step: int) -> float:
        """Get exploration noise standard deviation."""
        if isinstance(self.config.stddev_schedule, str):
            import re
            match = re.match(r"linear\((.+),(.+),(.+)\)", self.config.stddev_schedule)
            if match:
                init, final, duration = [float(g) for g in match.groups()]
                mix = np.clip(step / duration, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final
        return float(self.config.stddev_schedule)

    def _encode_decode_action(self, continuous_action: torch.Tensor) -> torch.Tensor:
        """
        Encode continuous action to discrete then decode back.

        This ensures actions align with the discretization grid.

        Args:
            continuous_action: Continuous action tensor.

        Returns:
            Discretized continuous action.
        """
        from src.cqn.cqn_utils import encode_action, decode_action

        discrete_action = encode_action(
            continuous_action,
            self.initial_low,
            self.initial_high,
            self.num_levels,
            self.num_bins,
        )
        continuous_action = decode_action(
            discrete_action,
            self.initial_low,
            self.initial_high,
            self.num_levels,
            self.num_bins,
        )
        return continuous_action

    def update(self, replay_iter, step: int) -> Dict[str, float]:
        """
        Update agent using a batch from replay buffer.

        Args:
            replay_iter: Iterator over replay buffer.
            step: Current training step.

        Returns:
            Dictionary of training metrics.
        """
        metrics = {}

        if step % self.config.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = self._to_torch(batch)

        if self.network.encoder is not None:
            obs = self.network.aug(obs.float())
            next_obs = self.network.aug(next_obs.float())

            obs = self.network.encoder(obs)
            with torch.no_grad():
                next_obs = self.network.encoder(next_obs)
        else:
            obs = obs.flatten(1)
            next_obs = next_obs.flatten(1)

        metrics["batch_reward"] = reward.mean().item()

        critic_metrics = self._update_critic(obs, action, reward, discount, next_obs)
        metrics.update(critic_metrics)

        self._soft_update_target()

        return metrics

    def _update_critic(
            self,
            obs: torch.Tensor,
            action: torch.Tensor,
            reward: torch.Tensor,
            discount: torch.Tensor,
            next_obs: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Update critic network using distributional RL loss.

        Args:
            obs: Observation features.
            action: Actions taken.
            reward: Rewards received.
            discount: Discount factors.
            next_obs: Next observation features.

        Returns:
            Dictionary of critic metrics.
        """
        metrics = {}

        with torch.no_grad():
            next_action, target_q_metrics = self.target_network.critic.get_action(next_obs)
            metrics.update(target_q_metrics)

            target_q_probs_a = self.target_network.critic.compute_target_q_dist(
                next_obs, next_action, reward, discount
            )

        _, _, _, log_q_probs_a = self.network.critic(obs, action)

        critic_loss = -torch.sum(target_q_probs_a * log_q_probs_a, -1).mean()

        metrics["critic_loss"] = critic_loss.item()

        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)

        critic_loss.backward()

        self.critic_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        return metrics

    def _soft_update_target(self) -> None:
        """Soft update target network parameters."""
        tau = self.config.critic_target_tau

        for param, target_param in zip(
                self.network.parameters(), self.target_network.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )

    def _to_torch(self, batch):
        """Convert batch to torch tensors."""
        return tuple(torch.as_tensor(x, device=self.device) for x in batch)

    def store_transition(
            self, obs, action: np.ndarray, reward: float, next_obs, done: bool
    ) -> None:
        """
        Store transition in replay buffer.

        Args:
            obs: Observation.
            action: Action taken.
            reward: Reward received.
            next_obs: Next observation.
            done: Whether episode is done.
        """
        discount = 1.0 if not done else 0.0
        self.replay_buffer.add(obs, action, reward, discount, next_obs)

    def save_checkpoint(self, filepath: str, episode: int) -> None:
        """
        Save agent checkpoint.

        Args:
            filepath: Path to save checkpoint.
            episode: Current episode number.
        """
        checkpoint = {
            "network_state_dict": self.network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "critic_opt_state_dict": self.critic_opt.state_dict(),
            "training_steps": self.training_steps,
            "episode": episode,
            "config": self.config,
        }
        if self.encoder_opt is not None:
            checkpoint["encoder_opt_state_dict"] = self.encoder_opt.state_dict()

        save_ckpt(checkpoint, filepath, self.is_tpu)

    def load_checkpoint(self, filepath: str) -> int:
        """
        Load agent checkpoint.

        Args:
            filepath: Path to checkpoint file.

        Returns:
            Episode number from checkpoint.
        """
        checkpoint = load_ckpt(filepath, self.device, self.is_tpu)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.critic_opt.load_state_dict(checkpoint["critic_opt_state_dict"])
        if self.encoder_opt is not None and "encoder_opt_state_dict" in checkpoint:
            self.encoder_opt.load_state_dict(checkpoint["encoder_opt_state_dict"])
        self.training_steps = checkpoint["training_steps"]
        return checkpoint.get("episode", 0)
