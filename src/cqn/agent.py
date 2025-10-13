from typing import Tuple, Dict

import numpy as np
import torch

from src.common.logger import Logger
from src.common.metrics_tracker import MetricsTracker
from src.common.replay_buffer import PrioritizedReplayBuffer
from src.common.training_utils import huber_loss
from src.cqn.discretizer import CoarseToFineDiscretizer
from src.cqn.networks import CQNNetwork


class CQNAgent(Logger):
    """
    Coarse-to-fine Q-Network Agent
    """

    def __init__(self, config, obs_shape: Tuple, action_spec: Dict, working_dir):
        super().__init__(working_dir)
        self.config = config
        self.obs_shape = obs_shape
        self.action_spec = action_spec
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_levels = config.num_levels
        self.num_bins = config.num_bins
        self.action_dim = len(action_spec["low"])
        self.network = CQNNetwork(
            config, obs_shape, self.action_dim, self.num_levels, self.num_bins
        ).to(self.device)
        self.discretizer = CoarseToFineDiscretizer(
            action_spec, self.num_levels, self.num_bins
        )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.lr)
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=config.replay_buffer_size,
            alpha=config.per_alpha,
            beta=config.per_beta,
            n_step=config.n_step,
            discount=config.discount,
        )
        self.replay_buffer.to_device(self.device)
        self.metrics_tracker = MetricsTracker(save_dir=config.save_dir)
        self.epsilon = config.initial_epsilon
        self.epsilon_decay = config.epsilon_decay
        self.min_epsilon = config.min_epsilon
        self.target_update_freq = config.target_update_freq
        self.training_steps = 0
        self.scaler = (
            torch.cuda.amp.GradScaler() if self.device.type == "cuda" else None
        )
        self.use_amp = self.device.type == "cuda"
        self.logger.info("CQN Agent initialized")

    def select_action(self, obs: torch.Tensor, evaluate: bool = False) -> torch.Tensor:
        """Select action using coarse-to-fine strategy."""
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        elif len(obs.shape) == len(self.obs_shape):
            obs = obs.unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_range = None
            prev_action = None

            for level in range(self.num_levels):
                if level > 0:
                    action_range = self.discretizer.get_action_range_for_level(
                        level, prev_action.squeeze(0)
                    )

                q1, q2 = self.network(obs, level, prev_action)
                q_combined = torch.max(q1, q2)
                if not evaluate and np.random.random() < self.epsilon:
                    level_actions = torch.randint(
                        0, self.num_bins, (self.action_dim,), device=self.device
                    )
                else:
                    level_actions = q_combined[0].argmax(dim=-1)

                if level == 0:
                    level_continuous = self.discretizer.discrete_to_continuous(
                        level_actions.unsqueeze(0), level
                    )[0]
                else:
                    level_continuous = torch.zeros(self.action_dim, device=self.device)
                    for dim in range(self.action_dim):
                        bin_idx = level_actions[dim].long()
                        range_min, range_max = (
                            action_range[0, dim],
                            action_range[1, dim],
                        )
                        bin_size = (range_max - range_min) / self.num_bins
                        level_continuous[dim] = range_min + (bin_idx + 0.5) * bin_size
                prev_action = level_continuous.unsqueeze(0)
            return level_continuous.cpu()

    def update(self, batch_size: int = 256) -> Dict[str, float]:
        """Update the agent using a batch from replay buffer"""
        if len(self.replay_buffer) < batch_size:
            return {}

        batch = self.replay_buffer.sample(batch_size)
        if batch is None:
            return {}

        obs, actions, rewards, next_obs, dones, discounts, weights, indices = batch

        if self.use_amp:
            with torch.cuda.amp.autocast():
                total_loss, td_errors, q1_current, q2_current = self._compute_loss(
                    obs, actions, rewards, next_obs, dones, discounts, weights
                )

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(total_loss).backward()

            if hasattr(self.config, "max_grad_norm"):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.config.max_grad_norm
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss, td_errors, q1_current, q2_current = self._compute_loss(
                obs, actions, rewards, next_obs, dones, discounts, weights
            )

            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()

            if hasattr(self.config, "max_grad_norm"):
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.config.max_grad_norm
                )

            self.optimizer.step()

        if self.training_steps % self.target_update_freq == 0:
            self.network.update_target_networks()

        avg_td_error = torch.stack(td_errors).mean(dim=0)
        self.replay_buffer.update_priorities(
            indices, avg_td_error.detach().cpu().numpy()
        )

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        self.training_steps += 1

        return {
            "loss": total_loss.item(),
            "epsilon": self.epsilon,
            "q_mean": (q1_current.mean() + q2_current.mean()).item() / 2,
            "mean_abs_td_error": avg_td_error.mean().item(),
        }

    def _compute_loss(
        self, obs, actions, rewards, next_obs, dones, discounts, weights
    ) -> Tuple[torch.Tensor, list, torch.Tensor, torch.Tensor]:
        """Compute loss for all levels - separated for AMP compatibility"""
        obs = obs.float()
        actions = actions.float()
        next_obs = next_obs.float()
        total_loss = 0.0
        td_errors = []
        prev_action = None
        prev_next_action = None
        q1_current_last = None
        q2_current_last = None

        for level in range(self.num_levels):
            q1_current, q2_current = self.network(
                obs, level, prev_action, use_target=False
            )

            if level == self.num_levels - 1:
                q1_current_last = q1_current
                q2_current_last = q2_current

            with torch.no_grad():
                q1_next_online, q2_next_online = self.network(
                    next_obs, level, prev_next_action, use_target=False
                )
                q_next_online = torch.max(q1_next_online, q2_next_online)
                next_actions = q_next_online.argmax(dim=2)

                q1_next_target, q2_next_target = self.network(
                    next_obs, level, prev_next_action, use_target=True
                )

                target_q1 = torch.gather(
                    q1_next_target, 2, next_actions.unsqueeze(2)
                ).squeeze(2)
                target_q2 = torch.gather(
                    q2_next_target, 2, next_actions.unsqueeze(2)
                ).squeeze(2)

                target_q = torch.min(target_q1, target_q2)

                td_target = (
                    rewards.unsqueeze(1)
                    + (1 - dones.float()).unsqueeze(1)
                    * discounts.unsqueeze(1)
                    * target_q
                )

            discrete_actions = self._continuous_to_discrete_for_level(actions, level)

            current_q1 = torch.gather(
                q1_current, 2, discrete_actions.unsqueeze(2)
            ).squeeze(2)
            current_q2 = torch.gather(
                q2_current, 2, discrete_actions.unsqueeze(2)
            ).squeeze(2)

            td_error1 = td_target - current_q1
            td_error2 = td_target - current_q2

            loss1 = huber_loss(td_error1, self.config.huber_loss_parameter)
            loss2 = huber_loss(td_error2, self.config.huber_loss_parameter)

            level_loss = ((loss1 + loss2) * weights.unsqueeze(1)).mean()
            total_loss += level_loss

            td_errors.append(
                torch.abs(td_error1).mean(dim=1) + torch.abs(td_error2).mean(dim=1)
            )
            if level < self.num_levels - 1:
                with torch.no_grad():
                    prev_action = self.discretizer.discrete_to_continuous(
                        discrete_actions, level
                    )
                    prev_next_action = self.discretizer.discrete_to_continuous(
                        next_actions, level
                    )

        return total_loss, td_errors, q1_current_last, q2_current_last

    def _continuous_to_discrete_for_level(
        self, continuous_actions: torch.Tensor, level: int
    ) -> torch.Tensor:
        """Convert continuous actions to discrete indices for a specific level"""
        bins_tensor = torch.stack(
            [
                self.discretizer.action_bins[level][dim]
                for dim in range(self.action_dim)
            ],
            dim=0,
        )
        distance = torch.abs(continuous_actions.unsqueeze(2) - bins_tensor.unsqueeze(0))
        discrete_actions = distance.argmin(dim=2)
        return discrete_actions

    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        """Store a transition in the replay buffer"""
        self.replay_buffer.add(obs, action, reward, next_obs, done)

    def save(self, filepath: str):
        """Save the agent"""
        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_steps": self.training_steps,
                "epsilon": self.epsilon,
                "config": self.config,
            },
            filepath,
        )
        self.logger.info(f"Agent saved to {filepath}")

    def load(self, filepath: str):
        """Load the agent"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_steps = checkpoint["training_steps"]
        self.epsilon = checkpoint["epsilon"]
        self.logger.info(f"Agent loaded from {filepath}")
