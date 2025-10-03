import numpy as np
import torch
from src.common.logger import Logger
from typing import Tuple, Dict
from networks import CQNNetwork
from discretizer import CoarseToFineDiscretizer
from src.common.replay_buffer import PrioritizedReplayBuffer
from src.common.metrics_tracker import MetricsTracker
from src.common.utils import huber_loss


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
        self.discretizer = CoarseToFineDiscretizer(action_spec, self.num_levels, self.num_bins)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.lr)
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=config.replay_buffer_size,
            alpha=config.per_alpha,
            beta=config.per_beta,
            n_step=config.n_step,
            discount=config.discount
        ).to_device(self.device)
        self.metrics_tracker = MetricsTracker(save_dir=config.save_dir)
        self.epsilon = config.initial_epsilon
        self.epsilon_decay = config.epsilon_decay
        self.min_epsilon = config.min_epsilon
        self.target_update_freq = config.target_update_freq
        self.training_steps = 0
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
                    action_range = self.discretizer.get_action_range_for_level(level, prev_action.squeeze(0))

                q1, q2 = self.network(obs, level, prev_action)
                q_combined = torch.max(q1, q2)
                level_actions = torch.zeros(self.action_dim, device=self.device)

                for dim in range(self.action_dim):
                    if not evaluate and np.random.random() < self.epsilon:
                        level_actions[dim] = torch.randint(0, self.num_bins, (1,), device=self.device)
                    else:
                        level_actions[dim] = q_combined[0, dim].argmax()

                if level == 0:
                    level_continuous = self.discretizer.discrete_to_continuous(
                        level_actions.unsqueeze(0), level
                    )[0]
                else:
                    level_continuous = torch.zeros(self.action_dim, device=self.device)
                    for dim in range(self.action_dim):
                        bin_idx = level_actions[dim].long()
                        range_min, range_max = action_range[0, dim], action_range[1, dim]
                        bin_size = (range_max - range_min) / self.num_bins
                        level_continuous[dim] = range_min + bin_idx * bin_size + bin_size / 2
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

        total_loss = 0.0
        td_errors = []

        prev_action = None
        for level in range(self.num_levels):
            q1_current, q2_current = self.network(obs, level, prev_action)

            with torch.no_grad():
                q1_next, q2_next = self.network(next_obs, level, prev_action)
                q_next_combined = torch.max(q1_next, q2_next)

                next_actions = q_next_combined.argmax(dim=2)  # [batch_size, action_dim]

                target_q1 = torch.gather(q1_next, 2, next_actions.unsqueeze(2)).squeeze(2)
                target_q2 = torch.gather(q2_next, 2, next_actions.unsqueeze(2)).squeeze(2)
                target_q = torch.min(target_q1, target_q2)

                td_target = rewards + (1 - dones.float()) * discounts * target_q.mean(dim=1)
                td_target = td_target.unsqueeze(1).expand(-1, self.action_dim)

            discrete_actions = self._continuous_to_discrete_for_level(actions, level)

            current_q1 = torch.gather(q1_current, 2, discrete_actions.unsqueeze(2)).squeeze(2)
            current_q2 = torch.gather(q2_current, 2, discrete_actions.unsqueeze(2)).squeeze(2)

            td_error1 = td_target - current_q1
            td_error2 = td_target - current_q2

            loss1 = huber_loss(td_error1, self.config.huber_loss_parameter)
            loss2 = huber_loss(td_error2, self.config.huber_loss_parameter)

            level_loss = ((loss1 + loss2) * weights.unsqueeze(1)).mean()
            total_loss += level_loss

            td_errors.append(torch.abs(td_error1).mean(dim=1) + torch.abs(td_error2).mean(dim=1))

            if level < self.num_levels - 1:
                with torch.no_grad():
                    prev_action = self.discretizer.discrete_to_continuous(discrete_actions, level)
        self.optimizer.zero_grad()
        total_loss.backward()

        if hasattr(self.config, 'max_grad_norm'):
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)

        self.optimizer.step()
        if self.training_steps % self.target_update_freq == 0:
            self.network.update_target_networks()

        avg_td_error = torch.stack(td_errors).mean(dim=0)
        self.replay_buffer.update_priorities(indices, avg_td_error.detach().cpu().numpy())

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        self.training_steps += 1

        metrics = {
            'loss': total_loss.item(),
            'epsilon': self.epsilon,
            'q_mean': (q1_current.mean() + q2_current.mean()).item() / 2,
            'td_error_mean': avg_td_error.mean().item()
        }

        return metrics

    def _continuous_to_discrete_for_level(self, continuous_actions: torch.Tensor, level: int) -> torch.Tensor:
        """Convert continuous actions to discrete indices for a specific level"""
        batch_size = continuous_actions.shape[0]
        discrete_actions = torch.zeros(batch_size, self.action_dim, dtype=torch.long, device=self.device)

        for dim in range(self.action_dim):
            bins = self.discretizer.action_bins[level][dim]
            distances = torch.abs(continuous_actions[:, dim].unsqueeze(1) - bins.unsqueeze(0))
            discrete_actions[:, dim] = distances.argmin(dim=1)

        return discrete_actions

    def store_transition(self, obs: np.ndarray, action: np.ndarray, reward: float,
                         next_obs: np.ndarray, done: bool):
        """Store a transition in the replay buffer"""
        self.replay_buffer.add(obs, action, reward, next_obs, done)

    def save(self, filepath: str):
        """Save the agent"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'epsilon': self.epsilon,
            'config': self.config
        }, filepath)
        self.logger.info(f"Agent saved to {filepath}")

    def load(self, filepath: str):
        """Load the agent"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_steps = checkpoint['training_steps']
        self.epsilon = checkpoint['epsilon']
        self.logger.info(f"Agent loaded from {filepath}")
