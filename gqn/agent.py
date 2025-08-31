from typing import Dict, Tuple

import torch.optim as optim

from common.agent_utils import *
from common.encoder import VisionEncoder
from common.replay_buffer import PrioritizedReplayBuffer
from critic import GrowingQCritic
from discretizer import GrowingActionDiscretizer
from scheduler import GrowingScheduler


class GrowingQNAgent:
    def __init__(self, config, obs_shape: Tuple, action_spec: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_discretizer = GrowingActionDiscretizer(
            action_spec, max_bins=config.num_bins, decouple=config.decouple
        )
        self.scheduler = GrowingScheduler(
            total_episodes=config.num_episodes,
            num_growth_stages=len(self.action_discretizer.growth_sequence) - 1,
            schedule_type=getattr(config, 'growing_schedule', 'adaptive')
        )

        if config.use_pixels:
            self.encoder = VisionEncoder(config, config.num_pixels).to(self.device)
            encoder_output_size = config.layer_size_bottleneck
            self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=config.learning_rate)
        else:
            self.encoder = None
            encoder_output_size = np.prod(obs_shape)
        self.q_network = GrowingQCritic(config, encoder_output_size, action_spec).to(self.device)
        self.target_q_network = GrowingQCritic(config, encoder_output_size, action_spec).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=config.max_replay_size,
            alpha=config.priority_exponent,
            beta=config.importance_sampling_exponent,
            n_step=config.adder_n_step,
            discount=config.discount
        )
        self.replay_buffer.device = self.device
        self.training_step = 0
        self.episode_count = 0
        self.epsilon = config.epsilon
        self.last_obs = None
        self.current_resolution_level = 0
        self.growth_history = [2]

    def update_epsilon(self, decay_rate: float = 0.995, min_epsilon: float = 0.01):
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

    def get_current_action_mask(self) -> torch.Tensor:
        return self.action_discretizer.get_action_mask(self.config.num_bins)

    def select_action(self, obs: torch.Tensor, evaluate: bool = False) -> torch.Tensor:
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        elif len(obs.shape) < 2:
            obs = obs.unsqueeze(0)

        with torch.no_grad():
            if self.encoder:
                encoded_obs = self.encoder(obs)
            else:
                encoded_obs = obs.flatten(1)

            action_mask = self.get_current_action_mask()
            q1, q2 = self.q_network(encoded_obs, action_mask)
            q_combined = torch.max(q1, q2)
            epsilon = 0.0 if evaluate else self.epsilon
            discrete_action = self._epsilon_greedy_action_selection(q_combined, epsilon)
            continuous_action = self.action_discretizer.discrete_to_continuous(discrete_action)

            return continuous_action[0].detach()

    def _epsilon_greedy_action_selection(self, q_values: torch.Tensor, epsilon: float) -> torch.Tensor:
        batch_size = q_values.shape[0]

        if self.config.decouple:
            num_dims = q_values.shape[1]
            current_bins = self.action_discretizer.current_bins
            return get_combined_random_and_greedy_actions(
                q_values, num_dims, current_bins, batch_size, epsilon, self.device)
        else:
            if torch.rand(1).item() < epsilon:
                current_bins = self.action_discretizer.current_bins
                total_actions = current_bins ** self.action_discretizer.action_dim
                return torch.randint(0, total_actions, (batch_size,), device=self.device)
            else:
                return q_values.argmax(dim=1)

    def observe_first(self, obs: torch.Tensor):
        self.last_obs = obs.detach() if isinstance(obs, torch.Tensor) else obs

    def observe(self, action: torch.Tensor, reward: float, next_obs: torch.Tensor, done: bool):
        if hasattr(self, 'last_obs') and self.last_obs is not None:
            discrete_action = continuous_to_discrete_action(self.config, self.action_discretizer, action)
            self.replay_buffer.add(self.last_obs, discrete_action, reward, next_obs, done)

        self.last_obs = next_obs.detach() if isinstance(next_obs, torch.Tensor) else next_obs

    def maybe_grow_action_space(self, episode_return: float) -> bool:
        if self.scheduler.should_grow(self.episode_count, episode_return):
            growth_occurred = self.action_discretizer.grow_action_space()
            if growth_occurred:
                current_bins = self.action_discretizer.current_bins
                self.growth_history.append(current_bins)
                print(f"Episode {self.episode_count}: Growing action space to {current_bins} bins per dimension")
                return True
        return False

    def update(self) -> Dict:
        if len(self.replay_buffer) < self.config.min_replay_size:
            return {}

        batch = self.replay_buffer.sample(self.config.batch_size)
        if batch is None:
            return {}

        obs, actions, rewards, next_obs, dones, discounts, weights, indices = batch

        obs = obs.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_obs = next_obs.to(self.device)
        dones = dones.to(self.device)
        discounts = discounts.to(self.device)
        weights = weights.to(self.device)

        if self.encoder:
            obs_encoded = self.encoder(obs)
            with torch.no_grad():
                next_obs_encoded = self.encoder(next_obs)
        else:
            obs_encoded = obs.flatten(1)
            next_obs_encoded = next_obs.flatten(1)

        action_mask = self.get_current_action_mask()

        q1_current, q2_current = self.q_network(obs_encoded, action_mask)

        # Next Q-values for target (Equation 3 from paper)
        with torch.no_grad():
            q1_next_target, q2_next_target = self.target_q_network(next_obs_encoded, action_mask)
            q1_next_online, q2_next_online = self.q_network(next_obs_encoded, action_mask)

            if self.config.decouple:
                # Double DQN: select with online, evaluate with target
                # Implement value decomposition as per Equation 2: Q(s,a) = Σ Q_j(s,a_j) / M

                q_next_online = 0.5 * (q1_next_online + q2_next_online)
                next_actions = q_next_online.argmax(dim=2)

                batch_indices = torch.arange(q1_next_target.shape[0], device=self.device)
                dim_indices = torch.arange(self.action_discretizer.action_dim, device=self.device)

                # Select Q-values for chosen actions
                q1_selected = q1_next_target[batch_indices.unsqueeze(1), dim_indices.unsqueeze(0), next_actions]
                q2_selected = q2_next_target[batch_indices.unsqueeze(1), dim_indices.unsqueeze(0), next_actions]

                # Average Q-values and sum over dimensions, then divide by M (Equation 2)
                q_target_per_dim = 0.5 * (q1_selected + q2_selected)
                q_target_values = q_target_per_dim.sum(dim=1) / self.action_discretizer.action_dim

                targets = rewards + discounts * q_target_values * (~dones).float()
            else:
                # Standard case for joint actions
                q_next_combined = 0.5 * (q1_next_target + q2_next_target)
                next_actions = self._epsilon_greedy_action_selection(
                    0.5 * (q1_next_online + q2_next_online), 0.0
                )
                q_target_values = q_next_combined.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                targets = rewards + discounts * q_target_values * ~dones

        # Current Q-values for selected actions
        if self.config.decouple:
            # Apply Equation 2: sum over dimensions and divide by M
            batch_indices = torch.arange(q1_current.shape[0], device=self.device)
            dim_indices = torch.arange(self.action_discretizer.action_dim, device=self.device)

            q1_per_dim = q1_current[batch_indices.unsqueeze(1), dim_indices.unsqueeze(0), actions]
            q2_per_dim = q2_current[batch_indices.unsqueeze(1), dim_indices.unsqueeze(0), actions]

            q1_selected = q1_per_dim.sum(dim=1) / self.action_discretizer.action_dim
            q2_selected = q2_per_dim.sum(dim=1) / self.action_discretizer.action_dim
        else:
            if len(actions.shape) > 1:
                actions = actions.squeeze(-1)
            q1_selected = q1_current.gather(1, actions.unsqueeze(1)).squeeze(1)
            q2_selected = q2_current.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute losses with Huber loss as specified in paper
        td_error1 = targets - q1_selected
        td_error2 = targets - q2_selected

        # Huber loss (paper uses huber_loss_parameter = 1.0)
        huber_param = getattr(self.config, "huber_loss_parameter", 1.0)
        loss1 = huber_loss(td_error1, huber_param)
        loss2 = huber_loss(td_error2, huber_param) if self.config.use_double_q else torch.zeros_like(loss1)

        # Apply importance sampling weights
        loss1 = (loss1 * weights).sum() / weights.sum()
        loss2 = (loss2 * weights).sum() / weights.sum() if self.config.use_double_q else torch.zeros_like(loss1)

        total_loss = loss1 + loss2

        # Optimize
        self.q_optimizer.zero_grad()
        if self.encoder:
            self.encoder_optimizer.zero_grad()

        total_loss.backward()

        # Gradient clipping as specified in paper (40.0)
        if getattr(self.config, "clip_gradients", False):
            clip_norm = getattr(self.config, "clip_gradients_norm", 40.0)
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), clip_norm)
            if self.encoder:
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip_norm)

        self.q_optimizer.step()
        if self.encoder:
            self.encoder_optimizer.step()

        # Update priorities
        priorities1 = torch.abs(td_error1).detach().cpu().numpy()
        priorities2 = torch.abs(td_error2).detach().cpu().numpy()
        priorities = 0.5 * (priorities1 + priorities2) if self.config.use_double_q else priorities1
        self.replay_buffer.update_priorities(indices, priorities)

        # Update target network
        self.training_step += 1
        if self.training_step % self.config.target_update_period == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        return {
            "loss": total_loss.item(),
            "q1_mean": q1_selected.mean().item(),
            "q2_mean": q2_selected.mean().item() if self.config.use_double_q else 0,
            "current_bins": self.action_discretizer.current_bins,
            "resolution_level": self.current_resolution_level
        }

    def end_episode(self, episode_return: float):
        self.episode_count += 1
        self.maybe_grow_action_space(episode_return)

    def get_growth_info(self) -> Dict:
        return {
            "current_bins": self.action_discretizer.current_bins,
            "resolution_level": self.current_resolution_level,
            "growth_history": self.growth_history,
            "max_bins": self.config.num_bins,
            "growth_sequence": self.action_discretizer.growth_sequence
        }
