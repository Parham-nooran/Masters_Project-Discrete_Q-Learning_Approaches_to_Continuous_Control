from typing import Dict, Tuple
import numpy as np
import torch
import torch.optim as optim

from src.common.agent_utils import get_combined_random_and_greedy_actions, huber_loss, continuous_to_discrete_action
from src.common.encoder import VisionEncoder
from src.common.replay_buffer import PrioritizedReplayBuffer
from src.gqn.critic import GrowingQCritic
from src.gqn.discretizer import GrowingActionDiscretizer
from src.gqn.scheduler import GrowingScheduler


class GrowingQNAgent:
    def __init__(self, config, obs_shape: Tuple, action_spec: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._cached_action_mask = None
        self._cached_mask_bins = None

        # Set up growth sequence based on task complexity
        if hasattr(config, 'growth_sequence') and config.growth_sequence is not None:
            growth_sequence = config.growth_sequence
        elif any(task in config.task.lower() for task in ['sawyer', 'metaworld', 'manipulation']):
            growth_sequence = [9, 17, 33, 65]  # As used for MetaWorld experiments
        elif 'myo' in config.task.lower():
            growth_sequence = [2, 3, 5, 9, 17, 33, 65]  # Extended sequence for MyoSuite
        else:
            growth_sequence = [2, 3, 5, 9]

        self.action_discretizer = GrowingActionDiscretizer(
            action_spec,
            max_bins=config.max_bins,
            decouple=config.decouple,
            growth_sequence=growth_sequence
        )

        self.scheduler = GrowingScheduler(
            total_episodes=config.num_episodes,
            num_growth_stages=len(self.action_discretizer.growth_sequence) - 1,
            schedule_type=getattr(config, 'growing_schedule', 'adaptive')
        )

        # Set up encoder if using pixels
        if config.use_pixels:
            self.encoder = VisionEncoder(config, config.num_pixels).to(self.device)
            encoder_output_size = config.layer_size_bottleneck
            self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=config.learning_rate)
        else:
            self.encoder = None
            encoder_output_size = np.prod(obs_shape)

        # Create Q-networks with proper initialization
        self.q_network = GrowingQCritic(config, encoder_output_size, action_spec).to(self.device)
        self.target_q_network = GrowingQCritic(config, encoder_output_size, action_spec).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)

        # Initialize replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=config.max_replay_size,
            alpha=config.priority_exponent,
            beta=config.importance_sampling_exponent,
            n_step=config.adder_n_step,
            discount=config.discount
        )
        self.replay_buffer.device = self.device

        # Training state
        self.training_step = 0
        self.episode_count = 0
        self.epsilon = config.epsilon
        self.last_obs = None
        self.current_resolution_level = 0
        self.growth_history = [self.action_discretizer.current_bins]

    def update_epsilon(self, decay_rate: float = 0.995, min_epsilon: float = 0.01):
        """Update exploration epsilon with decay."""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

    def get_current_action_mask(self) -> torch.Tensor:
        """Get action mask for current resolution level."""
        if (self._cached_action_mask is None or
                self._cached_mask_bins != self.action_discretizer.current_bins):
            self._cached_action_mask = self.action_discretizer.get_action_mask(self.config.max_bins)
            self._cached_mask_bins = self.action_discretizer.current_bins
        return self._cached_action_mask

    def select_action(self, obs: torch.Tensor, evaluate: bool = False) -> torch.Tensor:
        """Select action using epsilon-greedy policy."""
        # Ensure proper tensor format
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        elif len(obs.shape) < 2:
            obs = obs.unsqueeze(0)

        with torch.no_grad():
            # Encode observation
            if self.encoder:
                encoded_obs = self.encoder(obs)
            else:
                encoded_obs = obs.flatten(1)

            # Get Q-values with action masking
            action_mask = self.get_current_action_mask()
            q1, q2 = self.q_network(encoded_obs, action_mask)

            # Combine Q-values for action selection
            q_combined = torch.max(q1, q2)
            epsilon = 0.0 if evaluate else self.epsilon

            # Select discrete action
            if self.config.decouple:
                discrete_action = self._epsilon_greedy_action_selection_decoupled(q1, q2, epsilon)
            else:
                discrete_action = self._epsilon_greedy_action_selection(q_combined, epsilon)

            # Convert to continuous action
            continuous_action = self.action_discretizer.discrete_to_continuous(discrete_action)

            # Clamp to action bounds
            continuous_action = torch.clamp(
                continuous_action,
                torch.tensor(self.action_discretizer.action_spec["low"]).to(self.device),
                torch.tensor(self.action_discretizer.action_spec["high"]).to(self.device)
            )

            return continuous_action[0].detach()

    def _epsilon_greedy_action_selection_decoupled(self, q1: torch.Tensor, q2: torch.Tensor,
                                                   epsilon: float) -> torch.Tensor:
        """Epsilon-greedy action selection for decoupled case."""
        batch_size = q1.shape[0]
        num_dims = q1.shape[1]
        current_bins = self.action_discretizer.current_bins

        # Combine Q-values properly
        q_combined = torch.max(q1, q2)  # Shape: [batch, dims, bins]

        return get_combined_random_and_greedy_actions(
            q_combined, num_dims, current_bins, batch_size, epsilon, self.device)

    def _epsilon_greedy_action_selection(self, q_values: torch.Tensor, epsilon: float) -> torch.Tensor:
        """Epsilon-greedy action selection for joint case."""
        batch_size = q_values.shape[0]

        if torch.rand(1).item() < epsilon:
            # Random action from valid masked actions only
            action_mask = self.get_current_action_mask()
            valid_actions = torch.where(action_mask)[0]
            if len(valid_actions) > 0:
                random_indices = torch.randint(0, len(valid_actions), (batch_size,), device=self.device)
                return valid_actions[random_indices]
            else:
                # Fallback if no valid actions (shouldn't happen)
                return torch.randint(0, q_values.shape[1], (batch_size,), device=self.device)
        else:
            # Greedy action selection
            return q_values.argmax(dim=1)

    def observe_first(self, obs: torch.Tensor):
        """Observe first state of episode."""
        self.last_obs = obs.detach() if isinstance(obs, torch.Tensor) else obs

    def observe(self, action: torch.Tensor, reward: float, next_obs: torch.Tensor, done: bool):
        """Store transition in replay buffer."""
        if hasattr(self, 'last_obs') and self.last_obs is not None:
            discrete_action = continuous_to_discrete_action(self.config, self.action_discretizer, action)
            self.replay_buffer.add(self.last_obs, discrete_action, reward, next_obs, done)
        self.last_obs = next_obs.detach()

    def maybe_grow_action_space(self, episode_return: float) -> bool:
        """Check if action space should grow based on scheduler."""
        if (self.episode_count > 50 and
                self.episode_count - self.scheduler.last_growth_episode > 100):
            if self.scheduler.should_grow(self.episode_count, episode_return):
                growth_occurred = self.action_discretizer.grow_action_space()
                if growth_occurred:
                    current_bins = self.action_discretizer.current_bins
                    self.growth_history.append(current_bins)
                    print(f"Episode {self.episode_count}: Growing action space to {current_bins} bins per dimension")
                    return True
        return False

    def update(self) -> Dict:
        """Update Q-networks using sampled batch."""
        if len(self.replay_buffer) < self.config.min_replay_size:
            return {}

        batch = self.replay_buffer.sample(self.config.batch_size)
        if batch is None:
            return {}

        obs, actions, rewards, next_obs, dones, discounts, weights, indices = batch

        # Move all tensors to device
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_obs = next_obs.to(self.device)
        dones = dones.to(self.device)
        discounts = discounts.to(self.device)
        weights = weights.to(self.device)

        # Encode observations
        if self.encoder:
            obs_encoded = self.encoder(obs)
            with torch.no_grad():
                next_obs_encoded = self.encoder(next_obs)
        else:
            obs_encoded = obs.flatten(1)
            next_obs_encoded = next_obs.flatten(1)

        # Get action mask for current resolution
        action_mask = self.get_current_action_mask()

        # Current Q-values
        q1_current, q2_current = self.q_network(obs_encoded, action_mask)

        # Next Q-values for target computation
        with torch.no_grad():
            q1_next_target, q2_next_target = self.target_q_network(next_obs_encoded, action_mask)

            if self.config.use_double_q:
                # Double DQN: select actions with online network, evaluate with target
                q1_next_online, q2_next_online = self.q_network(next_obs_encoded, action_mask)

                if self.config.decouple:
                    q_next_online = torch.max(q1_next_online, q2_next_online)
                    next_actions = q_next_online.argmax(dim=2)

                    batch_indices = torch.arange(q1_next_target.shape[0], device=self.device)
                    dim_indices = torch.arange(self.action_discretizer.action_dim, device=self.device)

                    # Select Q-values for chosen actions using target network
                    q1_selected = q1_next_target[batch_indices.unsqueeze(1), dim_indices.unsqueeze(0), next_actions]
                    q2_selected = q2_next_target[batch_indices.unsqueeze(1), dim_indices.unsqueeze(0), next_actions]

                    # FIXED: Apply Equation 2 correctly - AVERAGE instead of sum
                    q_target_per_dim = torch.max(q1_selected, q2_selected)
                    q_target_decomposed = q_target_per_dim.mean(dim=1)  # CHANGED: sum -> mean
                    targets = rewards + discounts * q_target_decomposed * (~dones).float()
                else:
                    # Joint case with Double DQN
                    q_next_online = torch.max(q1_next_online, q2_next_online)
                    next_actions = q_next_online.argmax(dim=1)
                    q_next_target_combined = torch.max(q1_next_target, q2_next_target)
                    q_target_values = q_next_target_combined.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    targets = rewards + discounts * q_target_values * (~dones).float()
            else:
                # Standard DQN: use target network for both selection and evaluation
                if self.config.decouple:
                    q_next_target_combined = torch.max(q1_next_target, q2_next_target)
                    next_actions = q_next_target_combined.argmax(dim=2)

                    batch_indices = torch.arange(q1_next_target.shape[0], device=self.device)
                    dim_indices = torch.arange(self.action_discretizer.action_dim, device=self.device)

                    q1_selected = q1_next_target[batch_indices.unsqueeze(1), dim_indices.unsqueeze(0), next_actions]
                    q2_selected = q2_next_target[batch_indices.unsqueeze(1), dim_indices.unsqueeze(0), next_actions]

                    q_target_per_dim = torch.max(q1_selected, q2_selected)
                    q_target_total = q_target_per_dim.mean(dim=1)  # CHANGED: sum -> mean
                    targets = rewards + discounts * q_target_total * (~dones).float()
                else:
                    q_next_target_combined = torch.max(q1_next_target, q2_next_target)
                    next_actions = q_next_target_combined.argmax(dim=1)
                    q_target_values = q_next_target_combined.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    targets = rewards + discounts * q_target_values * (~dones).float()

        # FIXED: Clamp targets to prevent exploding values
        targets = torch.clamp(targets, -100, 100)

        # Current Q-values for selected actions
        if self.config.decouple:
            # FIXED: Apply Equation 2 correctly - AVERAGE over dimensions
            batch_indices = torch.arange(q1_current.shape[0], device=self.device)
            dim_indices = torch.arange(self.action_discretizer.action_dim, device=self.device)

            q1_per_dim = q1_current[batch_indices.unsqueeze(1), dim_indices.unsqueeze(0), actions]
            q2_per_dim = q2_current[batch_indices.unsqueeze(1), dim_indices.unsqueeze(0), actions]

            q1_selected = q1_per_dim.mean(dim=1)  # CHANGED: sum -> mean
            q2_selected = q2_per_dim.mean(dim=1)  # CHANGED: sum -> mean
        else:
            if len(actions.shape) > 1:
                actions = actions.squeeze(-1)
            q1_selected = q1_current.gather(1, actions.unsqueeze(1)).squeeze(1)
            q2_selected = q2_current.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute TD errors
        td_error1 = targets - q1_selected
        td_error2 = targets - q2_selected

        # Huber loss (paper uses huber_loss_parameter = 1.0)
        huber_param = getattr(self.config, "huber_loss_parameter", 1.0)
        loss1 = huber_loss(td_error1, huber_param)
        loss2 = huber_loss(td_error2, huber_param) if self.config.use_double_q else torch.zeros_like(loss1)

        # Apply importance sampling weights - FIXED normalization
        loss1_weighted = loss1 * weights
        loss2_weighted = loss2 * weights if self.config.use_double_q else torch.zeros_like(loss1)

        # Normalize by sum of weights to maintain proper loss scale
        total_loss = (loss1_weighted.sum() + loss2_weighted.sum()) / (weights.sum() + 1e-8)

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
        # FIXED: Clamp priorities to prevent numerical issues
        priorities = np.clip(priorities, 1e-6, 100.0)
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
            "resolution_level": self.current_resolution_level,
            "td_error_mean": torch.abs(td_error1).mean().item()
        }

    def end_episode(self, episode_return: float):
        """Handle end of episode, potentially growing action space."""
        self.episode_count += 1
        self.maybe_grow_action_space(episode_return)

    def get_growth_info(self) -> Dict:
        """Get information about current growth state."""
        return {
            "current_bins": self.action_discretizer.current_bins,
            "resolution_level": self.current_resolution_level,
            "growth_history": self.growth_history,
            "max_bins": self.config.max_bins,
            "growth_sequence": self.action_discretizer.growth_sequence
        }