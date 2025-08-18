import random
import torch.optim as optim
from actors import CustomDiscreteFeedForwardActor
from critic import *
from encoder import *
from replay_buffer import *


def huber_loss(td_error, huber_loss_parameter=1.0):
    """
    Huber loss implementation matching TensorFlow's acme.tf.losses.huber

    Args:
        td_error: Temporal difference error tensor
        huber_loss_parameter: Threshold for switching from quadratic to linear loss

    Returns:
        Huber loss tensor
    """
    abs_error = torch.abs(td_error)
    quadratic = torch.minimum(abs_error, torch.tensor(huber_loss_parameter, device=abs_error.device))
    linear = abs_error - quadratic
    return 0.5 * quadratic ** 2 + huber_loss_parameter * linear

class ActionDiscretizer:
    """Handles continuous to discrete action conversion."""

    def __init__(self, action_spec, num_bins, decouple=False, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.num_bins = num_bins
        self.decouple = decouple
        self.device = device

        # Handle different action_spec formats
        if isinstance(action_spec, dict):
            if 'low' in action_spec and 'high' in action_spec:
                self.action_min = torch.tensor(action_spec['low'], dtype=torch.float32, device=device)
                self.action_max = torch.tensor(action_spec['high'], dtype=torch.float32, device=device)
            else:
                raise ValueError(f"Invalid action_spec format: {action_spec}")
        elif hasattr(action_spec, 'low') and hasattr(action_spec, 'high'):
            # Handle gym-style Box spaces
            self.action_min = torch.tensor(action_spec.low, dtype=torch.float32, device=device)
            self.action_max = torch.tensor(action_spec.high, dtype=torch.float32, device=device)
        else:
            raise ValueError(f"Unsupported action_spec format: {type(action_spec)}")

        self.action_dim = len(self.action_min)

        if decouple:
            # Per-dimension discretization - create bins for each dimension separately
            self.action_bins = []
            for dim in range(self.action_dim):
                bins = torch.linspace(self.action_min[dim], self.action_max[dim], num_bins, device=device)
                self.action_bins.append(bins)
            self.action_bins = torch.stack(self.action_bins)  # Shape: [action_dim, num_bins]
        else:
            # Joint discretization - create all combinations
            bins_per_dim = [torch.linspace(self.action_min[i], self.action_max[i], num_bins, device=device)
                            for i in range(self.action_dim)]
            # Create cartesian product
            mesh = torch.meshgrid(*bins_per_dim, indexing='ij')
            self.action_bins = torch.stack([m.flatten() for m in mesh], dim=1)

    def discrete_to_continuous(self, discrete_actions):
        """Convert discrete actions to continuous."""
        if not isinstance(discrete_actions, torch.Tensor):
            discrete_actions = torch.tensor(discrete_actions, device=self.device)

        if discrete_actions.device != self.device:
            discrete_actions = discrete_actions.to(self.device)

        if self.decouple:
            # discrete_actions shape: [batch_size, action_dim]
            batch_size = discrete_actions.shape[0]
            continuous_actions = torch.zeros(batch_size, self.action_dim, device=self.device)

            for dim in range(self.action_dim):
                bin_indices = discrete_actions[:, dim]
                continuous_actions[:, dim] = self.action_bins[dim][bin_indices]

            return continuous_actions
        else:
            # discrete_actions shape: [batch_size]
            return self.action_bins[discrete_actions.flatten()]


class DecQNAgent:
    """DecQN Agent with PyTorch implementation."""

    def __init__(self, config, obs_shape, action_spec, device='cpu'):
        self.config = config
        self.device = device
        self.obs_shape = obs_shape
        self.action_spec = action_spec

        # Action discretization
        self.action_discretizer = ActionDiscretizer(action_spec, config.num_bins, config.decouple, device=device)

        # Networks
        if config.use_pixels:
            self.encoder = VisionEncoder(config, config.num_pixels).to(device)
            encoder_output_size = config.layer_size_bottleneck
            self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=config.learning_rate)
        else:
            self.encoder = None
            encoder_output_size = np.prod(obs_shape)

        # Q-networks
        self.q_network = CriticDQN(config, encoder_output_size, action_spec).to(device)
        self.target_q_network = CriticDQN(config, encoder_output_size, action_spec).to(device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        # Optimizers
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)

        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=config.max_replay_size,
            alpha=config.priority_exponent,
            beta=config.importance_sampling_exponent,
            n_step=config.adder_n_step,
            discount=config.discount
        )

        # Create actors for training and evaluation
        self.actor = CustomDiscreteFeedForwardActor(
            policy_network=self.q_network,
            encoder=self.encoder,
            action_discretizer=self.action_discretizer,
            epsilon=config.epsilon,
            decouple=config.decouple,
            device=device
        )

        self.eval_actor = CustomDiscreteFeedForwardActor(
            policy_network=self.q_network,
            encoder=self.encoder,
            action_discretizer=self.action_discretizer,
            epsilon=0.0,  # No exploration during evaluation
            decouple=config.decouple,
            device=device
        )

        # Training state
        self.training_step = 0
        self.epsilon = config.epsilon

    def select_action(self, obs, evaluate=False):
        """Select action using the appropriate actor."""
        if evaluate:
            return self.eval_actor.select_action(obs)
        else:
            return self.actor.select_action(obs)

    def observe_first(self, obs):
        """Handle first observation in an episode."""
        # Store initial observation if needed
        self.last_obs = obs

    def observe(self, action, reward, next_obs, done):
        """Observe transition and store in replay buffer."""
        if hasattr(self, 'last_obs'):
            # Keep everything as tensors, only convert to discrete for storage
            if isinstance(action, torch.Tensor):
                discrete_action = self._continuous_to_discrete_action(action)
            else:
                discrete_action = action

            self.store_transition(self.last_obs, discrete_action, reward, next_obs, done)

        # Handle tensor storage properly
        if isinstance(next_obs, torch.Tensor):
            self.last_obs = next_obs.detach()
        else:
            self.last_obs = next_obs

    def _continuous_to_discrete_action(self, continuous_action):
        """Convert continuous action back to discrete indices for storage."""
        if isinstance(continuous_action, torch.Tensor):
            continuous_action = continuous_action.cpu().numpy()

        continuous_action = np.array(continuous_action)

        if self.config.decouple:
            # Find closest bin for each dimension
            discrete_action = []
            for dim in range(len(continuous_action)):
                bins = self.action_discretizer.action_bins[dim].cpu().numpy()
                closest_idx = np.argmin(np.abs(bins - continuous_action[dim]))
                discrete_action.append(closest_idx)
            return np.array(discrete_action, dtype=np.int64)
        else:
            # Find closest action in joint space
            action_bins_cpu = self.action_discretizer.action_bins.cpu().numpy()
            distances = np.linalg.norm(action_bins_cpu - continuous_action, axis=1)
            return np.argmin(distances)

    def store_transition(self, obs, action, reward, next_obs, done):
        """Store transition in replay buffer."""
        # Convert discrete action back if needed
        self.replay_buffer.add(obs, action, reward, next_obs, done)

    def save_checkpoint(self, path, episode):
        """Save agent checkpoint."""
        checkpoint = {
            'episode': episode,
            'q_network_state_dict': self.q_network.state_dict(),
            'target_q_network_state_dict': self.target_q_network.state_dict(),
            'q_optimizer_state_dict': self.q_optimizer.state_dict(),
            'config': self.config,
            'training_step': self.training_step
        }

        if self.encoder:
            checkpoint['encoder_state_dict'] = self.encoder.state_dict()
            checkpoint['encoder_optimizer_state_dict'] = self.encoder_optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
        self.training_step = checkpoint.get('training_step', 0)

        if self.encoder and 'encoder_state_dict' in checkpoint:
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])

        return checkpoint['episode']

    def make_epsilon_greedy_policy(self, q_values, epsilon, decouple=False):
        """Create epsilon-greedy policy from Q-values."""
        batch_size = q_values[0].shape[0]

        if isinstance(q_values, tuple):  # Double Q
            q_max = torch.max(q_values[0], q_values[1])
        else:
            q_max = q_values

        if decouple:
            # For decoupled actions, sample each dimension independently
            actions = []
            for b in range(batch_size):
                action_per_dim = []
                for dim in range(q_max.shape[1]):
                    if random.random() < epsilon:
                        action_per_dim.append(random.randint(0, q_max.shape[2] - 1))
                    else:
                        action_per_dim.append(q_max[b, dim].argmax().item())
                actions.append(action_per_dim)
            return torch.tensor(actions, device=self.device, dtype=torch.long)
        else:
            # Standard epsilon-greedy
            if random.random() < epsilon:
                return torch.randint(0, q_max.shape[1], (batch_size,), device=self.device)
            else:
                return q_max.argmax(dim=1)

    def update(self):
        """Update the agent."""
        if len(self.replay_buffer) < self.config.min_replay_size:
            return {}

        # Sample batch
        batch = self.replay_buffer.sample(self.config.batch_size)
        if batch is None:
            return {}

        obs, actions, rewards, next_obs, dones, discounts, weights, indices = batch

        # Move to device
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

        # Current Q-values
        q1_current, q2_current = self.q_network(obs_encoded)

        # Next Q-values for target
        with torch.no_grad():
            q1_next_target, q2_next_target = self.target_q_network(next_obs_encoded)
            q1_next_online, q2_next_online = self.q_network(next_obs_encoded)

            # Double Q-learning target
            if self.config.decouple:
                # For decoupled actions
                next_actions = self.make_epsilon_greedy_policy((q1_next_online, q2_next_online), 0.0, True)
                q_next_target = 0.5 * (q1_next_target + q2_next_target)  # Average Q-values

                # For decoupled actions, each dimension is selected independently
                batch_size, action_dim = next_actions.shape
                q_target_values = torch.zeros(batch_size, action_dim, device=self.device)
                for dim in range(action_dim):
                    q_target_values[:, dim] = q_next_target[:, dim, next_actions[:, dim]]


                # Expand rewards and discounts for decoupled case
                rewards_expanded = rewards.unsqueeze(-1).expand(-1, actions.shape[1])
                discounts_expanded = discounts.unsqueeze(-1).expand(-1, actions.shape[1])
                dones_expanded = dones.unsqueeze(-1).expand(-1, actions.shape[1])

                targets = rewards_expanded + discounts_expanded * q_target_values * ~dones_expanded
            else:
                # Standard case
                next_actions = self.make_epsilon_greedy_policy((q1_next_online, q2_next_online), 0.0, False)
                q_next_target = 0.5 * (q1_next_target + q2_next_target)
                q_target_values = q_next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                targets = rewards + discounts * q_target_values * ~dones

        # Handle action shape for decoupled case
        if self.config.decouple:
            expected_action_dims = self.action_discretizer.action_dim

            if len(actions.shape) == 1:  # Single dimension actions
                actions = actions.unsqueeze(-1).expand(-1, expected_action_dims)
            elif actions.shape[1] != expected_action_dims:
                if actions.shape[1] < expected_action_dims:
                    actions = actions.repeat(1, expected_action_dims)[:, :expected_action_dims]

            batch_indices = torch.arange(q1_current.shape[0], device=self.device)
            action_indices = torch.arange(actions.shape[1], device=self.device)

            # Use advanced indexing to select Q-values for each action dimension
            q1_selected = q1_current[batch_indices.unsqueeze(1), action_indices.unsqueeze(0), actions]
            q2_selected = q2_current[batch_indices.unsqueeze(1), action_indices.unsqueeze(0), actions]
        else:
            if len(actions.shape) > 1:
                actions = actions.squeeze(-1)
            q1_selected = q1_current.gather(1, actions.unsqueeze(1)).squeeze(1)
            q2_selected = q2_current.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute losses
        td_error1 = targets - q1_selected
        td_error2 = targets - q2_selected

        if self.config.decouple:
            td_error1 = td_error1.mean(dim=1)  # Average across action dimensions
            td_error2 = td_error2.mean(dim=1)

        # Huber loss
        loss1 = huber_loss(td_error1, self.config.get('huber_loss_parameter', 1.0))
        loss2 = huber_loss(td_error2, self.config.get('huber_loss_parameter', 1.0)) if self.config.use_double_q else torch.zeros_like(loss1)


        # Apply importance sampling weights
        loss1 = (loss1 * weights).mean()
        loss2 = (loss2 * weights).mean()

        total_loss = loss1 + loss2

        # Optimize
        self.q_optimizer.zero_grad()
        if self.encoder:
            self.encoder_optimizer.zero_grad()

        total_loss.backward()

        # Gradient clipping
        if self.config.clip_gradients:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.clip_gradients_norm)
            if self.encoder:
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.config.clip_gradients_norm)

        self.q_optimizer.step()
        if self.encoder:
            self.encoder_optimizer.step()

        # Update priorities in replay buffer
        priorities1 = torch.abs(td_error1).detach().cpu().numpy()
        priorities2 = torch.abs(td_error2).detach().cpu().numpy()
        priorities = 0.5 * (priorities1 + priorities2) if self.config.use_double_q else priorities1
        self.replay_buffer.update_priorities(indices, priorities)

        # Update target network
        self.training_step += 1
        if self.training_step % self.config.target_update_period == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        return {
            'loss': total_loss.item(),
            'q1_mean': q1_selected.mean().item(),
            'q2_mean': q2_selected.mean().item() if self.config.use_double_q else 0,
        }