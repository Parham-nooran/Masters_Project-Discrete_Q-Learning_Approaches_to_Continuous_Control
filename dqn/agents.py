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
    quadratic = torch.minimum(
        abs_error, torch.tensor(huber_loss_parameter, device=abs_error.device)
    )
    linear = abs_error - quadratic
    return 0.5 * quadratic**2 + huber_loss_parameter * linear


class ActionDiscretizer:
    """Handles continuous to discrete action conversion."""

    def __init__(self, action_spec, num_bins, decouple=False):
        self.num_bins = num_bins
        self.decouple = decouple
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Handle different action_spec formats
        if isinstance(action_spec, dict):
            if "low" in action_spec and "high" in action_spec:
                self.action_min = torch.tensor(
                    action_spec["low"], dtype=torch.float32, device=self.device
                )
                self.action_max = torch.tensor(
                    action_spec["high"], dtype=torch.float32, device=self.device
                )
            else:
                raise ValueError(f"Invalid action_spec format: {action_spec}")
        elif hasattr(action_spec, "low") and hasattr(action_spec, "high"):
            # Handle gym-style Box spaces
            self.action_min = torch.tensor(
                action_spec.low, dtype=torch.float32, device=self.device
            )
            self.action_max = torch.tensor(
                action_spec.high, dtype=torch.float32, device=self.device
            )
        else:
            raise ValueError(f"Unsupported action_spec format: {type(action_spec)}")

        self.action_dim = len(self.action_min)

        if decouple:
            # Per-dimension discretization - create bins for each dimension separately
            self.action_bins = []
            for dim in range(self.action_dim):
                bins = torch.linspace(
                    self.action_min[dim],
                    self.action_max[dim],
                    num_bins,
                    device=self.device,
                )
                self.action_bins.append(bins)
            self.action_bins = torch.stack(
                self.action_bins
            )  # Shape: [action_dim, num_bins]
        else:
            bins_per_dim = torch.stack(
                [
                    torch.linspace(
                        self.action_min[i],
                        self.action_max[i],
                        num_bins,
                        device=self.device,
                    )
                    for i in range(self.action_dim)
                ]
            )
            # Create cartesian product more efficiently
            mesh = torch.meshgrid(*bins_per_dim, indexing="ij")
            self.action_bins = torch.stack([m.flatten() for m in mesh], dim=1)

    def discrete_to_continuous(self, discrete_actions):
        """Convert discrete actions to continuous."""
        if not isinstance(discrete_actions, torch.Tensor):
            discrete_actions = torch.tensor(discrete_actions, device=self.device)

        if discrete_actions.device != self.device:
            discrete_actions = discrete_actions.to(self.device)

        if self.decouple:
            # discrete_actions shape: [batch_size, action_dim]
            if len(discrete_actions.shape) == 1:
                discrete_actions = discrete_actions.unsqueeze(0)

            batch_size = discrete_actions.shape[0]
            continuous_actions = torch.zeros(
                batch_size, self.action_dim, device=self.device
            )

            for dim in range(self.action_dim):
                bin_indices = discrete_actions[:, dim].long()
                continuous_actions[:, dim] = self.action_bins[dim][bin_indices]

            return continuous_actions
        else:
            if len(discrete_actions.shape) == 0:
                discrete_actions = discrete_actions.unsqueeze(0)
            return self.action_bins[discrete_actions.long()]


class DecQNAgent:
    """DecQN Agent with PyTorch implementation."""

    def __init__(self, config, obs_shape, action_spec):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_shape = obs_shape
        self.action_spec = action_spec
        self.action_discretizer = ActionDiscretizer(
            action_spec, config.num_bins, config.decouple
        )
        if config.use_pixels:
            self.encoder = VisionEncoder(config, config.num_pixels).to(self.device)
            encoder_output_size = config.layer_size_bottleneck
            self.encoder_optimizer = optim.Adam(
                self.encoder.parameters(), lr=config.learning_rate
            )
        else:
            self.encoder = None
            encoder_output_size = np.prod(obs_shape)
        self.q_network = CriticDQN(config, encoder_output_size, action_spec).to(
            self.device
        )
        self.target_q_network = CriticDQN(config, encoder_output_size, action_spec).to(
            self.device
        )
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.q_optimizer = optim.Adam(
            self.q_network.parameters(), lr=config.learning_rate
        )
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=config.max_replay_size,
            alpha=config.priority_exponent,
            beta=config.importance_sampling_exponent,
            n_step=config.adder_n_step,
            discount=config.discount,
        )
        self.replay_buffer.device = self.device
        self.actor = CustomDiscreteFeedForwardActor(
            policy_network=self.q_network,
            encoder=self.encoder,
            action_discretizer=self.action_discretizer,
            epsilon=config.epsilon,
            decouple=config.decouple,
        )
        self.eval_actor = CustomDiscreteFeedForwardActor(
            policy_network=self.q_network,
            encoder=self.encoder,
            action_discretizer=self.action_discretizer,
            epsilon=0.0,  # No exploration during evaluation
            decouple=config.decouple,
        )

        # Training state
        self.training_step = 0
        self.epsilon = config.epsilon

    def update_epsilon(self, decay_rate=0.995, min_epsilon=0.01):
        """Update epsilon for exploration decay."""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
        self.actor.epsilon = self.epsilon

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
        if hasattr(self, "last_obs"):
            # Keep everything as tensors, only convert to discrete for storage
            if isinstance(action, torch.Tensor):
                discrete_action = self._continuous_to_discrete_action(action)
            else:
                discrete_action = action

            self.store_transition(
                self.last_obs, discrete_action, reward, next_obs, done
            )

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
        """Save agent checkpoints."""
        checkpoint = {
            "episode": episode,
            "q_network_state_dict": self.q_network.state_dict(),
            "target_q_network_state_dict": self.target_q_network.state_dict(),
            "q_optimizer_state_dict": self.q_optimizer.state_dict(),
            "config": self.config,
            "training_step": self.training_step,
        }

        if self.encoder:
            checkpoint["encoder_state_dict"] = self.encoder.state_dict()
            checkpoint["encoder_optimizer_state_dict"] = (
                self.encoder_optimizer.state_dict()
            )

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """Load agent checkpoints."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if "replay_buffer_buffer" in checkpoint:
            self.replay_buffer.buffer = checkpoint["replay_buffer_buffer"]
            self.replay_buffer.position = checkpoint["replay_buffer_position"]
            self.replay_buffer.priorities = checkpoint["replay_buffer_priorities"]
            self.replay_buffer.max_priority = checkpoint["replay_buffer_max_priority"]
            self.replay_buffer.to_device(self.device)

        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_q_network.load_state_dict(checkpoint["target_q_network_state_dict"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
        self.training_step = checkpoint.get("training_step", 0)

        if self.encoder and "encoder_state_dict" in checkpoint:
            self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
            self.encoder_optimizer.load_state_dict(
                checkpoint["encoder_optimizer_state_dict"]
            )

        # Load epsilon if available
        if "epsilon" in checkpoint:
            self.epsilon = checkpoint["epsilon"]
            self.actor.epsilon = self.epsilon

        # Handle both old config format and new config_dict format
        if "config_dict" in checkpoint:
            # New format - config saved as dictionary
            config_dict = checkpoint["config_dict"]
            for key, value in config_dict.items():
                setattr(self.config, key, value)
        # Old format with Config object will work with weights_only=False

        return checkpoint["episode"]

    def make_epsilon_greedy_policy(self, q_values, epsilon, decouple=False):
        """Create epsilon-greedy policy from Q-values."""
        batch_size = q_values[0].shape[0]

        if isinstance(q_values, tuple):  # Double Q
            q_max = torch.max(q_values[0], q_values[1])
        else:
            q_max = q_values

        if decouple:
            # For decoupled actions, sample each dimension independently

            num_dims = q_max.shape[1]
            num_bins = q_max.shape[2]

            # Create random mask for exploration decisions (independent for each batch and dimension)
            random_mask = torch.rand(batch_size, num_dims, device=self.device) < epsilon

            # Random actions for exploration
            random_actions = torch.randint(
                0, num_bins, (batch_size, num_dims), device=self.device
            )

            # Greedy actions for exploitation
            greedy_actions = q_max.argmax(dim=2)

            # Combine using the mask
            actions = torch.where(random_mask, random_actions, greedy_actions)

            return actions
        else:
            # Standard epsilon-greedy
            if torch.rand(1).item() < epsilon:
                return torch.randint(
                    0, q_max.shape[1], (batch_size,), device=self.device
                )
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

            if self.config.decouple:
                # Double DQN: select with online, evaluate with target
                q1_next_online_reshaped = q1_next_online.view(
                    q1_next_online.shape[0], self.action_discretizer.action_dim, -1
                )
                q2_next_online_reshaped = q2_next_online.view(
                    q2_next_online.shape[0], self.action_discretizer.action_dim, -1
                )

                q_next_online = 0.5 * (
                    q1_next_online_reshaped + q2_next_online_reshaped
                )
                next_actions = q_next_online.argmax(dim=2)

                q1_target_reshaped = q1_next_target.view(
                    q1_next_target.shape[0], self.action_discretizer.action_dim, -1
                )
                q2_target_reshaped = q2_next_target.view(
                    q2_next_target.shape[0], self.action_discretizer.action_dim, -1
                )

                batch_indices = torch.arange(
                    q1_target_reshaped.shape[0], device=self.device
                )
                dim_indices = torch.arange(
                    self.action_discretizer.action_dim, device=self.device
                )

                q1_selected = q1_target_reshaped[
                    batch_indices.unsqueeze(1), dim_indices.unsqueeze(0), next_actions
                ]
                q2_selected = q2_target_reshaped[
                    batch_indices.unsqueeze(1), dim_indices.unsqueeze(0), next_actions
                ]

                q_target_per_dim = 0.5 * (q1_selected + q2_selected)
                q_target_values = (
                    q_target_per_dim.sum(dim=1) / self.action_discretizer.action_dim
                )

                targets = rewards + discounts * q_target_values * (~dones).float()
            else:
                # Standard case
                next_actions = self.make_epsilon_greedy_policy(
                    (q1_next_online, q2_next_online), 0.0, False
                )
                q_next_target = 0.5 * (q1_next_target + q2_next_target)
                q_target_values = q_next_target.gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
                targets = rewards + discounts * q_target_values * ~dones

        # Handle action shape for decoupled case
        if self.config.decouple:
            # For decoupled case: implement Equation 2 from paper
            # Q(st, at) = (Σⱼ Qⱼ(st, aⱼₜ)) / M

            q1_reshaped = q1_current.view(
                q1_current.shape[0], self.action_discretizer.action_dim, -1
            )
            q2_reshaped = q2_current.view(
                q2_current.shape[0], self.action_discretizer.action_dim, -1
            )

            batch_indices = torch.arange(q1_reshaped.shape[0], device=self.device)
            dim_indices = torch.arange(
                self.action_discretizer.action_dim, device=self.device
            )

            q1_per_dim = q1_reshaped[
                batch_indices.unsqueeze(1), dim_indices.unsqueeze(0), actions
            ]
            q2_per_dim = q2_reshaped[
                batch_indices.unsqueeze(1), dim_indices.unsqueeze(0), actions
            ]

            q1_selected = q1_per_dim.sum(dim=1) / self.action_discretizer.action_dim
            q2_selected = q2_per_dim.sum(dim=1) / self.action_discretizer.action_dim
        else:
            if len(actions.shape) > 1:
                actions = actions.squeeze(-1)
            q1_selected = q1_current.gather(1, actions.unsqueeze(1)).squeeze(1)
            q2_selected = q2_current.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute losses
        td_error1 = targets - q1_selected
        td_error2 = targets - q2_selected

        # Huber loss
        loss1 = huber_loss(td_error1, getattr(self.config, "huber_loss_parameter", 1.0))
        loss2 = (
            huber_loss(td_error2, getattr(self.config, "huber_loss_parameter", 1.0))
            if self.config.use_double_q
            else torch.zeros_like(loss1)
        )

        # Apply importance sampling weights properly
        loss1 = (loss1 * weights).sum() / weights.sum()
        loss2 = (
            (loss2 * weights).sum() / weights.sum()
            if self.config.use_double_q
            else torch.zeros_like(loss1)
        )

        total_loss = loss1 + loss2

        # Optimize
        self.q_optimizer.zero_grad()
        if self.encoder:
            self.encoder_optimizer.zero_grad()

        total_loss.backward()

        # Gradient clipping as specified in paper (40.0)
        if getattr(self.config, "clip_gradients", False):
            clip_norm = getattr(
                self.config, "clip_gradients_norm", 40.0
            )  # Paper uses 40.0
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), clip_norm)
            if self.encoder:
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip_norm)

        self.q_optimizer.step()
        if self.encoder:
            self.encoder_optimizer.step()

        # Update priorities in replay buffer
        priorities1 = torch.abs(td_error1).detach().cpu().numpy()
        priorities2 = torch.abs(td_error2).detach().cpu().numpy()
        priorities = (
            0.5 * (priorities1 + priorities2)
            if self.config.use_double_q
            else priorities1
        )
        self.replay_buffer.update_priorities(indices, priorities)

        # Update target network
        self.training_step += 1
        if self.training_step % self.config.target_update_period == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        return {
            "loss": total_loss.item(),
            "q1_mean": q1_selected.mean().item(),
            "q2_mean": q2_selected.mean().item() if self.config.use_double_q else 0,
        }
