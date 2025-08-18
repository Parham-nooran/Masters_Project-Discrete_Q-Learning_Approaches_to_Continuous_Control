import torch
import random
import numpy as np


class CustomDiscreteFeedForwardActor:
    """
    A feed-forward actor for discrete actions.

    This is the PyTorch equivalent of the TensorFlow CustomDiscreteFeedForwardActor.
    It handles action selection using the policy network (Q-network + epsilon-greedy)
    and manages interaction with the replay buffer.
    """

    def __init__(self, policy_network, encoder=None, action_discretizer=None, epsilon=0.1, decouple=False,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            policy_network: The Q-network that outputs Q-values
            encoder: Optional encoder for visual inputs
            action_discretizer: Converts discrete actions to continuous
            epsilon: Exploration probability
            decouple: Whether using decoupled discretization
            device: Torch device
        """
        self.policy_network = policy_network
        self.encoder = encoder
        self.action_discretizer = action_discretizer
        self.epsilon = epsilon
        self.decouple = decouple
        self.device = device

        # Set to evaluation mode by default
        self.policy_network.eval()
        if self.encoder:
            self.encoder.eval()

    def select_action(self, observation):
        """
        Select action using epsilon-greedy policy.

        Args:
            observation: Raw observation from environment

        Returns:
            Continuous action for the environment
        """
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
        elif isinstance(observation, torch.Tensor):
            if observation.device != self.device:
                observation = observation.to(self.device)
            if len(observation.shape) < 2 or (self.encoder and len(observation.shape) == 3):
                observation = observation.unsqueeze(0)

        with torch.no_grad():
            # Encode observation if using vision
            if self.encoder:
                encoded_obs = self.encoder(observation)
            else:
                encoded_obs = observation.flatten(1)

            # Get Q-values
            q1, q2 = self.policy_network(encoded_obs)

            # Select discrete action using epsilon-greedy
            discrete_action = self._epsilon_greedy_action((q1, q2))

            # Convert to continuous action
            if self.action_discretizer:
                continuous_action = self.action_discretizer.discrete_to_continuous(
                    discrete_action)[0]  # Remove batch dimension
                return continuous_action.detach()
            else:
                return discrete_action[0]

    def _epsilon_greedy_action(self, q_values):
        """Apply epsilon-greedy policy to Q-values."""
        q1, q2 = q_values
        q_combined = torch.max(q1, q2)
        batch_size = q_combined.shape[0]

        if self.decouple:
            # For decoupled actions, each dimension is selected independently
            actions = torch.zeros(batch_size, q_combined.shape[1], dtype=torch.long, device=self.device)
            for dim in range(q_combined.shape[1]):
                if random.random() < self.epsilon:
                    actions[:, dim] = torch.randint(0, q_combined.shape[2], (batch_size,), device=self.device)
                else:
                    actions[:, dim] = q_combined[:, dim].argmax(dim=1)
            return actions
        else:
            # Standard epsilon-greedy
            if random.random() < self.epsilon:
                return torch.randint(0, q_combined.shape[1], (batch_size,), device=self.device)
            else:
                return q_combined.argmax(dim=1)